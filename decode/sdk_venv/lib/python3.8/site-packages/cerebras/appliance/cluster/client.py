#!/usr/bin/env python3
# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

""" Client implementation for interacting with Cluster Management."""

import copy
import datetime
import functools
import getpass
import json
import logging
import os
import re
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import timezone
from os import path
from pathlib import Path
from queue import Queue
from typing import (
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import grpc
import jwt
import psutil
from google.protobuf.json_format import MessageToJson
from packaging.version import parse as parse_version

from cerebras.appliance.cluster import WORKFLOW_ID, cluster_logger
from cerebras.appliance.cluster.config import get_cs_cluster_config
from cerebras.appliance.cluster.job_timer import JobTimer
from cerebras.appliance.cluster.surrogate_job import get_surrogate_job
from cerebras.appliance.cluster.types import MountDir
from cerebras.appliance.cluster_config import NotificationTarget
from cerebras.appliance.errors import (
    ClusterJobCancelledByCsctl,
    ClusterJobInitError,
    ClusterJobScheduleError,
)
from cerebras.appliance.pb.workflow.appliance.cluster_mgmt import cluster_pb2
from cerebras.appliance.pb.workflow.appliance.cluster_mgmt.cluster_pb2 import (
    JobEvent,
    JobStatus,
    UserNotification,
)
from cerebras.appliance.pb.workflow.appliance.cluster_mgmt.cluster_pb2_grpc import (
    ClusterManagementStub,
)
from cerebras.appliance.pb.workflow.appliance.cluster_mgmt.csctl import api_pb2
from cerebras.appliance.pb.workflow.appliance.cluster_mgmt.csctl.api_pb2_grpc import (
    CsCtlV1Stub,
)
from cerebras.appliance.pb.workflow.appliance.cluster_mgmt.csctl.v1.resources_pb2 import (
    JobList,
    VolumeList,
)
from cerebras.appliance.pb.workflow.appliance.common.common_config_pb2 import (
    ClusterDetails,
    DebugArgs,
    JobMode,
)
from cerebras.appliance.utils import Tracker

logger = cluster_logger.getChild("client")


class MissingVolumeError(RuntimeError):
    """Exception to indicate no valid cluster volume for the user node."""


@dataclass
class HeartBeatOptions:
    """Options to control appliance heartbeat signals."""

    cycle_seconds: int = 10
    lease_duration_seconds_override: int = 0

    def __post_init__(self) -> None:
        if self.cycle_seconds <= 0:
            raise ValueError(
                f"`cycle_seconds` must be greater than 0. "
                f"Got {self.cycle_seconds}."
            )
        if self.lease_duration_seconds_override < 0:
            raise ValueError(
                f"`lease_duration_seconds_override` must be no less than 0. "
                f"Got {self.lease_duration_seconds_override}."
            )


class UserSidecarEnv(NamedTuple):
    """
    Configurations to be set in the user sidecar container.
    """

    mount_dir: Optional[MountDir]
    python_path: str


CHUNK_SIZE = 1024 * 100
VENV_CLUSTER_VOLUME_TAG = "allow-venv"

RETRY_POLICY = {
    "methodConfig": [
        {
            "name": [
                {"service": "cluster.cluster_mgmt_pb.ClusterManagement"},
                {"service": "cluster.cluster_mgmt_pb.csctl.CsCtlV1"},
            ],
            "retryPolicy": {
                "maxAttempts": 5,
                "initialBackoff": "2s",
                "maxBackoff": "10s",
                "backoffMultiplier": 2,
                "retryableStatusCodes": [
                    "UNAVAILABLE",
                    "UNKNOWN",
                    "RESOURCE_EXHAUSTED",
                ],
            },
        }
    ]
}

KEEPALIVE = [
    # Send keepalive ping after 5 minutes idle
    ('grpc.keepalive_time_ms', 300000),
    # Close connection after 30s not pingable
    ('grpc.keepalive_timeout_ms', 30000),
    # Allow unlimited pings without data in stream calls
    ('grpc.http2.max_pings_without_data', 0),
]

AUTH_UID = "auth-uid"
AUTH_GID = "auth-gid"
AUTH_USERNAME = "auth-username"
AUTH_TOKEN = "auth-token"
WORKFLOW_KEY = "workflow-id"
CLIENT_BUILD_VERSION = "client-build-version"

# The actual user_auth_enabled will be obtained through GetServerConfig call.
is_user_auth_enabled = False


class _ClientCallDetailsFields(NamedTuple):
    method: str
    timeout: Optional[float]
    metadata: Optional[Sequence[Tuple[str, Union[str, bytes]]]]
    credentials: Optional[grpc.CallCredentials]
    wait_for_ready: Optional[bool]
    compression: Any


class ClientCallDetails(_ClientCallDetailsFields, grpc.ClientCallDetails):
    """Describes an RPC to be invoked.
    See https://grpc.github.io/grpc/python/grpc.html#grpc.ClientCallDetails.
    """


class JWToken:
    """
    Encoded JWT with helpers for decoding and returning properties.
    """

    def __init__(self, encoded: str):
        self._encoded = encoded
        if encoded:
            # Decode into a dict without verifying the HMAC
            self._decoded = jwt.decode(
                encoded, options={"verify_signature": False}
            )
        else:
            # Empty encoded token is empty claim
            self._decoded = {}

    def __str__(self):
        return self._encoded

    def will_expire_within(self, seconds: int):
        expiration_time = self._decoded.get("exp", 0)
        return expiration_time < time.time() + seconds


class ClientInterceptor(grpc.UnaryUnaryClientInterceptor):
    def __init__(self, workflow_id):
        def default_username() -> str:
            try:
                return getpass.getuser()
            # pylint: disable=broad-except
            except Exception as ex:
                logging.info(f"failed call to getpass.getuser: {ex}")
                return f"pid.{os.getpid()}"

        self._workflow_id = workflow_id
        self._auth_metadata = [
            (AUTH_UID, str(os.getuid())),
            (AUTH_GID, str(os.getgid())),
            (AUTH_USERNAME, default_username()),
        ]
        if token := os.environ.get("CSAUTH_TOKEN", ""):
            self.token = JWToken(token)
        else:
            self.token = None

    def set_workflow_id(self, workflow_id):
        self._workflow_id = workflow_id

    def intercept_unary_unary(self, continuation, client_call_details, request):
        """Intercept all gRPC requests to add uid/gid/workflow.

        Params:
            continuation: A function that proceeds with the invocation by executing the next
                interceptor in the chain or invoking the actual RPC on the underlying
                channel.
            request: RPC request message.
            call_details: Describes an RPC to be invoked.

        Returns:
            The type of the return should match the type of the return value received
            by calling `continuation`. This is an object that is both a
            `Call <https://grpc.github.io/grpc/python/grpc.html#grpc.Call>`_ for the
            RPC and a `Future <https://grpc.github.io/grpc/python/grpc.html#grpc.Future>`_.
        """

        if self.token and self.token.will_expire_within(60):
            # Check the expiration date. If there is less than a minute left,
            # renew the JWT by expiring this token.
            self.token = None

        if not self.token and is_user_auth_enabled:
            get_cerebras_token_binary = "/usr/local/bin/get-cerebras-token"

            if not os.path.exists(get_cerebras_token_binary):
                raise RuntimeError(
                    f"UserAuth is enabled in the cluster but "
                    f"{get_cerebras_token_binary} does not exist on the "
                    f"user node. Please upgrade the user nodes."
                )

            try:
                output = subprocess.check_output(
                    [get_cerebras_token_binary]
                ).decode()
                if not output.startswith("Token="):
                    raise RuntimeError(
                        f"Unexpected token from "
                        f"{get_cerebras_token_binary}: {output}"
                    )
                self.token = JWToken(output[len("Token=") :])
            except subprocess.CalledProcessError as exp:
                raise RuntimeError(
                    f"Failed to call {get_cerebras_token_binary}: {exp}"
                )
        token = str(self.token) if self.token else ""
        interceptor_headers = self._auth_metadata.copy()
        interceptor_headers.append((AUTH_TOKEN, token))
        interceptor_headers.append((WORKFLOW_KEY, self._workflow_id))

        from cerebras.appliance import __version__

        interceptor_headers.append((CLIENT_BUILD_VERSION, __version__))

        new_details = ClientCallDetails(
            client_call_details.method,
            client_call_details.timeout,
            interceptor_headers,
            client_call_details.credentials,
            client_call_details.wait_for_ready,
            client_call_details.compression,
        )

        return continuation(new_details, request)


class ClusterManagementClient:
    """
    Cluster Management Client library that defines the interfaces to Cerebras appliance.
    """

    def __init__(
        self,
        server=None,
        crt_file=None,
        namespace="",
        workflow_id="",
        job_timer: JobTimer = None,
        enable_client_lease_strategy=True,
        heartbeat_options=HeartBeatOptions(),
        options=None,
        workdir=None,
        tracker_execute=None,
        fabric_type_blacklist: Optional[List[str]] = None,
        notifications: Optional[List[NotificationTarget]] = None,
    ):
        csconfig_path, self.cs_cluster_config = get_cs_cluster_config()
        if namespace:
            self.namespace = namespace
            if not self.cs_cluster_config.namespace_exists(self.namespace):
                if not crt_file:
                    raise RuntimeError(
                        f"Usernode config {csconfig_path} does not have access to the "
                        f"{self.namespace} namespace. Please contact sysadmins for support."
                    )
                else:
                    logger.info(
                        f"Certificate {crt_file} is used to access the {namespace} namespace."
                    )
                    self._crt_bytes = Path(crt_file).read_bytes()
            else:
                self._crt_bytes = (
                    self.cs_cluster_config.get_namespace_certificate_authority(
                        self.namespace
                    ).certificate_authority_data
                )
        elif self.cs_cluster_config.namespaces:
            if len(self.cs_cluster_config.namespaces) == 1:
                self.namespace = self.cs_cluster_config.namespaces[0].name
                self._crt_bytes = (
                    self.cs_cluster_config.get_namespace_certificate_authority(
                        self.namespace
                    ).certificate_authority_data
                )
                logger.debug(
                    f"Defaulted to use the {self.namespace} namespace as the usernode config {csconfig_path} "
                    "only has access to that namespace."
                )
            elif len(self.cs_cluster_config.namespaces) == 0:
                raise RuntimeError(
                    f"Usernode config {csconfig_path} does not have access to any namespace. "
                    f"Please contact sysadmins for support."
                )
            else:
                namespace_names = [
                    namespace.name
                    for namespace in self.cs_cluster_config.namespaces
                ]
                raise RuntimeError(
                    f"Usernode config {csconfig_path} has access to multiple namespaces. "
                    f"Please select a namespace with the '--mgmt_namespace' option with one of {namespace_names}."
                )
        else:
            # This case most likely happens in local testing.
            # "namespace" was not provided and usernode config does not have access to any namespace
            self.namespace = 'job-operator'
            logger.warning(
                f"Defaulted to use the {self.namespace} namespace as it appears to be a dev environment."
            )
            if not crt_file:
                logger.warning(
                    "TLS certificate was not provided. Attempting to communicate with "
                    "the appliance in non-TLS insecure mode."
                )
                self._crt_bytes = b''
            else:
                logger.info(
                    f"Certificate {crt_file} is used to access the {self.namespace} namespace."
                )
                self._crt_bytes = Path(crt_file).read_bytes()

        if (
            fabric_type_blacklist is not None
            and self.cs_cluster_config.fabric_type in fabric_type_blacklist
        ):
            raise RuntimeError(
                f"Fabric type '{self.cs_cluster_config.fabric_type}' is not supported"
            )
        self.authority = f"{self.namespace}.{self.cs_cluster_config.authority}"

        self.server = server
        if not self.server:
            self.server = self.cs_cluster_config.mgmt_address
        if options is None:
            options = []
        self.enable_client_lease_strategy = enable_client_lease_strategy
        self.heartbeat_options = heartbeat_options
        self._heartbeat_threads = {}
        self._heartbeat_responses = {}
        self._heartbeat_stops = {}
        self.options = options
        self.options.append(('grpc.enable_retries', 1))
        self.options.append(('grpc.service_config', json.dumps(RETRY_POLICY)))
        self.options.extend(KEEPALIVE)
        # only add authority when crt specified
        if self._crt_bytes and self.authority:
            self.options.append(('grpc.default_authority', self.authority))
            # We usually connect via IP, but nginx-ingress is set up to do SNI
            # dependent cert lookup. In gRPC versions > 1.39, this is set
            # automatically to the gprc.default_authority, but in older
            # versions it needs to be manually set.
            self.options.append(
                ('grpc.ssl_target_name_override', self.authority)
            )
        self.channel = None
        self.stub = None
        self.csctl_stub = None
        self.hostname = socket.gethostname()
        self.workdir = Path(workdir) if workdir is not None else Path.cwd()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.job_events = []
        self.job_timer = job_timer
        self.notifications = notifications
        # When using subdirectories, we expect artifact_dir to look like this:
        # .../cerebras_logs/<train|eval>/<timestamp>/executors/<id>/
        # run_meta.json is a shared file across all runs so it needs to be outside
        # of all subdirectories.
        if self.workdir.parent.parent.parent.parent.name == "cerebras_logs":
            self.run_meta_file = (
                self.workdir.parent.parent.parent.parent / "run_meta.json"
            )
        # In the new Trainer flow, there is no train/eval subdirectory.
        # So, the artifact directory is expected to look like this instead:
        # .../cerebras_logs/<timestamp>/executors/<id>/
        elif self.workdir.parent.parent.parent.name == "cerebras_logs":
            self.run_meta_file = (
                self.workdir.parent.parent.parent / "run_meta.json"
            )
        else:
            self.run_meta_file = self.workdir / "run_meta.json"
        self.run_meta_file = str(self.run_meta_file)
        self.image_build_log_file = path.join(
            self.workdir, f"custom_worker_build.out"
        )
        self._lock = threading.Lock()
        self._wsjobs = []
        self._image_jobs = []
        self.grpc_fork_support_value = None
        self.is_message_broker_available = False
        self._workflow_id = workflow_id if workflow_id else WORKFLOW_ID
        self._interceptor = ClientInterceptor(self._workflow_id)

        self.server_config = None

        # create a dummy tracker if one is not passed
        self._tracker_execute = tracker_execute or Tracker()

        logger.debug(
            f"ClusterClient"
            f": server={self.server}"
            f", authority={self.authority}"
            f", cert={'EMPTY' if not self._crt_bytes else 'OMITTED'}"
            f", workflow_id={self._workflow_id}"
            f", enable-client-lease-strategy={self.enable_client_lease_strategy}"
            f", heartbeat_options={self.heartbeat_options}"
            f", options={self.options}"
        )

        self.surrogate_job = get_surrogate_job(logger, self.workdir)

    def __enter__(self):
        self.validate_connectivity()
        self.grpc_fork_support_value = os.environ.get(
            'GRPC_ENABLE_FORK_SUPPORT', None
        )
        # SW-89390: To suppress the spam messages from gRPC library
        os.environ.update({'GRPC_ENABLE_FORK_SUPPORT': '0'})
        self.connect()
        self.server_config = self.get_server_config()
        self.check_user_auth_enabled()
        self.check_message_broker_availability()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_heartbeat()
        self.stop_surrogate()

        self.server_config = None

        if self.job_timer:
            self.job_timer.unregister_client(self)

        self.channel.close()
        if self.grpc_fork_support_value is not None:
            os.environ.update(
                {'GRPC_ENABLE_FORK_SUPPORT': self.grpc_fork_support_value}
            )
        else:
            os.environ.pop('GRPC_ENABLE_FORK_SUPPORT', None)

    def connect(self):
        if self._crt_bytes:
            creds = grpc.ssl_channel_credentials(self._crt_bytes)
            channel = grpc.secure_channel(self.server, creds, self.options)
        else:
            channel = grpc.insecure_channel(self.server, self.options)
        self.channel = grpc.intercept_channel(channel, self._interceptor)
        self.stub = ClusterManagementStub(self.channel)
        self.csctl_stub = CsCtlV1Stub(self.channel)

    def _is_ingress_ready(self, response, job_id):
        if response.is_scheduled:
            self._tracker_execute.stop("scheduler_wait")
            if response.is_ready:
                logger.info(f"Poll ingress success: {response.message}")
                self.dump_job_meta(job_id=job_id)
                return True

        if response.job_failed:
            logger.error(
                f"Job {job_id} failed during initialization: {response.message}"
            )
            if not response.is_scheduled:
                raise ClusterJobScheduleError(job_id, response.message)
            else:
                raise ClusterJobInitError(job_id, response.message)
        return False

    def _poll_ingress_readiness(self, job_id):
        self._tracker_execute.start("scheduler_wait")
        if (
            self.heartbeat_has_job_status()
            and self.enable_client_lease_strategy
        ):
            last_message = ""
            sleep_times = 0
            while True:
                with self._lock:
                    response = self._heartbeat_responses[job_id]
                    # print message every 10 minutes or status update
                    if last_message != response.message or sleep_times >= 60:
                        last_message = response.message
                        sleep_times = 0
                        logger.info(f"Poll ingress status: {response.message}")
                        if response.is_scheduled:
                            self._log_scheduling_events(job_id)
                    if self._is_ingress_ready(response, job_id):
                        return response
                time.sleep(10)
                sleep_times += 1

        # for backwards compatible/testing, todo: remove at rel-2.8/update tests
        retries = 0
        while True:
            # request grpc error will auto retry based on our policy
            responses = self.stub.PollIngressV2(
                cluster_pb2.GetIngressRequest(job_id=job_id),
                wait_for_ready=True,
            )
            # stream iteration is not caught as grpc retryable error, need explicit retry
            try:
                for response in responses:
                    logger.info(f"Poll ingress status: {response.message}")
                    if response.is_scheduled:
                        self._log_scheduling_events(job_id)
                    if self._is_ingress_ready(response, job_id):
                        return response
                    retries = 0
            except grpc.RpcError as e:
                # retry for 1min since it should not happen
                if retries < 12:
                    logger.warning(
                        f"Retry on poll ingress error: {e.code()}, {e.details()}"
                    )
                    retries += 1
                    # give enough time in case of server restart
                    time.sleep(5)
                    continue
                logger.error(f"Poll ingress for {job_id} failed: {e.details()}")
                raise e

    def _log_scheduling_events(self, job_id):
        job_events_resp = self.stub.GetJobEvents(
            cluster_pb2.GetJobEventsRequest(job_id=job_id),
        )
        for event in job_events_resp.job_events:
            msg = str(
                f"Event {event.lastTimestamp} reason={event.reason.strip()} "
                f"wsjob={event.name.strip()} message='{event.message.strip()}'"
            )
            if msg not in self.job_events:
                logger.warning(msg)
                self.job_events.append(msg)

    def get_active_jobs(self):
        with self._lock:
            return self._wsjobs.copy()

    def get_active_img_jobs(self):
        with self._lock:
            return self._image_jobs.copy()

    def _put_run_meta(self, run_meta_dict):
        with open(self.run_meta_file, "w+") as meta_file:
            json.dump(run_meta_dict, meta_file, indent=4)

    def _update_job_run_meta(self, job_id, new_prop_dict):
        run_meta_dict = self.get_run_meta()
        for key in run_meta_dict:
            for job in run_meta_dict[key]:
                if job["id"] == job_id:
                    job.update(new_prop_dict)
                    self._put_run_meta(run_meta_dict)
                    return

        logger.warning(
            f"There is no job record with the id {job_id} in run meta."
        )

    def get_image_build_log_path(self):
        return self.image_build_log_file

    def _put_image_build_logs(self, build_log_content):
        with open(self.image_build_log_file, "w+") as log_file:
            log_file.write(build_log_content)

    def start_heartbeat(self, lease_id) -> None:
        if not self.enable_client_lease_strategy:
            return

        # job level or workflow heartbeat
        if not lease_id:
            logger.warning("No job/workflow id for heartbeat, abort")
            return

        logger.debug(
            f"Starting heartbeat thread for {lease_id}. Heartbeat requests will be sent "
            f"every {self.heartbeat_options.cycle_seconds} seconds."
        )

        self._heartbeat_stops[lease_id] = threading.Event()
        self._heartbeat_responses[lease_id] = cluster_pb2.HeartbeatResponse()
        self._heartbeat_threads[lease_id] = HeartBeatThread(
            self.stub,
            copy.deepcopy(self.heartbeat_options),
            lease_id,
            self.hostname,
            self._heartbeat_stops[lease_id],
            self._lock,
            self._heartbeat_responses[lease_id],
            lease_id != self.workflow_id and self.heartbeat_has_job_status(),
        )
        self._heartbeat_threads[lease_id].start()

    def stop_heartbeat(self, job_id=None) -> None:
        """Command to stop heartbeats with cluster server."""
        if not self.enable_client_lease_strategy:
            return

        def _stop_hb(_job_id) -> None:
            if (
                _job_id not in self._heartbeat_threads
                or self._heartbeat_stops[_job_id].is_set()
            ):
                return
            logger.debug(f"Signaling heartbeat thread to stop for {_job_id}")
            self._heartbeat_stops[_job_id].set()
            self._heartbeat_threads[_job_id].join(
                self.heartbeat_options.cycle_seconds
            )

        if job_id:
            _stop_hb(job_id)
            return

        for job_id in self._heartbeat_threads:
            _stop_hb(job_id)

    def _record_new_job(
        self,
        workflow_id,
        job_id,
        mode,
        log_path,
        image_reference="",
        image_ready=False,
    ):
        with self._lock:
            self.log_job(
                job_id,
                mode,
                log_path,
                workflow_id,
            )
            if not image_reference:
                self._wsjobs.append(job_id)
            else:
                self._image_jobs.append(job_id)
                self._log_image_reference(
                    job_id,
                    image_reference,
                    image_ready,
                )

    def init_workflow(self, resource_reserve=False, override=False):
        """
        resource_reserve: First compile/execute job will be reserved if set.
        override: If True, start a new workflow even if one is already active (testing purposes).
        """
        logger.debug(
            f"Initiating a new workflow, resource_reserve: {resource_reserve}"
        )
        # workflow id is expected to be singleton for entire run.py lifecycle
        if (
            self._workflow_id
            and self._workflow_id != WORKFLOW_ID
            and not override
        ):
            raise Exception(f"Workflow id {self._workflow_id} exists already!")
        request = cluster_pb2.InitWorkflowRequest(
            resource_reserve=resource_reserve,
        )
        init_response = self.stub.InitWorkflow(request)
        self._workflow_id = init_response.workflow_id
        self._interceptor.set_workflow_id(self._workflow_id)
        self.start_heartbeat(self._workflow_id)
        logger.debug(f"Workflow {self._workflow_id} init success.")

    @property
    def workflow_id(self):
        return self._workflow_id

    def check_storage_connectivity(self, endpoint_url):
        request = cluster_pb2.CheckStorageConnectivityRequest()
        request.s3_endpoint.url = endpoint_url
        response = self.stub.CheckStorageConnectivity(request)

        if not response.is_reachable:
            raise RuntimeError(
                f"Storage endpoint {endpoint_url} is not reachable due to:\n"
                f"{response.message}"
            )

    def init_compile_job(
        self,
        compile_dir_relative_path,
        job_mode,
        debug_args: DebugArgs,
        # This is only a test handle and will not be used in production
        skip_ingress_creation=False,
    ) -> Dict[str, Any]:
        if self.surrogate_job:
            (label_key, label_value) = self.surrogate_job.get_job_label()
            debug_args.debug_mgr.labels[label_key] = label_value

        request = cluster_pb2.CompileJobRequest(
            compile_dir_relative_path=compile_dir_relative_path,
            job_mode=job_mode,
            client_workdir=f"{self.hostname}:{self.workdir.absolute()}",
            debug_args=debug_args,
            debug_args_json=MessageToJson(
                debug_args, sort_keys=True, indent=None
            ),
            user_notifications=_create_user_notifications(self.notifications),
        )
        logger.info(
            "Initiating a new compile wsjob against the cluster server."
        )
        init_response = self.stub.InitCompileJob(request)
        self._record_new_job(
            init_response.workflow_id,
            init_response.job_id,
            job_mode.job,
            init_response.log_path,
        )

        job_props = {
            "compile_dir_absolute_path": init_response.compile_dir_absolute_path,
        }
        return self.get_job_handle(
            init_response.job_id, "", job_props, skip_ingress_creation
        )

    def init_execute_job(
        self,
        compile_dir_relative_path,
        compile_artifact_dir,
        job_mode,
        debug_args: DebugArgs,
        mount_dirs: Optional[List[MountDir]] = None,
        python_paths: Optional[List[str]] = None,
        user_sidecar_env: Optional[UserSidecarEnv] = None,
        # This is only a test handle and will not be used in production
        skip_ingress_creation=False,
    ) -> Dict[str, Any]:
        if self.surrogate_job:
            (label_key, label_value) = self.surrogate_job.get_job_label()
            debug_args.debug_mgr.labels[label_key] = label_value

        if mount_dirs:
            mount_dirs = [
                cluster_pb2.MountDir(**md._asdict()) for md in mount_dirs
            ]

        request = cluster_pb2.ExecuteJobRequest(
            compile_dir_relative_path=compile_dir_relative_path,
            compile_artifact_dir=compile_artifact_dir,
            job_mode=job_mode,
            debug_args=debug_args,
            client_workdir=f"{self.hostname}:{self.workdir.absolute()}",
            mount_dirs=mount_dirs,
            python_paths=python_paths,
            debug_args_json=MessageToJson(
                debug_args, sort_keys=True, indent=None
            ),
            user_notifications=_create_user_notifications(self.notifications),
        )

        if user_sidecar_env is not None:
            request.user_sidecar_env.python_path = user_sidecar_env.python_path
            if user_sidecar_env.mount_dir is not None:
                request.user_sidecar_env.mount_dir.CopyFrom(
                    cluster_pb2.MountDir(**user_sidecar_env.mount_dir._asdict())
                )

        logger.info(
            "Initiating a new execute wsjob against the cluster server."
        )
        with self._tracker_execute.entry("job_initializing"):
            init_response = self.stub.InitExecuteJob(request)
        self._record_new_job(
            init_response.workflow_id,
            init_response.job_id,
            job_mode.job,
            init_response.log_path,
        )

        job_props = {
            "compile_artifact_dir": compile_artifact_dir,
        }

        # Get the number of CS2s so we can append it to the SLURM surrogate job name
        num_cs2 = 0
        for task in job_mode.cluster_details.tasks:
            if task.task_type == ClusterDetails.TaskInfo.TaskType.WSE:
                num_cs2 = len(task.task_map)
                break
        if num_cs2 > 0:
            num_cs2_str = "-csx" + str(num_cs2)
        else:
            num_cs2_str = ""

        return self.get_job_handle(
            init_response.job_id,
            num_cs2_str,
            job_props,
            skip_ingress_creation,
        )

    def init_sdk_compile_job(
        self,
        job_mode,
        compile_dir_relative_path,
        debug_args=None,
        skip_ingress_creation=False,
    ):
        if job_mode.job != JobMode.Job.SDK_COMPILE:
            raise RuntimeError(
                f"Unexpected job {job_mode.job} (expecting SDK_COMPILE)"
            )

        request = cluster_pb2.SdkCompileJobRequest(
            job_mode=job_mode,
            compile_dir_relative_path=compile_dir_relative_path,
            debug_args=debug_args,
        )
        logger.info(
            "Initiating a new SDK compile job against the cluster server"
        )
        init_response = self.stub.InitSdkCompileJob(request)
        self._record_new_job(
            init_response.workflow_id,
            init_response.job_id,
            job_mode.job,
            init_response.log_path,
        )
        return self.get_job_handle(
            init_response.job_id, "", {}, skip_ingress_creation
        )

    def init_sdk_execute_job(
        self, job_mode, debug_args=None, skip_ingress_creation=False
    ):
        if job_mode.job != JobMode.Job.SDK_EXECUTE:
            raise RuntimeError(
                f"Unexpected job {job_mode.job} (expecting SDK_EXECUTE)"
            )

        request = cluster_pb2.SdkExecuteJobRequest(
            job_mode=job_mode,
            debug_args=debug_args,
        )
        logger.info(
            "Initiating a new SDK execute job against the cluster server"
        )
        init_response = self.stub.InitSdkExecuteJob(request)
        self._record_new_job(
            init_response.workflow_id,
            init_response.job_id,
            job_mode.job,
            init_response.log_path,
        )
        return self.get_job_handle(
            init_response.job_id, "", {}, skip_ingress_creation
        )

    def init_image_build_job(
        self,
        debug_args: DebugArgs,
        pip_options: Optional[str] = None,
        frozen_dependencies: Optional[List[str]] = None,
        base_image_override: Optional[str] = None,
    ) -> cluster_pb2.ImageBuildResponse:
        request = cluster_pb2.InitImageBuildRequest(
            pip_options=pip_options,
            frozen_dependencies=frozen_dependencies,
            base_image_override=base_image_override,
            debug_args=debug_args,
        )
        logger.info(
            f"Initiating a new image build job against the cluster server."
        )
        init_response = self.stub.InitImageBuildJob(request)
        self._record_new_job(
            init_response.workflow_id,
            init_response.job_id,
            JobMode.Job.IMAGE_BUILD,
            init_response.log_path,
            image_reference=init_response.image_reference,
            image_ready=init_response.image_ready,
        )
        return init_response

    def get_server_versions(self) -> Dict[str, str]:
        """Return a map of server versions, including cluster-server and job-operator."""
        request = cluster_pb2.GetServerVersionsRequest()
        response = self.stub.GetServerVersions(request)
        return response.versions

    def get_job_handle(
        self,
        job_id,
        num_cs2_suffix,
        job_props,
        skip_ingress_creation,
    ) -> Dict[str, Any]:
        self.start_heartbeat(job_id)

        response = {"job_id": job_id}
        if skip_ingress_creation:
            return response

        ingress_response = self._poll_ingress_readiness(job_id)
        response["service_authority"] = f"{ingress_response.service_authority}"
        response["service_url"] = self.server
        # check if returned endpoint reachable in case nginx pod not exists on crd node
        # fall back to default address(VIP)
        if ingress_response.service_url and self.validate_connectivity(
            ingress_response.service_url, False
        ):
            response["service_url"] = str(ingress_response.service_url)
        response["certificate_bytes"] = self._crt_bytes

        # Append the number of CSX
        self.start_surrogate(job_id, num_cs2_suffix)

        logger.debug(f"Cluster mgmt job handle: {response}")

        if self.job_timer:
            self.job_timer.register_client(self)

        response.update(job_props)
        return response

    def list_workflow_jobs(self):
        compile_jobs = []
        execute_jobs = []
        job_list = JobList()
        result = self.csctl_stub.Get(
            api_pb2.GetRequest(
                type="jobs",
                accept=api_pb2.PROTOBUF_METHOD,
                representation=api_pb2.OBJECT_REPRESENTATION,
                options=api_pb2.GetOptions(
                    namespace=self.namespace,
                    workflow_id=self._workflow_id,
                    all_states=True,
                    current_user_only=False,
                    sort_by="age",
                ),
            )
        )
        job_list.ParseFromString(result.raw)
        logger.debug(f"Workflow jobs: {job_list}")
        for job in job_list.items:
            if "compile" in job.spec.type:
                compile_jobs.append(job.meta.name)
            else:
                execute_jobs.append(job.meta.name)
        return compile_jobs, execute_jobs

    def reserve_job_resources(self, job_id):
        request = cluster_pb2.ReserveJobResourcesRequest(
            job_id=job_id,
        )
        logger.debug(f"Reserving job resource: {job_id}")
        response = self.stub.ReserveJobResources(request)
        logger.debug(response.message)
        return response

    def release_job_resources(self, job_id):
        request = cluster_pb2.ReleaseJobResourcesRequest(
            job_id=job_id,
        )
        logger.debug(f"Releasing reserved job resource: {job_id}")
        response = self.stub.ReleaseJobResources(request)
        logger.debug(response.message)
        return response

    def release_workflow_resources(self):
        request = cluster_pb2.ReleaseWorkflowResourcesRequest(
            workflow_id=self.workflow_id,
        )
        logger.debug(
            f"Releasing reserved workflow resource: {self.workflow_id}"
        )
        response = self.stub.ReleaseWorkflowResources(request)
        logger.debug(response.message)
        return response

    def get_reservation_status(self, job_id):
        request = cluster_pb2.GetReservationStatusRequest(
            job_id=job_id,
        )
        response = self.stub.GetReservationStatus(request)
        logger.debug(f"Reservable status: {response}")
        return response.system_count

    def get_run_meta(self):
        run_meta_dict = {}
        if path.exists(self.run_meta_file):
            try:
                with open(self.run_meta_file, "r") as meta_file:
                    run_meta_dict = json.load(meta_file)
            except:
                pass
        return run_meta_dict

    def dump_job_meta(self, job_id):
        """Retrieve fully populated cluster details, software versions and system versions."""
        request = cluster_pb2.GetJobMetaRequest(job_id=job_id)
        response = self.stub.GetJobMeta(request)

        with open(self.workdir / f"{job_id}-cluster-details.json", "w") as f:
            f.write(MessageToJson(response.cluster_details, indent=4))

        software_versions = dict(response.software_versions)
        from cerebras.appliance import __version__

        software_versions["appliance-client"] = __version__
        _sorted = {k: software_versions[k] for k in sorted(software_versions)}
        self._update_job_run_meta(job_id, {"software_versions": _sorted})

        system_versions = dict(response.system_versions)
        _sorted = {}
        for k in sorted(system_versions):
            serialized_software_versions = system_versions[k]
            system_version_details = json.loads(serialized_software_versions)
            attrs = ["product", "execmode", "components"]
            _sorted[k] = {
                _k: system_version_details[_k]
                for _k in attrs
                if _k in system_version_details
            }
        self._update_job_run_meta(job_id, {"system_versions": _sorted})

    def log_job(
        self,
        job_id,
        job_mode,
        log_path,
        workflow_id="",
    ):
        """
        Write various job metadata (including namespace, job_id, workflow_id) to run_meta.json.
        """
        run_meta_dict = self.get_run_meta()
        category = "jobs"
        if job_mode == JobMode.Job.IMAGE_BUILD:
            category = "image_build_jobs"
        elif "compile" in JobMode.Job.Name(job_mode).lower():
            category = "compile_jobs"
        elif "execute" in JobMode.Job.Name(job_mode).lower():
            category = "execute_jobs"
        if category not in run_meta_dict:
            run_meta_dict[category] = []

        run_meta_dict[category].append(
            {
                "namespace": self.namespace,
                "id": job_id,
                "workflow_id": workflow_id,
                "log_path": log_path,
                "start_time": _get_curr_time(),
            }
        )

        self._put_run_meta(run_meta_dict)

        logger.debug(f"Run meta is available at {self.run_meta_file}.")
        if job_id and log_path:
            workflow_info = (
                f" workflow id: {workflow_id}," if workflow_id else ""
            )
            logger.info(
                f"Job id: {job_id},"
                f"{workflow_info} "
                f"namespace: {self.namespace}, "
                f"remote log path: {log_path}"
            )

    def log_cache_compile(
        self, job_id, cache_compile_dir, cluster_callback=True
    ):
        prop_dict = {
            "cache_compile": {
                "location": cache_compile_dir,
                "available_time": _get_curr_time(),
            },
        }
        self._update_job_run_meta(job_id, prop_dict)
        if not cluster_callback:
            return
        try:
            self.stub.FinalizeCompileJob(
                cluster_pb2.CompileResultRequest(
                    job_id=job_id,
                    compile_artifact_dir=cache_compile_dir,
                )
            )
        except grpc.RpcError as exp:
            logger.warning(f"Unable to finalize compile job {job_id}: {exp}")

    def log_debugviz_url(self, job_id):
        if (
            self.server_config is not None
            and self.server_config.debugviz_url_prefix
        ):
            prop_dict = {
                "debugviz_url": f"{self.server_config.debugviz_url_prefix}/{self.namespace}/{job_id}",
            }
            self._update_job_run_meta(job_id, prop_dict)

    def _log_image_reference(self, job_id, image_ref, image_ready):
        prop_dict = {
            "image": {
                "reference": image_ref,
                "ready": image_ready,
            },
        }
        if image_ready:
            prop_dict["image"]["available_time"] = _get_curr_time()
        self._update_job_run_meta(job_id, prop_dict)

    def gather_cluster_job_failure(self, job_id):
        """
        Best effort to gather job/pods failure from cluster side.
        Raise exception it's a csctl cancel caused failure to notify FW client.
        """
        num_tries = 0
        max_tries = 12
        sleep_time = 5
        timeout = max_tries * sleep_time
        logger.info(
            f"Trying to fetch failure info from the cluster for job {job_id}. "
            f"This may take up to {timeout} seconds."
        )
        try:
            # Exception can be thrown at the appliance client
            # before job status change on the server.
            # Wait for at most 60s to retrieve the pod failures.
            while True:
                response = self.get_job(job_id)
                if num_tries == 0 and response.dashboard:
                    logger.info(
                        f"For more info, please check on dashboard: {response.dashboard}, "
                        f"or run `csctl get job {job_id} -n{self.namespace} -oyaml`"
                    )
                if response.status == JobStatus.JOB_STATUS_IN_PROGRESS:
                    num_tries += 1
                    if num_tries < max_tries:
                        logger.debug(
                            f"Job {job_id} is still in progress. Retrying in "
                            f"{sleep_time} seconds."
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.info(
                            f"Job {job_id} is still in progress after "
                            f"{timeout} seconds. Abort fetching failure info."
                        )
                        break
                else:
                    logger.error(
                        f"Job {job_id} failed due to: {response.message}"
                    )
                    log_failed_job_events(response.job_events)
                    if (
                        response.status == JobStatus.JOB_STATUS_CANCELLED
                        and "canceled-by-csctl" in response.message
                    ):
                        raise ClusterJobCancelledByCsctl(response.message)
                    break
        except ClusterJobCancelledByCsctl:
            raise
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                f"Failed to fetch failure info for job {job_id}", exc_info=True
            )

    def get_latest_job(self):
        """
        Retrieve the latest job reference mostly for testing purposes.
        """
        run_meta_dict = self.get_run_meta()
        if not run_meta_dict:
            return None

        # Flattens the job list regardless of job types
        jobs = [job for job_list in run_meta_dict.values() for job in job_list]
        jobs_sorted_by_time = sorted(jobs, key=lambda x: x["start_time"])

        return None if not jobs_sorted_by_time else jobs_sorted_by_time[-1]

    def get_latest_execute_job(self) -> Optional[dict]:
        """
        Retrieve the latest execute job reference.
        """
        run_meta_dict = self.get_run_meta()
        job_mode_string = JobMode.Job.Name(JobMode.Job.EXECUTE).lower()
        if f"{job_mode_string}_jobs" not in run_meta_dict:
            logger.warning(
                f"There is no existing {job_mode_string} job record."
            )
            return None

        jobs = run_meta_dict[f"{job_mode_string}_jobs"]
        jobs_sorted_by_time = sorted(jobs, key=lambda x: x["start_time"])
        return None if not jobs_sorted_by_time else jobs_sorted_by_time[-1]

    def get_job(self, job_id):
        """Get Job Request."""
        request = cluster_pb2.GetJobRequest(
            job_id=job_id,
        )
        return self.stub.GetJob(request)

    def get_image_build_job(self, job_id, image_reference=None):
        """Get image build job request."""
        request = cluster_pb2.ImageBuildRequest(
            job_id=job_id,
            image_reference=image_reference,
        )
        response = self.stub.GetImageBuildJob(request)
        if response.image_ready:
            self._log_image_reference(
                response.job_id, response.image_reference, response.image_ready
            )
        if response.build_log_content:
            self._put_image_build_logs(response.build_log_content)
        return response

    def delete_job(self, job_id):
        """Delete existing Job Request, for testing only."""
        request = cluster_pb2.DeleteJobRequest(
            job_id=job_id,
        )
        return self.stub.DeleteJob(request)

    def delete_image_build_job(self, job_id):
        """Delete existing image job request, for testing only."""
        request = cluster_pb2.ImageBuildRequest(
            job_id=job_id,
        )
        response = self.stub.DeleteImageBuildJob(request)
        with self._lock:
            if job_id in self._image_jobs:
                self._image_jobs.remove(job_id)
        return response

    def cancel_job(self, job_id, job_status, message=""):
        """Cancel existing Job Request."""
        message = message.split("Stack Trace", 1)[0].strip()
        request = cluster_pb2.CancelJobRequest(
            job_id=job_id,
            job_status=job_status,
            message=message,
        )
        response = self.stub.CancelJobV2(request)
        with self._lock:
            if job_id in self._wsjobs:
                self._wsjobs.remove(job_id)
                if len(self._wsjobs) == 0 and self.job_timer:
                    self.job_timer.unregister_client(self)
        self.stop_heartbeat(job_id)
        return response

    def start_surrogate(self, job_id, num_cs2_suffix=""):
        """Start surrogate job."""
        if self.surrogate_job:
            # TODO(joy): should lift this up to the constructor later.
            namespace = self.namespace if self.namespace else 'job-operator'
            self.surrogate_job.start(job_id, namespace, num_cs2_suffix)

    def stop_surrogate(self):
        """Stop surrogate job."""
        if self.surrogate_job:
            # There is a time gap between the wsjob completing and the client exiting.
            # Here, we wait for all appliance jobs to complete before stopping the surrogate
            # jobs. Otherwise, the surrogate job might find the job in progress and try to
            # cancel it.
            # Note: this may need optimize to be called earlier for multiple jobs in one client.
            for job in self.surrogate_job.get_appliance_jobs():
                while job and True:
                    response = self.get_job(job)
                    logger.debug(
                        f"Wait for job {job} to complete. Current status {response.status}"
                    )
                    if response.status != cluster_pb2.JOB_STATUS_IN_PROGRESS:
                        break
                    time.sleep(1)
            self.surrogate_job.stop()

    def cancel(self):
        """Cancel all active jobs."""
        for job_id in self.get_active_jobs():
            self.cancel_job(job_id, "JOB_STATUS_CANCELLED")
        for job_id in self.get_active_img_jobs():
            self.delete_image_build_job(job_id)

    # check if address is accessible to avoid grpc infinite retry at connect time
    # this can also help validate the preferred CRD address has a healthy nginx endpoint
    def validate_connectivity(self, address=None, ping_only=True):
        """Validate the connectivity by ping or curl."""
        if not address:
            address = self.server

        # skip for test cases
        if "localhost" in address:
            return True

        if ping_only:
            address = address.split(':')[0]
            cmd = f"ping -c 1 -W 3 {address}"
        else:
            cmd = f"curl --connect-timeout 3 {address}"

        try:
            out = subprocess.run(cmd, capture_output=True, shell=True)
            if out.returncode == 0:
                return True

            if ping_only:
                err = f"Failed to ping cluster mgmt node: {address}, please check usernode network"
                logger.error(err)
                raise Exception(err)
            logger.debug(
                f"Failed to curl preferred cluster mgmt ingress svc: {address}, fallback to default server"
            )
            return False
        except subprocess.CalledProcessError as e:
            # there can be resource contention causing subprocess failed to launch
            logger.debug(
                f"Unable to run connectivity test: {e}, default to success as best effort only"
            )
            return True

    def check_message_broker_availability(self):
        response = self.stub.IsMessageBrokerAvailable(
            cluster_pb2.MessageBrokerAvailabilityCheckRequest()
        )
        self.is_message_broker_available = response.is_available

    def get_server_config(self):
        try:
            return self.stub.GetServerConfig(
                cluster_pb2.GetServerConfigRequest()
            )
        except grpc.RpcError as exp:
            if exp.code() == grpc.StatusCode.UNIMPLEMENTED:
                # Assume that user auth is not enabled, to support backward compatibility
                logger.error(
                    f"Unimplemented error occurred while getting server config: {exp}."
                )
                return None
            else:
                raise exp

    def check_user_auth_enabled(self):
        global is_user_auth_enabled
        if self.server_config is None:
            logger.warning(
                "Server config is not available. Assume user auth disabled."
            )
            is_user_auth_enabled = False
        else:
            is_user_auth_enabled = self.server_config.is_user_auth_enabled

    def is_sidecar_available(self):
        if (
            self.server_config is None
            or not self.server_config.job_operator_semantic_version
        ):
            return False
        else:
            # Sidecar is available starting from job operator 1.0.4 and higher
            return parse_version(
                self.server_config.job_operator_semantic_version
            ) >= parse_version("1.0.4")

    def is_mount_dir_dedupe_available(self):
        if (
            self.server_config is None
            or not self.server_config.job_operator_semantic_version
        ):
            return False
        else:
            # Mount dir dedupe is available starting from job operator 1.0.8 and higher
            return parse_version(
                self.server_config.job_operator_semantic_version
            ) >= parse_version("1.0.8")

    def is_debug_volume_available(self):
        if (
            self.server_config is None
            or not self.server_config.cluster_server_semantic_version
        ):
            return False
        else:
            # Debug volume is available starting from cluster server 1.0.8 and higher
            return parse_version(
                self.server_config.cluster_server_semantic_version
            ) >= parse_version("1.0.8")

    def heartbeat_has_job_status(self):
        if (
            self.server_config is None
            or not self.server_config.cluster_server_semantic_version
        ):
            return False
        else:
            return parse_version(
                self.server_config.cluster_server_semantic_version
            ) >= parse_version("1.0.10")

    def publish_messages(self, job_id, message_iterator, max_attempts=5):
        if not self.is_message_broker_available:
            logger.warning(
                "Message broker is not available hence skipping publishing messages."
            )
            return

        def request_generator():
            init_msg = cluster_pb2.PublishMessagesRequest(job_id=job_id)
            yield init_msg  # send the initialization message first
            for message in message_iterator:
                yield cluster_pb2.PublishMessagesRequest(message=message)

        for i in range(max_attempts):
            try:
                self.stub.PublishMessages(request_generator())
                logger.debug(
                    "Published all messages from the message iterator."
                )
                break
            except grpc.RpcError as e:
                logger.error(f"Error occurred while publishing messages: {e}")
                if i < max_attempts - 1:
                    time.sleep(2**i)
                else:
                    raise e

    def subscribe_messages(self, job_id, from_beginning, max_attempts=5):
        if not self.is_message_broker_available:
            logger.warning(
                "Message broker is not available hence skipping subscribing messages."
            )
            return

        consecutive_errors = 0
        while True:
            try:
                req = cluster_pb2.SubscribeMessagesRequest(
                    job_id=job_id, from_beginning=from_beginning
                )
                responses = self.stub.SubscribeMessages(req)
                consecutive_errors = 0
                for response in responses:
                    yield response.message
            except grpc.RpcError as e:
                logger.error(f"Error occurred while polling messages: {e}")
                if consecutive_errors < max_attempts:
                    consecutive_errors += 1
                    time.sleep(2**consecutive_errors)
                else:
                    raise e

    @functools.lru_cache
    def get_volumes(self, include_internal_volumes=False):
        """
        Retrieve cluster volumes.

        Prior to rel-2.5, the appliance backend does not support deduplication of
        mount directories. As a result, we need to filter out internal volumes
        that are not suitable.

        In rel-2.5 and later, the appliance backend supports deduplication of
        mount directories. We can include all volumes in the list.

        include_internal_volumes: Include internal volumes in the list for older
        appliance backends.
        """
        _volume_list = VolumeList()
        try:
            result = self.csctl_stub.Get(
                api_pb2.GetRequest(
                    type="volumes",
                    accept=api_pb2.PROTOBUF_METHOD,
                    representation=api_pb2.OBJECT_REPRESENTATION,
                    options=api_pb2.GetOptions(namespace=self.namespace),
                )
            )
            _volume_list.ParseFromString(result.raw)
        except Exception as e:
            raise RuntimeError(f"Failed to get volumes: {self.namespace}/{e}")

        volume_list = VolumeList()
        for volume in _volume_list.items:
            # TODO: Remove the backend check in rel-2.7
            if (
                not self.is_mount_dir_dedupe_available()
                and not include_internal_volumes
            ):
                if (
                    volume.meta.labels.get("cached-compile", None) == "true"
                    or volume.meta.labels.get("workdir-logs", None) == "true"
                    or volume.meta.labels.get(VENV_CLUSTER_VOLUME_TAG, None)
                    == "true"
                ):
                    continue
            volume_list.items.append(volume)
        return volume_list

    def get_user_venv_cluster_volume_path(
        self, venv_src: Path
    ) -> Tuple[bool, str]:
        """
        Retrieve cluster volume path for staging the user virtual environment over NFS.
        Return value is a tuple where the first element in the tuple indicates whether
        venv copying should be honored, and the second element is the cluster volume path
        that the user-node virtual environment should be replicated to.

        In the event where the user-node virtual environment is on a cluster volume path
        already, we do not need to replicate the environment again over NFS.
        """
        logger.debug(f"start checking venv with src: {venv_src}")
        volume_list = self.get_volumes(include_internal_volumes=True)
        try:
            for vol in volume_list.items:
                logger.debug(f"start validating cluster vol: {vol}")
                mount_path = (
                    vol.nfs.container_path
                    if vol.nfs.container_path
                    else vol.host_path.container_path
                )
                is_nfs_volume = True if vol.nfs.container_path else False
                if is_nfs_volume and str(venv_src).startswith(mount_path):
                    # Does not require replicating the virtual environment
                    # if the source venv is already on a NFS volume
                    return False, ""

                if vol.meta.labels.get(VENV_CLUSTER_VOLUME_TAG) != "true":
                    logger.debug(
                        f"Cluster volume {vol.meta.name} is not a venv-allowed volume, skip to check next volume"
                    )
                    continue

                if not Path(mount_path).exists():
                    logger.debug(
                        f"Cluster volume {vol.meta.name} is not set up on the user node: {mount_path}, "
                        f"skip to check next volume"
                    )
                    continue

                try:
                    cmd = f"df {mount_path} --output=fstype"
                    result = subprocess.run(
                        cmd.split(), capture_output=True, text=True, check=True
                    )
                    lines = str(result.stdout).strip().split("\n")
                    if len(lines) > 1:
                        fs_type = lines[1]
                        if fs_type.startswith('nfs') == is_nfs_volume:
                            logger.info(
                                f"User venv cluster volume {vol.meta.name} on {mount_path} validated."
                            )
                            return True, mount_path
                        else:
                            logger.warning(
                                f"{vol.meta.name} on {mount_path} is {fs_type} but inconsistent with cluster setup, "
                                f"skip to check next volume"
                            )
                except subprocess.CalledProcessError as e:
                    logger.warning(
                        f"An error occurred while checking the filesystem for {mount_path}: {e}"
                    )
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse cluster volume path for user venv: {e}"
            )

        raise MissingVolumeError(
            f"No valid cluster volume was set up to allow user venv mounting on this user node. "
            f"Please contact your organization's sysadmin to configure one."
        )

    def request_log_export_debug_artifacts(
        self, job_id, signature: Optional[Literal["onwafer"]] = None
    ):
        """log-export debug artifacts upon a job failure (stall, NaN, etc.)"""

        # future extensions, None means all servers
        servers_to_export = (
            "chief,coordinator" if signature == "onwafer" else None
        )

        logger.info(f"Requesting log-export debug artifacts for job {job_id}")

        try:
            timeout = 3600  # seconds
            create_request = api_pb2.CreateLogExportRequest(
                name=job_id,
                binaries=False,
                level=0,
                include_compile_artifacts=False,
                skip_cluster_logs=True,
                timeout_seconds=timeout,
                tasks=servers_to_export,
                copy_nfs_logs=True,
                export_target_type=api_pb2.ExportTargetType.EXPORT_TARGET_TYPE_DEBUG_VOLUME,
            )
            response = self.csctl_stub.CreateLogExport(create_request)

            get_request = api_pb2.GetLogExportRequest(
                export_id=response.export_id
            )

            # Note: this call does NOT block, this is only here to get the current log-export status
            # Expected log output is "RUNNING"
            response = self.csctl_stub.GetLogExport(get_request)
            status = json.loads(MessageToJson(response)).get(
                "status", "UNKNOWN"
            )
            logger.debug(
                f"Current log-export status for job {job_id}: {status}"
            )
        except Exception:
            logger.warning("Failed to export debug artifacts.", exc_info=True)


def _get_curr_time():
    return datetime.datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_failed_job_events(job_events: List[JobEvent], log=logger):
    """Friendly logging explaining JobEvents and their relevant config."""
    replica_type_mem_runconfig = {
        "activation": "act_memory_gi",
        "command": "cmd_memory_gi",
        "coordinator": "compile_crd_memory_gi, execute_crd_memory_gi",
        "weight": "wgt_memory_gi",
        "worker": "wrk_memory_gi",
    }
    pod_replica_type_re = re.compile(r"^wsjob-[^-\s]+-([^-\s]+)-\d+$")

    for event in job_events:
        msg = str(
            f"Event {event.lastTimestamp} reason={event.reason.strip()} "
            f"object={event.name.strip()} message='{event.message.strip()}'"
        )
        details = ""
        if event.reason == "OOMKilled":
            m = pod_replica_type_re.match(event.name)
            if m:
                rt = m.groups()[0]
                if rt in replica_type_mem_runconfig:
                    params = replica_type_mem_runconfig[rt]
                    details = str(
                        "You can attempt to override the default "
                        f"memory limit using runconfig params ({params}) and "
                        "check node capacity with `csctl get nodegroups`. "
                        "Note that increasing memory can lead to unschedulable "
                        "jobs."
                    )
        elif event.reason == "SchedulingFailed":
            params_str = ", ".join(sorted(replica_type_mem_runconfig.values()))
            details = str(
                "If cluster unhealthy, you can run `csctl get cluster --error-only` "
                "for more info. If memory was the limiting resource, you can try "
                f"override the default using a runconfig param ({params_str}) "
                "and check node capacity with `csctl get nodegroups`"
            )

        if details:
            msg += f" details={details}"
        log.warning(msg)


class HeartBeatThread(threading.Thread):
    def __init__(
        self,
        stub: ClusterManagementStub,
        options: HeartBeatOptions,
        lease_id: str,
        hostname: str,
        stop_event: threading.Event,
        lock: threading.Lock,
        response: cluster_pb2.HeartbeatResponse,
        status_poll: bool,
    ):
        """
        stub: The client to use for sending the heartbeat signals.
        options: HeartBeat configuration options.
        lease_id: The job/workflow id which should send heartbeats to.
        hostname: Current user node name.
        stop_event: Event that will stop the heartbeat thread when set.
        lock: Lock for updating response.
        response: Last response from server side.
        status_poll: Poll job status as part of heartbeat response.
        """
        super().__init__(name="cluster_client_heartbeat_thread", daemon=True)
        self.stub = stub
        self.options = options
        self.lease_id = lease_id
        self.hostname = hostname
        self.stop_event = stop_event
        self.lock = lock
        self.response = response
        self.stream = None
        self.last_beat_timestamp = None
        self.status_poll = status_poll

    def hb_request(self) -> cluster_pb2.HeartbeatRequest:
        host_mem_total_bytes = 0
        host_mem_used_bytes = 0
        host_cpu_used_percent = 0
        process_id = 0
        process_mem_rss_bytes = 0
        process_cpu_used_percent = 0
        try:
            # metrics of host
            virtual_memory = psutil.virtual_memory()
            host_mem_total_bytes = virtual_memory.total
            host_mem_used_bytes = virtual_memory.used
            host_cpu_used_percent = psutil.cpu_percent()
            # metrics the current process
            process = psutil.Process()
            process_id = process.pid
            process_mem_rss_bytes = process.memory_info().rss
            process_cpu_used_percent = process.cpu_percent()
        except Exception as exception:
            logger.debug(
                f"user node/process metrics scrape failed: {exception}"
            )
        finally:
            return cluster_pb2.HeartbeatRequest(
                message=f"client timestamp at: {_get_curr_time()}",
                job_id=self.lease_id,  # for backwards compatible
                lease_id=self.lease_id,
                lease_duration_seconds_override=self.options.lease_duration_seconds_override,
                host_name=self.hostname,
                host_mem_total_bytes=host_mem_total_bytes,
                host_mem_used_bytes=host_mem_used_bytes,
                host_cpu_used_percent=host_cpu_used_percent,
                process_id=process_id,
                process_mem_rss_bytes=process_mem_rss_bytes,
                process_cpu_used_percent=process_cpu_used_percent,
                status_poll=self.status_poll,
            )

    def request_generator(self):
        while not self.stop_event.is_set():
            # sleep first to give time for lease creation
            time.sleep(self.options.cycle_seconds)
            self.last_beat_timestamp = _get_curr_time()
            yield self.hb_request()

    def run(self):
        retries = 0
        while not self.stop_event.is_set():
            try:
                self.stream = self.stub.HeartbeatV2(self.request_generator())
                for response in self.stream:
                    if self.stop_event.is_set():
                        break
                    retries = 0
                    if self.response.is_ready or self.response.job_failed:
                        self.status_poll = False
                    if self.status_poll:
                        with self.lock:
                            self.response.CopyFrom(response)
                    else:
                        logger.debug(
                            f"Heartbeat({self.lease_id}) response: {response.message}"
                        )
            except grpc.RpcError as e:
                if self.stop_event.is_set():
                    break
                # retry for 1min since it should not happen
                if retries < 6:
                    logger.debug(
                        f"Retry on heartbeat error: {e.code()}, {e.details()}"
                    )
                    retries += 1
                    # give enough time in case of server restart
                    time.sleep(self.options.cycle_seconds)
                    continue
                self.response.job_failed = True
                self.response.message = (
                    f"Heartbeat for {self.lease_id} failed: {e.details()}"
                )
                logger.debug(self.response.message)
                raise e
        logger.debug(
            f"Heartbeat thread stopped for {self.lease_id}, "
            f"last heartbeat at: {self.last_beat_timestamp}."
        )

    def join(self, timeout=None):
        super().join(timeout)
        if self.stream and self.stream.is_active():
            try:
                logger.debug(
                    f"Attempting to cancel the heartbeat stream for {self.lease_id}"
                )
                self.stream.cancel()
            except grpc.RpcError as e:
                logger.debug(
                    f"Failed to cancel the heartbeat stream for {self.lease_id}: {e.details()}"
                )


def _create_user_notifications(
    notifications: Optional[List[NotificationTarget]],
) -> List[UserNotification]:
    """Convert NotificationTarget objects to UserNotification protos.

    Args:
        notifications: List of NotificationTarget objects containing notification preferences

    Returns:
        List of UserNotification protos configured with the notification targets
    """
    result = []
    if not notifications:
        return result

    for notification in notifications:
        # Add email notifications
        for email in notification.mailto:
            item = UserNotification()
            item.notification_type = cluster_pb2.NOTIFICATION_TYPE_EMAIL
            item.target = email
            if notification.severity_threshold is not None:
                item.severity_threshold = notification.severity_threshold
            result.append(item)

        # Add Slack notifications
        for slack_webhook in notification.slack:
            item = UserNotification()
            item.notification_type = cluster_pb2.NOTIFICATION_TYPE_SLACK
            item.target = slack_webhook
            if notification.severity_threshold is not None:
                item.severity_threshold = notification.severity_threshold
            result.append(item)

        # Add PagerDuty notifications
        for pagerduty_key in notification.pagerduty:
            item = UserNotification()
            item.notification_type = cluster_pb2.NOTIFICATION_TYPE_PAGERDUTY
            item.target = pagerduty_key
            if notification.severity_threshold is not None:
                item.severity_threshold = notification.severity_threshold
            result.append(item)

    return result


class TelemetryClient:
    """A client to report metrics to the cluster management server."""

    def __init__(
        self,
        mgmt_client: ClusterManagementClient,
        max_buffer_size: int = 128,
        max_buffer_seconds: int = 150,
    ):
        """
        Args:
            mgmt_client: A connected cluster management client.
            max_buffer_size: The maximum number of metrics to buffer before flushing.
            max_buffer_seconds: The maximum time to wait before flushing the buffer.
        """

        self._mgmt_client = mgmt_client
        self.max_buffer_seconds = max_buffer_seconds
        self._allow_push = False

        self._buffer = Queue(maxsize=max_buffer_size)
        self._last_flush_time = None

    def __enter__(self):
        """Enable pushing to buffer and start a thread to begin the flush loop."""

        self._allow_push = True

        self._last_flush_time = time.time()
        self._flush_data_cv = threading.Condition()

        self._flush_thread = threading.Thread(target=self._flush_loop)
        self._stop_flush_loop = threading.Event()
        self._flush_thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up the TelemetryClient instance and disable pushing to buffer."""

        self._cleanup()

        self._allow_push = False

    @functools.cached_property
    def job_id(self):
        """Get latest execute job ID from management client."""

        # execute jobs list is sorted by create time, hence
        # last execute is the current running job
        return self._mgmt_client.list_workflow_jobs()[1][-1]

    def push(
        self,
        metrics: Dict[str, float],
        labels: Optional[Dict[str, str]] = None,
    ):
        """
        Push a set of metrics to the buffer for reporting.

        Args:
            metrics: A dictionary of metric names and values.
            labels: A dictionary of labels that associate with
                the given metrics.
        """

        if not self._allow_push:
            return

        if labels is None:
            labels = dict()

        curr_timestamp = int(time.time())

        for name, value in metrics.items():
            self._buffer.put(
                cluster_pb2.Metric(
                    name=name,
                    value=value,
                    timestamp_sec=curr_timestamp,
                    labels=labels,
                )
            )

            if self._buffer.full():
                with self._flush_data_cv:
                    self._flush_data_cv.notify()

    def flush_data(self):
        """Flush the buffer to the cluster management server."""

        metrics = []
        while not self._buffer.empty():
            metrics.append(self._buffer.get())
            self._buffer.task_done()

        if metrics:
            request = cluster_pb2.ReportMetricRequest(
                metrics=metrics,
                job_id=self.job_id,
            )

            try:
                response = self._mgmt_client.stub.ReportMetrics(request)

                if response.code != 0:
                    logger.debug(
                        f"Failed to report metrics due to: {response.message}"
                    )
            except grpc.RpcError as e:
                logger.debug(f"Failed to report metrics due to: {e.details()}")

        self._last_flush_time = time.time()

    def _cleanup(self):
        """
        Clean up the TelemetryClient instance by joining flush thread
        and flushing remaining data on buffer.
        """

        self._stop_flush_loop.set()

        # Wake up flush thread if it's waiting
        with self._flush_data_cv:
            self._flush_data_cv.notify()

        self._flush_thread.join()

        if not self._buffer.empty():
            self.flush_data()

    def _should_flush(self):
        """Determine if we should flush the buffer based on size or time."""

        return self._stop_flush_loop.is_set() or (
            self._buffer.full()
            or (time.time() - self._last_flush_time) >= self.max_buffer_seconds
        )

    def _flush_loop(self):
        """Flush the buffer periodically or when it reaches the maximum size."""

        while not self._stop_flush_loop.is_set():
            time_since_last_flush = time.time() - self._last_flush_time
            remaining_time = max(
                1.0,  # Minimum wait time to prevent CPU spinning
                self.max_buffer_seconds - time_since_last_flush,
            )

            with self._flush_data_cv:
                self._flush_data_cv.wait_for(
                    self._should_flush, timeout=remaining_time
                )

            if self._should_flush():
                self.flush_data()
