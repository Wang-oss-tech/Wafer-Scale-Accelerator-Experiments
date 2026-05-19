# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

""" GRPC Client Used by Framework User to connect to SDK Worker
"""
import ast
import json
import logging
import math
import os
import tarfile
import tempfile
import textwrap
from contextlib import ExitStack
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Optional

import grpc
import numpy as np

from cerebras.appliance import __version__
from cerebras.appliance.cluster.client import ClusterManagementClient
from cerebras.appliance.cluster.config import get_cs_cluster_config
from cerebras.appliance.cluster_config import ClusterConfig
from cerebras.appliance.errors import (
    ApplianceUnknownError,
    ApplianceVersionError,
    cerebras_support_msg,
)
from cerebras.appliance.pb.sdk import sdk_appliance_pb2, sdk_common_pb2
from cerebras.appliance.pb.sdk.sdk_appliance_pb2_grpc import sdk_applianceStub
from cerebras.appliance.pb.workflow.appliance.common.common_config_pb2 import (
    ClusterDetails,
    JobMode,
    ResourceInfo,
)
from cerebras.appliance.utils import version_check
from cerebras.sdk.client import sdk_utils
from cerebras.sdk.client.sdk_appliance_utils import (
    GRPC_CONNECT_TIMEOUT,
    MAX_MESSAGE_LENGTH,
    get_artifact_hashes,
    get_artifact_id,
    get_csl_files,
)

RETRY_POLICY = {
    "methodConfig": [
        {
            "name": [{"service": "cerebras.Appliance"}],
            "retryPolicy": {
                "maxAttempts": 5,
                "initialBackoff": "2s",
                "maxBackoff": "10s",
                "backoffMultiplier": 2,
                "retryableStatusCodes": ["UNAVAILABLE", "UNKNOWN"],
            },
        }
    ]
}

# Setup logger
LOGGER = logging.getLogger(__name__)
hdlr = logging.StreamHandler()
FORMAT = "%(asctime)s %(levelname)-8s %(message)s"
formatter = logging.Formatter(FORMAT)
hdlr.setFormatter(formatter)
LOGGER.addHandler(hdlr)
LOGGER.setLevel(logging.INFO)


class SdkMode(Enum):
    """Enum describing the type of SDK job requested by the user.

    SDK_COMPILE will result in the reservation of a single compute node
    SDK_SIMULATE will result in the reservation of a single compute node
    SDK_EXECUTE will result in a compute node and a CSX
    """

    SDK_COMPILE = 0
    SDK_EXECUTE = 1
    SDK_SIMULATE = 2


class SdkContext:
    """Appliance context used by SdkClient classes
    (SdkCompiler and SdkRuntime)
    """

    def __init__(self, sdk_mode: SdkMode, **kwargs):
        self._stub = None
        self._sdk_mode = sdk_mode
        self._stack = None
        self._job_id = ""

        self._default_authority = None
        self._credentials = None
        grpc_address = kwargs.get("grpc_address", None)
        if grpc_address is None:
            self._init_mgmt_client(**kwargs)
        else:
            self._grpc_address = grpc_address

        self._init_appliance_stub()

    def stub(self):
        """Returns the internal sdk_applianceStub"""
        if not hasattr(self, '_created_from_with'):
            raise RuntimeError(
                "SdkContext and its derived classes must be created using a 'with' statement"
            )

        return self._stub

    def _init_mgmt_client(self, **kwargs):
        """This function requests resources via a ClusterManagementClient"""

        resource_cpu = int(kwargs.get("resource_cpu", 24000))
        resource_mem = int(kwargs.get("resource_mem", 67 << 30))
        namespace = kwargs.get("mgmt_namespace", None)
        disable_version_check = kwargs.get("disable_version_check", False)

        compile_dir_relative_path = kwargs.get(
            "relative_compile_dir", "sdk_cached_compile"
        )

        if self._sdk_mode in [SdkMode.SDK_COMPILE, SdkMode.SDK_SIMULATE]:
            job = JobMode.Job.SDK_COMPILE
            num_csx = 0
            task_type = ClusterDetails.TaskInfo.TaskType.CRD
        else:
            job = JobMode.Job.SDK_EXECUTE
            num_csx = 1
            task_type = ClusterDetails.TaskInfo.TaskType.WRK

        _, cluster_config = get_cs_cluster_config()

        self._cluster_config = ClusterConfig(
            mgmt_address=cluster_config.mgmt_address,
            mgmt_namespace=namespace,
            disable_version_check=disable_version_check,
            num_csx=num_csx,
            max_wgt_servers=0,
            num_workers_per_csx=0,
        )

        self._grpc_address = None
        self._mgmt_client_args = {
            "server": self._cluster_config.mgmt_address,
            "namespace": self._cluster_config.mgmt_namespace,
            "job_timer": self._cluster_config.job_timer,
        }

        # defaults to 0 which is no op currently (unlimited memory/cpu)
        # once restrictions are known, update the value
        compile_memory_bytes = (
            resource_mem  # current default ram memory for coord
        )
        compile_cpu_millicore = (
            resource_cpu  # current default num of cpus for coord
        )
        cluster_details = ClusterDetails()
        task_type = ClusterDetails.TaskInfo.TaskType.CRD
        if job == JobMode.Job.SDK_EXECUTE:
            task_type = ClusterDetails.TaskInfo.TaskType.WRK

        task_info = ClusterDetails.TaskInfo(
            task_type=task_type,
            resource_info=ResourceInfo(
                memory_bytes=compile_memory_bytes,
                cpu_millicore=compile_cpu_millicore,
            ),
        )
        cluster_details.tasks.append(task_info)  # pylint: disable=E1101
        job_mode = JobMode(job=job, cluster_details=cluster_details)

        self._mgmt_client = None
        with ExitStack() as stack:
            self._mgmt_client = stack.enter_context(
                ClusterManagementClient(**self._mgmt_client_args)
            )
            self._stack = stack.pop_all()

        try:
            mgmt_versions = self._mgmt_client.get_server_versions()
            # Handle in manager because cluster management test don't include appliance
            # SW-91475
        except grpc.RpcError as rpc_error:
            rpc_error_code = rpc_error.code()  # pylint: disable=no-member
            if rpc_error_code == grpc.StatusCode.UNIMPLEMENTED:
                # Catch version 1.7 where we didn't have version checks
                release_version = __version__.split("+", maxsplit=1)[0]
                raise ApplianceVersionError(
                    "Cluster management server version is out of date. Please install version: "
                    f"{release_version}"
                ) from rpc_error
            raise

        if not self._cluster_config.disable_version_check:
            for component_details in mgmt_versions:
                component_name = component_details.name
                version = component_details.version
                if "+" in version:
                    version, githash = version.split("+")
                else:
                    # Possible future cluster version epoch.
                    # Definitely incompatible, so pass None for the githash to
                    # fail the check.
                    githash = None
                version_check(component_name, version, githash)

        compile_dir_relative_path = kwargs.get(
            "relative_compile_dir", "sdk_cached_compile"
        )
        if job == JobMode.Job.SDK_COMPILE:
            response = self._mgmt_client.init_sdk_compile_job(
                job_mode, compile_dir_relative_path=compile_dir_relative_path
            )
        else:
            response = self._mgmt_client.init_sdk_execute_job(
                job_mode,
            )

        self._job_id = "[job_id = " + response["job_id"] + "]"
        LOGGER.debug("SdkContext executing job %s", self._job_id)
        service_authority = response["service_authority"]
        self._default_authority = service_authority

        # TODO workaround for mocking cluster management responses
        if response['service_url'].count(':') == 1:
            self._grpc_address = response['service_url']
        else:
            second_colon_idx = response['service_url'].rfind(':')
            self._grpc_address = response['service_url'][:second_colon_idx]
        self._set_credentials(response["certificate_bytes"])

    def _init_appliance_stub(self):
        """This function creates the GRPC client that sends requests to the
        Coordinator"""
        self.grpc_fork_support_value = os.environ.get(
            'GRPC_ENABLE_FORK_SUPPORT', None
        )
        # SW-89390: To suppress the spam messages from gRPC library
        os.environ.update({'GRPC_ENABLE_FORK_SUPPORT': '0'})
        channel_options = [
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ("grpc.per_rpc_retry_buffer_size", MAX_MESSAGE_LENGTH),
            ("grpc.enable_retries", 1),  # Only required until grpc > 1.39
            ('grpc.service_config', json.dumps(RETRY_POLICY)),
        ]
        if self._default_authority is not None:
            channel_options.append(
                ('grpc.default_authority', self._default_authority)
            )

        wrk_address = self._grpc_address
        if self._credentials:
            self._channel = grpc.secure_channel(
                wrk_address,
                self._credentials,
                options=channel_options,
            )
        else:
            self._channel = grpc.insecure_channel(
                wrk_address,
                options=channel_options,
            )

        try:
            grpc.channel_ready_future(self._channel).result(
                timeout=GRPC_CONNECT_TIMEOUT
            )
        except:  # pylint: disable-W0702
            raise ApplianceUnknownError(
                cerebras_support_msg(
                    f"{self._job_id} Could not connect to the cluster."
                )
            ) from None

        self._stub = sdk_applianceStub(self._channel)

        LOGGER.debug("sdk_applianceStub started at address: %s", wrk_address)

    def _set_credentials(self, value: bytes):
        """Sets the credentials, loading the byte string cert."""
        if value:
            self._credentials = grpc.ssl_channel_credentials(value)

    def job_id(self):
        """Returns the current job id assigned by the Management Node"""
        return self._job_id

    def __enter__(self):
        self._created_from_with = True
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self._stack is not None:
            self._stack.close()
        self._created_from_with = False


class SdkClient:
    """SdkClient is the base class of all SDK appliance clients.
    SdkRuntime and SdkCompiler derive from SdkClient. SdkClient offers
    common functionalities such as download_artifacts.
    """

    def __init__(self, sdk_mode: SdkMode, **kwargs):
        self._options = kwargs
        self.sdk_mode = sdk_mode
        self._stub = None
        self._stack = None
        self._context = None

    def __enter__(self):
        self._created_from_with = True

        # check if there is a sdk_test_params.json file
        if os.path.exists("sdk_test_params.json"):
            with open("sdk_test_params.json", "r", encoding="utf8") as fparam:
                data = json.load(fparam)
                self._options["resource_mem"] = data["resource_mem"]
                self._options["resource_cpu"] = data["resource_cpu"]

        with ExitStack() as stack:
            self._context = stack.enter_context(
                SdkContext(self.sdk_mode, **self._options)
            )
            self._stub = self._context.stub()
            self._stack = stack.pop_all()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stack is not None:
            self._stack.close()
        self._created_from_with = False

    def stub(self):
        """Returns the internal sdk_applianceStub"""
        if not hasattr(self, '_created_from_with'):
            raise RuntimeError(
                f"{self.__class__.__name__} must be created using a 'with' statement"
            )

        return self._stub

    def sdk_support_msg(self, msg):
        """Prepends the current job id (if available) to the standard
        Cerebras support message printed when an error occurs.
        """

        return cerebras_support_msg(f"{self._context.job_id()} {msg}")

    def download_artifact(self, artifact_name: str, out_path: str):
        """Allows to download an artifact from the appliance.

        artifact_name - name of the artifact to download. If it is a directory,
        a tarball of that directory will be created and transferred.

        Returns the name of the file that has been written (can contain a
        .tar.gz extension if the artifact_name was a directory)
        """

        request = sdk_appliance_pb2.SdkArtifactDownloadArgs(
            file_name=artifact_name,
        )
        response_iterator = self.stub().sdk_download_files(request)

        out_filename = ""
        out_file = None
        code = None
        tot_bytes_written = 0
        for i, response in enumerate(response_iterator):
            if response.HasField("status"):
                if response.status.code in [
                    sdk_common_pb2.StatusCode.SDK_RT_INTERNAL_ERROR,
                    sdk_common_pb2.StatusCode.SDK_RT_FILE_ERROR,
                ]:
                    raise ApplianceUnknownError(
                        self.sdk_support_msg(f"{response.status.message}")
                    )
                else:
                    raise ApplianceUnknownError(response.status.message)

            if out_filename != response.data.file_name:
                if out_file is not None:
                    out_file.flush()
                    out_file.close()
                    if tot_bytes_written != total_filesize:
                        message = f"""{type(self)}.download_artifact() error:
                        {out_filename} total size conflict:
                        {tot_bytes_written} bytes received and written while
                        {total_filesize} bytes were expected."""
                        raise ApplianceUnknownError(
                            self.sdk_support_msg(message)
                        )
                    tot_bytes_written = 0

            out_filename = response.data.file_name
            total_filesize = response.data.total_bytes

            if i == 0:
                if not os.path.isdir(out_path):
                    out_filename = out_path
                    parent_dir = os.path.dirname(out_filename)
                else:
                    out_filename = out_path + "/" + out_filename
                    parent_dir = out_path

                os.makedirs(parent_dir, exist_ok=True)

            # write the block of data
            out_file = open(out_filename, "wb")
            bytes_written = out_file.write(response.data.data_chunk)
            if bytes_written != response.data.num_bytes:
                out_file.flush()
                out_file.close()
                message = f"""{type(self)}.download_artifact() error:
                could not write chunk of {out_filename}. Only {bytes_written}
                bytes out of {response.data.num_bytes} bytes were written."""
                raise ApplianceUnknownError(self.sdk_support_msg(message))

            tot_bytes_written += bytes_written

        if out_file is not None:
            out_file.flush()
            out_file.close()
            if code is None and tot_bytes_written != total_filesize:
                message = f"""{type(self)}.download_artifact() error:
                {out_filename} total size conflict: {tot_bytes_written}
                bytes received and written while {total_filesize} bytes
                were expected."""
                raise ApplianceUnknownError(self.sdk_support_msg(message))

        return out_filename

    def upload_artifact(
        self, artifact_path: str, destination_rel_path: Optional[str] = None
    ):
        """Allows to upload an artifact to the appliance.

        artifact_path - path to the artifact to upload. If it is a directory,
        a tarball of that directory will be created, transferred, and
        decompressed on the other end.
        destination_rel_path - optional relative path in the appliance where
        to place the artifact being transferred
        """

        if destination_rel_path is None:
            compile_hash = get_artifact_id(artifact_path)
        else:
            compile_hash = destination_rel_path

        # if directory, compress first
        if os.path.isdir(artifact_path):
            # if it is a directory, create a tar ball
            output_filename = artifact_path + ".tar.gz"
            with tarfile.open(output_filename, "w:gz") as tar:
                tar.add(artifact_path, arcname=os.path.basename(artifact_path))
            artifact_path = output_filename

        files = [artifact_path]

        app_path = Path(artifact_path).parent.absolute()

        return self._upload_files(
            app_path, compile_hash, files, destination_rel_path
        )

    def _upload_files(
        self,
        app_path: str,
        compile_hash: str,
        files: List[str],
        destination_rel_path: Optional[str] = None,
    ):

        if destination_rel_path is None:
            destination_rel_path = ""

        mode = "SDK_COMPILE" if isinstance(self, SdkCompiler) else "SDK_EXECUTE"

        def generate_artifacts_args():
            for cur_file in files:
                with open(cur_file, "rb") as in_file:
                    data = in_file.read()
                    num_bytes = len(data)
                    num_chunks = max(
                        math.ceil(num_bytes / MAX_MESSAGE_LENGTH), 1
                    )
                    for i in range(num_chunks):
                        chunk_bytes = (
                            MAX_MESSAGE_LENGTH
                            if i < num_chunks - 1
                            else num_bytes - i * MAX_MESSAGE_LENGTH
                        )
                        subrequest = sdk_appliance_pb2.SdkArtifactsArgs()
                        subrequest.file_name = os.path.relpath(
                            cur_file, app_path
                        )
                        subrequest.data_chunk = data[
                            i * MAX_MESSAGE_LENGTH : i * MAX_MESSAGE_LENGTH
                            + chunk_bytes
                        ]
                        subrequest.num_bytes = chunk_bytes
                        subrequest.total_bytes = num_bytes
                        subrequest.compile_hash = compile_hash
                        subrequest.mode = mode
                        subrequest.destination_rel_path = destination_rel_path
                        yield subrequest

        response = self.stub().sdk_upload_files(generate_artifacts_args())
        if response.code in [
            sdk_common_pb2.StatusCode.SDK_RT_INTERNAL_ERROR,
            sdk_common_pb2.StatusCode.SDK_RT_FILE_ERROR,
        ]:
            raise ApplianceUnknownError(
                self.sdk_support_msg(f"{response.message}")
            )

        return response

    def _check_sdk_response(
        self, response: sdk_appliance_pb2.SdkResponse, preamble: str
    ) -> None:

        if response.code != sdk_common_pb2.StatusCode.SDK_RT_SUCCESS:
            # Add the contact support message if it is an internal error
            if response.code in [
                sdk_common_pb2.StatusCode.SDK_RT_INTERNAL_ERROR,
                sdk_common_pb2.StatusCode.SDK_RT_FILE_ERROR,
            ]:
                message = self.sdk_support_msg(response.message)

                raise ApplianceUnknownError(f"{message}")
            elif response.code == sdk_common_pb2.StatusCode.SDK_COMMAND_ERROR:
                raise RuntimeError(
                    f"{self.__class__.__name__}.run() command error:\n%s" % response.message
                )
            else:
                if len(response.message) > 0:
                    raise ApplianceUnknownError(
                        f"{preamble}: {response.message}"
                    )
                else:
                    raise ApplianceUnknownError(f"{preamble}")


class SdkLauncher(SdkClient):
    """SdkLauncher allows to run SDK programs on CSX and simulator.
    The SdkLauncher allows to upload artifacts, run custom commands in the
    appliance, and use custom scripts written as if the system was not set
    in appliance mode to run on the attached CSX. The user must use the
    %CMADDR% template string to pass the system address to their run script.

    The constructor takes the following parameters:
    artifact_id - the id of a compiled artifact as returned by a SdkCompiler
    object

    kwargs:
    simulator - boolean indicating whether the run should be done on simulator
    or not. In the case of a simulator run, no CSX will be allocated in the
    Wafer Scale Cluster
    """

    def __init__(
        self,
        artifact_path: str,
        **kwargs,
    ) -> None:
        self._simfab = kwargs.get("simulator", False)
        sdk_mode = (
            SdkMode.SDK_EXECUTE if not self._simfab else SdkMode.SDK_SIMULATE
        )
        self._artifact_path = artifact_path
        self._artifact_id = -1

        # call the parent class constructor
        super().__init__(sdk_mode, **kwargs)

    def __enter__(self):
        # call the parent method, allowing the use of stub
        super().__enter__()
        # upload the artifact to the appliance
        response = self.upload_artifact(self._artifact_path)
        self._check_sdk_response(
            response, "SdkLauncher creation (upload phase)"
        )

        self._artifact_id = get_artifact_id(self._artifact_path)

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self._stub is not None:
            response = self.stub().sdk_shutdown(
                sdk_appliance_pb2.SdkEmptyMessage()
            )
            self._check_sdk_response(response, "SdkLauncher shutdown")
        super().__exit__(exc_type, exc_value, exc_tb)

    def stage(self, artifact_path: str):
        """Stages additional artifacts in the remote working
        directory within the appliance
        """

        # upload in the this._artifact destination directory
        response = self.upload_artifact(artifact_path, self._artifact_id)
        self._check_sdk_response(response, "SdkLauncher staging artifact ")

    def run(self, *args):
        """Runs a command in the appliance.

        The %CMADDR% placeholder can be used where
        the address of the CS2/3 machine is needed.
        """
        for cmd in args:
            if not isinstance(cmd, str):
                raise RuntimeError(
                    "SdkLauncher.run() can only be used with string arguments"
                )
            elif cmd.find("%CMADDR%") != -1 and self._simfab:
                raise RuntimeError(
                    """SdkLauncher.run() was asked to run a command containing %CMADDR%
                    which is not compatible with the simulator argument provided during
                    SdkLauncher creation"""
                )
        commands = list(args)

        # then replace what is in the keyword list by their replacements
        keywords = {
            "cs_python": "python",
        }
        for word, replacement in keywords.items():
            for i, command in enumerate(commands):
                commands[i] = command.replace(word, replacement)

        def generate_command_params():
            for command in commands:
                subrequest = sdk_appliance_pb2.SdkRunCommandParams()
                subrequest.app_hash = self._artifact_id
                subrequest.command = command
                yield subrequest

        response = self.stub().sdk_run_command(generate_command_params())
        self._check_sdk_response(response, "SdkLauncher running command(s) ")

        # return the stdout output if needed
        return response.message

    def download_artifact(self, artifact_name: str, out_path: str):
        """Allows to download an artifact from the appliance.

        artifact_name - name of the artifact to download. If it is a directory,
        a tarball of that directory will be created and transferred.

        Returns the name of the file that has been written (can contain a
        .tar.gz extension if the artifact_name was a directory)
        """

        # prepend the artifact_id to the filename as everything is relative
        # to that path when using SdkLauncher

        artifact_name = self._artifact_id + "/" + artifact_name
        super().download_artifact(artifact_name, out_path)


class SdkRuntime(SdkClient):
    """SdkRuntime allows to run SDK programs on CSX and simulator.
    The SdkRuntime creates, loads, runs, and stops the execution of a compiled
    CSL program on actual hardware or on simulator. The runtime provides a high
    level interface to create data transfers and RPC commands.

    The constructor takes the following parameters:
    artifact_id - the id of a compiled artifact as returned by a SdkCompiler
    object

    kwargs:
    simulator - boolean indicating whether the run should be done on simulator
    or not. In the case of a simulator run, no CSX will be allocated in the
    Wafer Scale Cluster
    """

    class Task:
        """The Task class allows to keep track of non-blocking SdkRuntime
        operations, such as memcpy_h2d, memcpy_d2h, call, or launch requests.

        For memcpy_d2h operations, a dest numpy array needs to be provided to store
        the result of the operation."""

        def __init__(self, runtime, op_index: int, **kwargs):
            self._runtime = runtime

            # this constains the index of the operation, which is also what
            # is used to retrieve the handle on the worker side
            self.op_index = op_index
            self.is_done = False

            self.dest = kwargs.get("dest", None)

        def done(self) -> bool:
            """Returns a boolean indicating if the operation op_index has
            completed"""
            return self._runtime.is_task_done(self)

        def wait(self) -> None:
            """Blocks until the operation op_index has completed"""
            return self._runtime.task_wait(self)

    def __init__(
        self,
        artifact_path: str,
        **kwargs,
    ) -> None:
        self._simfab = kwargs.get("simulator", False)
        self._op_index = -1
        self._last_finished_op_index = -1
        # _active can guarantee that the pair {start(), stop()} is called properly.
        self._active = False
        # _has_run can prevent the following sequence
        # start()
        # stop()
        # start() --> emits an error
        self._has_run = False
        self._artifact_path = artifact_path

        # dictionnary to keep track of all non blocking tasks
        self._nonblock_task = {}

        sdk_mode = (
            SdkMode.SDK_EXECUTE if not self._simfab else SdkMode.SDK_SIMULATE
        )

        self._api_info_dict = {}

        # call the parent class constructor
        super().__init__(sdk_mode, **kwargs)

    def is_simulator(self):
        """Indicates whether the run is a simulation or an actual run on CSX"""

        return self._simfab

    def _get_next_op_index(self):
        """Returns the index of the next sequential operation/command that
        will be executed by the SdkRuntime object
        """

        self._op_index += 1
        return self._op_index

    def start(self):
        """This function starts the execution of the program on the
        CSX/simulator. This step also encompass loading the program on
        the CSX/simulator.

        SdkRuntime.start() can be called only once per SdkRuntime object.
        """

        if self._has_run:
            raise RuntimeError(
                """SdkRuntime.start() can only be called once
                               per SdkRuntime object"""
            )
        if self._active:
            raise RuntimeError("SdkRuntime.start() has already been called")

        response = self.stub().sdk_runtime_start(
            sdk_appliance_pb2.SdkEmptyMessage()
        )
        self._check_sdk_response(response, "SdkRuntime.start()")
        self._nonblock_task = {}
        self._active = True
        self._has_run = True

    def stop(self):
        """This function stops the execution of the program on the
        CSX/simulator.

        SdkRuntime.start() must have been called prior to calling this function.
        SdkRuntime.stop() should be called before the SdkRuntime object
        destruction.
        """

        if not self._active:
            raise RuntimeError(
                "SdkRuntime.start() must be called before SdkRuntime.stop()"
            )

        LOGGER.debug(
            "SdkRuntime.stop() len(self._nonblock_task) = %d",
            len(self._nonblock_task),
        )
        if len(self._nonblock_task) > 0:
            LOGGER.debug("SdkRuntime.stop() is waiting for unfinished commands")
            # get the op_index of the last operation:
            last_op_index = self._op_index

            # since SdkRuntime executes operations in the op_index order,
            # waiting on the last operation is sufficient to ensure that
            # every operation have completed.
            if last_op_index in self._nonblock_task:
                self._nonblock_task[last_op_index].wait()
            keys = list(self._nonblock_task.keys())
            for op_index in keys:
                t = self._nonblock_task[op_index]
                # if it is a D2H operation, we need to get the data back
                if t.dest is not None:
                    # wait() also removes the operation from self._nonblock_task
                    self._nonblock_task[op_index].wait()
            self._nonblock_task.clear()

        response = self.stub().sdk_runtime_stop(
            sdk_appliance_pb2.SdkEmptyMessage()
        )
        self._active = False
        self._check_sdk_response(response, "SdkRuntime.stop()")

    def __enter__(self):
        # call the parent method, allowing the use of stub
        super().__enter__()

        # upload the artifact to the appliance
        response = self.upload_artifact(self._artifact_path)
        self._check_sdk_response(response, "SdkRuntime creation (upload phase)")

        artifact_id = get_artifact_id(self._artifact_path)

        # then instantiate the SdkRuntime on the worker
        response = self.stub().sdk_runtime_create(
            sdk_appliance_pb2.SdkCreateArgs(
                app_hash=artifact_id,
            )
        )
        self._check_sdk_response(
            response, "SdkRuntime creation (runtime creation phase)"
        )

        # This is required to be able to verify launch parameters
        with tempfile.TemporaryDirectory() as tmpdirname:
            # get the appliance csl_bin_name
            self.download_artifact(
                artifact_id + "/.csl_bin_name",
                f"{tmpdirname}/.tmp_csl_bin_name",
            )
            with open(f"{tmpdirname}/.tmp_csl_bin_name", encoding="utf8") as f:
                oname = f.readline().strip()

            # download the rpc_out json file
            self.download_artifact(
                artifact_id + "/" + oname + "/bin/out_rpc.json",
                f"{tmpdirname}/.tmp_out_rpc.json",
            )

            self._api_info_dict = sdk_utils.get_api_info_dict_from_json(
                f"{tmpdirname}/.tmp_out_rpc.json"
            )

        self.start()

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is None:
            # if we didn't have an exception, we can gracefully
            # stop the runtime
            if self._active:
                self.stop()
        else:
            # that avoids trigerring an RPC in the destructor
            self._active = False
        super().__exit__(exc_type, exc_value, exc_tb)

    def memcpy_h2d(
        self,
        dest: int,
        src: np.array,
        x: int,
        y: int,
        w: int,
        h: int,
        elt_per_pe: int,
        **kwargs,
    ) -> Optional[Task]:
        """Transfer data to the CSX/simulator via the memcpy interface.

        dest - the symbol id (as returned by SdkRuntime.get_id()) of the
        destination variable (or the destination color if streaming mode is
        used).
        src - a numpy array containing the input data. The shape of the array
        must be (h, w, elt_per_pe).
        x - x-coordinate of origin of target rectangle of processing elements
        y - y-coordinate of origin of target rectangle of processing elements
        w - width of target rectangle of processing elements
        h - height of target rectangle of processing elements
        elt_per_pe - number of array elements per processing element in src/dest

        kwargs:
            streaming - memcpy-streaming mode if True, memcpy API otherwise
            data_type - MemcpyDataType.MEMCPY_32BIT or MEMCPY_16BIT
            order - MemcpyOrder.ROW_MAJOR or COL_MAJOR
            nonblock - True for non-blocking operations, False otherwise. The
            function will return a SdkRuntime.Task in the nonblocking case.
        """

        if not self._active:
            raise RuntimeError(
                """SdkRuntime.memcpy_h2d() cannot be called while
                the runtime is not active"""
            )

        for param in ["streaming", "data_type", "order", "nonblock"]:
            if not param in kwargs:
                raise RuntimeError(
                    f"""Must specify {param} in kwargs of
                SdkRuntime.memcpy_h2d()"""
                )

        if src.dtype.itemsize != 4:
            raise RuntimeError(
                """Internal data type of any
                SdkRuntime.memcpy_h2d() or memcpy_d2h() operation should be 32
                bit"""
            )

        op_index = self._get_next_op_index()

        # create a generator to split the numpy array in multiple blocks
        def generate_h2d_requests():
            data = src.tobytes()
            num_bytes = len(data)
            num_chunks = math.ceil(num_bytes / MAX_MESSAGE_LENGTH)
            for i in range(num_chunks):
                chunk_bytes = (
                    MAX_MESSAGE_LENGTH
                    if i < num_chunks - 1
                    else num_bytes - i * MAX_MESSAGE_LENGTH
                )
                subrequest = sdk_appliance_pb2.MemcpyH2DParams(
                    op_index=op_index,
                    dest=dest,
                    src=sdk_appliance_pb2.MemcpyData(
                        data_chunk=data[
                            i * MAX_MESSAGE_LENGTH : i * MAX_MESSAGE_LENGTH
                            + chunk_bytes
                        ],
                        num_bytes=chunk_bytes,
                        total_bytes=num_bytes,
                    ),
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    elt_per_pe=elt_per_pe,
                    options=sdk_appliance_pb2.MemcpyOpOptions(
                        streaming=kwargs["streaming"],
                        data_type=kwargs["data_type"],
                        order=kwargs["order"],
                        nonblock=kwargs["nonblock"],
                    ),
                )
                yield subrequest

        nonblock = kwargs["nonblock"]
        response = self.stub().sdk_runtime_memcpy_h2d(generate_h2d_requests())

        self._check_sdk_response(response, "SdkRuntime::memcpy_h2d()")

        if nonblock:
            t = SdkRuntime.Task(self, op_index)
            self._nonblock_task[op_index] = t
            return t
        else:
            return None

    def memcpy_d2h(
        self,
        dest: np.array,
        src: int,
        x: int,
        y: int,
        w: int,
        h: int,
        elt_per_pe: int,
        **kwargs,
    ) -> Optional[Task]:
        """Transfer data back from the CSX/simulator via the memcpy interface.

        dest - a numpy array where the output data will be copied back.
        The shape of the array must be (h, w, elt_per_pe).
        src - the symbol id (as returned by SdkRuntime.get_id()) of the
        source variable (or the source color if streaming mode is used).
        x - x-coordinate of origin of target rectangle of processing elements
        y - y-coordinate of origin of target rectangle of processing elements
        w - width of target rectangle of processing elements
        h - height of target rectangle of processing elements
        elt_per_pe - number of array elements per processing element in src/dest

        kwargs:
            streaming - memcpy-streaming mode if True, memcpy API otherwise
            data_type - MemcpyDataType.MEMCPY_32BIT or MEMCPY_16BIT
            order - MemcpyOrder.ROW_MAJOR or COL_MAJOR
            nonblock - True for non-blocking operations, False otherwise. The
            function will return a SdkRuntime.Task in the nonblocking case.
        """

        if not self._active:
            raise RuntimeError(
                """SdkRuntime.memcpy_d2h() cannot be called
            while the runtime is not active"""
            )

        for param in ["streaming", "data_type", "order", "nonblock"]:
            if not param in kwargs:
                raise RuntimeError(
                    f"""Must specify {param} in kwargs of
                memcpy_d2h()"""
                )

        if dest.dtype.itemsize != 4:
            raise RuntimeError(
                """Internal data type of any memcpy_d2h() or
            memcpy_h2d() operation should be 32 bit"""
            )

        op_index = self._get_next_op_index()

        request = sdk_appliance_pb2.MemcpyD2HParams(
            op_index=op_index,
            src=src,
            x=x,
            y=y,
            w=w,
            h=h,
            elt_per_pe=elt_per_pe,
            options=sdk_appliance_pb2.MemcpyOpOptions(
                streaming=kwargs["streaming"],
                data_type=kwargs["data_type"],
                order=kwargs["order"],
                nonblock=kwargs["nonblock"],
            ),
        )
        nonblock = kwargs["nonblock"]
        response_iterator = self.stub().sdk_runtime_memcpy_d2h(request)

        if nonblock:
            for response in response_iterator:
                if response.HasField("status"):
                    if (
                        response.status.code
                        != sdk_common_pb2.StatusCode.SDK_RT_SUCCESS
                    ):
                        if response.status.code in [
                            sdk_common_pb2.StatusCode.SDK_RT_INTERNAL_ERROR,
                            sdk_common_pb2.StatusCode.SDK_RT_FILE_ERROR,
                        ]:
                            raise ApplianceUnknownError(
                                self.sdk_support_msg(
                                    f"{response.status.message}"
                                )
                            )
                        else:
                            raise ApplianceUnknownError(
                                f"""SdkRuntime.memcpy_d2h()
                                request failed: {response.status.message}"""
                            )
                elif not response.HasField("d2h_data"):
                    raise ApplianceUnknownError(
                        self.sdk_support_msg(
                            """Received SdkD2HResponse without status
                            or d2h_data fields"""
                        )
                    )

            t = SdkRuntime.Task(self, op_index, dest=dest)
            self._nonblock_task[op_index] = t
            return t
        else:

            def memcpy_d2h_watcher(
                dest: np.ndarray,
                request: sdk_appliance_pb2.MemcpyD2HParams,
                response_iterator: Iterator[sdk_appliance_pb2.SdkD2HResponse],
            ) -> None:
                dtype = np.dtype(dest.dtype)
                offset = 0
                for response in response_iterator:
                    if response.HasField("status"):
                        if (
                            response.status.code
                            != sdk_common_pb2.StatusCode.SDK_RT_SUCCESS
                        ):
                            if response.status.code in [
                                sdk_common_pb2.StatusCode.SDK_RT_INTERNAL_ERROR,
                                sdk_common_pb2.StatusCode.SDK_RT_FILE_ERROR,
                            ]:
                                raise ApplianceUnknownError(
                                    self.sdk_support_msg(
                                        f"{response.status.message}"
                                    )
                                )
                            else:
                                raise ApplianceUnknownError(
                                    f"""SdkRuntime.memcpy_d2h() request failed:
                                    {response.status.message}"""
                                )
                    elif response.HasField("d2h_data"):
                        data = response.d2h_data
                        num_elems = int(data.num_bytes / dtype.itemsize)
                        dest.ravel()[offset : offset + num_elems] = (
                            np.frombuffer(data.data_chunk, dtype=dtype)
                        )
                        offset += num_elems
                    else:
                        raise ApplianceUnknownError(
                            self.sdk_support_msg(
                                """Received SdkD2HResponse without
                                status or d2h_data fields"""
                            )
                        )
                # make sure we have received everything we expect
                total_size = request.w * request.h * request.elt_per_pe
                if offset != total_size:
                    raise ApplianceUnknownError(
                        self.sdk_support_msg(
                            f"""Missing SdkRuntime.memcpy_d2h() data: received
                            {offset} while expecting {total_size}"""
                        )
                    )

                return

            memcpy_d2h_watcher(dest, request, response_iterator)
            return None

    def call(self, name: str, args: List[int], **kwargs) -> Optional[Task]:
        """Performs a RPC call of function name on the CSX.
        Parameters must be given as a list of integers (so they need to be cast
        beforehand)

        name - the name of the function to be called on the CSX
        args - the list of function arguments (cast as integers) to pass to the
        function

        kwargs:
            nonblock - True for non-blocking operations, False otherwise. The
            function will return a SdkRuntime.Task in the nonblocking case.
            This parameter must be provided.
        """

        if not self._active:
            raise RuntimeError(
                """SdkRuntime.call() cannot be called while the
            runtime is not active"""
            )

        # first check the kwargs
        for option in ["streaming", "data_type", "order"]:
            if option in kwargs:
                raise RuntimeError(
                    f"""The {option} option is not valid for
                SdkRuntime.call()"""
                )
        if "nonblock" not in kwargs:
            raise RuntimeError(
                """SdkRuntime.call() requires to specify the
            nonblock kwarg"""
            )

        u32_args = [sdk_utils.cast_uint32(x) for x in args]

        op_index = self._get_next_op_index()

        request = sdk_appliance_pb2.SdkCallArgs(
            op_index=op_index,
            name=name,
            params=u32_args,
            options=sdk_appliance_pb2.MemcpyOpOptions(
                nonblock=kwargs["nonblock"],
            ),
        )

        response = self.stub().sdk_runtime_call(request)

        nonblock = kwargs["nonblock"]
        if nonblock:
            t = SdkRuntime.Task(self, op_index)
            self._nonblock_task[op_index] = t
            return t
        else:
            self._check_sdk_response(response, "SdkRuntime.call()")
            return None

    def launch(self, name: str, *args, **kwargs) -> Optional[Task]:
        """Performs a RPC call of function name on the CSX.
        Parameters are given as a list and type consistency will be checked.

        name - the name of the function to be called on the CSX
        args - the list of function arguments to pass to the function

        kwargs:
            nonblock - True for non-blocking operations, False otherwise. The
            function will return a SdkRuntime.Task in the nonblocking case.
            This parameter must be provided.
        """

        if not self._active:
            raise RuntimeError(
                """SdkRuntime.launch() cannot be called while
            the runtime is not active"""
            )

        # first check the kwargs
        for option in ["streaming", "data_type", "order"]:
            if option in kwargs:
                raise RuntimeError(
                    f"""The {option} option is not valid for
                SdkRuntime.launch()"""
                )
        if "nonblock" not in kwargs:
            raise RuntimeError(
                """SdkRuntime.launch() requires to specify the
            nonblock kwarg"""
            )

        # now check the arguments and types
        try:
            sdk_utils.check_rpc_api(name, args, self._api_info_dict)
        except RuntimeError as e:
            raise RuntimeError(
                f"""SdkRuntime.launch() argument validation
            error:\n{e}"""
            ) from None

        # convert args to uint32
        u32_args = [sdk_utils.cast_uint32(x) for x in args]

        op_index = self._get_next_op_index()

        request = sdk_appliance_pb2.SdkCallArgs(
            op_index=op_index,
            name=name,
            params=u32_args,
            options=sdk_appliance_pb2.MemcpyOpOptions(
                nonblock=kwargs["nonblock"],
            ),
        )

        response = self.stub().sdk_runtime_call(request)

        nonblock = kwargs["nonblock"]
        if nonblock:
            t = SdkRuntime.Task(self, op_index)
            self._nonblock_task[op_index] = t
            return t
        else:
            self._check_sdk_response(response, "SdkRuntime.launch()")
            return None

    def get_id(self, symbol: str) -> int:
        """This function returns the id of a symbol in a compiled CSL program.

        The id of a symbol is required to be able to write data in a symbol
        (using SdkRuntime.memcpy_h2d()) or read the value of a symbol (using
        SdkRuntime.memcpy_d2h())

        symbol - a string containing the name of the symbol exported in the
        CSL program
        """

        request = sdk_appliance_pb2.SdkGetIdParams(symbol=symbol)
        response = self.stub().sdk_runtime_get_id(request)
        symbol_id = -1
        self._check_sdk_response(response.status, "SdkRuntime.get_id()")
        symbol_id = response.id

        return symbol_id

    def _check_task_status(self, task: Task, wait: bool) -> bool:
        """This functions checks the status of a Task task if wait is False,
        or waits for the completion of the tasks if wait is True.
        If the task is done, any output numpy array return by the operation
        will be copied in the task.dest attribute.

        task - a Task object
        wait - whether to blocking wait until task is complete or not"""

        op_index = task.op_index
        if task.is_done:
            return True
        else:
            # if the request has no associated output data (not a D2H)
            # we may not have to query the server
            if task.dest is None:
                if op_index < self._last_finished_op_index:
                    task.is_done = True
                    return True

            request = sdk_appliance_pb2.SdkRuntimeTask(op_index=op_index)
            if wait:
                response_iterator = self.stub().sdk_runtime_task_wait(request)
                funcname = "SdkRuntime.Task.wait()"
            else:
                response_iterator = self.stub().sdk_runtime_is_task_done(
                    request
                )
                funcname = "SdkRuntime.Task.done()"
            offset = 0
            for response in response_iterator:
                if response.HasField("status"):
                    if (
                        response.status.code
                        != sdk_common_pb2.StatusCode.SDK_RT_SUCCESS
                    ):
                        if response.status.code in [
                            sdk_common_pb2.StatusCode.SDK_RT_INTERNAL_ERROR,
                            sdk_common_pb2.StatusCode.SDK_RT_FILE_ERROR,
                        ]:
                            raise ApplianceUnknownError(
                                self.sdk_support_msg(
                                    f"{funcname} error: {response.status.message}"
                                )
                            )
                        else:
                            raise ApplianceUnknownError(
                                f"{funcname} error: {response.status.message}"
                            )
                elif response.HasField("info"):
                    if response.info.HasField("state"):
                        task.is_done = response.info.state.done
                    elif response.info.HasField("d2h_data"):
                        # done is implicit in that case
                        task.is_done = True
                        dtype = np.dtype(task.dest.dtype)
                        data = response.info.d2h_data
                        num_elems = int(data.num_bytes / dtype.itemsize)
                        task.dest.ravel()[offset : offset + num_elems] = (
                            np.frombuffer(data.data_chunk, dtype=dtype)
                        )
                        offset += num_elems
                    else:
                        raise ApplianceUnknownError(
                            self.sdk_support_msg(
                                """SdkTaskDoneResponse.info doesn't
                            have a state of a d2h_data field"""
                            )
                        )
                else:
                    raise ApplianceUnknownError(
                        self.sdk_support_msg(
                            """SdkTaskDoneResponse doesn't have a
                        status or an info field"""
                        )
                    )
        if task.is_done and task.dest is not None:
            total_size = task.dest.size
            if offset != total_size:
                raise ApplianceUnknownError(
                    self.sdk_support_msg(
                        f"""Missing SdkRuntime.memcpy_d2h()
                    data: received {offset} while expecting {total_size}"""
                    )
                )

        # if task is done, remove it from the dictionnary of pending nonblocking
        # operations
        if task.is_done:
            # update the index of the last done task
            self._last_finished_op_index = max(
                self._last_finished_op_index, op_index
            )
            del self._nonblock_task[op_index]

        return task.is_done

    def is_task_done(self, task: Task) -> bool:
        """Returns a boolean indicating if the SdkRuntime.Task task has
        completed"""

        return self._check_task_status(task, False)

    def task_wait(self, task: Task):
        """Blocks until the SdkRuntime.Task task has completed"""

        self._check_task_status(task, True)

    def download_artifact(self, artifact_name: str, out_path: str):
        """Allows to download an artifact from the appliance.

        artifact_name - name of the artifact to download. If it is a directory,
        a tarball of that directory will be created and transferred.

        Returns the name of the file that has been written (can contain a
        .tar.gz extension if the artifact_name was a directory)
        """

        if self._active:
            raise RuntimeError(
                """SdkRuntime.download_artifact() cannot be
            called while the runtime is active"""
            )
        SdkClient.download_artifact(self, artifact_name, out_path)


class SdkCompiler(SdkClient):
    """SdkCompiler allows to compile SDK programs written in CSL for CSX.
    In a separate stage, compiled programs can be run on CSX or simulator
    using a SdkRuntime object.
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        """Creates initial connection and configures client.
        Args:
            app_path(str): Path to the compile artifacts
        """
        sdk_mode = SdkMode.SDK_COMPILE
        # call the parent class constructor
        super().__init__(sdk_mode, **kwargs)

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self._stub is not None:
            response = self.stub().sdk_shutdown(
                sdk_appliance_pb2.SdkEmptyMessage()
            )
            self._check_sdk_response(response, "SdkCompiler shutdown")
        super().__exit__(exc_type, exc_value, exc_tb)

    def compile(
        self, app_path: str, csl_main: str, options: str, out_path: str
    ) -> str:
        """Compiles every .csl files contained in app_path, using csl_main as
        the main csl file. Returns the artifact_id corresponding to the
        compiled program.

        app_path - a directory containing the .csl source files. Sub-directories
        containing other .csl files will be considered as well.
        csl_main - the .csl file that will be used as the main file of the
        program (usually the file where the "layout" is set).
        options - a string containing CSL compiler options.
        out_path - the path where to place the compile artifact on the user's
        machine. Note that the compiled artifact will be a tar.gz file.

        Returns the name of the compiled program artifact
        """

        options = csl_main + " " + options
        option_list = options.split()
        csl_hash, _ = get_artifact_hashes(app_path, options)
        # add a cs_ prefix to be compatible with cluster_mgmt
        # automatic cleanup process of the compile cache
        compile_hash = "cs_" + csl_hash

        # STEP 1 transfer artifacts
        csl_files = get_csl_files(app_path)
        response = self._upload_files(app_path, compile_hash, csl_files)

        if response.code != sdk_common_pb2.StatusCode.SDK_RT_ALREADY_COMPILED:
            # STEP 2 compile artifacts
            response = self.stub().sdk_compile(
                sdk_appliance_pb2.SdkCompilerArgs(
                    compile_hash=compile_hash,
                    options=option_list,
                )
            )

            # response.message typically contains a string representation of a bytes object.
            # To properly display, we must remove the b'' wrapper and convert escaped chars.
            if response.message.startswith("b'"):
                cleaned_message = ast.literal_eval(response.message).decode(
                    'utf-8'
                )
                if not cleaned_message.strip():
                    cleaned_message = "CSL compiler produced no messages. Compilation successful."
            else:
                cleaned_message = response.message
                if not cleaned_message.strip():
                    cleaned_message = "CSL compiler produced no messages. Compilation successful."

            cleaned_message = textwrap.indent(cleaned_message, prefix="\t")
            LOGGER.info(f"CSL compiler output:\n{cleaned_message}")

            response.message = ""
            self._check_sdk_response(response, "SdkCompiler.compile() failed.")
        else:
            LOGGER.info("Application was found in the compile cache.")

        artifact_name = self.download_artifact(compile_hash, out_path)
        return artifact_name
