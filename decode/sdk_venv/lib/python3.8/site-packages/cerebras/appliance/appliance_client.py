# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

""" GRPC Client Used by Framework User to connect to Coordinator.
"""
import copy
import itertools
import json
import logging
import os
import queue
import signal
import threading
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import dill
import grpc
import numpy as np
from grpc import ChannelCredentials
from grpc._channel import _InactiveRpcError

from cerebras.appliance import logger
from cerebras.appliance.data.conversions import (
    np_dtype_from_rtfx_dtype,
    rtfx_dtype_from_np_dtype,
)
from cerebras.appliance.errors import (
    ApplianceClientException,
    ApplianceDeadlockError,
    ApplianceRequestCancelled,
    ApplianceResourceExhausted,
    ApplianceRuntimeServerError,
    ApplianceServiceInterrupted,
    ApplianceTensorDropped,
    ApplianceUnknownError,
)
from cerebras.appliance.pb.framework.appliance_service_pb2 import (
    CancelPromptRequest,
    CarryOverFromPTRRequest,
    CheckCompileCompatibilityRequest,
    CheckCompileCompatibilityResponse,
    CompileRequest,
    CompileResponse,
    DataCheckpointRequest,
    DeleteFromPTRRequest,
    DoneRequest,
    FinalizeRequest,
    FinalizeResponse,
    GetCMDStateIdsRequest,
    GetCMDStateRequest,
    GetFromPTRRequest,
    GetOutputRequest,
    HeartBeatRequest,
    InferenceRequest,
    InferenceResponse,
    InferenceStatsRequest,
    InferenceStatsResponse,
    InitRequest,
    LoadRequest,
    MonitorErrorRequest,
    MoveToPTRRequest,
    MsgQueueRequest,
    ReconfigureRequest,
    RunInferenceRequest,
    RunRequest,
    SaveWeightsRequest,
    SendCheckGroup,
    SendCheckRequest,
    SendDeferredInputRequest,
    SendInputRequest,
    StartRequest,
    StartStreamingRequest,
    StartWeightIngestionRequest,
    SyncRequest,
)
from cerebras.appliance.pb.framework.appliance_service_pb2_grpc import (
    ApplianceStub,
)
from cerebras.appliance.pb.workflow.appliance.common.common_config_pb2 import (
    ClusterDetails,
)
from cerebras.appliance.pb.workflow.appliance.common.message_queue_pb2 import (
    ValidTopics,
)
from cerebras.appliance.pb.ws.common_pb2 import (
    WS_RT_DROPPED_TENSOR,
    WS_RT_MONITOR_CLEAN,
    WS_RT_REQUEST_CANCELLED,
    WS_RT_SERVICE_INTERRUPTED,
    WS_RT_STALL_DETECTED,
    WS_RT_SUCCESS,
    ClockSyncRequest,
    PingRequest,
    StatusCode,
)
from cerebras.appliance.pb.ws.rtfx_pb2 import RtFxProto
from cerebras.appliance.utils import version_check
from cerebras.appliance.utils.interceptors import ExceptionFormattingInterceptor

MAX_MESSAGE_LENGTH = (1024 * 1024 * 1024 * 2) - 1024  # 2GB - 1 KB
MAX_TRANSFER_BYTES = 256 * 1024 * 1024  # Use 256MiB chunks

RETRY_POLICY = {
    "methodConfig": [
        {
            "name": [{"service": "cerebras.Appliance"}],
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


@dataclass
class HeartBeatOptions:
    """Options to control appliance heartbeat signals."""

    cycle_seconds: int = 30
    cycle_threshold: int = 10

    def __post_init__(self) -> None:
        if self.cycle_seconds <= 0:
            raise ValueError(
                f"`cycle_seconds` must be greater than 0. "
                f"Got {self.cycle_seconds}."
            )
        if self.cycle_threshold <= 0:
            raise ValueError(
                f"`cycle_threshold` must be greater than 0. "
                f"Got {self.cycle_threshold}."
            )


OnErrorCallable = Callable[[Exception], None]


def make_channel_options(
    default_authority: Optional[str] = None,
    service_config: Optional[dict] = None,
):
    """Creates initial connection and configures client.
    Args:
        default_authority: Authority to authorize communication.
        service_config: The gRPC service config to use for things like retry
          policy, etc.
    """
    if service_config is None:
        service_config = RETRY_POLICY
    channel_options = [
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_metadata_size', MAX_MESSAGE_LENGTH),
        ("grpc.per_rpc_retry_buffer_size", MAX_MESSAGE_LENGTH),
        ("grpc.enable_retries", 1),  # Only required until grpc > 1.39
        ('grpc.service_config', json.dumps(service_config)),
    ]
    if default_authority is not None:
        channel_options.append(('grpc.default_authority', default_authority))
        # We usually connect via IP, but nginx-ingress is set up to do SNI
        # dependent cert lookup. In gRPC versions > 1.39, this is set
        # automatically to the gprc.default_authority, but in older
        # versions it needs to be manually set.
        channel_options.append(
            ('grpc.ssl_target_name_override', default_authority)
        )
    return channel_options


class ApplianceClient:
    """Manages connections to Coordinator GRPC Server."""

    def __init__(
        self,
        crd_address: str,
        credentials: Optional[ChannelCredentials] = None,
        default_authority: Optional[str] = None,
        heartbeat_options: Optional[HeartBeatOptions] = None,
        execution_strategy: Optional[int] = None,
        disable_version_check: bool = False,
        retry_small_payload: bool = False,
        max_transfer_bytes: Optional[int] = None,
        service_config: Optional[dict] = None,
        on_error: Optional[OnErrorCallable] = None,
    ) -> None:
        """Creates initial connection and configures client.
        Args:
            crd_address: Address of grpc server to conect to.
            credentials: GRPC Channel Credentials to establish secure channel.
            default_authority: Authority to authorize communication.
            heartbeat_options: Options to control appliance heartbeat signals.
                If None, heartbeat is disabled.
            execution_strategy (ExecutionStrategy): The execution strategy to
                initialize the server for. This is either pipeline or weight
                streaming. If None, it assumes the server has already been
                initialized with the appropriate execution strategy. Defaults to
                None.
            disable_version_check: Whether to disable checking versions across components.
            retry_small_payload: Whether to retry sending small payloads upon server exhaustion.
            max_transfer_bytes: Maximum chunk size to use for data transfer.
            service_config: The gRPC service config to use for things like retry policy, etc.
        """
        self.grpc_fork_support_value = os.environ.get(
            'GRPC_ENABLE_FORK_SUPPORT', None
        )
        # SW-89390: To suppress the spam messages from gRPC library
        os.environ.update({'GRPC_ENABLE_FORK_SUPPORT': '0'})
        self.retry_small_payload = retry_small_payload
        self.max_transfer_bytes = max_transfer_bytes
        if not max_transfer_bytes:
            self.max_transfer_bytes = MAX_TRANSFER_BYTES

        def def_on_error(exc):
            raise exc

        self._on_error = on_error or def_on_error

        channel_options = make_channel_options(
            default_authority, service_config
        )
        if credentials:
            self._channel = grpc.secure_channel(
                crd_address,
                credentials,
                options=channel_options,
            )
        else:
            self._channel = grpc.insecure_channel(
                crd_address,
                options=channel_options,
            )

        self._channel = grpc.intercept_channel(
            self._channel,
            ExceptionFormattingInterceptor(),
        )
        self._heartbeat_stop = threading.Event()

        self.stub = ApplianceStub(self._channel)
        logger.debug(f"ApplianceStub started at address: {crd_address}")

        # Clock Sync on connection to verify server in READY state
        self.clock_sync()

        # Initialize service for given execution strategy
        if execution_strategy is not None:
            logger.debug("Sending start request")
            start_response = self.stub.UnaryStart(
                StartRequest(
                    execution_strategy=execution_strategy,
                    client_version_enforce=True,
                )
            )
            _check_rpc_status(start_response)
            if not disable_version_check:
                crd_version = start_response.version_info.version
                crd_hash = start_response.version_info.githash
                version_check("CBCORE", crd_version, crd_hash)

        if heartbeat_options is not None:
            if not isinstance(heartbeat_options, HeartBeatOptions):
                raise ValueError(
                    f"`heartbeat_options` must either be `None` (to disable "
                    f"hearbeats or an object of type `HeartBeatOptions`, "
                    f"but got `{type(heartbeat_options)}`."
                )

            logger.debug(
                f"Starting heartbeat thread. Heartbeat requests will be sent "
                f"every {heartbeat_options.cycle_seconds} seconds."
            )
            threading.Thread(
                target=_heartbeat_thread,
                args=(
                    self.stub,
                    copy.deepcopy(heartbeat_options),
                    self._heartbeat_stop,
                ),
                name="heartbeat_thread",
                daemon=True,
            ).start()

        # Monitor Coordinator for potential Runtime server error
        self._monitor_result = None
        self._is_monitoring = False
        # Track activities occuring during shutdown
        self._shut_down = threading.Event()

    def __del__(self):
        self.stop_heartbeat()

        if self.grpc_fork_support_value is not None:
            os.environ.update(
                {'GRPC_ENABLE_FORK_SUPPORT': self.grpc_fork_support_value}
            )
        else:
            os.environ.unsetenv('GRPC_ENABLE_FORK_SUPPORT')

    def set_on_error(self, on_error: OnErrorCallable) -> None:
        """Update the error handler called for any fatal monitoring result."""
        self._on_error = on_error

    def clock_sync(self, timeout=300) -> bool:
        """Command to do a clock sync with Coordinator Server."""
        response = self.stub.UnaryClockSync(
            ClockSyncRequest(),
            wait_for_ready=True,
            timeout=timeout,
        )
        return response.code == WS_RT_SUCCESS

    def stop_heartbeat(self) -> None:
        """Command to stop heart beat with Coordinator Server."""
        if not self._heartbeat_stop.is_set():
            logger.debug("Signalling heartbeat thread to stop")
            self._heartbeat_stop.set()

    def stop_monitor_error(self) -> None:
        """Command to stop async thread monitoring Coordinator Errors."""
        if self._is_monitoring:
            logger.debug("Signalling monitor thread to stop")
            self._is_monitoring = False

    def ping(self, timeout=None) -> bool:
        """Command to ping Coordinator Server."""
        logger.debug("Pinging coordinator")
        request = PingRequest(message="Hello, Coordinator!")
        response = self.stub.UnaryPing(request, timeout=timeout)
        logger.debug("Pinged coordinator")
        return response.code == WS_RT_SUCCESS

    @property
    def shutdown(self):
        """Shutdown event to avoid access to server after termination."""
        return self._shut_down

    def done(self, timeout=300) -> bool:
        """Command to shutdown Coordinator Server."""
        success = False
        logger.debug("Sending done request")
        self._shut_down.set()
        try:
            response = self.stub.UnaryDone(DoneRequest(), timeout=timeout)
        except grpc.RpcError as rpc_error:
            rpc_error_code = rpc_error.code()  # pylint: disable=no-member
            if rpc_error_code == grpc.StatusCode.DEADLINE_EXCEEDED:
                logger.debug(
                    f"{timeout}-sec deadline exceeded,"
                    "Check whether the server stub is live and able to receive"
                    "requests from the client."
                )
            elif rpc_error_code == grpc.StatusCode.UNAVAILABLE:
                logger.debug(
                    "Coordinator is unavailable,"
                    "Check whether the server stub is live and"
                    "able to receive requests from the client."
                )
                success = True
            else:
                logger.debug(
                    f"RPC error code was: {rpc_error_code},"
                    "Check whether the server stub is live"
                    "and able to receive requests from the client."
                )
        else:
            logger.debug(f"Response code: {response.code}")
            logger.debug(f"Response message: {response.message}")
            success = response.code == WS_RT_SUCCESS
        self.stop_heartbeat()
        return success

    def compile(self, request: CompileRequest) -> CompileResponse:
        """Command to send artifacts to Coordinator."""
        logger.debug("Sending init request")
        response = self.stub.UnaryCompile(request)
        logger.debug(f"Compile dir: {response.cache_compile_dir}")
        return response

    def check_compile_compatibility(
        self, request: CheckCompileCompatibilityRequest
    ) -> CheckCompileCompatibilityResponse:
        """Command to check compile compatibility on Coordinator."""
        logger.debug("Sending compile compatibility check request")
        return self.stub.UnaryCheckCompileCompatibility(request)

    def finalize(self, request: FinalizeRequest) -> FinalizeResponse:
        """Finalize the run and allow server to run any cleanup actions."""
        logger.debug("Sending finalize request")
        response = self.stub.UnaryFinalize(request)
        _check_rpc_status(response)
        return response

    def init_servers(self, request: InitRequest) -> None:
        """Command to initialize Runtime Command and Weight servers."""
        logger.debug("Sending init request")
        response = self.stub.UnaryInit(request)
        _check_rpc_status(response)

    def load_rtir(self, request: LoadRequest) -> None:
        """Command to load RT IR to Runtime Command and Weight servers."""
        logger.debug("Sending load request")
        response = self.stub.UnaryLoad(request)
        _check_rpc_status(response)

    def sync(self, task_type: ClusterDetails.TaskInfo.TaskType = None) -> None:
        """Command to synchronize Runtime Command and Weight servers."""
        logger.debug("Sending sync request")
        try:
            request = SyncRequest()
            if task_type is not None:
                request.task_type = task_type

            response = self.stub.UnarySync(request, wait_for_ready=True)
        except grpc.RpcError as rpc_error:
            # pylint: disable=no-member
            rpc_error_code = rpc_error.code()
            # pylint: disable=no-member
            rpc_error_details = rpc_error.details()
            error_msg = (
                f"Received gRPC error ({rpc_error_code}) : "
                f"'{rpc_error_details}' while sending sync request"
            )
            if rpc_error_code == grpc.StatusCode.RESOURCE_EXHAUSTED:
                raise ApplianceResourceExhausted(error_msg) from rpc_error
            else:
                raise ApplianceUnknownError(error_msg) from rpc_error
        _check_rpc_status(response)

    def run_deferred(self, request: RunRequest) -> None:
        """
        Command to run Runtime Command and Weight servers for a given iteration.
        """
        logger.debug("Sending run request")
        response = self.stub.UnaryRun(request, wait_for_ready=True)
        _check_rpc_status(response)

    def run_inference(self, request: RunInferenceRequest) -> None:
        """
        Command to start the inference hosting SWDRIVER.
        """
        logger.debug("Sending run request")
        response = self.stub.UnaryRunInference(request, wait_for_ready=True)
        _check_rpc_status(response)

    def start_streaming(self) -> None:
        """Command to put Runtime in streaming mode."""
        logger.debug("Sending start_streaming request")
        response = self.stub.UnaryStartStreaming(
            StartStreamingRequest(), wait_for_ready=True
        )
        _check_rpc_status(response)

    def start_weight_ingestion(self, tensor_id_shard_ids) -> None:
        """Command to put Runtime in streaming mode."""
        logger.debug("Sending start_weight_ingestion request")

        def create_id_pair(tensor_id, shard_id):
            if isinstance(tensor_id, str):
                return StartWeightIngestionRequest.IDPair(
                    tensor_name=tensor_id,
                    shard_id=shard_id,
                )
            else:
                return StartWeightIngestionRequest.IDPair(
                    tensor_id=tensor_id,
                    shard_id=shard_id,
                )

        response_generator = self.stub.StartWeightIngestion(
            StartWeightIngestionRequest(
                tensor_id_shard_ids=[
                    create_id_pair(tensor_id, shard_id)
                    for tensor_id, shard_id in tensor_id_shard_ids
                ]
            ),
            wait_for_ready=True,
        )

        for response in response_generator:
            # The server responded, but we still need to check if
            # StartWeightIngestionResponse has any indication for an error
            _check_rpc_status(response)

            yield response

    def send_check(
        self, iteration, info_type=SendCheckRequest.InfoType.ID
    ) -> Union[List[int], List[str], List[SendCheckGroup]]:
        """
        Command to check which tensors are expected by Runtime Weight servers.
        """
        logger.debug("Sending send_check request")
        request = SendCheckRequest(iteration=iteration, info_type=info_type)
        response = self.stub.UnarySendCheck(request, wait_for_ready=True)
        _check_rpc_status(response)
        if info_type == SendCheckRequest.InfoType.ID:
            return response.tensor_ids
        elif info_type == SendCheckRequest.InfoType.NAME:
            return response.tensor_names
        elif info_type == SendCheckRequest.InfoType.GROUP:
            return response.tensor_groups
        else:
            raise ValueError(
                f"Invalid info type: {SendCheckRequest.InfoType.Name(info_type)}"
            )

    def send_weight(
        self,
        iteration: int,
        tensor_info: Union[int, str],
        tensor_value: np.ndarray,
        scalar_broadcast: bool = False,
        cancel: Optional[threading.Event] = None,
    ) -> None:
        """Command to send weight tensors to Runtime Weight servers.

        Args:
            iteration: Iteration number that this weight is targeted for.
            tensor_info: Name or ID of the tensor.
            tensor_value: The tensor content.
            scalar_broadcast: If true, the tensor (usually a scalar) will be
                              broadcasted to the larger tensor at server side.
            cancel: Event to cancel the request if set.
        """
        logger.debug(f"Sending weight tensor `{tensor_info}`")
        return self.send_tensor(
            iteration,
            tensor_info,
            tensor_value,
            self.stub.SendInputBidirStream,
            scalar_broadcast,
            cancel,
        )

    def send_cmd_state(
        self,
        tensor_info: int,
        tensor_value: np.ndarray,
    ) -> None:
        """Command to send weight tensors to Runtime Weight servers.

        Args:
            tensor_info: The ID of the CMD state tensor.
            tensor_value: The CMD state tensor content.
        """
        logger.debug(f"Sending CMD state `{tensor_info}`")
        return self.send_tensor(
            0,
            tensor_info,
            tensor_value,
            self.stub.SendCMDStateBidirStream,
        )

    def send_tensor(
        self,
        iteration: int,
        tensor_info: Union[int, str],
        tensor_value: np.ndarray,
        send_tensor_callback,
        scalar_broadcast: bool = False,
        cancel: Optional[threading.Event] = None,
    ) -> None:
        """Command to send tensors to Runtime servers.

        Args:
            iteration: Iteration number that this tensor is targeted for.
            tensor_info: Name or ID of the tensor.
            tensor_value: The tensor content.
            scalar_broadcast: If true, the tensor (usually a scalar) will be
                              broadcasted to the larger tensor at server side.
        """

        # gRPC streaming API logs a generic error message when there's an error.
        # So we cache the exception here and reraise it in the request handler.
        generator_error = None
        flow_control = queue.Queue()

        def request_generator(bytes_per_chunk):
            try:
                yield from _chunked_tensor_stream(
                    flow_control,
                    iteration,
                    tensor_info,
                    tensor_value,
                    bytes_per_chunk,
                    scalar_broadcast,
                )
            except Exception as e:
                nonlocal generator_error
                generator_error = e
                raise

        transfer_bytes = self.max_transfer_bytes
        transfer_retries = 0

        retries = 0
        while True:
            try:
                response = None
                response_generator = send_tensor_callback(
                    request_generator(transfer_bytes)
                )
                for response in response_generator:
                    # Technically Only the first needs to be put in...
                    flow_control.put(response)

                    if cancel is not None and cancel.is_set():
                        return
                break
            except grpc.RpcError as rpc_error:
                rpc_error_code = rpc_error.code()  # pylint: disable=no-member
                rpc_error_details = (
                    rpc_error.details()  # pylint: disable=no-member
                )
                logger.warning(
                    f"gRPC error code {rpc_error_code} when "
                    f"sending weight tensor {tensor_info}"
                )
                if (
                    isinstance(rpc_error, _InactiveRpcError)
                    and rpc_error_details == "GOAWAY received"
                    and retries < 1
                ):
                    logger.warning(f"Retrying GOAWAY for tensor: {tensor_info}")
                    retries += 1
                elif rpc_error_code == grpc.StatusCode.RESOURCE_EXHAUSTED:
                    # We will retry with smaller chunks if provided via debug_usr.
                    if self.retry_small_payload and transfer_retries < 10:
                        # Clamp transfer bytes down to tensor size
                        transfer_bytes = min(
                            tensor_value.nbytes, transfer_bytes
                        )
                        # Also, keep decreasing by half with every retry
                        transfer_bytes = max(1, transfer_bytes // 2)
                        transfer_retries += 1
                        logger.warning(
                            f"Retrying RESOURCE_EXHAUSTED for tensor "
                            f"'{tensor_info}' for {transfer_retries} time(s)."
                        )
                    else:
                        raise ApplianceResourceExhausted(
                            f"Failed to send weight tensor '{tensor_info}' with "
                            f"{tensor_value.nbytes} bytes at iteration {iteration} "
                            f"due to exhaustion of resources at Coordinator server. "
                            f"Number of transfer retries = {transfer_retries}. "
                            f"Smallest amount of transfer bytes = {transfer_bytes}. "
                        ) from rpc_error
                elif generator_error is not None:
                    raise ApplianceClientException(
                        f"Failed to send weight tensor {tensor_info} at "
                        f"iteration {iteration} due to error when generating "
                        f"requests."
                    ) from generator_error
                elif not self.retry_small_payload and transfer_bytes > 1:
                    raise ApplianceUnknownError(
                        f"Failed to send weight tensor {tensor_info} at "
                        f"iteration {iteration}."
                    ) from rpc_error

        _check_rpc_status(response)

    def send_deferred_tensor_group(
        self,
        iteration: int,
        tensors: List[SendDeferredInputRequest.Tensor],
        tensor_names: List[str] = None,
        shard_id: int = -1,
    ) -> None:
        """Command to send init graphs to Runtime servers.

        Args:
            iteration: Iteration number that this tensor is targeted for.
            tensors: List of deferred tensors to send.
            tensor_names: Optional list of names of tensors to send
            shard_id: id of the shard to send the tensors to.
        """

        request = SendDeferredInputRequest(
            iteration=iteration,
            tensors=tensors,
            shard_id=shard_id,
        )

        try:
            response = self.stub.UnarySendDeferredInput(request)
        except grpc.RpcError as rpc_error:
            rpc_error_code = rpc_error.code()  # pylint: disable=no-member

            if tensor_names is None:
                tensor_names = (t.tensor_id for t in tensors)
            tensor_names = ", ".join(map(str, tensor_names))

            logger.warning(
                f"gRPC error code {rpc_error_code} when "
                f"sending weight tensor(s): {tensor_names}"
            )
            raise ApplianceUnknownError(
                f"Failed to send weight tensor(s) {tensor_names} at "
                f"iteration {iteration}."
            ) from rpc_error

        _check_rpc_status(response)

    def get_from_ptr(self, request: GetFromPTRRequest) -> np.ndarray:
        """Receive output tensors from persistent tensor repository."""
        logger.debug(f"Receiving output tensor {request.ptid} from PTR")
        return _recv_output_stream(
            self.stub.GetFromPTRStream(request),
            f"tensor {request.ptid} from persistent tensor repository",
        )

    def recv_output(
        self, iteration, tensor_info: Union[str, int]
    ) -> np.ndarray:
        """Receive output tensors from Runtime Weight servers."""
        logger.debug(f"Receiving output tensor {tensor_info}")
        request = GetOutputRequest(iteration=iteration)
        if isinstance(tensor_info, str):
            request.tensor_name = tensor_info
        else:
            request.tensor_id = tensor_info

        return _recv_output_stream(
            self.stub.GetOutputStream(request),
            f"tensor {tensor_info} for runtime iteration {iteration}",
        )

    def save_weights(
        self,
        iteration,
        weight_infos: List,
        bucket: str,
        key_prefix: str,
        compress_data: bool = True,
    ) -> np.ndarray:
        """Receive output tensors from Runtime Weight servers."""
        logger.debug(f"Saving weights: {[name for name, _, _ in weight_infos]}")

        response_generator = self.stub.SaveWeights(
            SaveWeightsRequest(
                iteration=iteration,
                compress_data=compress_data,
                weights=[
                    SaveWeightsRequest.Weight(
                        name=name,
                        shape=shape,
                        dtype=rtfx_dtype_from_np_dtype(dtype),
                    )
                    for name, shape, dtype in weight_infos
                ],
                s3_path=SaveWeightsRequest.S3(
                    bucket=bucket, key_prefix=key_prefix
                ),
            ),
            wait_for_ready=True,
        )

        for response in response_generator:
            # The server responded, but we still need to check if
            # SaveWeightsResponse has any indication for an error
            _check_rpc_status(response)

            yield response

    def monitor_error_async(self, poll_duration: int = 10) -> None:
        """
        Command to monitor Coordinator Server, which also monitors Runtime
        servers for any crashes, assertions, etc.

        Args:
            poll_duration: Seconds to poll for errors before reconnecting
        """
        if self._is_monitoring:
            return

        def run():
            monitor_error_stop = False
            while (
                not monitor_error_stop
                and not self.shutdown.is_set()
                and self._is_monitoring
            ):
                monitor_error_stop = True
                try:
                    response = self.stub.UnaryMonitorError(
                        MonitorErrorRequest(
                            message="Appliance Client monitoring Coordinator",
                            poll_seconds=poll_duration,
                        ),
                    )
                except grpc.RpcError as rpc_error:
                    # pylint: disable=no-member
                    rpc_error_code = rpc_error.code()
                    rpc_error_details = rpc_error.details()
                    # Skip spurious errors after we've triggered shutdown
                    # Ignore RESOURCE_EXHAUSTED since a memory pool may be
                    # exhausted while transferring weights...
                    if (
                        not self.shutdown.is_set()
                        and self._is_monitoring
                        and self._monitor_result is None
                        and rpc_error_code != grpc.StatusCode.RESOURCE_EXHAUSTED
                    ):
                        self._monitor_result = ApplianceUnknownError(
                            f"Received unexpected gRPC error ({rpc_error_code}): "
                            f"'{rpc_error_details}' while monitoring Coordinator "
                            f"for Runtime server errors"
                        )
                else:
                    if response.code == WS_RT_SUCCESS:
                        logger.debug(
                            f"Monitoring returned success with message: {response.message}"
                        )
                    elif response.code == WS_RT_MONITOR_CLEAN:
                        monitor_error_stop = False
                    elif self.shutdown.is_set() or not self._is_monitoring:
                        # If shutdown is set already, ignore any errors
                        pass
                    elif self._monitor_result is None:
                        if (
                            response.code == WS_RT_SERVICE_INTERRUPTED
                            and response.srv_type
                            in [
                                ClusterDetails.TaskInfo.TaskType.CRD,
                                ClusterDetails.TaskInfo.TaskType.CHF,
                            ]
                        ):
                            self._monitor_result = ApplianceServiceInterrupted(
                                f"{ClusterDetails.TaskInfo.TaskType.Name(response.srv_type)} "
                                f"service was interrupted due to: '{response.message}'"
                            )
                        else:
                            self._monitor_result = ApplianceRuntimeServerError(
                                f"Error detected in "
                                f"{ClusterDetails.TaskInfo.TaskType.Name(response.srv_type)} "
                                f"#{response.host_index}. "
                                f"Status code: {StatusCode.Name(response.code)}. "
                                f"Status message: {response.message}"
                            )
            # Send SIGURG to the process that is running ApplianceClient
            os.kill(os.getpid(), signal.SIGURG)

        # Register SIGURG handler for alarm sent from monitor callback
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "Asynchronously monitoring for errors can only be initiated from the main thread."
            )
        signal.signal(signal.SIGURG, self._alarm_handler)

        # Start two interleaved threads for monitoring
        logger.debug("Starting async error monitoring thread")
        self._is_monitoring = True
        threading.Thread(target=run).start()
        threading.Timer(poll_duration // 2, run).start()

    def close(self) -> None:
        """Close GRPC Channel for client."""
        self.stop_heartbeat()
        self._channel.close()

    def _alarm_handler(self, signum, frame):
        """Handles alarm sent from monitor callback."""
        logger.debug("Inside alarm handler")
        # If monitor is inflight while shutting down can hit closed server
        if self._monitor_result is not None:
            self.stop_heartbeat()
            logger.warning(
                f"Monitor came back with error: '{self._monitor_result}'"
            )
            self._on_error(self._monitor_result)

    def fetch_dataloader_state(self, current_step: int) -> List[Any]:
        """Command to fetch state of WRK(s) feeding the input data pipeline
        at the current appliance step."""
        logger.debug(f"Fetching dataloader state")
        request = DataCheckpointRequest(iteration=current_step)
        response = self.stub.UnaryDataCheckpoint(request)
        _check_rpc_status(response)
        return [
            fw_user_deserialize(
                state, name="Worker DataLoader state", from_usr=False
            )
            for state in response.state_dict_serialized
        ]

    def reconfigure(self, request: ReconfigureRequest):
        """
        Reconfigure the service at runtime.
        """
        response = self.stub.UnaryReconfigure(request)
        _check_rpc_status(response)
        return response

    def get_msg(self, topic: ValidTopics, timeout: int = 10):
        """Retrieve Message from Server Message Queue."""
        request = MsgQueueRequest(topic=topic, timeout_seconds=timeout)
        return self.stub.UnaryMsgQueue(request)

    def carry_over_from_ptr(
        self,
        iteration: int,
        tensor_name: str,
        tensor_id: int,
        keep_in_repo: bool = False,
    ) -> None:
        """Carryover tensor from persistent tensor repository to this session."""
        logger.debug(
            f"Sending carry_over_from_repo request: "
            f"{tensor_name=}, {tensor_id=}, {iteration=}"
        )
        request = CarryOverFromPTRRequest(
            iteration=iteration,
            tensor_name=tensor_name,
            ptid=tensor_id,
            keep_in_repo=keep_in_repo,
        )
        response = self.stub.UnaryCarryOverFromPTR(request, wait_for_ready=True)
        _check_rpc_status(response)

    def move_to_ptr(self, request: MoveToPTRRequest) -> None:
        """Move output tensor to the persistent tensor repository."""
        logger.debug("Sending move_to_ptr request")
        response = self.stub.UnaryMoveToPTR(request, wait_for_ready=True)
        _check_rpc_status(response)

    def delete_from_ptr(self, request: DeleteFromPTRRequest) -> None:
        """Delete tensor from persistent tensor repository."""
        logger.debug("Sending delete_from_ptr request")
        response = self.stub.UnaryDeleteFromPTR(request, wait_for_ready=True)
        _check_rpc_status(response)

    def get_cmd_state_ids(self) -> None:
        """Get the CMD state IDs."""
        logger.debug("Sending get_cmd_state_ids request")
        response = self.stub.UnaryGetCMDStateIds(
            GetCMDStateIdsRequest(), wait_for_ready=True
        )
        return response.ids

    def get_cmd_state(self, request: GetCMDStateRequest) -> None:
        """Get the CMD state IDs."""
        logger.debug("Sending get_cmd_state request")
        return _recv_output_stream(
            self.stub.GetCMDStateStream(request),
            f"tensor {request.id} from CMD state",
        )


class AsyncApplianceClient:
    """Exposes a subset of ApplianceClient endpoints for asyncio."""

    @classmethod
    def from_grpc_client_args(cls, grpc_args) -> "AsyncApplianceClient":
        """
        Consruct a new instance from the grpc_args from
        ApplianceManager.grpc_client_args.
        """
        if cred_string := grpc_args.get("credentials"):
            # Make a copy to avoid modifying the passed in dict
            grpc_args = grpc_args.copy()
            grpc_args["credentials"] = grpc.ssl_channel_credentials(
                cred_string.encode("ascii")
            )
        else:
            grpc_args["credentials"] = None

        return cls(**grpc_args)

    def __init__(
        self,
        crd_address: str,
        credentials: Optional[ChannelCredentials] = None,
        default_authority: Optional[str] = None,
        service_config: Optional[dict] = None,
        num_channels: int = 1,
    ):
        """Creates initial connection and configures client.
        Args:
            crd_address: Address of grpc server to conect to.
            credentials: GRPC Channel Credentials to establish secure channel.
            default_authority: Authority to authorize communication.
            service_config: The gRPC service config to use for things like
              retry policy, etc.
            num_channels: the appliance nginx ingress limits the number of
              streams per HTTP 2 connection to 128. In the main use case for
              async appliance client, we need lots of parallel streams, so
              force the creation of multiple channels, each with distinct
              channels options to prevent the global subchannel pooling from
              combining them.
        """

        channel_options = make_channel_options(
            default_authority, service_config
        )
        channel_options.append(None)
        self._stubs = []
        for i in range(num_channels):
            # Set a custom channel option just to prevent the global subchannel
            # reuse from pooling these all into the same HTTP connection anyway
            channel_options[-1] = ("channel_number", i)

            if credentials:
                channel = grpc.aio.secure_channel(
                    crd_address,
                    credentials,
                    options=channel_options,
                )
            else:
                channel = grpc.aio.insecure_channel(
                    crd_address,
                    options=channel_options,
                )
            self._stubs.append(ApplianceStub(channel))
        self._stub_iterator = itertools.cycle(self._stubs)

        logger.debug(f"ApplianceStub started at address: {crd_address}")

    @property
    def stub(self):
        """
        Return the next stub that is round-robin'd amongst the channels.
        """
        return next(self._stub_iterator)

    async def inference_stats(
        self,
        update_frequency_s: float = 1,
    ) -> AsyncIterator[InferenceStatsResponse]:
        """
        Poll for inference service stats.
        """
        async for resp in self.stub.InferenceStatsStream(
            InferenceStatsRequest(
                update_frequency_s=update_frequency_s,
            )
        ):
            _check_rpc_status(resp)
            yield resp

    async def infer(
        self,
        request: InferenceRequest,
    ) -> AsyncIterator[InferenceResponse]:
        """
        Submit the given prompt to the currently loaded inference model and
        yield tokens as they are generated.

        Args:
            prompt: tokens of the prompt to seed generation
            out_final_resp: This object is modified to contain the final
            response with timing information
        """
        # Since this is used as a service, we may have a delay between the
        # monitor thread detecting an error and shutting down. Early error this
        # request if there are any fatal errors.
        logger.debug("Beginning inference")
        async for resp in self.stub.InferenceStream(request):
            _check_rpc_status(resp)
            yield resp

    async def cancel_request(
        self,
        request: CancelPromptRequest,
    ):
        """
        Cancel the given sequence id.
        """
        response = await self.stub.UnaryCancelPrompt(request)
        _check_rpc_status(response)

    async def reconfigure(self, request: ReconfigureRequest):
        """
        Reconfigure the service at runtime.
        """
        response = await self.stub.UnaryReconfigure(request)
        _check_rpc_status(response)
        return response


def _check_rpc_status(response, name=None, raise_error=True):
    """Raises an exception if gRPC response code is anything but success.

    Args:
        response: gRPC response from appliance service.
        name: Name of the operation to print in debug message.
        raise_error: Raise error if response code is anything but success.
    Raises:
        ApplianceClientException if response code is an error.
    """
    messages = [f"Response code: {response.code}"]
    if hasattr(response, "message"):
        messages.append(f"Response message: {response.message}")

    for msg in messages:
        if name:
            msg = f"{name} {msg}"

        if response.code == WS_RT_SUCCESS or raise_error:
            logger.debug(msg)
        else:
            logger.error(msg)

    if response.code == WS_RT_SUCCESS:
        return

    error_cls = ApplianceDeadlockError
    if response.code == WS_RT_STALL_DETECTED:
        error_cls = ApplianceDeadlockError
    elif response.code == WS_RT_REQUEST_CANCELLED:
        error_cls = ApplianceRequestCancelled
    elif response.code == WS_RT_DROPPED_TENSOR:
        error_cls = ApplianceTensorDropped
    else:
        error_cls = ApplianceUnknownError

    raise error_cls(
        getattr(
            response,
            "message",
            f"Server call failed with {StatusCode.Name(response.code)}",
        )
    )


def _chunked_tensor_stream(
    flow_control: queue.Queue,
    iteration: int,
    tensor_name: Union[int, str],
    tensor_value: np.ndarray,
    bytes_per_chunk: int,
    scalar_broadcast: bool = False,
) -> Generator[SendInputRequest, None, None]:
    """Chunks a numpy array into a stream of SendInputRequests.

    gRPC has a 2GB transfer size limit. As such, any tensor over this limit is
    chunked into a list of smaller tensors that can be streamed to a gRPC
    endpoint.

    Args:
        iteration: Iteration number that this tensor is targeted for.
        tensor_name: Name or ID of the tensor.
        tensor_value: The tensor content.
        bytes_per_chunk: Number of bytes per chunk of tensor.
        scalar_broadcast: If true, the tensor (usually a scalar) will be
                          broadcasted to the larger tensor at server side.
    """
    rtfx_dtype = rtfx_dtype_from_np_dtype(tensor_value.dtype)
    if rtfx_dtype == RtFxProto.T_I1:
        # I1 needs to be encoded as int16
        tensor_value = tensor_value.astype(np.int16)

    tensor_size = tensor_value.size
    tensor_shape = tensor_value.shape
    num_dims = tensor_value.ndim
    num_bytes = tensor_value.nbytes
    num_chunks = 1
    if num_bytes > bytes_per_chunk:
        assert num_dims, "Expects an array"
        logging.debug(f"Shape of tensor '{tensor_name}': {tensor_shape}")
        logging.debug(f"Size of tensor '{tensor_name}': {tensor_size}")
        logging.debug(f"Storage size of tensor '{tensor_name}': {num_bytes}")
        # We will use the number of chunks that is power of 2
        num_chunks = num_bytes // bytes_per_chunk
        num_chunks = 1 << num_chunks.bit_length()
        # We are also making sure every chunk is equal in size
        if num_bytes % num_chunks:
            logging.debug(
                f"Tensor '{tensor_name}' with size of {num_bytes} bytes can't "
                "be put into chunks with an equal size. Dropping back to "
                "sending entire tensor in a single chunk."
            )
            num_chunks = 1
    logging.debug(f"Number of chunks to send: {num_chunks}")
    chunk_size = tensor_size // num_chunks
    tensor_nbytes = num_bytes // num_chunks
    logging.debug(f"Number of items in each chunk: {chunk_size}")
    logging.debug(f"Number of bytes in each chunk: {tensor_nbytes}")
    # Here, we will try to avoid using `np.ravel()` as well by working on
    # memory view directly. Iterating over memory view might be tricky.
    # However, most of the large tensors have the similar dimensionality.
    # Therefore, we can start with very simple special case. We will improve
    # the special cases later if needed.
    use_ravel = True
    chunk_offset = chunk_size
    # Special case where we can avoid `np.ravel()`
    if num_dims == 2:
        row_count = tensor_shape[0]
        col_count = tensor_shape[1]
        if row_count >= num_chunks and row_count % num_chunks == 0:
            if col_count < chunk_size and chunk_size % col_count == 0:
                use_ravel = False
                chunk_offset = chunk_size // col_count
    if use_ravel:
        logging.debug(f"Using 'ravel()' for tensor '{tensor_name}'")
        tensor_memory_view = tensor_value.ravel(order='A').data
    else:
        logging.debug(f"Skipping 'ravel()' for tensor '{tensor_name}'")
        tensor_memory_view = tensor_value.data
    # Now, we are putting the weight tensor into chunk(s)
    for k in range(num_chunks):
        start = k * chunk_offset
        end = start + chunk_offset

        request = SendInputRequest(
            iteration=iteration,
            num_bytes=tensor_nbytes,
            has_more=k < num_chunks - 1,
        )

        if isinstance(tensor_name, str):
            request.tensor_name = tensor_name
        elif isinstance(tensor_name, int):
            request.tensor_id = tensor_name
        else:
            raise TypeError(
                f"Expected tensor info to be an integer ID or string name, "
                f"but got {type(tensor_name)}"
            )

        request.rtfx_proto.dtype = rtfx_dtype
        if scalar_broadcast:
            # Send the original tensor name
            request.rtfx_proto.scalar.name = str(tensor_name)
            request.rtfx_proto.scalar.data = tensor_memory_view[
                start:end
            ].tobytes()
        else:
            # Send the original tensor name and shape
            request.rtfx_proto.tensor.name = str(tensor_name)
            request.rtfx_proto.tensor.shape.extend(tensor_shape)

            if k == 0:
                # Send an empty request so coordinator can allocate buffers and
                # backpressure gRPC if needed
                has_more = request.has_more
                request.has_more = True
                logging.debug(f"Sending metadata only chunk for {tensor_name}")
                yield request
                request.has_more = has_more
                # Block until server gives us the first "go ahread" response.
                _ = flow_control.get()
                logging.debug(f"Received go ahead for {tensor_name}")
                # continue!
            request.rtfx_proto.tensor.data = tensor_memory_view[
                start:end
            ].tobytes()

        logging.debug(f"Sending chunk {k} for {tensor_name}")
        yield request


def _recv_output_stream(output_stream: Iterable, msg: str) -> np.ndarray:
    """Receive an output array through a chunked stream."""
    # Receive large tensor in chunks
    try:
        dtype = None
        shape = None
        buffer = bytearray()
        for response in output_stream:
            # The server responded, but we still need to check if
            # GetOutputResponse has any indication for an error
            _check_rpc_status(response)

            assert response.HasField(
                "rtfx_proto"
            ), "Expects the type of received tensor chunk to be RtFxProto"
            logger.debug(f"Response code: {response.code}")
            logger.debug(f"Response message: {response.message}")
            if dtype is not None:
                assert dtype == response.rtfx_proto.dtype
            else:
                dtype = response.rtfx_proto.dtype
            if shape is not None:
                assert shape == response.rtfx_proto.tensor.shape
            else:
                shape = response.rtfx_proto.tensor.shape
            buffer.extend(response.rtfx_proto.tensor.data)
    except grpc.RpcError as rpc_error:
        rpc_error_code = rpc_error.code()  # pylint: disable=no-member
        logger.debug(f"gRPC error code {rpc_error_code} when receiving {msg}")
        raise ApplianceUnknownError(
            f"Ran into error while receiving {msg}"
        ) from rpc_error

    # Construct np.ndarray
    return _np_from_rtfx(buffer, dtype, shape)


def _np_from_rtfx(
    buffer: bytearray, rtfx_dtype: int, shape: Tuple[int]
) -> np.ndarray:
    """Returns a numpy array from the given buffer with the given rtfx dtype.

    Args:
        buffer: The buffer containing the data.
        rtfx_dtype: The RtFxProto dtype.
        shape: The shape of the tensor.
    Returns:
        The numpy array matching the given buffer.
    """
    # Construct np.ndarray
    dtype = np_dtype_from_rtfx_dtype(rtfx_dtype)
    if dtype == bool:  # RtFx T_I1 is stored as 16-bit int
        dtype = np.int16

    logger.debug(f"Buffer size: {len(buffer)}, {dtype = }, {shape = }")
    if not shape:
        shape = []
    array = np.frombuffer(buffer, dtype=dtype).reshape(shape)

    # I1 comes through as int16, but it _should_ be bool...
    # Might need dtype conversion.
    if rtfx_dtype == RtFxProto.T_I1 and array.dtype != bool:
        array = array.astype(bool)

    return array


def fw_user_serialize(
    obj: Any,
    name: str = "object",
    from_usr: bool = True,
    recurse: bool = False,
    byref: bool = False,
) -> str:
    """Serialized information that client can send the appliance.

    Currently, this handles the input_fn and params for the input_fn.
    These are separate because python can't serialized generators directly,
    so the worker creates the generator locally.

    Args:
        obj: An object to serialize.
        name: A user-friendly name for the object being serialized.
        from_usr: Whether this is a method is being called for an object to send from the user node
            to the Wafer-Scale Cluster or vice-versa.
        recurse: Whether to recursively serialize the object. If enabled, the entire context for the
            closure will be serialized.

    Returns:
        serialized_data: Assumed will be deserialized by fw_user_deserialize.
    """

    src, dst = _get_src_dst_names(from_usr)

    try:
        return dill.dumps(obj, recurse=recurse, byref=byref).hex()
    except Exception as e:
        if recurse:
            logger.debug(
                f"Failed to recursively serialize `{name}` to send from the {src} to the {dst} "
                f"due to the following error: {e}. Attempting to serialize without recursion. "
                "This may result in a partial serialization."
            )
            try:
                return fw_user_serialize(
                    obj, name, from_usr, recurse=False, byref=byref
                )
            except Exception as inner_exception:
                raise inner_exception from None

        if not byref:
            logger.debug(
                f"Failed to serialize `{name}` using byref=False to send from the {src} to "
                f"the {dst} due to the following error: {e}. Attempting to serialize with "
                f"byref=True. This may result in a partial serialization."
            )
            try:
                return fw_user_serialize(
                    obj, name, from_usr, recurse=recurse, byref=True
                )
            except Exception as inner_exception:
                raise inner_exception from None

        raise RuntimeError(
            f"Failed to serialize `{name}` to send from the {src} to the {dst} "
            f"due to the following error: {e}.\nPlease make sure `{name}` is "
            f"picklable using the `dill` package."
        ) from e


def fw_user_deserialize(
    serialized_data: str,
    name: str = "object",
    from_usr: bool = True,
) -> Any:
    """Deserialize consistent with fw_user_serialize.

    Args:
        serialized_data: Output of fw_user_serialize.
        name: A user-friendly name for the object being deserialized.
        from_usr: Whether this is a method is being called for an object to send from the user node
            to the Wafer-Scale Cluster or vice-versa.

    Returns:
        original_data: Whatever was originally passed in to fw_user_serialize.
    """

    src, dst = _get_src_dst_names(from_usr)

    try:
        return dill.loads(bytes.fromhex(serialized_data))
    except ImportError as e:
        raise ImportError(
            f"Failed to deserialize `{name}` sent from the {src} to the {dst} "
            f"due to the following import error: {e}.\nThis is often due to a "
            f"missing path in PYTHONPATH and/or due to the directories required to "
            f"find packages used by the input workers not being included in "
            f"the mounted directories. Please pass all the necessary "
            f"`python_paths` and `mount_dirs` to capture dependencies of the "
            f"input pipeline (e.g., dataloader)."
        ) from e
    except Exception as e:
        raise ValueError(
            f"Failed to deserialize `{name}` sent from the {src} to the {dst} "
            f"due to the following error: {e}.\nThis could be due to corrupted "
            f"data or a mismatch between the data format sent and the data format "
            f"expected. Currently, the only supported formats are JSON and "
            f"pickled python encoded in hex."
        ) from e


def _get_src_dst_names(from_usr: bool) -> Tuple[str, str]:
    """Returns user-friendly names for user and worker nodes."""

    src = "user node"
    dst = "input workers (running in the Wafer-Scale Cluster)"
    return (src, dst) if from_usr else (dst, src)


def _heartbeat_thread(
    stub: ApplianceStub,
    options: HeartBeatOptions,
    stop_event: threading.Event,
    timeout: int = 1,
) -> None:
    """Thread that continuously sends heartbeat signals to the Appliance.

    Args:
        stub: The client to use for sending the heartbeat signals.
        options: HeartBeat configuration options.
        stop_event: Event that will stop the heartbeat thread when set.
        timeout: Timeout for the heartbeat RPC.
    """
    failure_count = 0
    first_request = True
    while (first_request and not stop_event.is_set()) or not stop_event.wait(
        options.cycle_seconds
    ):
        first_request = False

        try:
            stub.UnaryHeartBeat(
                HeartBeatRequest(
                    cycle_seconds=options.cycle_seconds,
                    failure_cycle_threshold=options.cycle_threshold,
                ),
                timeout=timeout,
            )
            if failure_count:
                logger.info(
                    f"Heartbeat to the Appliance succeeded after "
                    f"{failure_count} consecutive failures."
                )
            failure_count = 0
        except grpc.RpcError as rpc_error:
            rpc_error_code = rpc_error.code()  # pylint: disable=no-member
            failure_count += 1
            logger.warning(
                f"Heartbeat to the Appliance failed with error code "
                f"`{rpc_error_code.name}` (consecutive failure count: "
                f"{failure_count}). Retrying in {options.cycle_seconds} "
                f"seconds."
            )
            if failure_count == options.cycle_threshold:
                logger.warning(
                    f"Heartbeat failure count, {failure_count}, has exceeded "
                    f"the threshold. The Appliance will likely begin "
                    f"self-destructing soon as it hasn't heard from the client "
                    f"in a while."
                )

    logger.debug("Heartbeat thread stopped.")
