# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""gRPC channel interceptors."""

import grpc

from cerebras.appliance.errors import PicklableRpcError
from cerebras.appliance.utils.signal import on_sigint


class ExceptionFormattingInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
    grpc.StreamUnaryClientInterceptor,
    grpc.StreamStreamClientInterceptor,
):  # pylint: disable=no-init
    """gRPC interceptor class that replaces pure gRPC error with a custom gRPC exception."""

    def intercept_unary_unary(self, continuation, client_call_details, request):
        """Intercept "Single request, single response" RPC calls."""
        return self._intercept_call(continuation, client_call_details, request)

    def intercept_unary_stream(
        self, continuation, client_call_details, request
    ):
        """Intercept "Single request, streaming response" RPC calls."""
        try:
            response_generator = continuation(client_call_details, request)
            with on_sigint(lambda *a: response_generator.cancel()):
                yield from response_generator
        except grpc.RpcError as e:
            raise PicklableRpcError.from_grpc_error(e) from None

    def intercept_stream_unary(
        self, continuation, client_call_details, request_iterator
    ):
        """Intercept "Streaming request, single response" RPC calls."""
        return self._intercept_call(
            continuation, client_call_details, request_iterator
        )

    def intercept_stream_stream(
        self, continuation, client_call_details, request_iterator
    ):
        """Intercept "Streaming request, streaming response" RPC calls."""
        try:
            response_generator = continuation(
                client_call_details, request_iterator
            )
            with on_sigint(lambda *a: response_generator.cancel()):
                yield from response_generator
        except grpc.RpcError as e:
            raise PicklableRpcError.from_grpc_error(e) from None

    @staticmethod
    def _intercept_call(continuation, client_call_details, request_or_iterator):
        response = continuation(client_call_details, request_or_iterator)
        return (
            PicklableRpcError.from_grpc_error(response.exception())
            if isinstance(response.exception(), grpc.RpcError)
            else response
        )
