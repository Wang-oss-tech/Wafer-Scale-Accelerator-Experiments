# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""signal handling utilities"""
import signal
import threading
from contextlib import contextmanager


@contextmanager
def on_sigint(callback):
    """
    Register a callback to be run on sigint.

    Callback is unregistered once the context is exited.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    prev_sigint_handler = signal.getsignal(signal.SIGINT)

    def wrapped_callback(signum, frame):
        callback(signum, frame)

        if prev_sigint_handler is not None and prev_sigint_handler not in (
            signal.SIG_DFL,
            signal.SIG_IGN,
        ):
            prev_sigint_handler(signum, frame)

    signal.signal(signal.SIGINT, wrapped_callback)

    try:
        yield
    finally:
        # restore original signal handler
        if prev_sigint_handler is not None:
            signal.signal(signal.SIGINT, prev_sigint_handler)
