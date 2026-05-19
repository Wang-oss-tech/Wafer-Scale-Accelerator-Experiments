#!/usr/bin/env python3
# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

""" Timer to keep track of appliance jobs and cancel the jobs when the timer is hit """

import os
import signal
import time
from threading import Lock, Timer

from cerebras.appliance.cluster import cluster_logger

logger = cluster_logger.getChild("job_timer")


class JobTimer:
    """
    JobTimer that keeps track of the total time that appliance jobs are taking, and
    cancels the running appliance jobs after a given time limit (in seconds) is reached.

    The cluster client will register itself to this timer when its appliance job is in
    running state. This timer will keep track of all concurrent cluster clients, and cancel
    them when the time limit is reached.

    When the time limit is hit, an ApplianceRuntimeJobTimeoutError exception is raised in
    USR2 signal handler.
    """

    def __init__(self, time_limit_sec: int):
        self._pid = os.getpid()
        self._time_limit_sec = time_limit_sec

        # Total runtime (in seconds) for all appliance jobs that have been tallied since
        # _last_record_timestamp.
        self._total_time_sec = 0
        self.hit_time_limit = False

        # Last timestamp when total time was tallied.
        self._last_record_timestamp = 0

        # A list of concurrent cluster clients that are in running state.
        self._clients = []

        self._lock = Lock()
        self._timer = None
        self._error_message = (
            f"Appliance job timer time out after {self._time_limit_sec}s"
        )
        logger.debug(f"Init timer with timeout of {self._total_time_sec}s")

        # Use SIGUSR2 to raise an exception when a time limit is hit.
        def _raise_timeout(signum, frame):
            from cerebras.appliance.errors import (
                ApplianceRuntimeJobTimeoutError,
            )

            raise ApplianceRuntimeJobTimeoutError(self._error_message)

        signal.signal(signal.SIGUSR2, _raise_timeout)

    def __getstate__(self):
        """Return state values to be pickled"""
        return (
            self._pid,
            self._time_limit_sec,
            self._total_time_sec,
            self.hit_time_limit,
            self._last_record_timestamp,
            self._clients,
        )

    def __setstate__(self, state):
        """Restore state from the unpickled state values"""
        (
            pid,
            time_limit_sec,
            total_time_sec,
            hit_time_limit,
            last_record_timestamp,
            clients,
        ) = state
        self._pid = pid
        self._time_limit_sec = time_limit_sec
        self._total_time_sec = total_time_sec
        self.hit_time_limit = hit_time_limit
        self._last_record_timestamp = last_record_timestamp
        self._clients = clients

        self._lock = Lock()
        self._timer = None

    def _handle_timeout(self, timer_duration: float):
        self._lock.acquire()
        try:
            if len(self._clients) == 0:
                # Timer was canceled but _handle_timeout was still executed
                return

            # Time limit is hit. Cancel the current running jobs
            self.hit_time_limit = True
            self._total_time_sec += timer_duration
            self._last_record_timestamp = time.time()
        finally:
            self._lock.release()

        logger.error(self._error_message)
        for client in self._clients:
            client.cancel()

        # Send SIGUSR2 to the process that is running JobTimer
        os.kill(self._pid, signal.SIGUSR2)
        logger.debug(f"Sending SIGUSR2 signal to {self._pid}")

    def _start_timer(self):
        """Start the timer"""
        timer_duration = self._time_limit_sec - self._total_time_sec
        self._timer = Timer(
            timer_duration,
            self._handle_timeout,
            args=[timer_duration],
        )
        self._timer.daemon = True
        self._timer.start()
        self._last_record_timestamp = time.time()

    def _stop_timer(self):
        """Stop the current timer"""
        now_time = time.time()
        self._total_time_sec += now_time - self._last_record_timestamp
        logger.debug(f"Total elapsed seconds in timer: {self._total_time_sec}")

        self._last_record_timestamp = now_time
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def register_client(self, client):
        """Register a cluster client whose job is currently running."""
        from cerebras.appliance.cluster.client import ClusterManagementClient

        if not isinstance(client, ClusterManagementClient):
            raise RuntimeError("Only support ClusterManagementClient")
        logger.debug(f"Register a client with timer: {client}")

        if client in self._clients:
            logger.debug(
                f"{client} already registered with timer, skip registering"
            )
            return

        self._lock.acquire()
        try:
            if self.hit_time_limit:
                client.cancel()
                from cerebras.appliance.errors import (
                    ApplianceRuntimeJobTimeoutError,
                )

                raise ApplianceRuntimeJobTimeoutError(self._error_message)

            if len(self._clients) == 0:
                self._start_timer()
            self._clients.append(client)
        finally:
            self._lock.release()

    def unregister_client(self, client):
        """Unregister a cluster client"""
        from cerebras.appliance.cluster.client import ClusterManagementClient

        if not isinstance(client, ClusterManagementClient):
            raise RuntimeError("Only support ClusterManagementClient")

        if client not in self._clients:
            logger.debug(
                f"{client} was not register with timer, skip unregistering"
            )
            return

        logger.debug(f"Unregister a client with timer: {client}")
        self._lock.acquire()
        try:
            if self.hit_time_limit:
                return

            self._clients.remove(client)
            if len(self._clients) == 0:
                self._stop_timer()
        finally:
            self._lock.release()
