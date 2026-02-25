# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause
"""
Global perf view is a tool aimed to help triage perf degradations. It collects
TTFL, network errors and training/eval steps and generates Perfetto trace. It
also generates a performance related run summary for each session.
"""

import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tabulate import tabulate

from cerebras.appliance.log import ClassLogger, named_class_logger
from cerebras.appliance.pb.perfetto.perfetto_trace_pb2 import (
    DebugAnnotation,
    Trace,
    TrackEvent,
)
from cerebras.appliance.pb.workflow.appliance.common.global_perf_view_pb2 import (
    ExecuteJobPerf,
)

# pylint doesn't recognize types in protobuf generated python file
# pylint: disable=no-member


@named_class_logger("GlobalPerfView")
class GPVTraceWriter(ClassLogger):
    """The class to write Global Performance View (GPV) traces in Perfetto.
    Visualized events include training/eval steps, TTFL events and network
    errors."""

    # Each "process" appears as a collapsible top level track in the Perfetto
    # UI. For GPV use cases, we predefine these top level tracks.
    # The pids are used to assign UUID of the actual Perfetto Packets and need
    # to be unique.
    pid_ttfl = 1
    ttfl_name = "TTFL and Session Setups"
    pid_network_errors = 2
    network_errors_name = "Network Errors"
    pid_train_eval_sessions = 3
    train_eval_sessions_name = "Train / Eval"

    # Start of the UUID range for events.
    uuid_start = 1000

    def __init__(self):
        super().__init__()
        self.pb = Trace()
        self.current_session = 1
        self.current_uuid = self.uuid_start
        # Create TTFL "Process"
        packet = self.pb.packet.add()
        packet.trusted_packet_sequence_id = 1
        track_desc = packet.track_descriptor
        track_desc.uuid = self.pid_ttfl
        track_desc.process.pid = self.pid_ttfl
        track_desc.process.process_name = self.ttfl_name
        # Create Network Errors "Process"
        packet = self.pb.packet.add()
        packet.trusted_packet_sequence_id = 1
        track_desc = packet.track_descriptor
        track_desc.uuid = self.pid_network_errors
        track_desc.process.pid = self.pid_network_errors
        track_desc.process.process_name = self.network_errors_name
        # Create Train Eval Sessions "Process"
        packet = self.pb.packet.add()
        packet.trusted_packet_sequence_id = 1
        track_desc = packet.track_descriptor
        track_desc.uuid = self.pid_train_eval_sessions
        track_desc.process.pid = self.pid_train_eval_sessions
        track_desc.process.process_name = self.train_eval_sessions_name

    def _get_uuid(self):
        self.current_uuid += 1
        return self.current_uuid

    def _add_event(
        self,
        pid,
        start_ts,  # timestamp in nanoseconds from epoch
        end_ts,
        track_name,
        event_name,
        debug_annotations: List[DebugAnnotation] = None,
    ):
        if start_ts == 0 or end_ts == 0:
            # This can happen when ltc_backend.py:on_run_end() is called in an
            # early exit scenario.
            self.logger.debug(
                f"Skip logging event {event_name} with zero or negative timestamps"
            )
            return
        if start_ts > end_ts:
            # Should not happen
            raise ValueError("start_ts must be less than or equal to end_ts")
        # First create the "Track" (can be thought of as threads) under the
        # process.
        packet = self.pb.packet.add()
        packet.trusted_packet_sequence_id = 1
        track_desc = packet.track_descriptor
        event_uuid = self._get_uuid()
        track_desc.uuid = event_uuid
        track_desc.name = track_name
        track_desc.parent_uuid = pid

        # Then create the actual event that happens inside the track/thread.
        start = self.pb.packet.add()
        start.trusted_packet_sequence_id = 1
        start.timestamp = start_ts
        start.track_event.name = event_name
        start.track_event.track_uuid = event_uuid
        if start_ts == end_ts:
            # instant event
            start.track_event.type = TrackEvent.Type.TYPE_INSTANT
        else:
            # duration event
            start.track_event.type = TrackEvent.Type.TYPE_SLICE_BEGIN
            end = self.pb.packet.add()
            end.trusted_packet_sequence_id = 1
            end.timestamp = end_ts
            end.track_event.track_uuid = event_uuid
            end.track_event.type = TrackEvent.Type.TYPE_SLICE_END
        if debug_annotations:
            start.track_event.debug_annotations.extend(debug_annotations)

    def add_session_init_stats(self, events: Dict[str, Tuple[int, int]]):
        """Add events for the initialization of a session. Called at the end of
        a session.
        """
        for name, (start, end) in events.items():
            self._add_event(self.pid_ttfl, start, end, "Events", name)

    def add_session(self, start_ts, end_ts):
        """Write a session event to the Perfetto trace."""
        track_name = f"Session {self.current_session}"
        self._add_event(
            self.pid_train_eval_sessions,
            start_ts,
            end_ts,
            track_name,
            track_name,  # Use the same name "Session X" for this event
        )
        self.current_session += 1

    def add_step(
        self,
        start_ts: int,
        end_ts: int,
        step_index: int,  # user_iteration that starts with 1
        samples_per_second: float,
        global_samples_per_second: float,
        is_checkpoint_step: bool,
    ):
        """Write a step event to the Perfetto trace."""
        track_name = f"Session {self.current_session}"
        debug_annotations = []
        anno_sps = DebugAnnotation()
        anno_sps.name = "samples_per_second"
        anno_sps.double_value = samples_per_second
        anno_gsps = DebugAnnotation()
        anno_gsps.name = "global_samples_per_second"
        anno_gsps.double_value = global_samples_per_second
        anno_ckpt = DebugAnnotation()
        anno_ckpt.name = "is_checkpoint_step"
        anno_ckpt.bool_value = is_checkpoint_step
        debug_annotations = [anno_sps, anno_gsps, anno_ckpt]
        self._add_event(
            self.pid_train_eval_sessions,
            start_ts,
            end_ts,
            track_name,
            f"Step {step_index}",
            debug_annotations,
        )

    def write_perfetto(self, filepath):
        """Write the Perfetto trace to a file."""
        try:
            with open(filepath, "wb") as f:
                f.write(self.pb.SerializeToString())
        except FileNotFoundError:
            self.logger.warning(
                f"Cannot write to file {filepath}. GPV trace not saved."
            )


class GlobalPerfView(GPVTraceWriter):
    """Core GPV class that interacts with cerebras_pytorch"""

    def __init__(self):
        super().__init__()
        # Start of 1st step, end of 1st step (also start of 2nd step), ...
        self.this_session_step_timestamps: List[int]
        # Samples per second of each step
        self.this_session_step_samples_per_second: List[float]
        # The samples per second of the previous step in the same session, 0 if
        # we are recording the first step of a session
        self.previous_step_samples_per_second: float
        # The starting timestamp of the latest session
        self.this_session_starting_timestamp: int
        # Network errors returned from UnaryFinalize response from the latest
        # session. Only need to save the latest session's response because it
        # uses session_start_timestamp to query, so will cover the wsjob's all
        # sessions.
        # Type is protobuf repeated field of NetworkErrorTrace
        self.network_errors = None
        # Used to skip GPV in not supported use cases i.e. inference or some
        # unit tests
        self.first_step_started = False

    def start_session(self):
        """Record the start of a session."""
        self.this_session_starting_timestamp = time.time_ns()
        self.this_session_step_timestamps = []
        self.this_session_step_samples_per_second = []
        self.previous_step_samples_per_second = 0
        self.first_step_started = False

    def _generate_session_summary(
        self,
        path: str,
        network_errors,  # Type is protobuf repeated field of NetworkErrorTrace
        perf: ExecuteJobPerf,
        wsjob_dashboard: Optional[str],
    ):
        """Writes the perf summary txt of the session."""
        # No data
        if not self.first_step_started:
            return

        # Gather data first before writing the summary file

        # First get average rate and slowest step in steady state. For now take
        # the steps after the first step as steady state.
        start_steady_state = 1
        if len(self.this_session_step_samples_per_second) <= start_steady_state:
            # Report the only first step if there is only one step
            start_steady_state = 0
        steady_state_samples_per_second = (
            self.this_session_step_samples_per_second[start_steady_state:]
        )
        avg_samples_per_second = statistics.mean(
            steady_state_samples_per_second
        )
        median_samples_per_second = statistics.median(
            steady_state_samples_per_second
        )
        fastest_samples_per_second = max(steady_state_samples_per_second)
        fastest_step = (
            steady_state_samples_per_second.index(fastest_samples_per_second)
            + start_steady_state
        )
        fastest_percentage_of_avg = (
            fastest_samples_per_second / avg_samples_per_second
        )
        slowest_samples_per_second = min(steady_state_samples_per_second)
        slowest_percentage_of_avg = (
            slowest_samples_per_second / avg_samples_per_second
        )
        slowest_step = (
            steady_state_samples_per_second.index(slowest_samples_per_second)
            + start_steady_state
        )
        slowest_step_start, slowest_step_end = (
            self.this_session_step_timestamps[slowest_step],
            self.this_session_step_timestamps[slowest_step + 1],
        )

        # Then get network errors that overlap with the slowest step
        overlapping_network_errors = []
        for ne in network_errors:
            # One specific counter on one port/nic
            counts = []
            for o in ne.occurrences:
                # occurance overlaps with the slowest step
                if max(o.start_timestamp, int(slowest_step_start / 1e9)) <= min(
                    o.end_timestamp, int(slowest_step_end / 1e9)
                ):
                    counts.append(float(o.count))
            if counts:
                overlapping_network_errors.append(
                    f"{ne.description} on {ne.node_switch_system} {ne.interface},"
                    f" counter average {statistics.mean(counts)}"
                )

        def format_rate(rate):
            """Print rate in the same format as in user log. Copied from
            src/models/src/cerebras/modelzoo/trainer/loggers/progress.py"""
            if rate < 1.0:
                return f"{rate:.3g}"
            return f"{rate:.2f}"

        f = open(path, "w")
        f.write(
            f"{len(self.this_session_step_samples_per_second)} steps in session\n"
        )
        f.write(
            f"Average steady state samples per second: "
            f"{format_rate(avg_samples_per_second)}, "
        )
        f.write(f"median: {format_rate(median_samples_per_second)}\n")
        f.write(
            f"Fastest step: step {fastest_step + 1}, rate: "
            f"{format_rate(fastest_samples_per_second)} "
        )
        f.write(f"({fastest_percentage_of_avg:.2%} of average rate)\n")
        f.write(
            f"Slowest step: step {slowest_step + 1}, rate: "
            f"{format_rate(slowest_samples_per_second)} "
        )
        f.write(f"({slowest_percentage_of_avg:.2%} of average rate)\n\n")

        if not wsjob_dashboard:
            wsjob_dashboard = "Not available"
        f.write("Link to wsjob Dashboard:\n")
        f.write(f"{wsjob_dashboard}\n\n")

        if not perf.system:
            f.write("System Data not collected!\n")
        else:
            for system in perf.system:
                # System info
                system_info = [
                    {
                        "System Name": system.node_name,
                        "cmaddr": system.cm_addr,
                        "workdir": system.chief_workdir,
                    }
                ]
                f.write(tabulate(system_info, headers="keys", tablefmt="grid"))
                f.write("\n")
                # IO timestamp summary
                f.write(
                    f"IO timestamp Summary: data collected from runtime iteration "
                    f"{system.io_timestamp.start_rt_iter} to "
                    f"{system.io_timestamp.end_rt_iter}, user step "
                    f"{system.io_timestamp.start_user_step} to "
                    f"{system.io_timestamp.end_user_step} (inclusive):\n"
                    f"{system.io_timestamp.table}\n"
                )
                # FPGA reconfig drops
                f.write(
                    f"FPGA Summary: Total WIO reconfig drops={system.fpga.total_reconfig_drops}\n"
                )
                f.write("\n")

        f.write("\nNetwork errors that overlap the slowest step:\n")
        if overlapping_network_errors:
            f.write("\n".join(overlapping_network_errors))
        else:
            f.write("None")
        f.write("\n")

        f.close()

    def end_session(
        self,
        perf: ExecuteJobPerf = None,
        wsjob_dashboard: str = None,
        summary_path: str = None,
    ):
        """Record the end of a session."""
        self.add_session(
            self.this_session_starting_timestamp,
            time.time_ns(),
        )
        # network errors are used in writing GPV perfetto trace in end_run(), so
        # need to save it.
        self.network_errors = perf.network_errors
        self._generate_session_summary(
            summary_path, perf.network_errors, perf, wsjob_dashboard
        )

    def _smooth(self, current_rate):
        if self.previous_step_samples_per_second != 0:
            # If not first step, apply smoothing like in tracker.py
            smoothed_rate = (
                0.6 * current_rate + 0.4 * self.previous_step_samples_per_second
            )
        else:
            smoothed_rate = current_rate
        self.previous_step_samples_per_second = smoothed_rate
        return smoothed_rate

    def end_run(self, filepath: Path):
        """Finalize GPV for a run/wsjob and dump the trace to the given path."""
        # The purpose of this early return is to avoid failures in inference.
        # ltc_backend.py will pass an inexistent path so can't write gpv.pb.
        # In inference, we don't collect any data for GPV either, so use it as
        # the condition to skip.
        if not self.first_step_started:
            return

        # Add the saved network errors to the Perfetto trace
        def add_network_errors():
            if not self.network_errors:
                return
            for e in self.network_errors:  # pylint: disable=not-an-iterable
                # e.g. FEC Uncorrected Errors (net005-ax-sr01 eth400g0)
                track_name = (
                    f"{e.description} ({e.node_switch_system} {e.interface})"
                )
                for occurrence in e.occurrences:
                    self._add_event(
                        self.pid_network_errors,
                        occurrence.start_timestamp * int(1e9),
                        occurrence.end_timestamp * int(1e9),
                        track_name,
                        occurrence.count,  # For each occurence, name it with the error count
                    )

        add_network_errors()
        # Write the Perfetto trace to the given file
        self.write_perfetto(filepath)

    def end_step(
        self, step_index: int, batch_size: int, is_checkpoint_step: bool
    ):
        """Called in a step closure to record the ending timestamp and the rates
        of a step to perfetto trace."""
        if self.first_step_started is False:
            return
        now = time.time_ns()
        start = self.this_session_step_timestamps[-1]
        raw_samples_per_second = batch_size * 1e9 / (now - start)
        samples_per_second = self._smooth(raw_samples_per_second)
        # Calculate global rate, assuming batch size is same for all steps
        global_samples_per_second = (
            batch_size
            * step_index
            * 1e9
            / (now - self.this_session_step_timestamps[0])
        )

        self.add_step(
            start,
            now,
            step_index,
            samples_per_second,
            global_samples_per_second,
            is_checkpoint_step,
        )

        self.this_session_step_timestamps.append(now)
        self.this_session_step_samples_per_second.append(samples_per_second)

    def start_first_step(self):
        """Called in a step closure to log the starting timestamp of the first step."""
        self.first_step_started = True
        now = time.time_ns()
        self.this_session_step_timestamps.append(now)
