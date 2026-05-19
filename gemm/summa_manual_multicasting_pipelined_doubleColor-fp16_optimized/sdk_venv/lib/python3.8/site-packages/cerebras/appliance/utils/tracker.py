# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
Tracker class and associated stats classes for profiling and tracking
"""
import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List


class ProfileStats(object):
    """Class for profiling one metric and providing aggregate stats."""

    def __init__(self, name):
        """Construct a `ProfileStats` instance."""
        self.name = name
        self._count = 0
        self._min = float("inf")
        self._max = 0.0
        self._total = 0.0

        self.is_running = False
        self._start_time = 0.0
        self._end_time = 0.0
        self._round_precision = 7

    @property
    def count(self):
        """Returns total number of times `start()` was called."""
        return self._count

    @property
    def min(self):
        """Returns minimum duration in seconds."""
        if self._min != float("inf"):
            return self._to_seconds(self._min)
        else:
            return 0.0

    @property
    def max(self):
        """Returns maximum duration in seconds."""
        return self._to_seconds(self._max)

    @property
    def total(self):
        """Returns aggregate duration in seconds."""
        return self._to_seconds(self._total)

    @property
    def avg(self):
        """Returns average duration in seconds."""
        if self._count != 0:
            avg = self._to_seconds(self._total / self._count)
        else:
            avg = 0.0
        return avg

    @property
    def throughput(self):
        """Returns throughput in counts per seconds."""
        total_time = self.total
        if total_time != 0.0:
            throughput = round(
                float(self._count) / total_time, self._round_precision
            )
        else:
            throughput = 0.0
        return throughput

    @property
    def start_time(self):
        """Returns the start time in nanoseconds from epoch time."""
        return self._start_time

    @property
    def end_time(self):
        """Returns the end time in nanoseconds from epoch time."""
        return self._end_time

    def start(self):
        """Starts profiling. Increments count and starts the timer."""
        if self.is_running:
            raise RuntimeError("Profiler is already running")

        self.is_running = True
        self._count += 1
        self._start_time = time.time_ns()

    def stop(self, overwrite=False):
        """Stops profiling. Stops timer and updates stats.

        Args:
            overwrite: If True, set the total to be the current
                run time. Otherwise, add the current run time to the total.
        """
        if not self.is_running:
            return

        self._end_time = time.time_ns()
        run_time = self._end_time - self._start_time
        self._min = min(self._min, run_time)
        self._max = max(self._max, run_time)
        if overwrite:
            self._total = run_time
        else:
            self._total += run_time
        self.is_running = False

    def to_dict(self):
        """Returns a dict representation of the stats."""
        return {
            "min": self.min,
            "max": self.max,
            "total": self.total,
            "average": self.avg,
            "count": self.count,
            "throughput": self.throughput,
        }

    def _to_seconds(self, val):
        """Converts val in nanoseconds to seconds and rounds it.."""
        return round(float(val) / 1e9, self._round_precision)


@dataclass
class NestedStats:
    """Nested structure that represents the hierarchy of the stats"""

    stats: ProfileStats = None
    children: List["NestedStats"] = field(default_factory=list)

    def add_child(self, stats: ProfileStats):
        """Add stats as a child to the current next stat"""
        ns = NestedStats(stats=stats)
        self.children.append(ns)
        return ns

    @property
    def total(self):
        """Returns a nested dictionary of [total time, children] pairs"""
        children_total = {
            child.stats.name: child.total for child in self.children
        }
        if self.stats:
            return [self.stats.total, children_total]
        else:
            return children_total

    @property
    def timestamps(self):
        """Returns a dictionary of (start, end) timestamps of each event"""
        complete_timestamps = {}
        for child in self.children:
            complete_timestamps.update(child.timestamps)
        if self.stats:
            complete_timestamps[self.stats.name] = (
                self.stats.start_time,
                self.stats.end_time,
            )
        return complete_timestamps


class Tracker:
    """Class to track a ProfileStat."""

    def __init__(self):
        """Tracker instance to consume ProfilerStats"""
        self.root = NestedStats()
        self.profile_stats_stack = [self.root]

    @lru_cache(maxsize=None)
    def profile_stats(self, name):  # pylint: disable=no-self-use
        """Returns a profile stats object with a given name

        The returned object gets cached so that it can be fetched again
        """
        return ProfileStats(name)

    def start(self, name):
        """Starts the profiler specified"""
        ps = self.profile_stats(name)
        if ps.is_running:
            raise NameError(
                f"Stat with name {name} cannot be started as it is already used "
                f"in Tracker collection"
            )

        ps.start()
        ns = self.profile_stats_stack[-1].add_child(ps)
        self.profile_stats_stack.append(ns)

    def is_running(self, name):
        """Returns True if the profiler is running"""
        return self.profile_stats(name).is_running

    def stop(self, name, overwrite=False):
        """Closes specified profiler and captures the result"""
        if not self.profile_stats(name).is_running:
            return

        # Stop all stats that are within the name's context. This is to handle
        # the case where some nested stat is started but never stopped. All
        # nested stats should be stopped when the parent is stopped.
        while True:
            if len(self.profile_stats_stack) <= 1:
                raise RuntimeError(
                    f"Tracker cannot stop {name} as it is not tracking any stats"
                )

            ps = self.profile_stats_stack.pop()
            ps.stats.stop(overwrite)
            if ps.stats.name == name:
                break

    def flush(self, filepath):
        """
        Saves the results from the profilers to a file
        """
        with open(filepath, "w") as f:
            json.dump(self.root.total, f, indent=4)

    def get_timestamps(self):
        """Returns a dictionary of [start, end] timestamps of each event"""
        return self.root.timestamps

    class Entry:
        """re-entrant context manager that can start and stop a stat"""

        def __init__(self, tracker, name, overwrite):
            self.tracker = tracker
            self.name = name
            self.overwrite = overwrite

        def __enter__(self):
            self.tracker.start(self.name)

        def __exit__(self, *args):
            self.tracker.stop(self.name, self.overwrite)

    def entry(self, name, overwrite=False):
        """returns a re-entrant context manager wrapped around start and stop"""
        return Tracker.Entry(self, name, overwrite)
