# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Cerebras Cluster Configuration Class."""
import functools
import os
import re
from dataclasses import MISSING, dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

import yaml

from cerebras.appliance.cluster.client import MountDir
from cerebras.appliance.cluster.config import (
    DEFAULT_JOB_PRIORITY,
    JOB_PRIORITY_P1,
    JOB_PRIORITY_P2,
    JOB_PRIORITY_P3,
    VALID_JOB_PRIORITIES,
)
from cerebras.appliance.cluster.job_timer import JobTimer
from cerebras.appliance.cluster.types import NotificationTarget
from cerebras.appliance.utils.debug_args import DebugArgs
from cerebras.appliance.utils.descriptor import Descriptor


class _JobLabels(Descriptor):
    """Descriptor for job labels."""

    def __init__(self):
        super().__init__(default_factory=list)

    def sanitize(self, value: Optional[List[str]]) -> List[str]:
        pattern = r'^([A-Za-z0-9][-A-Za-z0-9_.]{0,61})?[A-Za-z0-9]$'

        labels = value or []

        for kv_pair in labels:
            tokens = kv_pair.split("=")
            if len(tokens) != 2:
                raise ValueError(
                    f"'{kv_pair}' is an invalid label. Expecting the label key and "
                    f"the label value to be separated by a single equal sign(=) character."
                )
            for token in tokens:
                if re.match(pattern, token) is None:
                    raise ValueError(
                        f"'{kv_pair}' is an invalid label. Expecting the label key and the label "
                        f"value to match regex '{pattern}'."
                    )

        return labels


class _JobPriority(Descriptor):
    """Descriptor for job labels."""

    def __init__(self):
        super().__init__(default=DEFAULT_JOB_PRIORITY)

    # pylint: disable=no-self-use
    def sanitize(self, value: str) -> DebugArgs.DebugMGR.JobPriority:
        """Sanitize job priority."""
        priority_mapping = {
            JOB_PRIORITY_P1: DebugArgs.DebugMGR.JobPriority.JOB_PRIORITY_P1,
            JOB_PRIORITY_P2: DebugArgs.DebugMGR.JobPriority.JOB_PRIORITY_P2,
            JOB_PRIORITY_P3: DebugArgs.DebugMGR.JobPriority.JOB_PRIORITY_P3,
        }
        # Valid job priorities are "p1", "p2" and "p3".
        if value not in VALID_JOB_PRIORITIES:
            if any(value == v for v in priority_mapping.values()):
                return value

            raise ValueError(
                f"Invalid job priority value: {value} "
                f"(should be one of {VALID_JOB_PRIORITIES})"
            )
        return priority_mapping[value]


class _MgmtAddress(Descriptor):
    """Descriptor for management address."""

    pattern = re.compile(r'.+:[0-9]+')

    def sanitize(self, value: Optional[str]) -> str:
        """Sanitize mgmt_address."""
        mgmt_address = value
        if (
            mgmt_address is not None
            and _MgmtAddress.pattern.match(mgmt_address) is None
        ):
            raise ValueError(
                f"mgmt_address '{mgmt_address}' should be in the form of '<name>:<port>'"
            )
        return mgmt_address


class _MountDirs(Descriptor):
    """Descriptor for mounted directories."""

    def __init__(self):
        super().__init__(default_factory=list)

    def sanitize(
        self, value: Optional[List[Union[str, MountDir]]]
    ) -> List[MountDir]:
        """Sanitize mount directories."""
        s = set()
        for md in (value or []) + _get_cluster_defaults().get("mount_dirs", []):
            if isinstance(md, (str, Path)):
                real_path = Path(md).resolve()
                if not real_path.exists():
                    raise ValueError(f"Mount dir {real_path} does not exist")
                md = MountDir(path=md, container_path=md)
            s.add(md)
        return list(s)


class _PythonPaths(Descriptor):
    """Descriptor for python paths."""

    def __init__(self):
        super().__init__(default_factory=list)

    def sanitize(self, value: Optional[List[str]]) -> List[str]:
        """Sanitize python paths by turning them into their canonical path."""
        s = set()
        for x in (value or []) + _get_cluster_defaults().get(
            "python_paths", []
        ):
            real_path = Path(x).resolve()
            if not real_path.exists():
                raise ValueError(f"{real_path} does not exist")
            s.add(str(real_path))

        return list(s)


class _PositiveInt(Descriptor):
    """Descriptor for positive integers."""

    def sanitize(self, value: int) -> int:
        if not value:
            return self.default
        if not isinstance(value, int) or value < 1:
            raise ValueError(
                f"{self.name} must be a positive integer or None. Got {value}"
            )

        return value


class _NonNegativeInt(Descriptor):
    """Descriptor for non-negative integers."""

    def sanitize(self, value: int) -> int:
        if not value:
            return self.default
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{self.name} must be a non-negative integer or None. Got {value}"
            )

        return value


class _Notifications(Descriptor):
    """Descriptor for notifications."""

    def __init__(self):
        super().__init__(default_factory=list)

    def sanitize(
        self, value: List[Union[NotificationTarget, dict]]
    ) -> List[NotificationTarget]:
        """Sanitize notifications by turning them into proper config objects."""
        targets = []

        for item in value:
            if isinstance(item, NotificationTarget):
                targets.append(item)
            if isinstance(item, dict):
                targets.append(NotificationTarget(**item))
            else:
                raise TypeError(
                    f"Invalid notification target. Expected a dict or `NotificationTarget`. "
                    f"Got type {type(item)}."
                )

        return targets


@dataclass
class ClusterConfig:
    """Hold config details for Wafer Scale Cluster.

    Args:
        mgmt_address: Address to connect to appliance.
        mgmt_namespace: Namespace of cluster-mgmt software
            for internal multi-version support only.
        credentials_path: Credentials for connecting to appliance.
        num_csx: Number of Cerebras Systems to run on.
        max_wgt_servers: Maximum number of weight servers to support run. If 0, an appropriate
            default is chosen based on the cluster architecture.
        max_act_per_csx: Maximum number of activation servers per system. If 0, an appropriate
            default is chosen based on the cluster architecture.
        num_workers_per_csx: Number of streaming workers per system.
        job_labels: A list of equal-sign-separated key-value pairs that
            get applied as part of job metadata.
        job_priority: Priority of the job in scheduling queue.
        job_time_sec: Time limit for the appliance jobs, not including the queue time.
        mount_dirs: Local storage to mount to appliance (ex. training data).
        python_paths: A list of path that worker pods respect as PYTHONPATH
            in addition to the PYTHONPATH set in the container image.
        automount: Whether to automount available mount dirs and python paths. If False, any
            directories/python paths that are needed by the workers in the Appliance must be
            manually mounted by specifying them in `mount_dirs` and `python_paths`.
        mount_all: Whether to always mount all available cluster volumes. This is only used
            when `automount` is True.
        disable_mount_check: Whether to disable checking validity of `mount_dirs` against
            available mount dirs on the cluster.
        disable_version_check: Whether to disable version check across client/server components.
        workflow_id: Unique identifier for the workflow, which may comprise multiple ML jobs. Note
            that cluster resources are reserved for the entirety of the workflow duration.
        notifications: List of targets to notify when there are cluster-side events.
    """

    DEFAULT = object()

    mgmt_address: Optional[str] = _MgmtAddress(default=None)
    mgmt_namespace: Optional[str] = None
    credentials_path: Optional[str] = None

    num_csx: int = _PositiveInt(default=1)
    max_wgt_servers: int = _NonNegativeInt(default=0)
    max_act_per_csx: int = _NonNegativeInt(default=0)
    num_workers_per_csx: int = _PositiveInt(default=1)

    job_labels: List[str] = _JobLabels()
    job_priority: str = _JobPriority()
    job_time_sec: Optional[int] = _PositiveInt(default=None)

    mount_dirs: List[MountDir] = _MountDirs()
    python_paths: List[str] = _PythonPaths()
    automount: bool = True
    mount_all: bool = True
    disable_mount_check: bool = True

    disable_version_check: bool = False
    workflow_id: Optional[str] = None

    notifications: List[NotificationTarget] = _Notifications()

    def __hash__(self):
        return hash(
            tuple(
                tuple(value) if isinstance(value, list) else value
                for f in fields(self)
                if (value := getattr(self, f.name, None))
            )
        )

    @property
    def job_timer(self) -> Optional[JobTimer]:
        """Returns a cached job timer instance."""
        if self.job_time_sec is not None and self.job_time_sec > 0:
            return self.get_job_timer(self.job_time_sec)
        return None

    @lru_cache(maxsize=1)
    def get_job_timer(  # pylint: disable=no-self-use
        self, job_time_sec
    ) -> JobTimer:
        """Returns a cached JobTimer object."""
        return JobTimer(job_time_sec)

    def __setattr__(self, key, val):
        """Override setter to sanitize values if needed."""
        if val is ClusterConfig.DEFAULT:
            val = self.get_default(key)

        super().__setattr__(key, val)

    @classmethod
    def get_default(cls, key: str):
        """Get the default value of a field."""
        for f in fields(cls):
            if f.name == key:
                if f.default is not MISSING:
                    return f.default
                elif f.default_factory is not MISSING:
                    return f.default_factory()
                else:
                    raise ValueError(
                        f"{key} does not specify a "
                        f"default value or default factory."
                    )

        raise AttributeError(f"Field {key} not found in ClusterConfig")


@functools.lru_cache()
def _get_cluster_defaults():
    if filepath := os.getenv('CEREBRAS_WAFER_SCALE_CLUSTER_DEFAULTS'):
        with open(filepath, "r") as f:
            return yaml.safe_load(f)
    return {}
