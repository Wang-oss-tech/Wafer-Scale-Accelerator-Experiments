#!/usr/bin/env python3
# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Shared type definitions for cluster management.

This module contains shared data structures used across the cluster management system.
It was created to avoid circular dependencies between client.py and cluster_config.py,
by providing a common location for types that are needed by both modules.
"""

from dataclasses import dataclass, field, fields
from typing import List, NamedTuple


@dataclass(frozen=True)
class NotificationTarget:
    """Notification target configuration."""

    mailto: List[str] = field(default_factory=list)
    slack: List[str] = field(default_factory=list)
    pagerduty: List[str] = field(default_factory=list)
    severity_threshold: int = 3  # Medium-severity default

    def __post_init__(self):
        if not all(
            isinstance(x, (list, tuple)) and all(isinstance(y, str) for y in x)
            for x in [self.mailto, self.slack, self.pagerduty]
        ):
            raise ValueError("Notification targets must be a list of strings.")

        if not any([self.mailto, self.slack, self.pagerduty]):
            raise ValueError(
                "Empty notification target. At least one email, slack, or pagerduty "
                "target must be specified."
            )

        if not isinstance(self.severity_threshold, int):
            raise ValueError("severity_threshold must be an integer.")

    def __hash__(self):
        return hash(
            tuple(
                tuple(value) if isinstance(value, list) else value
                for f in fields(self)
                if (value := getattr(self, f.name, None))
            )
        )


class MountDir(NamedTuple):
    """
    A path to be mounted into the appliance containers.

    Parameters:
        path: Path to be mounted.
        container_path: Path that appears in the container. If no value was provided,
            then it will default to use the value of "path"
    """

    path: str
    container_path: str
