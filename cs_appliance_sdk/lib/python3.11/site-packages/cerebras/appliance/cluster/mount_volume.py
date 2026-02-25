# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import os
from functools import cached_property
from pathlib import Path
from typing import List

from cerebras.appliance.pb.workflow.appliance.cluster_mgmt.csctl.v1.resources_pb2 import (
    VolumeList,
)
from cerebras.appliance.utils.file import find_mount_point, get_mount_device


class ClusterVolumeManager:
    """Manager class for cluster volumes."""

    def __init__(self, volumes: VolumeList):
        self._volumes = dict()
        self._unknown = set()
        self._mismatching = dict()
        self._visible = set()
        self._local = set()

        for volume in volumes.items:
            if volume.HasField("host_path"):
                cluster_mountpoint = volume.host_path.container_path
                cluster_mountdevice = None
            elif volume.HasField("nfs"):
                cluster_mountpoint = volume.nfs.container_path
                cluster_mountdevice = ":".join(
                    [volume.nfs.server, volume.nfs.server_path]
                )
            else:
                raise ValueError(
                    f"Cluster volume has no source container path: {volume}"
                )

            if not cluster_mountpoint:
                raise ValueError(
                    f"Got an empty cluster container path! "
                    f"Here's the volume list: {volumes}"
                )

            self._volumes[cluster_mountpoint] = cluster_mountdevice

            if cluster_mountdevice is None:
                self._local.add(cluster_mountpoint)
                continue

            try:
                client_mountpoint = find_mount_point(
                    os.path.realpath(cluster_mountpoint)
                )
                client_mountdevice = get_mount_device(client_mountpoint)

                if _is_path_visible(
                    cluster_mountpoint,
                    cluster_mountdevice,
                    client_mountpoint,
                    client_mountdevice,
                ):
                    self._visible.add(cluster_mountpoint)
                else:
                    self._mismatching[cluster_mountpoint] = (
                        cluster_mountdevice,
                        client_mountdevice,
                    )
            except Exception:
                self._unknown.add(cluster_mountpoint)

    @cached_property
    def available_mountpoints(self) -> List[str]:
        return sorted(self._visible | self._local)

    @cached_property
    def unavailable_mountpoints(self) -> List[str]:
        return sorted(self._unknown | set(self._mismatching.keys()))

    @cached_property
    def all_mountpoints(self) -> List[str]:
        return sorted(self.available_mountpoints + self.unavailable_mountpoints)

    def get_mountpoint_for_path(self, path: str) -> str:
        for cluster_mountpoint in self._local:
            if _is_same_path(
                os.path.commonpath([path, cluster_mountpoint]),
                cluster_mountpoint,
            ):
                return cluster_mountpoint

        # Iterate from longest to shortest
        for cluster_mountpoint in sorted(
            self._volumes.keys(), key=lambda x: len(x), reverse=True
        ):
            if _is_same_path(
                os.path.commonpath([path, cluster_mountpoint]),
                cluster_mountpoint,
            ):
                if cluster_mountpoint in self._unknown:
                    raise ValueError(
                        f"Could not find the mount device for path {path}."
                    )

                client_mountpoint = find_mount_point(os.path.realpath(path))
                client_mountdevice = get_mount_device(client_mountpoint)

                if not _is_path_visible(
                    cluster_mountpoint,
                    self._volumes[cluster_mountpoint],
                    client_mountpoint,
                    client_mountdevice,
                ):
                    raise ValueError(
                        f"Path {path} appears to be on a different mount device on "
                        f"this machine ({client_mountdevice}) vs. "
                        f"on the appliance ({self._volumes[cluster_mountpoint]}). "
                    )
                elif cluster_mountpoint in self._mismatching:
                    raise ValueError(
                        f"Path {path} appears to be on a different mount device on "
                        f"this machine ({self._mismatching[cluster_mountpoint][1]}) vs. "
                        f"on the appliance ({self._mismatching[cluster_mountpoint][0]}). "
                    )

                return cluster_mountpoint

        raise ValueError(f"No cluster mountpoints found for {path}.")


def _is_path_visible(
    cluster_mountpoint,
    cluster_mountdevice,
    client_mountpoint,
    client_mountdevice,
):
    """Returns whether a given path on client node is visible on the cluster.

    This method handles cases where client and cluster have different mountpoints,
    but the client mountpoint is a superset of the cluster mountpoint. For example,
    consider the following setup:

    client:
        mountpoint: /root
        mountdevice: nfs.server.com
        mounted on: /qux

    cluster:
        mountpoint: /root/foo/bar
        mountdevice: nfs.server.com
        mounted on: /qux/foo/bar

    In this case, path "qux/foo/bar/hello" is visible on both side even though
    mountpoints are different on client vs. cluster.
    """
    return client_mountdevice == cluster_mountdevice or (
        (
            common_subpath := os.path.commonpath(
                [cluster_mountpoint, client_mountpoint]
            )
        )
        and _is_same_path(common_subpath, client_mountpoint)
        and _is_same_path(
            os.path.join(
                client_mountdevice,
                os.path.relpath(cluster_mountpoint, start=client_mountpoint),
            ),
            cluster_mountdevice,
        )
    )


def _is_same_path(a: str, b: str) -> bool:
    return Path(a) == Path(b)
