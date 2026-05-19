# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Dataclasses to represent appliance connectivity"""
from typing import List, Optional, Tuple

from cerebras.appliance.pb.workflow.appliance.common.common_config_pb2 import (
    ClusterDetails,
)


class ClusterDetailsParser:
    """Extracts Information From ClusterDetails proto"""

    def __init__(self, cluster_details: ClusterDetails) -> None:
        self._tasks = cluster_details.tasks
        self._cluster_details = cluster_details

    def extract_wse_details(
        self, role: ClusterDetails.TaskInfo.TaskType, role_id: int
    ) -> List[Tuple[int, str]]:
        """Extract the wse_id and address for this specific role"""
        for a_task in self._tasks:
            if a_task.task_type == role:
                for a_map in a_task.task_map:
                    if a_map.task_id.task_id == role_id:
                        wse_ids = (
                            a_map.task_id.wse_ids
                            if a_map.task_id.wse_ids
                            else [a_map.task_id.wse_id]
                        )
                        return [
                            (wse_id, self.extract_cm_address(wse_id))
                            for wse_id in wse_ids
                        ]
        raise ValueError(
            f"Couldn't extract wse information for {role}:{role_id}"
        )

    def extract_num_csx(self):
        """Helper to grab number of CSX"""
        num_csx = 0
        for a_task in self._tasks:
            if a_task.task_type == ClusterDetails.TaskInfo.TaskType.WSE:
                num_csx += len(a_task.task_map)
        return num_csx

    def extract_num_wgt_srvs(self) -> int:
        """Helper to grab number of wgt servers"""
        for a_task in self._tasks:
            if a_task.task_type == ClusterDetails.TaskInfo.TaskType.WGT:
                return len(a_task.task_map)
        raise ValueError(
            "Couldn't find weight servers in ClusterDetails"
            f"Received: {ClusterDetails}"
        )

    def extract_num_act_srvs(self, wse_id: Optional[int] = None) -> int:
        """Helper to grab number of act servers.
        Args:
            wse_id: If given, it will return number of workers dedicated to that
                particular WSE. If None (default), it will return total number
                of workers across all WSE's.
        """
        num_act_srvs = 0
        for a_task in self._tasks:
            if a_task.task_type == ClusterDetails.TaskInfo.TaskType.ACT:
                if wse_id is None:
                    num_act_srvs += len(a_task.task_map)
                else:
                    for a_map in a_task.task_map:
                        if (
                            wse_id in a_map.task_id.wse_ids
                            or wse_id == a_map.task_id.wse_id
                        ):
                            num_act_srvs += 1
        return num_act_srvs

    def extract_cm_address(self, wse_id: int = 0) -> str:
        """Helper to grab cm_address"""
        cm_address = ""
        for a_task in self._tasks:
            if a_task.task_type == ClusterDetails.TaskInfo.TaskType.WSE:
                # TODO: Handle for multiple wse_id. SW-85857
                task_map = a_task.task_map[wse_id]
                cm_address = task_map.address_book.wse_ip
                break
        return cm_address

    def extract_act_addresses(self, wse_id: int) -> List[str]:
        """Helper to grab act addresses"""
        act_addresses = []
        for a_task in self._tasks:
            if a_task.task_type == ClusterDetails.TaskInfo.TaskType.ACT:
                for a_map in a_task.task_map:
                    if (
                        wse_id == -1
                        or wse_id in a_map.task_id.wse_ids
                        or wse_id == a_map.task_id.wse_id
                    ):
                        act_addresses.append(a_map.address_book.task_ip_port)
                break
        return act_addresses

    def extract_coord_address(
        self,
    ) -> str:
        """Helper to grab coord_address"""
        coord_address = None
        for a_task in self._tasks:
            if a_task.task_type == ClusterDetails.TaskInfo.TaskType.CRD:
                assert (
                    len(a_task.task_map) == 1
                ), "There should only be one coordinator"
                task_map = a_task.task_map[0]
                coord_address = task_map.address_book.task_ip_port
                break
        return coord_address

    def extract_wrk_address(self, role_id: int) -> str:
        """Helper to grab wrk addresses"""
        for a_task in self._tasks:
            if a_task.task_type == ClusterDetails.TaskInfo.TaskType.WRK:
                for a_map in a_task.task_map:
                    if a_map.task_id.task_id == role_id:
                        return a_map.address_book.task_ip_port
        return None

    def extract_stm_address(self, role_id: int) -> str:
        """Helper to grab streamer addresses"""
        for a_task in self._tasks:
            if a_task.task_type == ClusterDetails.TaskInfo.TaskType.WRK:
                for a_map in a_task.task_map:
                    if a_map.task_id.task_id == role_id:
                        # Streamer runs on the same host as its paired WRK and
                        # only communicates with the WRK, so we spin it up on
                        # the same host.
                        # TODO: We should use task_comm_ports[0]
                        # return f"localhost:{a_map.address_book.task_comm_ports[0]}"
                        return a_map.address_book.task_debug_address
        return None

    def extract_sidecar_address(
        self, task_type: ClusterDetails.TaskInfo.TaskType, task_id: int
    ) -> str:
        if task_type not in (
            ClusterDetails.TaskInfo.TaskType.WGT,
            ClusterDetails.TaskInfo.TaskType.WRK,
            # Only required for swmodel
            ClusterDetails.TaskInfo.TaskType.ACT,
        ):
            return None

        for a_task in self._tasks:
            if a_task.task_type == task_type:
                for a_map in a_task.task_map:
                    if a_map.task_id.task_id == task_id:
                        return a_map.address_book.user_sidecar_address

        return None

    def extract_num_workers(self, wse_id: Optional[int] = None) -> int:
        """Helper to grab number of workers.

        Args:
            wse_id: If given, it will return number of workers dedicated to that
                particular WSE. If None (default), it will return total number
                of workers across all WSE's.
        """
        num_workers = 0
        for a_task in self._tasks:
            if a_task.task_type == ClusterDetails.TaskInfo.TaskType.WRK:
                if wse_id is None:
                    num_workers += len(a_task.task_map)
                else:
                    for a_map in a_task.task_map:
                        if (
                            wse_id in a_map.task_id.wse_ids
                            or wse_id == a_map.task_id.wse_id
                        ):
                            num_workers += 1
        return num_workers
