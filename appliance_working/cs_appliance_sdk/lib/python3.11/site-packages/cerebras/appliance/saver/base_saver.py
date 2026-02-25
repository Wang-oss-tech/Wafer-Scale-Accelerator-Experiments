# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
Abstract class for checkpoint store and restore.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from numpy import ndarray as nd


class BaseSaver(ABC):
    """Abstract class for checkpoint store and restore"""

    @abstractmethod
    def __init__(self):
        """Creates saver instance."""

    @abstractmethod
    def save(self, ckpt_file: str, tensor_dict: Dict[str, nd]) -> str:
        """Saves tensor_dict to the checkpoint with step as metadata.

        Args:
            ckpt_file: where to write checkpoint
            tensor_dict: Mapping from name to tensor values

        Returns:
            path: path to checkpoint
        """

    @abstractmethod
    def load(
        self, ckpt_file: str, tensor_names: Optional[List[str]] = None
    ) -> Dict[str, nd]:
        """Loads checkpoint.

        Args:
            step: Denotes which checkpoint to load.
                * Defaults uses the latest ckpt.
                * If set and exists in the ckpt_path else throws an error.

        Returns:
            tensor_dict: Mapping from name to tensor values
        """

    @abstractmethod
    def save_tensor(
        self, ckpt_file: str, tensor_name: str, tensor_value: nd
    ) -> None:
        """Saves a tensor to the ckpt_file"""

    @abstractmethod
    def load_tensor(self, ckpt_path: str, tensor_name: str) -> nd:
        """Loads a tensor from the ckpt_path"""
