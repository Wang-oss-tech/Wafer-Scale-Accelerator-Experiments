# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
Workaround before we can fully unify the protobuf message packages in SW-193835
Currently we have two packages in cerebras.pb.* and cerebras.appliance.pb.*, i.e.:
   - cerebras.pb.workflow.appliance.common.common_config_pb2
   - cerebras.appliance.pb.workflow.appliance.common.common_config_pb2
This may cause bugs in protobuf MergeFrom/CopyFrom when the class are not the same.
A long term solution is to unify the protobuf message packages in cerebras.appliance.pb
"""

import importlib
from typing import Optional

from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.message import Message


def common_config_wrapper(
    cls_name: str, obj: Optional[Message], with_prefix_appliance: bool = False
) -> Optional[Message]:
    """
    Workaround for the protobuf message package issue.

    Args:
        cls_name (str): The name of the target class in the protobuf module.
        obj (Optional[Message]): The original protobuf message object to convert.
        with_prefix_appliance (bool): Flag to indicate which protobuf package to use.
            If True, the target module will be imported from
            "cerebras.appliance.pb.workflow.appliance.common.common_config_pb2".
            Otherwise, "cerebras.pb.workflow.appliance.common.common_config_pb2" is used.

    Returns:
        Optional[Message]: A new protobuf message object of the specified
        class with data copied from `obj`. If `obj` is None, returns None,
        or if the module is not found then returns the original `obj`.
    """
    if obj is None:
        return None

    prefix = "cerebras.appliance.pb" if with_prefix_appliance else "cerebras.pb"
    target_module_name = f"{prefix}.workflow.appliance.common.common_config_pb2"

    try:
        target_module = importlib.import_module(target_module_name)
        new_obj = getattr(target_module, cls_name)()
        ParseDict(MessageToDict(obj), new_obj)
        return new_obj
    except ModuleNotFoundError:
        # Some tests may not have `cerebras.pb` packages, so no conversion is needed
        return obj
