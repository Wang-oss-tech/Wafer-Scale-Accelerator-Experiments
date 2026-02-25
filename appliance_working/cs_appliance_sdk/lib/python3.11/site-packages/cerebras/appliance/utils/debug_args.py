# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""DebugArgs related utilities."""
from collections import defaultdict
from typing import List, Optional, Union

from google.protobuf import json_format, text_format

from cerebras.appliance.pb.workflow.appliance.common.common_config_pb2 import (
    DebugArgs,
)
from cerebras.appliance.utils.descriptor import Descriptor


def write_debug_args(debug_args: DebugArgs, path: str):
    """Appliance mode write debug args file."""
    with open(path, 'w') as f:
        text_format.PrintMessage(debug_args, f)


def get_debug_args(value: Union[DebugArgs, str, None]) -> DebugArgs:
    """Appliance mode load debug args and apply defaults."""
    if isinstance(value, DebugArgs):
        return value

    debug_args = DebugArgs()
    if value:
        with open(value, 'r') as f:
            text_format.Parse(f.read(), debug_args)
    return debug_args


def update_debug_args_from_keys(debug_args: DebugArgs, flat_dargs: dict):
    """
    Update DebugArgs from a dict using dotted-field notation to refer to
    nested objects.
    """

    def nested_defaultdict():
        return defaultdict(nested_defaultdict)

    dict_dargs = nested_defaultdict()

    # First, handle both the dotted flat dict keys and nested dict values into
    # a full dict matching the proto message schema.
    def recursive_merge(d, nd):
        for key, value in nd.items():
            subkeys = key.split(".")
            obj = d
            for subkey in subkeys[:-1]:
                obj = obj[subkey]
            key = subkeys[-1]

            if isinstance(value, dict):
                recursive_merge(obj[key], value)
            else:
                obj[key] = value

    recursive_merge(dict_dargs, flat_dargs)

    # Now, convert this nested dict to the proto message
    json_format.ParseDict(dict_dargs, debug_args)


def update_debug_args_with_job_labels(
    debug_args: DebugArgs, job_labels: Optional[List[str]] = None
):
    """Update debug args with job labels."""
    if not job_labels:
        return

    for label in job_labels:
        tokens = label.split("=")
        label_key = tokens[0]
        label_val = tokens[1]
        debug_args.debug_mgr.labels[label_key] = label_val


class DebugArgsDescriptor(Descriptor):
    """Parses debug args from file at specified path."""

    def __init__(self):
        super().__init__(default=DebugArgs())

    def set_attr(self, obj, value):
        debug_args = getattr(obj, self._attr_name, None)
        if debug_args is None:
            debug_args = DebugArgs()
            setattr(obj, self._attr_name, debug_args)

        if isinstance(value, dict):
            update_debug_args_from_keys(debug_args, value)
        else:
            debug_args.MergeFrom(get_debug_args(value))
