# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Common routines for managing protobuf objects"""

import google.protobuf.json_format as json_format
from google.protobuf.message import Message


def proto_msg_from_jsondict(proto_cls, json_dict, ignore_unknown=False):
    """Parses the JSON dict, returns it as a protobuf object of the given cls"""
    if not issubclass(proto_cls, Message):
        raise ValueError("proto_cls must be a protobuf message type")
    try:
        proto_msg = json_format.ParseDict(
            json_dict, proto_cls(), ignore_unknown_fields=ignore_unknown
        )
        return proto_msg
    except json_format.ParseError as exc:
        raise ValueError("Invalid JSON dict") from exc


def proto_msg_from_jsontext(proto_cls, json_text, ignore_unknown=False):
    """Parses the JSON text, returns it as a protobuf object of the given cls"""
    if not issubclass(proto_cls, Message):
        raise ValueError("proto_cls must be a protobuf message type")
    try:
        proto_msg = json_format.Parse(
            json_text, proto_cls(), ignore_unknown_fields=ignore_unknown
        )
    except (UnicodeDecodeError, json_format.ParseError) as exc:
        raise ValueError("Invalid JSON text") from exc
    return proto_msg


def proto_msg_from_jsonfile(proto_cls, json_file, ignore_unknown=False):
    """Reads a JSON file and returns it as a protobuf object of the given cls"""
    try:
        with open(json_file, 'r') as filehandle:
            json_text = filehandle.read()
    except UnicodeDecodeError as exc:
        raise ValueError("Invalid JSON file") from exc
    return proto_msg_from_jsontext(
        proto_cls, json_text, ignore_unknown=ignore_unknown
    )


def proto_msg_to_jsondict(proto_msg):
    """Serializes the given protobuf object to a JSON dict."""
    if not isinstance(proto_msg, Message):
        raise ValueError("proto_msg must be a protobuf message")
    return json_format.MessageToDict(
        proto_msg,
        including_default_value_fields=True,
        preserving_proto_field_name=True,
    )


def proto_msg_to_jsontext(
    proto_msg, include_default_value_fields=True, indent=4
):
    """Serializes the given protobuf object to a JSON string."""
    if not isinstance(proto_msg, Message):
        raise ValueError("proto_msg must be a protobuf message")
    # Note: sort_keys=True is here just for the sake of having completely
    # deterministic behavior; this makes testing and bug reproduction easier.
    return json_format.MessageToJson(
        proto_msg,
        including_default_value_fields=include_default_value_fields,
        preserving_proto_field_name=True,
        indent=indent,
        sort_keys=True,
    )


def proto_msg_to_jsonfile(
    proto_msg, json_file, include_default_value_fields=True
):
    """Serializes the given protobuf object to a JSON file."""
    if not isinstance(proto_msg, Message):
        raise ValueError("proto_msg must be a protobuf message")
    jsontext = proto_msg_to_jsontext(proto_msg, include_default_value_fields)
    with open(json_file, 'w') as filehandle:
        filehandle.write(jsontext)
        filehandle.write('\n')
