# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import base64
import uuid

from cerebras.appliance.log import logger


def gen_uuid():
    """Generate a UUIDv4 and encode it in lowercase alphanumeric format with fixed length."""
    uuid_bytes = uuid.uuid4().bytes  # Get UUID as bytes
    base32_encoded = (
        base64.b32encode(uuid_bytes).decode('utf-8').lower()
    )  # Encode in Base32 and convert to lowercase
    truncated_result = base32_encoded.replace('=', '')  # Strip padding
    return truncated_result


cluster_logger = logger.getChild("cluster")

# Deprecated: this is only workaround for legacy flows
# todo: remove after all flows adopts server side workflow
WORKFLOW_ID = f"wflow-{gen_uuid()}"

# Expose WORKFLOW_ID at the package level
__all__ = ["WORKFLOW_ID"]
