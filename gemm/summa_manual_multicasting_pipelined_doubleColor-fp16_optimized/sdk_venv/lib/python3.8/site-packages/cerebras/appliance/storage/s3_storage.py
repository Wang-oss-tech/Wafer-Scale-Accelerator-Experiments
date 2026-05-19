# Copyright 2016-2025 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import logging
import os
import threading
import urllib
from functools import lru_cache
from io import BytesIO
from typing import Optional

import dill
import numpy
import zstandard

from cerebras.appliance.storage import (
    StorageDeleter,
    StorageReader,
    StorageWriter,
)

logging.getLogger("botocore").setLevel(logging.WARNING)

_THREAD_LOCAL = threading.local()


def get_boto_session(profile_name=None):
    config = (
        profile_name,
        *map(
            os.environ.get,
            (
                "AWS_CONFIG_FILE",
                "AWS_SHARED_CREDENTIALS_FILE",
                "AWS_PROFILE_NAME",
                "AWS_ENDPOINT_URL",
                "AWS_ENDPOINT_URL_S3",
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN",
            ),
        ),
    )

    _BOTO_SESSIONS = getattr(_THREAD_LOCAL, "boto_sessions", {})
    if config not in _BOTO_SESSIONS:
        import boto3

        _BOTO_SESSIONS[config] = boto3.session.Session(
            profile_name=profile_name
        )
        _THREAD_LOCAL.boto_sessions = _BOTO_SESSIONS

    return _BOTO_SESSIONS[config]


@lru_cache
def get_s3_resource(profile_name=None, endpoint_url=None):
    session = get_boto_session(profile_name)
    return session.resource("s3", endpoint_url=endpoint_url)


def get_s3_client(profile_name=None, endpoint_url=None):
    return get_s3_resource(profile_name, endpoint_url).meta.client


class S3Interface:
    def __init__(self, path: str):
        if not self.is_valid_path(path):
            raise ValueError(
                f"Invalid S3 path: {path}.\n"
                f"Expected format: s3://<bucket>/<key>"
            )

        super().__init__(path)

        parsed = self.parse_path(path)

        self.scheme = parsed["scheme"]
        self.bucket = parsed["bucket"]
        self.key = parsed["key"]

        self._s3 = get_s3_resource()

    @property
    def metadata_key(self):
        return f"{self.key}/__METADATA__"

    @classmethod
    def parse_path(cls, path: str):
        result = urllib.parse.urlparse(str(path))
        return {
            "scheme": result.scheme,
            "bucket": result.netloc,
            "key": result.path.lstrip("/"),
            "index": result.query,
        }

    @classmethod
    def is_valid_path(cls, path: str) -> bool:
        return cls.parse_path(path)["scheme"] == "s3"

    @classmethod
    def path_exists(cls, path: str) -> bool:
        assert cls.is_valid_path(path)
        parsed = cls.parse_path(path)
        bucket = parsed["bucket"]
        key = parsed["key"]

        from botocore.exceptions import ClientError

        try:
            for obj in (
                get_s3_resource().Bucket(bucket).objects.filter(Prefix=key)
            ):
                return True
        except ClientError:
            pass

        return False

    @property
    def s3(self):
        return self._s3

    @property
    def s3_client(self):
        return self.s3.meta.client


class S3Reader(S3Interface, StorageReader):
    def __init__(self, path: str):
        super().__init__(path)

        from botocore.exceptions import ClientError

        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            raise ValueError(
                f"Bucket {self.bucket} either does not exist "
                f"or you do not have permissions to access it."
            ) from e

        self.s3_bucket = self.s3.Bucket(self.bucket)

        try:
            self.stats  # noqa
        except ClientError as e:
            raise ValueError(
                f"{path} either does not exist "
                f"or you do not have permissions to access it."
            ) from e

    @classmethod
    @lru_cache
    def construct(cls, path):
        # Cached version of `construct` method as we are allowed to reuse the
        # same reader for S3 if the path is the same
        return cls(path)

    @property
    def stats(self) -> StorageReader.Stats:
        response = self.s3_client.head_object(
            Bucket=self.bucket, Key=self.metadata_key
        )

        last_modified = response['LastModified']
        st_mtime = int(last_modified.timestamp())

        return StorageReader.Stats(
            path=self.path,
            st_mtime=st_mtime,
        )

    def read(self, key):
        if key is None:  # Get global metadata
            bytes = BytesIO()
            self.s3_bucket.download_fileobj(self.metadata_key, bytes)

            return None, dill.loads(bytes.getbuffer())

        s3_object = self.s3_bucket.Object(f"{self.key}/{key}")

        metadata = json.loads(s3_object.metadata["metadata"])

        response = s3_object.get()
        body = response["Body"].read()

        if metadata.get("compressed"):
            # Decompress the data using zstandard
            bytes = BytesIO(zstandard.decompress(body))
        else:
            bytes = BytesIO(body)

        if numpy_metadata := metadata.get("__NUMPY__"):
            shape = numpy_metadata["shape"]
            dtype = numpy_metadata["dtype"]
            return (
                numpy.frombuffer(bytes.getbuffer(), dtype=dtype).reshape(shape),
                metadata,
            )

        return bytes, metadata


class S3Writer(S3Interface, StorageWriter):
    def __init__(self, path: str):
        super().__init__(path)

        self.s3_bucket = self.s3.Bucket(self.bucket)

        from botocore.exceptions import ClientError

        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.s3_bucket.create()  # Create bucket if it doesn't exist

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            try:
                S3Deleter(self.path).delete()
            except Exception as e:
                logging.error(
                    f"Exception occurred while trying to save {self.path}. "
                    f"Objects that were already written could not be cleaned up "
                    f"due to:\n{e}"
                )

    def write_numpy(self, key: str, array: numpy.ndarray, metadata: dict):
        bytes = BytesIO(array.data)
        metadata["__NUMPY__"] = {
            "shape": tuple(array.shape),
            "dtype": str(array.dtype),
        }
        return self.write_bytes(key, bytes, metadata)

    def write_bytes(self, key, bytes, metadata):
        object_key = f"{self.key}/{key}"
        if object_key == self.metadata_key:
            raise ValueError(
                f"{key} is a reserved key name and must not be used."
            )

        metadata["data_sizes"] = [len(bytes.getbuffer())]

        # Compress the data using zstandard
        if metadata.setdefault("compressed"):
            bytes = BytesIO(zstandard.compress(bytes.getbuffer()))

        metadata["boundaries"] = [len(bytes.getbuffer())]

        self.s3_bucket.Object(object_key).upload_fileobj(
            bytes,
            ExtraArgs=dict(
                Metadata={"metadata": json.dumps(metadata)},
            ),
        )

    def write_metadata(self, key, metadata):
        if key is not None:
            raise ValueError("Only global metadata may be written separately")

        self.s3_bucket.upload_fileobj(
            BytesIO(dill.dumps(metadata)), self.metadata_key
        )


class S3Deleter(S3Interface, StorageDeleter):
    def __init__(self, path: str):
        super().__init__(path)

    def delete(self, key: Optional[str] = None):
        if key is None:
            key = self.key
        else:
            key = f"{self.key}/{key}"

        bucket = self.s3.Bucket(self.bucket)
        for obj in bucket.objects.filter(Prefix=key):
            obj.delete()
