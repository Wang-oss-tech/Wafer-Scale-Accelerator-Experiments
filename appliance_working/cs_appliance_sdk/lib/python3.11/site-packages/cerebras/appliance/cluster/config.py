# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

""" Cluster configuration """

import base64
import functools
import json
import os
import pathlib
import typing

from cerebras.appliance.cluster import cluster_logger

logger = cluster_logger.getChild("config")


CSCONFIG_PATH = pathlib.Path("/opt/cerebras/config_v2")
FABRIC_TYPE_CS2 = "cs2"
FABRIC_TYPE_CS3 = "cs3"
JOB_PRIORITY_P0 = "p0"
JOB_PRIORITY_P1 = "p1"
JOB_PRIORITY_P2 = "p2"
JOB_PRIORITY_P3 = "p3"
VALID_JOB_PRIORITIES = [
    JOB_PRIORITY_P1,
    JOB_PRIORITY_P2,
    JOB_PRIORITY_P3,
]
DEFAULT_JOB_PRIORITY = JOB_PRIORITY_P2


class ClusterConfigError(RuntimeError):
    """Config loading error"""


class NamespaceCertAuthority:
    def __init__(
        self,
        name: str,
        certificate_authority_data: bytes,
        certificate_authority: typing.Optional[str] = None,
        fabric_type: typing.Optional[str] = None,
    ):
        """Initializes a NamespaceCertAuthority instance.

        Args:
            namespace (str): The namespace name.
            certificate_authority_data (bytes): Certificate for the cluster-server,
                PEM encoded cert byte string.
            certificate_authority (typing.Optional[str], optional): File path of the
                certificate if the certificate was not embedded in the cluster config.
                This is for logging purposes. Defaults to None.
        """
        self._name = name
        self._certificate_authority_data = certificate_authority_data
        self._certificate_authority = certificate_authority

    @property
    def name(self) -> str:
        """Get the namespace name."""
        return self._name

    @property
    def certificate_authority_data(self) -> bytes:
        """Get the PEM encoded certificate data."""
        return self._certificate_authority_data

    @property
    def certificate_authority(self) -> typing.Optional[str]:
        """Get the file path of the certificate."""
        return self._certificate_authority

    def __str__(self):
        if self._certificate_authority:
            cert_str = f"certificate_authority={self._certificate_authority}"
        else:
            cert_str = "certificate_authority_data="
            cert_str += (
                "EMPTY" if not self._certificate_authority_data else "OMITTED"
            )

        return str(f"NamespaceCertAuthority{{name={self._name}, {cert_str}}}")


class CSClusterConfig:
    """Base class for cluster connection config"""

    def __init__(
        self,
        mgmt_address: str,
        authority: str,
        namespaces: typing.List[NamespaceCertAuthority],
        fabric_type: typing.Optional[str] = None,
    ):
        """Initializes a CSClusterConfig instance.

        Args:
            mgmt_address (str): Address of the cluster-server.
            authority (str): Backend authority of the cluster-server.
            namespaces (typing.List[NamespaceCertAuthority]): namespaces
                and their corresponding certificate authorities.
        """
        self._mgmt_address = mgmt_address
        self._authority = authority
        self._namespaces = namespaces
        self._fabric_type = (
            fabric_type if fabric_type is not None else FABRIC_TYPE_CS2
        )

    @property
    def mgmt_address(self) -> str:
        """Cluster-server address."""
        return self._mgmt_address

    @property
    def authority(self) -> str:
        """Backend authority of the cluster-server."""
        return self._authority

    @property
    def namespaces(self) -> typing.List[NamespaceCertAuthority]:
        """Get the list of namespaces and their corresponding certificate authorities."""
        return self._namespaces

    @property
    def fabric_type(self) -> typing.Optional[str]:
        """Get the cluster fabric type."""
        return self._fabric_type

    def namespace_exists(self, namespace: str) -> bool:
        """Check if a namespace exists in the config"""
        return self._namespaces is not None and any(
            ns.name == namespace for ns in self._namespaces
        )

    def get_namespace_certificate_authority(
        self, namespace: str
    ) -> typing.Optional[NamespaceCertAuthority]:
        """Get the NamespaceCertAuthority associated with a given namespace."""
        for ns_cert_auth in self._namespaces:
            if ns_cert_auth.name == namespace:
                return ns_cert_auth
        raise ClusterConfigError(
            f"config error: namespace {namespace} is not configured."
        )

    def __str__(self):
        namespaces_str = ", ".join(map(str, self._namespaces))

        return str(
            f"CSClusterConfig{{mgmt_address={self.mgmt_address}, "
            f"authority={self.authority}, "
            f"namespaces=[{namespaces_str}], "
            f"fabric_type={self.fabric_type}}}"
        )


class CSClusterConfigDefault(CSClusterConfig):
    """
    Default values for cluster connection if config file not present.
    Intended for test usage.
    """

    def __init__(self):
        super().__init__("", "cluster-server.cerebrassc.local", None)

    @property
    def mgmt_address(self) -> str:
        raise ClusterConfigError(
            "config error: mgmt_address has no default value"
        )

    def __str__(self):
        return str("CSClusterConfigDefault")


def load_cs_cluster_config() -> typing.Tuple[pathlib.Path, CSClusterConfig]:
    """
    Load a cached csconfig file installed by the usernode installer. This file
    contains the management address for the cluster. This file also contains a
    list of namespaces the user node should have access to and the corresponding
    certificates in those namespaces.
    Returns:
        The CSClusterConfig with mgmt address, a list of namespaces and corresponding
        certificates if the file was loaded successfully, or None if no csconfig
        file is found.
    """
    csconfig_path = CSCONFIG_PATH
    csconfig_override = os.environ.get("CSCONFIG_PATH")
    if csconfig_override:
        csconfig_path = pathlib.Path(csconfig_override)

    if not csconfig_path.exists():
        logger.warning(
            f"csconfig file {csconfig_path} was not found. "
            f"Using default authority fallback and user must provide "
            f"mgmt_address, mgmt_namespace and credentials_path"
        )
        return (None, CSClusterConfigDefault())

    try:
        doc = json.loads(csconfig_path.read_text())
        ctx = doc["currentContext"]
        contexts = {c["name"]: c for c in doc["contexts"]}
        cluster_name = contexts[ctx]["cluster"]
        clusters = {c["name"]: c for c in doc["clusters"]}
        cluster = clusters[cluster_name]

        namespaces = []
        for ns_cert_auth in cluster.get("namespaces", []):
            ns_name = ns_cert_auth['name']
            pem_path = None
            if "certificateAuthorityData" in ns_cert_auth:
                pem_bytes = base64.b64decode(
                    ns_cert_auth["certificateAuthorityData"]
                )
            elif "certificateAuthority" in ns_cert_auth:
                pem_path = pathlib.Path(ns_cert_auth["certificateAuthority"])
                try:
                    pem_bytes = pem_path.read_bytes()
                except OSError:
                    raise ClusterConfigError(
                        f"config error: failed to read 'certificateAuthority' file {pem_path}"
                    )
            else:
                raise ClusterConfigError(
                    f"config error for namespace {ns_name}: required keys 'certificateAuthorityData' or 'certificateAuthority' missing."
                )

            namespace_cert_authority = NamespaceCertAuthority(
                name=ns_name,
                certificate_authority_data=pem_bytes,
                certificate_authority=pem_path,
            )
            namespaces.append(namespace_cert_authority)

        if "fabricType" in cluster:
            fabric_type = cluster["fabricType"]
        else:
            fabric_type = FABRIC_TYPE_CS2
        cfg = CSClusterConfig(
            mgmt_address=cluster["server"],
            authority=cluster["authority"],
            namespaces=namespaces,
            fabric_type=fabric_type,
        )
        logger.debug(f"CSClusterConfig loaded {csconfig_path}: {cfg}")
        return (csconfig_path, cfg)
    except KeyError as e:
        raise ClusterConfigError(
            f"config error: missing required key {e.args[0]} in {csconfig_path}"
        )


@functools.lru_cache()
def get_cs_cluster_config() -> typing.Tuple[pathlib.Path, CSClusterConfig]:
    return load_cs_cluster_config()
