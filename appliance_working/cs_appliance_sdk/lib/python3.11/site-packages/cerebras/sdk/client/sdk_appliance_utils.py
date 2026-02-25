# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""Module offering various utility functions and parameters used by the
appliance SDK GRPC Client and Server
"""
import argparse
import hashlib
import os
from pathlib import Path
from typing import List, Tuple

#COMPILE_CACHE_PATH = os.getenv("SDK_COMPILE_PATH", default=".")

MAX_MESSAGE_LENGTH = (1024 * 1024 * 1024 * 2) - 1024  # 2GB - 1 KB
MAX_TRACEBACK_LENGTH = 7200

GRPC_CONNECT_TIMEOUT = 60


def get_csl_files(app_path: str) -> List[str]:
    """get_csl_files returns the list of .csl files contained in a path
    and its sub-directories

    app_path - the path to explore
    """
    files = [
        os.path.join(dirpath, f)
        for (dirpath, dirnames, filenames) in os.walk(app_path)
        for f in filenames
    ]
    csl_files = [f for f in files if os.path.splitext(f)[1] == ".csl"]
    return csl_files

def get_non_csl_files(app_path: str) -> List[str]:
    """get_non_csl_files returns the list of all non .csl files
    contained in a path and its sub-directories

    app_path - the path to explore
    """
    files = [
        os.path.join(dirpath, f)
        for (dirpath, dirnames, filenames) in os.walk(app_path)
        for f in filenames
    ]
    non_csl_files = [f for f in files if os.path.splitext(f)[1] != ".csl"]
    return non_csl_files

def get_all_files(app_path: str) -> List[str]:
    """get_all_files returns the list of all files
    contained in a path and its sub-directories

    app_path - the path to explore
    """

    csl_files = get_csl_files(app_path)
    non_csl_files = get_non_csl_files(app_path)

    return csl_files + non_csl_files

def get_artifact_hashes(app_path: str, options: str) -> Tuple[str, str]:
    """get_artifact_hashes returns a tuple of hashes. The first hash
    is computed from the content of all .csl files contained in app_path
    and from the compile options. The second hash is computed from the content
    of all other files.

    app_path - the path containing the files
    options - compile options that will be provided to the CSL compiler

    returns tuple( hash(csl files, options), hash(non csl files) )
    """
    # get the list of all files in app_path and subdirectories
    csl_files = get_csl_files(app_path)
    other_files = get_non_csl_files(app_path)

    other_m = hashlib.sha256()
    for cur_file in other_files:
        with open(cur_file, "rb") as in_file:
            data = in_file.read()
            other_m.update(data)

    csl_m = hashlib.sha256()
    for cur_file in csl_files:
        with open(cur_file, "rb") as in_file:
            data = in_file.read()
            csl_m.update(data)

    # include the compile options in the hash as well
    csl_m.update(bytes(options, encoding='utf8'))

    return (csl_m.hexdigest(), other_m.hexdigest())


def get_cslc_parser() -> argparse.ArgumentParser:
    """get_cslc_parser returns an argparse.ArgumentParser object
    identical to the one used by the CSL compiler. It can be used
    to parse and validate compiler options
    """
    parser = argparse.ArgumentParser(
        prog="cslc",
        # description=__doc__
    )
    parser.add_argument(
        "csl_filename", help="Input CSL file",
    )
    parser.add_argument(
        "-o",
        dest="output_name",
        default="out",
        help="Output directory name (default: %(default)s)",
    )
    parser.add_argument(
        "--params",
        action="append",
        help="Comma-separated list of param-to-value mappings where a mapping is a \
              `name:value` pair where name is a string and value is an unsigned integer. \
              The parameter list is passed on to cslc-driver as-is.",
    )
    parser.add_argument(
        "--colors",
        action="append",
        help="Comma-separated list of color-to-value mappings where a mapping is a \
              `color:value` pair where color is a string and value is an unsigned integer. \
              The parameter list is passed on to cslc-driver as-is.",
    )
    parser.add_argument(
        "--memcpy",
        action="store_true",
        help="Add memcpy support to this program",
    )
    parser.add_argument(
        "--channels",
        default=0,
        type=int,
        help="Number of memcpy I/O channels to use when memcpy support is compiled with this \
             program. If this argument is not present, or is 0, then the previous single-I/O \
             channel version is compiled.",
    )
    parser.add_argument(
        "--import-path",
        action="append",
        default=[],
        help="Add the given directory to the list of directories searched for <...> paths in "
        "@import_module and @set_tile_code statements.",
    )
    parser.add_argument(
        "--width-west-buf",
        default=0,
        type=int,
        help="width of west buffer (default is zero, i.e. no buffer to mitigate slow input)",
    )
    parser.add_argument(
        "--width-east-buf",
        default=0,
        type=int,
        help="width of east buffer (default is zero, i.e. no buffer to mitigate slow output)",
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    return parser


def get_artifact_id(artifact_path: str):
    """Recursively" retrieve the compile_hash from the file name
       the recursion here is needed to handle filenames with multiple
       extensions such as foo.tar.gz. The compile_hash should just be
       foo in that case
    """
    artifact_id = Path(artifact_path).stem
    while Path(artifact_id).stem != artifact_id:
        artifact_id = Path(artifact_id).stem

    return artifact_id


