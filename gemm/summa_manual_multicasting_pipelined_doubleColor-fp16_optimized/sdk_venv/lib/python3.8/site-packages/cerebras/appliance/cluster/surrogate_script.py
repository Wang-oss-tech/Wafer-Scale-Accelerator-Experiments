#!/usr/bin/env python3
# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
A script to run in the external scheduler to serve as a surrogate job for
the appliance job. It waits for a signal from the appliance or the external
scheduler to terminate the process. If an exception is raise, it queries
the appliance to find the status for the appliance job, and exit based on
that status:
    SUCCEEDED: exit with 0.
    FAILED or CANCELLED: exit with 1.
    others: send a cancel job to the appliance to cancel the job and exit with 1.
"""

import argparse
import json
import logging
import shlex
import signal
import subprocess
import sys
import time

from cerebras.appliance.cluster import cluster_logger

logger = cluster_logger.getChild("surrogate_script")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


_shutdown = False
_complete = False


# Handle the case when slurm is shutting down a job
def _handle_shutdown(signal_no, frame):
    global _shutdown  # pylint: disable=global-statement
    _shutdown = True


# Handle the case when the appliance job is completed
def _handle_complete(signal_no, frame):
    global _complete  # pylint: disable=global-statement
    _complete = True


def _get_appliance_job(namespace, job_id):
    try:
        cmd = f'csctl get job {job_id} --namespace {namespace} -ojson'
        output = subprocess.check_output(shlex.split(cmd)).decode('utf-8')
        output_json = json.loads(output)
        return output_json
    except subprocess.CalledProcessError as exp:
        logger.error(f"cmd {cmd} failed; {exp}. Exit 1")
        sys.exit(1)


def _main(args):
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGUSR1, _handle_complete)

    job_name = args.job_name
    logger.info(f"Surrogate job for appliance job {job_name} started")

    # TODO(joy) When namespace feature is externalized, we might want to revisit this to make
    # the naming consistent.
    splits = job_name.split('/')
    if len(splits) == 1:
        namespace = "job-operator"
        job_id = splits[0]
    elif len(splits) == 2:
        namespace = splits[0]
        job_id = splits[1]
    else:
        raise RuntimeError(f"Unexpected job_name: {job_name}")
    try:
        while not _shutdown and not _complete:
            logger.debug("Waiting for shutdown or job completion")
            time.sleep(5)
    finally:
        output_json = _get_appliance_job(namespace, job_id)

        if _complete:
            # There is a gap between the appliance client considers a job completed
            # and the appliance job sets its status to complete. We wait for the appliance
            # job to be not in running state before continue.
            while output_json['status']['phase'] == 'RUNNING':
                time.sleep(1)
                output_json = _get_appliance_job(namespace, job_id)

        if output_json['status']['phase'] == 'SUCCEEDED':
            logger.info(
                f"Appliance job {job_name} {output_json['status']['phase']}. Exit normally"
            )
            sys.exit(0)
        elif (
            output_json['status']['phase'] == 'FAILED'
            or output_json['status']['phase'] == 'CANCELLED'
        ):
            logger.error(
                f"Appliance job {job_name} {output_json['status']['phase']}. Exit with error"
            )
            sys.exit(1)
        else:
            # Cancel the appliance job.
            cancel_cmd = f'csctl cancel job {job_id} --namespace {namespace}'
            logger.info(f"Cancelling {job_name}")
            subprocess.check_output(shlex.split(cancel_cmd))
            sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-name",
        default=None,
        help="Name of the appliance job. Could be <job_id> or <namespace>/<job_id>.",
    )
    _main(parser.parse_args())
