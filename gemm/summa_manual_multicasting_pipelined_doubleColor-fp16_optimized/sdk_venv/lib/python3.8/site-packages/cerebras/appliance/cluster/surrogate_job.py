#!/usr/bin/env python3
# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

""" Surrogate job that maps the appliance job onto an external scheduler """

import os
import pathlib
import shlex
import signal
import subprocess
import time
from abc import ABC, abstractmethod


class SurrogateJob(ABC):
    """Define a surrogate job that maps the appliance job onto an external scheduler"""

    @abstractmethod
    def start(self, job_name, namespace, num_cs2_suffix=""):
        """Start a surrogate job on the external scheduler"""

    @abstractmethod
    def stop(self):
        """Stop a surrogate job on the external scheduler"""

    @abstractmethod
    def get_job_label(self) -> (str, str):
        """
        Return a (key,value) pair for the external schedule job to be used as a label
        in its appliance jobs.
        """

    @abstractmethod
    def get_appliance_jobs(self) -> [str]:
        """Return a list of appliance jobs"""


class SlurmSurrogateJob(SurrogateJob):
    """For slurm external scheduler"""

    def __init__(self, logger, workdir):
        self._jobs = {}
        self._logger = logger
        self._workdir = workdir

    def _get_srun_job_id(self, job_name) -> str:
        """Find the job_id for a given job_name"""
        cmd = f"sacct -n --format JobId%50,JobName%50"
        try:
            output = subprocess.check_output(shlex.split(cmd)).decode("utf-8")

            # There might be multiple lines in `output`. Parse out the job_id for
            # the given job_name.
            entries = output.split('\n')
            for entry in entries:
                id_name = entry.split()
                if len(id_name) == 2:
                    if id_name[1] == job_name:
                        return id_name[0]

            # Didn't find the job, so return None.
            return None
        except subprocess.CalledProcessError as exp:
            self._logger.error(f"Command {cmd} failed: {exp}")
            return None

    def _terminate_srun_job(self, job_id) -> bool:
        """Terminate a given srun job"""
        cmd = f"scancel --signal USR1 {job_id}"
        try:
            self._logger.info(f"run {cmd} to cancel surrogate job {job_id}")
            subprocess.check_output(shlex.split(cmd))
            return True
        except subprocess.CalledProcessError as exp:
            self._logger.error(f"Command {cmd} failed: {exp}")
            return False

    def start(self, job_name, namespace, num_cs2_suffix=""):
        """Start a surrogate job on slurm, and return its process."""

        def _get_surrogate_script(job_name):
            """Return a script text for a surrogate job"""
            file_dir = pathlib.Path(os.path.realpath(__file__)).parent
            return f"{file_dir}/surrogate_script.py --job-name {job_name}"

        if namespace == "job-operator":
            full_job_name = f"{job_name}"
        else:
            full_job_name = f"{namespace}/{job_name}"

        job_script = _get_surrogate_script(full_job_name)
        node_list = ""
        if 'SLURM_JOB_NODELIST' in os.environ:
            node_list = f"-w {os.environ['SLURM_JOB_NODELIST']}"
        try:
            cmd = str(
                f"srun {node_list} --output {self._workdir}/surrogate-{full_job_name}{num_cs2_suffix}.out "
                f"--job-name {full_job_name}{num_cs2_suffix} python3 "
                f"{job_script}"
            )
            surrogate_job = subprocess.Popen(
                shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except FileNotFoundError as exp:
            self._logger.error(
                f"Cmd {cmd} caught exception {exp}. Skip starting surrogate job"
            )
            return

        # Obtain the job id for this srun, for stopping this process later on. Try a few time
        # to get the job_id, since there is a delay for the job to show up in sacct.
        for ntry in range(30):
            surrogate_job_id = self._get_srun_job_id(
                f"{full_job_name}{num_cs2_suffix}"
            )
            if surrogate_job_id is not None:
                break
            time.sleep(1)

        self._logger.info(
            f"Start the surrogate job with pid {surrogate_job.pid}, job_id {surrogate_job_id}"
        )

        self._jobs[job_name] = (surrogate_job, surrogate_job_id)

    def stop(self):
        """Stop all surrogate jobs on slurm"""
        for _, (surrogate_job, surrogate_job_id) in self._jobs.items():
            if surrogate_job:
                srun_terminated = False
                if surrogate_job_id:
                    srun_terminated = self._terminate_srun_job(surrogate_job_id)

                # If we can't retrieve surrogate job_id or terminate the surrogate job
                # for some reason, let us send SIGTERM to the process.
                if not srun_terminated:
                    self._logger.debug(
                        f"send SIGNTERM to pid {surrogate_job.pid}"
                    )
                    os.kill(surrogate_job.pid, signal.SIGTERM)
                _, serr = surrogate_job.communicate()
                if serr:
                    # When the surrogate job was cancelled when the appliance job completed,
                    # the process returns an error. We show the error logs in the debug level,
                    # not to pollute the main log.
                    self._logger.debug(f"Surrogate job returns error: {serr}")
        self._jobs = {}

    def get_job_label(self) -> (str, str):
        """
        Return a (key,value) pair for the external schedule job to be used as a label
        in its appliance jobs.
        """
        return ("SLURM_JOB_ID", os.environ['SLURM_JOB_ID'])

    def get_appliance_jobs(self) -> [str]:
        """Return a list of appliance jobs"""
        return self._jobs.keys()


def get_surrogate_job(logger, workdir):
    """Return a SurrogateJob based on environment"""
    if 'SLURM_JOB_ID' in os.environ:
        # Ideally, we should not create SlurmSurrogateJob instance if the run.py
        # is invoked in a SLURM step, for example, in a `srun`. However, using
        # SLURM_STEP_ID as an indicator is not right. Currently, comment this out
        # until we found a right way to error out.
        # if 'SLURM_STEP_ID' in os.environ:
        #    # To create a surrogate job, we shouldn't invoke run.py with srun.
        #    raise RuntimeError("the script can't be invoked with `srun` command")
        return SlurmSurrogateJob(logger, workdir)
    else:
        return None
