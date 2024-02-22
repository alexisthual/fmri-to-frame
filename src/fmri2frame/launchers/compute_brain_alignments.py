#!/usr/bin/env python
# coding: utf-8
"""Compute stimuli latent representations for a list of participants."""

# %%
from itertools import combinations, product
from pathlib import Path

import hydra
import omegaconf
import submitit
from hydra.core.hydra_config import HydraConfig

from fmri2frame.scripts.compute_alignment import compute_alignment
from fmri2frame.scripts.utils import get_logger, monitor_jobs


# %%
# Experiment parameters

# 1. Setup for aligning all pairs of IBC subjects

source_datasets = ["ibc_clips_seg-train"]
source_dataset_path = "/gpfsstore/rech/nry/uul79xi/data/ibc"

target_datasets = ["ibc_clips_seg-train"]
target_dataset_path = "/gpfsstore/rech/nry/uul79xi/data/ibc"

alpha = 0.5
eps = 1e-4
rho = 1
reg_mode = "joint"
divergence = "kl"
solver = "mm"

ibc_subjects = [4, 6, 8, 9, 11, 12, 14, 15]
args_map = list(combinations(ibc_subjects, 2))


def get_output_name(source_subject, target_subject):
    """Return output name."""
    return f"sub-{source_subject:02d}_sub-{target_subject:02d}_clips-train"


output_path = "/gpfsscratch/rech/nry/uul79xi/inter-species/alignments"

# 2. Setup for aligning all pairs of IBC humans and Leuven macaques

# source_datasets = ["leuven_mk_seg-1", "leuven_mk_seg-2"]
# source_dataset_path = "/gpfsstore/rech/nry/uul79xi/data/leuven"
# source_subjects = ["Luce", "Jack"]

# target_datasets = ["ibc_mk_seg-1", "ibc_mk_seg-2"]
# target_dataset_path = "/gpfsstore/rech/nry/uul79xi/data/ibc"
# target_subjects = [4, 6, 8, 9, 11, 12, 14, 15]


# def get_output_name(source_subject, target_subject):
#     """Return output name."""
#     return f"{source_subject}_sub-{target_subject:02d}_mk-seg-1-2"


# output_path = "/gpfsscratch/rech/nry/uul79xi/inter-species/alignments"

# alpha = 0.5
# eps = 100
# rho = 1e9
# reg_mode = "joint"
# divergence = "l2"
# solver = "mm"

# args_map = list(
#     product(
#         source_subjects,
#         target_subjects,
#     )
# )


def compute_alignment_wrapper(args):
    """Compute alignment between two subjects."""
    (source_subject, target_subject) = args
    output_name = get_output_name(source_subject, target_subject)

    return compute_alignment(
        source_datasets=source_datasets,
        source_dataset_path=source_dataset_path,
        source_subject=source_subject,
        source_lag=0,
        source_window_size=1,
        target_datasets=target_datasets,
        target_dataset_path=target_dataset_path,
        target_subject=target_subject,
        target_lag=0,
        target_window_size=1,
        alpha=alpha,
        eps=eps,
        rho=rho,
        reg_mode=reg_mode,
        divergence=divergence,
        solver=solver,
        output_name=output_name,
        output_path=output_path,
    )


# %%
@hydra.main(version_base="1.2", config_path="../conf", config_name="default")
def launch_jobs(config):
    """Launch all jobs."""
    # Load config
    hydra_config = HydraConfig.get()
    try:
        logger = get_logger(
            f"{hydra_config.job.id} {hydra_config.job.override_dirname}"
        )
    except omegaconf.errors.MissingMandatoryValue:
        logger = get_logger(
            f"{hydra_config.job.name} {hydra_config.job.override_dirname}"
        )

    # Set up executor
    executor = submitit.AutoExecutor(
        folder=Path(hydra_config.runtime.output_dir) / "logs"
    )
    executor.update_parameters(
        slurm_job_name="compute_alignment",
        slurm_time="00:30:00",
        # JZ config
        slurm_account="nry@v100",
        slurm_partition="gpu_p13",
        cpus_per_task=10,
        gpus_per_node=1,
    )

    # Launch jobs
    jobs = executor.map_array(compute_alignment_wrapper, args_map)
    monitor_jobs(jobs, logger=logger, poll_frequency=120)


if __name__ == "__main__":
    launch_jobs()
