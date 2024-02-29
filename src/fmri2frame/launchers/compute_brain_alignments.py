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

alignments_path = Path("/gpfsscratch/rech/nry/uul79xi/inter-species/alignments")

# %%
# 1. Setup for aligning all pairs of IBC subjects

source_datasets = [
    "ibc_clips_seg-train",
    "ibc_clips_seg-valid",
    "ibc_mk_seg-1",
    "ibc_mk_seg-2",
]
source_dataset_path = "/gpfsstore/rech/nry/uul79xi/datasets/ibc"
source_is_macaque = False

target_datasets = [
    "ibc_clips_seg-train",
    "ibc_clips_seg-valid",
    "ibc_mk_seg-1",
    "ibc_mk_seg-2",
]
target_dataset_path = "/gpfsstore/rech/nry/uul79xi/datasets/ibc"
target_is_macaque = False

alpha = 0.5
eps = 1e-4
rho = 1
reg_mode = "joint"
divergence = "kl"
solver = "mm"
# solver = "sinkhorn"

ibc_subjects = [4, 6, 8, 9, 11, 12, 14, 15]
args_map = list(combinations(ibc_subjects, 2))
# args_map = list(product(ibc_subjects, ibc_subjects))


def get_output_name(source_subject, target_subject):
    """Return output name."""
    return f"sub-{source_subject:02d}_sub-{target_subject:02d}"


output_path = alignments_path / f"clips-train-valid_mk-1-2_{solver}"
output_path.mkdir(parents=True, exist_ok=True)


# %%
# 2. Setup for aligning all pairs of IBC humans and Leuven macaques

# source_datasets = ["leuven_mk_seg-1", "leuven_mk_seg-2"]
# source_dataset_path = "/gpfsstore/rech/nry/uul79xi/datasets/leuven"
# source_subjects = ["Luce", "Jack"]
# source_is_macaque = True

# target_datasets = ["ibc_mk_seg-1", "ibc_mk_seg-2"]
# target_dataset_path = "/gpfsstore/rech/nry/uul79xi/datasets/ibc"
# target_subjects = [4, 6, 8, 9, 11, 12, 14, 15]
# target_is_macaque = False

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


# def get_output_name(source_subject, target_subject):
#     """Return output name."""
#     return f"{source_subject}_sub-{target_subject:02d}"


# output_path = (
#     Path("/gpfsscratch/rech/nry/uul79xi/inter-species/alignments") / f"mk-1-2_{solver}"
# )
# output_path.mkdir(parents=True, exist_ok=True)


# %%
def compute_alignment_wrapper(args):
    """Compute alignment between two subjects."""
    (source_subject, target_subject) = args
    output_name = get_output_name(source_subject, target_subject)
    print(f"Align {source_subject} {target_subject}")

    return compute_alignment(
        source_datasets=source_datasets,
        source_dataset_path=source_dataset_path,
        source_subject=source_subject,
        source_lag=0,
        source_window_size=1,
        source_is_macaque=source_is_macaque,
        target_datasets=target_datasets,
        target_dataset_path=target_dataset_path,
        target_subject=target_subject,
        target_lag=0,
        target_window_size=1,
        target_is_macaque=target_is_macaque,
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
        # slurm_time="00:15:00",
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
