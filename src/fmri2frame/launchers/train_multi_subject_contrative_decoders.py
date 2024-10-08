#!/usr/bin/env python
# coding: utf-8

# %%
from itertools import combinations, product
from pathlib import Path
from types import SimpleNamespace

import hydra
import omegaconf
import submitit
from hydra.core.hydra_config import HydraConfig

from fmri2frame.scripts.train_brain_decoder_contrastive import (
    train_multi_subject_brain_decoder,
)
from fmri2frame.scripts.utils import get_logger, monitor_jobs


# %%
# Experiment parameters

# 1. Sweep model parameters on one subject from the IBC dataset using Clips stimuli

dataset_path = "/lustre/fsn1/projects/rech/nry/uul79xi/store/datasets/ibc"

all_training_subjects = [4, 6, 8, 9, 11, 12, 14, 15]
train_dataset_ids = [
    "ibc_clips_seg-train",
    "ibc_clips_seg-valid",
    "ibc_mk_seg-1",
    "ibc_mk_seg-2",
    "ibc_mk_seg-3",
    # "ibc_mk_seg-4",
    # "ibc_mk_seg-5",
]

reference_subjects = [4, 6, 8, 9, 11, 12, 14, 15]
align = True

valid_dataset_ids = [
    # "ibc_clips_seg-valid-dedup",
    # "ibc_clips_seg-valid",
    # "ibc_mk_seg-1",
    "ibc_mk_seg-4",
    "ibc_mk_seg-5",
]

lag = 2
window_size = 2

pretrained_models = SimpleNamespace(
    **{
        "clip": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/clip",
        "sd": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/stable_diffusion",
        "vd": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/versatile_diffusion",
        "vdvae": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/vdvae",
    }
)
cache = "/lustre/fsn1/projects/rech/nry/uul79xi/cache"


# Baseline configuration
baseline_config = {
    "hidden_size_backbone": 512,
    "hidden_size_projector": 512,
    "dropout": 0.3,
    "n_res_blocks": 2,
    "n_proj_blocks": 1,
    "temperature": 0.01,
    "batch_size": 128,
    "lr": 1e-4,
    "weight_decay": 0,
    "n_epochs": 20,
}

# Baseline updates determined from sweep runs
# for clip_vision_latents
# latent_type = "clip_vision_latents"
# baseline_config.update({
#     "temperature": 0.005,
#     "dropout": 0.7,
# })

# for clip_vision_cls
latent_type = "clip_vision_cls"
n_augmentations = 20
baseline_config.update(
    {
        "temperature": 0.03,
        "lr": 1e-4,
        "dropout": 0.8,
        "batch_size": 128,
    }
)


exps_path = Path("/lustre/fsn1/projects/rech/nry/uul79xi/inter-species")
alignments_path = exps_path / "alignments" / "clips-train_mm_alpha-0.5"
alignments_path = (
    exps_path
    / "alignments"
    / "clips-train_mm_div-kl_alpha-0.5_rho-1.0e+00_eps-0.0001_reg-joint"
)
# alignments_path = exps_path / "alignments" / "clips-train-valid_mk-1-2_mm_alpha-0.5"

# train_subjects_str = f"{'-'.join([f'{s:02d}' for s in train_subjects])}"

# wandb_project_postfix = None
# wandb_project_postfix = "train-clips-train-valid-mk-1-2-3_valid-mk-4"
# wandb_project_postfix = "train-clips-train_test-clips-valid-dedup"
wandb_project_postfix = "train-clips-train-valid_mk-1-2-3_test-mk-4-5"
# wandb_project_postfix = "train-clips-train-valid_mk-2-3-4-5_test-mk-1"
# wandb_project_postfix = "train-clips-train_test-clips-valid"

args_map = product(
    reference_subjects,
    combinations(
        all_training_subjects,
        # len(all_training_subjects) - 1
        len(all_training_subjects)
    )
)


# %%
def train_brain_decoder_wrapper(args):
    """Train decoder in one individual."""
    (reference_subject, train_subjects) = args
    valid_subject = reference_subject

    train_subjects_str = f"{'-'.join([f'{s:02d}' for s in train_subjects])}"

    print(
        f"Train multi-subject decoder {reference_subject} {latent_type}"
    )

    # checkpoints_path = None
    checkpoints_path = (
        exps_path
        / "decoders"
        / "multi-subject"
        / "contrastive"
        # brain decoder training data
        # / "clips-train"
        # / "clips-train"
        / "clips-train-valid-mk-1-2-3"
        # / "clips-train-valid_mk-2-3-4-5"
        # alignment plans
        / "clips-train_mm_alpha-0.5"
        # / "clips-train-valid_mk-1-2-3_mm_alpha-0.5"
        # reference subject, training subjects and latent type
        / f"ref-{reference_subject:02d}_train-{train_subjects_str}_{latent_type}"
    )
    if checkpoints_path is not None:
        checkpoints_path.mkdir(parents=True, exist_ok=True)

    wandb_tags = []

    train_multi_subject_brain_decoder(
        reference_subject=reference_subject,
        alignments_path=alignments_path,
        align=align,
        train_dataset_ids=train_dataset_ids,
        train_subjects=train_subjects,
        valid_dataset_ids=valid_dataset_ids,
        valid_subject=valid_subject,
        dataset_path=dataset_path,
        lag=lag,
        window_size=window_size,
        latent_type=latent_type,
        pretrained_models_path=pretrained_models,
        n_augmentations=n_augmentations,
        cache=cache,
        # model + training parameters
        wandb_project_postfix=wandb_project_postfix,
        wandb_tags=wandb_tags,
        **baseline_config,
        checkpoints_path=checkpoints_path,
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
        slurm_job_name="train_decoder",
        slurm_time="03:00:00",
        # Jean Zay cluster config
        slurm_account="nry@v100",
        slurm_partition="gpu_p13",
        cpus_per_task=30,
        gpus_per_node=1,
    )

    # Launch jobs
    jobs = executor.map_array(train_brain_decoder_wrapper, args_map)
    monitor_jobs(jobs, logger=logger, poll_frequency=120)


# %%
if __name__ == "__main__":
    launch_jobs()
