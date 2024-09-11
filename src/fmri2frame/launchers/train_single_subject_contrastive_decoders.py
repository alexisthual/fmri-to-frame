#!/usr/bin/env python
# coding: utf-8

# %%
from pathlib import Path
from types import SimpleNamespace

import hydra
import omegaconf
import submitit
from hydra.core.hydra_config import HydraConfig

from fmri2frame.scripts.train_brain_decoder_contrastive import (
    train_single_subject_brain_decoder,
)
from fmri2frame.scripts.utils import get_logger, monitor_jobs


# %%
# Experiment parameters

train_dataset_ids = [
    "ibc_clips_seg-train",
    "ibc_clips_seg-valid",
    "ibc_mk_seg-1",
    "ibc_mk_seg-2",
    "ibc_mk_seg-3",
]
valid_dataset_ids = [
    # "ibc_clips_seg-valid-dedup"
    "ibc_mk_seg-4",
    "ibc_mk_seg-5",
]
dataset_path = "/lustre/fsn1/projects/rech/nry/uul79xi/store/datasets/ibc"

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
    # "alpha": 1,
    "temperature": 0.01,
    "batch_size": 128,
    "lr": 1e-4,
    "weight_decay": 1e-1,
    "n_epochs": 20,
}

# Baseline updates determined from sweep runs
# For clip_vision_latents
# latent_type = "clip_vision_latents"
# baseline_config.update({})

# For clip_vision_cls
latent_type = "clip_vision_cls"
n_augmentations = 20
baseline_config.update({
    "temperature": 0.03,
    "dropout": 0.7,
    # "alpha": 0.5,
})


exps_path = Path("/lustre/fsn1/projects/rech/nry/uul79xi/inter-species")

# wandb_project_postfix = None
# wandb_project_postfix = "train-clips-train_test-clips-valid1"
# wandb_project_postfix = "train-clips-train_test-clips-valid-dedup"
wandb_project_postfix = "train-clips-train-valid_mk-1-2-3_test-mk-4-5"


# 1. Train decoder on each subject

subjects = [4, 6, 8, 9, 11, 12, 14, 15]
args_map = subjects


# %%
def train_brain_decoder_wrapper(args):
    """Train decoder in one individual."""
    subject = args
    print(f"Train decoder {subject} {latent_type}")

    # checkpoints_path = None
    checkpoints_path = (
        exps_path
        / "decoders"
        / "single-subject"
        # decoder type
        / "contrastive"
        # / f"fused_alpha-{run_config['alpha']}"
        # training data
        / "clips-train-valid_mk-1-2-3"
        # # alignment data
        # / "clips-train-valid_mk-1-2_mm"
        / f"sub-{subject:02d}_{latent_type}"
    )
    if checkpoints_path is not None:
        checkpoints_path.mkdir(parents=True, exist_ok=True)

    train_single_subject_brain_decoder(
        train_dataset_ids=train_dataset_ids,
        valid_dataset_ids=valid_dataset_ids,
        dataset_path=dataset_path,
        subject=subject,
        lag=lag,
        window_size=window_size,
        pretrained_models_path=pretrained_models,
        latent_type=latent_type,
        n_augmentations=n_augmentations,
        cache=cache,
        # model + training parameters
        wandb_project_postfix=wandb_project_postfix,
        # wandb_tags=wandb_tags,
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
        slurm_time="00:20:00",
        # Jean Zay cluster config
        slurm_account="nry@v100",
        slurm_partition="gpu_p13",
        cpus_per_task=20,
        gpus_per_node=1,
    )

    # Launch jobs
    jobs = executor.map_array(train_brain_decoder_wrapper, args_map)
    monitor_jobs(jobs, logger=logger, poll_frequency=120)


# %%
if __name__ == "__main__":
    launch_jobs()
