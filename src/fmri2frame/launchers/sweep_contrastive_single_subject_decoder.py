#!/usr/bin/env python
# coding: utf-8

# %%
from pathlib import Path
from types import SimpleNamespace

import hydra
import omegaconf
import submitit
from hydra.core.hydra_config import HydraConfig

from fmri2frame.scripts.brain_decoder_contrastive import (
    train_single_subject_brain_decoder,
)
from fmri2frame.scripts.utils import get_logger, monitor_jobs


# %%
# Experiment parameters

# 1. Sweep model parameters on one subject from the IBC dataset using Clips stimuli

subject = 4
train_dataset_ids = ["ibc_clips_seg-train"]
valid_dataset_ids = ["ibc_clips_seg-valid"]
dataset_path = "/gpfsstore/rech/nry/uul79xi/datasets/ibc"

lag = 2
window_size = 2

pretrained_models = SimpleNamespace(
    **{
        "clip": "/gpfsstore/rech/nry/uul79xi/models/clip",
        "sd": "/gpfsstore/rech/nry/uul79xi/models/stable_diffusion",
        "vd": "/gpfsstore/rech/nry/uul79xi/models/versatile_diffusion",
        "vdvae": "/gpfsstore/rech/nry/uul79xi/models/vdvae",
    }
)
cache = "/gpfsscratch/rech/nry/uul79xi/cache"

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
# baseline_config.update({})

# for clip_vision_cls
latent_type = "clip_vision_cls"
baseline_config.update({})

# Possible fine-tuned values
finetuned_values = {
    # "hidden_size_backbone": [256, 1024],
    # "hidden_size_projector": [256, 1024],
    # "dropout": [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    # "n_res_blocks": [0, 1, 3],
    # "n_proj_blocks": [0, 2],
    "temperature": [0.1, 0.03, 0.003, 0.001, 0.0001],
    # "batch_size": [32, 64, 128, 256, 512, 1024],
    # "lr": [1e-3, 1e-4, 1e-5],
    # "weight_decay": [1e-1, 1e-2, 1e-3, 1e-4],
}

# Launch 1 job with baseline config
# and then jobs for which the value of only one parameter has changed
# args_map = [{}] + [{k: v} for k, values in finetuned_values.items() for v in values]
args_map = [{k: v} for k, values in finetuned_values.items() for v in values]
# args_map = args_map[:2]


# %%
def train_brain_decoder_wrapper(args):
    """Train decoder in one individual."""
    finetuned_config = args
    print(f"Train decoder {subject} {latent_type} {finetuned_config}")

    a = list(finetuned_config.keys())
    wandb_tags = [a[0] if len(a) > 0 else "baseline"]

    train_single_subject_brain_decoder(
        train_dataset_ids=train_dataset_ids,
        valid_dataset_ids=valid_dataset_ids,
        dataset_path=dataset_path,
        subject=subject,
        lag=lag,
        window_size=window_size,
        latent_type=latent_type,
        pretrained_models_path=pretrained_models,
        cache=cache,
        # model + training parameters
        wandb_tags=wandb_tags,
        **{
            **baseline_config,
            **finetuned_config,
        }
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
