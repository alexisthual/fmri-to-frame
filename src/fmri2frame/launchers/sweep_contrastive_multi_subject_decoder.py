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
    train_multi_subject_brain_decoder,
)
from fmri2frame.scripts.utils import get_logger, monitor_jobs


# %%
# Experiment parameters

# 1. Sweep model parameters on one subject from the IBC dataset using Clips stimuli

dataset_path = "/gpfsstore/rech/nry/uul79xi/datasets/ibc"

train_subjects = [4, 6, 8, 9, 11, 12, 14, 15]
train_dataset_ids = [
    "ibc_clips_seg-train",
    "ibc_clips_seg-valid",
    "ibc_mk_seg-1",
    "ibc_mk_seg-2",
]
reference_subject = 4
align = False

valid_subject = 4
valid_dataset_ids = [
    "ibc_mk_seg-4",
]

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
# baseline_config.update({
#     "temperature": 0.005,
#     "dropout": 0.7,
# })

# for clip_vision_cls
latent_type = "clip_vision_cls"
baseline_config.update(
    {
        "temperature": 0.03,
        "lr": 1e-4,
        "dropout": 0.8,
        "batch_size": 128,
    }
)

# Possible fine-tuned values
finetuned_values = {
    # "hidden_size_backbone": [128, 256, 756, 1024],
    # "hidden_size_projector": [128, 256, 756, 1024],
    # "dropout": [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    # "n_res_blocks": [0, 1, 3],
    # "n_proj_blocks": [0, 2],
    "temperature": [0.1, 0.05, 0.01, 0.005, 0.003, 0.001],
    # "batch_size": [32, 64, 256, 512, 1024],
    # "lr": [1e-2, 3e-3],
    # "weight_decay": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
}

# Launch 1 job with baseline config
# and then jobs for which the value of only one parameter has changed
args_map = [{}] + [{k: v} for k, values in finetuned_values.items() for v in values]
# args_map = [{k: v} for k, values in finetuned_values.items() for v in values]

exps_path = Path("/gpfsscratch/rech/nry/uul79xi/inter-species")
# alignments_path = exps_path / "alignments" / "clips-train_mm"
alignments_path = exps_path / "alignments" / "clips-train-valid_mk-1-2_mm"
checkpoints_path = (
    exps_path
    / "decoders_multi-subject"
    / "contrastive"
    / "clips-train-valid_mk-1-2"
    / "clips-train-valid_mk-1-2_mm"
    / f"sub-{reference_subject:02d}_{latent_type}"
)
if checkpoints_path is not None:
    checkpoints_path.mkdir(parents=True, exist_ok=True)

# wandb_project_postfix = None
wandb_project_postfix = "train-clips-train-valid_mk-1-2_test-mk-4"


# %%
def train_brain_decoder_wrapper(args):
    """Train decoder in one individual."""
    finetuned_config = args
    print(
        f"Train multi-subject decoder {reference_subject} {latent_type} {finetuned_config}"
    )

    a = list(finetuned_config.keys())
    wandb_tags = [a[0] if len(a) > 0 else "baseline"]

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
        cache=cache,
        # model + training parameters
        wandb_project_postfix=wandb_project_postfix,
        wandb_tags=wandb_tags,
        **{
            **baseline_config,
            **finetuned_config,
        },
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
