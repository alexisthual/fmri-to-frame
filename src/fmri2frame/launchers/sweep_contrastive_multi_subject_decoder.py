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
    train_multi_subject_brain_decoder,
)
from fmri2frame.scripts.utils import get_logger, monitor_jobs


# %%
# Experiment parameters

# 1. Sweep model parameters on one subject from the IBC dataset using Clips stimuli

dataset_path = "/lustre/fsstor/projects/rech/nry/uul79xi/datasets/ibc"

train_subjects = [4, 6, 8, 9, 11, 12, 14, 15]
train_dataset_ids = [
    "ibc_clips_seg-train",
    # "ibc_clips_seg-valid",
    # "ibc_mk_seg-1",
    # "ibc_mk_seg-2",
    # "ibc_mk_seg-3",
    # "ibc_mk_seg-4",
    # "ibc_mk_seg-5",
]
# reference_subjects = [4, 6, 8, 9, 11, 12, 14, 15]
reference_subject = 4
align = True

valid_subject = 4
valid_dataset_ids = [
    "ibc_clips_seg-valid",
    # "ibc_mk_seg-1",
]

lag = 2
window_size = 2

pretrained_models = SimpleNamespace(
    **{
        "clip": "/lustre/fsstor/projects/rech/nry/uul79xi/models/clip",
        "sd": "/lustre/fsstor/projects/rech/nry/uul79xi/models/stable_diffusion",
        "vd": "/lustre/fsstor/projects/rech/nry/uul79xi/models/versatile_diffusion",
        "vdvae": "/lustre/fsstor/projects/rech/nry/uul79xi/models/vdvae",
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
    # "alpha": 0,
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

# Possible fine-tuned values
finetuned_values = {
    # "hidden_size_backbone": [128, 256, 756, 1024],
    # "hidden_size_projector": [128, 256, 756, 1024],
    # "dropout": [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    # "n_res_blocks": [0, 1, 3],
    # "n_proj_blocks": [0, 2],
    # "temperature": [0.1, 0.05, 0.01, 0.005, 0.003, 0.001],
    # "batch_size": [32, 64, 256, 512, 1024],
    # "lr": [1e-2, 3e-3],
    # "weight_decay": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    # "alpha": [],
}

# Launch 1 job with baseline config
# and then jobs for which the value of only one parameter has changed
args_map = [{}] + [{k: v} for k, values in finetuned_values.items() for v in values]
# args_map = [{k: v} for k, values in finetuned_values.items() for v in values]

exps_path = Path("/lustre/fsn1/projects/rech/nry/uul79xi/inter-species")
alignments_path = exps_path / "alignments" / "clips-train_mm_alpha-0.5"
# alignments_path = exps_path / "alignments" / "clips-train-valid_mk-1-2_mm_alpha-0.5"

train_subjects_str = f"{'-'.join([f'{s:02d}' for s in train_subjects])}"


# wandb_project_postfix = None
# wandb_project_postfix = "train-clips-train_test-clips-valid"
# wandb_project_postfix = "train-clips-train-valid_mk-1-2_test-mk-4"
# wandb_project_postfix = "train-clips-train-valid_mk-2-3-4-5_test-mk-1"
wandb_project_postfix = "train-clips-train_test-clips-valid"


# %%
def train_brain_decoder_wrapper(args):
    """Train decoder in one individual."""
    finetuned_config = args
    print(
        f"Train multi-subject decoder {reference_subject} {latent_type} "
        f"{finetuned_config}"
    )

    run_config = {
        **baseline_config,
        **finetuned_config,
    }

    checkpoints_path = None
    checkpoints_path = (
        exps_path
        / "decoders"
        / "multi-subject"
        / f"fused_alpha-{run_config['alpha']}"
        # brain decoder training data
        / "clips-train"
        # alignment training data
        / "clips-train_mm_alpha-0.5"
        # reference subject, training subjects and latent type
        / f"ref-{reference_subject:02d}_train-{train_subjects_str}_{latent_type}"
    )
    if checkpoints_path is not None:
        checkpoints_path.mkdir(parents=True, exist_ok=True)

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
        n_augmentations=n_augmentations,
        cache=cache,
        # model + training parameters
        wandb_project_postfix=wandb_project_postfix,
        wandb_tags=wandb_tags,
        **run_config,
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
