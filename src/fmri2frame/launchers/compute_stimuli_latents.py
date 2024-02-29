#!/usr/bin/env python
# coding: utf-8
"""Compute and cache stimuli latent representations for a list of subjects."""

# %%
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import hydra
import omegaconf
import submitit
from hydra.core.hydra_config import HydraConfig

from fmri2frame.scripts.compute_latents import compute_latents
from fmri2frame.scripts.utils import get_logger, monitor_jobs


# %%
pretrained_models = SimpleNamespace(
    **{
        "clip": "/gpfsstore/rech/nry/uul79xi/models/clip",
        "sd": "/gpfsstore/rech/nry/uul79xi/models/stable_diffusion",
        "vd": "/gpfsstore/rech/nry/uul79xi/models/versatile_diffusion",
        "vdvae": "/gpfsstore/rech/nry/uul79xi/models/vdvae",
    }
)

seed = 0
batch_size = 32
cache = "/gpfsscratch/rech/nry/uul79xi/cache"

# 1. Human subjects

# dataset_ids = ["ibc_clips_seg-train", "ibc_clips_seg-valid"]
# dataset_ids = [
#     "ibc_clips_seg-train",
#     "ibc_clips_seg-valid",
#     "ibc_mk_seg-1",
#     "ibc_mk_seg-2",
#     "ibc_mk_seg-3",
#     "ibc_mk_seg-4",
#     "ibc_mk_seg-5",
# ]
# dataset_path = "/gpfsstore/rech/nry/uul79xi/datasets/ibc"
# subjects = [4, 6, 8, 9, 11, 12, 14, 15]

# latent_types = [
#     "clip_vision_cls",
#     # "sd_autokl",
#     # "clip_vision_latents",
#     # "vdvae_encoder_31l_latents",
# ]

# args_map = list(
#     product(
#         dataset_ids,
#         latent_types,
#         subjects,
#     )
# )


# 2. Non-human subjects

dataset_ids = [
    "leuven_mk_seg-1",
    "leuven_mk_seg-2",
    "leuven_mk_seg-3",
    "leuven_mk_seg-4",
    "leuven_mk_seg-5",
]
dataset_path = "/gpfsstore/rech/nry/uul79xi/datasets/leuven"
subjects = ["Luce", "Jack"]

latent_types = [
    "clip_vision_cls",
    # "sd_autokl",
    "clip_vision_latents",
    # "vdvae_encoder_31l_latents",
]

args_map = list(
    product(
        dataset_ids,
        latent_types,
        subjects,
    )
)


# %%
# Load brain features and latent representations
def init_latent(dataset_id, latent_type, subject):
    """Load brain features and latent representations."""
    print("init_latent", dataset_id, latent_type, subject)

    if latent_type in ["clip_vision_latents", "clip_text_latents"]:
        model_path = pretrained_models.vd
    elif latent_type in ["vdvae_encoder_31l_latents"]:
        model_path = pretrained_models.vdvae
    elif latent_type in ["clip_vision_cls"]:
        model_path = pretrained_models.clip
    elif latent_type in ["sd_autokl"]:
        model_path = pretrained_models.sd
    else:
        raise NotImplementedError()

    latents, _ = compute_latents(
        dataset_id=dataset_id,
        dataset_path=dataset_path,
        subject=subject,
        latent_type=latent_type,
        model_path=model_path,
        seed=seed,
        batch_size=batch_size,
        cache=cache,
    )

    return latents


def init_latent_wrapper(args):
    """Wrap plot_individual_surf to be used with submitit."""
    (dataset_id, latent_type, subject) = args

    return init_latent(
        dataset_id,
        latent_type,
        subject,
    )


# %%
@hydra.main(version_base="1.2", config_path="../conf", config_name="default")
def launch_jobs(config):
    """Launch computation of all latents."""
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
        # slurm_mem_per_gpu=30,
        # slurm_partition="parietal,gpu",
        # slurm_partition="gpu",
        slurm_job_name="init_latent",
        # JZ config for computing latents (clip_vision_cls, sd_autokl)
        # slurm_time="00:40:00",
        # slurm_account="nry@v100",
        # slurm_partition="gpu_p13",
        # cpus_per_task=10,
        # gpus_per_node=2,
        # JZ config for computing latents (clip_vision_latents, vdvae)
        slurm_time="00:40:00",
        slurm_account="nry@v100",
        slurm_partition="gpu_p13",
        cpus_per_task=40,
        gpus_per_node=2,
        # JZ config for saving precomputed latents
        # slurm_account="nry@cpu",
        # cpus_per_task=20,
    )

    # Launch jobs
    jobs = executor.map_array(init_latent_wrapper, args_map)
    monitor_jobs(jobs, logger=logger, poll_frequency=120)


if __name__ == "__main__":
    launch_jobs()
