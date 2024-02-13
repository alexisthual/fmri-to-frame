#!/usr/bin/env python
# coding: utf-8

# %%
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import hydra
import omegaconf
import submitit
from hydra.core.hydra_config import HydraConfig

from fmri2frame.scripts.train_brain_decoder import train_brain_decoder
from fmri2frame.scripts.utils import get_logger, monitor_jobs


# %%
# Experiment parameters

# 1. Train brain decoders on individual IBC subjects on clips-train
subjects = [4, 6, 8, 9, 11, 12, 14, 15]
dataset_ids = ["ibc_clips_seg-train"]
dataset_path = "/gpfsstore/rech/nry/uul79xi/data/ibc"

lag = 2
window_size = 2

pretrained_models = SimpleNamespace(
    **{
        "vdvae": "/gpfsstore/rech/nry/uul79xi/data/vdvae",
        "vd": "/gpfsstore/rech/nry/uul79xi/data",
        "sd": "/gpfsstore/rech/nry/uul79xi/data/stable_diffusion",
    }
)
cache = "/gpfsscratch/rech/nry/uul79xi/fmri2frame/cache"

latent_types = [
    "clip_vision_cls",
    # "sd_autokl",
    # "clip_vision_latents",
    # "vdvae_encoder_31l_latents",
]

args_map = list(
    product(
        subjects,
        latent_types,
    )
)

output_path = Path(
    "/gpfsscratch/rech/nry/uul79xi/inter-species/brain-decoders/clips-train"
)


# %%
def train_brain_decoder_wrapper(args):
    """Train decoder in one individual."""
    subject, latent_type = args

    train_brain_decoder(
        dataset_ids=dataset_ids,
        dataset_path=dataset_path,
        subject=subject,
        lag=lag,
        window_size=window_size,
        latent_type=latent_type,
        pretrained_models_path=pretrained_models,
        cache=cache,
        output_path=output_path / f"sub-{subject:02d}_{latent_type}.pkl",
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
        slurm_job_name="train_brain_decoder",
        slurm_time="00:30:00",
        # JZ config
        slurm_account="nry@cpu",
        slurm_partition="prepost",
        cpus_per_task=10,
    )

    # Launch jobs
    jobs = executor.map_array(train_brain_decoder_wrapper, args_map)
    monitor_jobs(jobs, logger=logger, poll_frequency=120)


# %%
if __name__ == "__main__":
    launch_jobs()
