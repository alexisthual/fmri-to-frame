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

from fmri2frame.scripts.brain_decoder_linear import train_multi_subject_brain_decoder
from fmri2frame.scripts.utils import get_logger, monitor_jobs


# %%
# Experiment parameters

# 1. Train brain decoders on individual IBC subjects on clips-train

subjects = [4, 6, 8, 9, 11, 12, 14, 15]
dataset_ids = [
    "ibc_clips_seg-train",
    "ibc_clips_seg-valid",
    "ibc_mk_seg-1",
    "ibc_mk_seg-2",
]
dataset_path = "/gpfsstore/rech/nry/uul79xi/datasets/ibc"

lag = 2
window_size = 2

pretrained_models = SimpleNamespace(
    **{
        "clip": "/gpfsstore/rech/nry/uul79xi/models/clip",
        "sd": "/gpfsstore/rech/nry/uul79xi/models/stable_diffusion",
        "vd": "/gpfsstore/rech/nry/uul79xi/models",
        "vdvae": "/gpfsstore/rech/nry/uul79xi/models/vdvae",
    }
)
cache = "/gpfsscratch/rech/nry/uul79xi/cache"

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

exps_path = Path("/gpfsscratch/rech/nry/uul79xi/inter-species")
# alignments_path = exps_path / "alignments" / "clips-train_mm"
alignments_path = exps_path / "alignments" / "clips-train-valid_mk-1-2_mm"
# output_path = exps_path / "decoders_multi-subject" / "clips-train"
output_path = (
    exps_path
    / "decoders_multi-subject"
    / "clips-train-valid_mk-1-2"
    / "clips-train-valid_mk-1-2_mm"
)
output_path.mkdir(parents=True, exist_ok=True)


# %%
def train_brain_decoder_wrapper(args):
    """Train decoder in one individual."""
    reference_subject, latent_type = args
    print(f"Train decoder {reference_subject} {latent_type}")

    train_multi_subject_brain_decoder(
        dataset_ids=dataset_ids,
        dataset_path=dataset_path,
        reference_subject=reference_subject,
        training_subjects=subjects,
        lag=lag,
        window_size=window_size,
        latent_type=latent_type,
        pretrained_models_path=pretrained_models,
        cache=cache,
        alignments_path=alignments_path,
        output_path=output_path / f"sub-{reference_subject:02d}_{latent_type}.pkl",
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
        slurm_time="02:00:00",
        # JZ config
        # slurm_account="nry@cpu",
        # # slurm_partition="prepost",
        # slurm_partition="cpu_p1",
        # cpus_per_task=10,
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
