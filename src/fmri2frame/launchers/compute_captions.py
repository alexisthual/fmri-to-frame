#!/usr/bin/env python
# coding: utf-8
"""Compute and save captions generated from ground truth clip latents."""

# %%
import pickle
from pathlib import Path
from types import SimpleNamespace

import hydra
import omegaconf
import submitit
from hydra.core.hydra_config import HydraConfig

from fmri2frame.scripts.compute_latents import compute_latents
from fmri2frame.scripts.generate_captions import generate_captions
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

dataset_ids = [
    "ibc_clips_seg-train",
    "ibc_clips_seg-valid",
    "ibc_clips_seg-valid-dedup",
    # "ibc_clips_seg-valid2",
    # "ibc_clips_seg-valid3",
    "ibc_mk_seg-1",
    "ibc_mk_seg-2",
    "ibc_mk_seg-3",
    "ibc_mk_seg-4",
    "ibc_mk_seg-5",
]

dataset_path = "/gpfsstore/rech/nry/uul79xi/datasets/ibc"
subject = 4
latent_type = "clip_vision_cls"
model_path = "/gpfsstore/rech/nry/uul79xi/models/clip"
seed = 0
batch_size = 32
cache = "/gpfsscratch/rech/nry/uul79xi/cache"

output_path = Path("/gpfsscratch/rech/nry/uul79xi/inter-species/captions")
output_path.mkdir(exist_ok=True, parents=True)

args_map = dataset_ids


# %%
# Load brain features and latent representations
def compute_captions(dataset_id):
    """Load brain features and latent representations."""
    print("compute_captions", dataset_id)

    latents, _ = compute_latents(
        dataset_id,
        dataset_path,
        subject,
        latent_type,
        model_path=model_path,
        seed=seed,
        batch_size=batch_size,
        cache=cache,
    )

    print(latents.shape)

    captions = generate_captions(latents)

    with open(output_path / f"{dataset_id}.pkl", "wb") as f:
        pickle.dump(captions, f)


# %%
@hydra.main(version_base="1.2", config_path="../conf", config_name="default")
def launch_jobs(config):
    """Launch computation of all captions."""
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
        slurm_job_name="compute_captions",
        # JZ config for computing latents (clip_vision_cls, sd_autokl)
        # slurm_time="00:40:00",
        # slurm_account="nry@v100",
        # slurm_partition="gpu_p13",
        # cpus_per_task=10,
        # gpus_per_node=2,
        # JZ config for computing latents (clip_vision_latents, vdvae)
        slurm_time="06:00:00",
        slurm_account="nry@v100",
        slurm_partition="gpu_p13",
        cpus_per_task=40,
        gpus_per_node=2,
        # JZ config for saving precomputed latents
        # slurm_account="nry@cpu",
        # cpus_per_task=20,
    )

    # Launch jobs
    jobs = executor.map_array(compute_captions, args_map)
    monitor_jobs(jobs, logger=logger, poll_frequency=120)


if __name__ == "__main__":
    launch_jobs()
