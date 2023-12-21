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

from fmri2frame.scripts.compute_latents import compute_latents
from fmri2frame.scripts.utils import get_logger, monitor_jobs


# %%
# Load brain features and latent representations
def init_latent(latent_type, subject):
    """Load brain features and latent representations."""

    print("init_latent", latent_type, subject)

    dataset_id = "ibc_gbu"
    dataset_path = "/gpfsstore/rech/nry/uul79xi/data/ibc"

    pretrained_models = SimpleNamespace(
        **{
            "vdvae": "/gpfsstore/rech/nry/uul79xi/data/vdvae",
            "vd": "/gpfsstore/rech/nry/uul79xi/data",
            "sd": "/gpfsstore/rech/nry/uul79xi/data/stable_diffusion",
        }
    )

    if latent_type in [
        "clip_vision_latents",
        "clip_text_latents",
    ]:
        model_path = pretrained_models.vd
    elif latent_type in [
        "vdvae_encoder_31l_latents",
    ]:
        model_path = pretrained_models.vdvae
    elif latent_type in ["clip_vision_cls"]:
        model_path = None
    elif latent_type in ["sd_autokl"]:
        model_path = pretrained_models.sd
    else:
        raise NotImplementedError()

    latents, metadata = compute_latents(
        dataset_id=dataset_id,
        dataset_path=dataset_path,
        subject=subject,
        latent_type=latent_type,
        model_path=model_path,
        seed=0,
        batch_size=32,
        cache="/gpfsscratch/rech/nry/uul79xi/fmri2frame/cache",
    )

    return latents


# %%
@hydra.main(version_base="1.2", config_path="../conf", config_name="default")
def launch_jobs(config):
    """Launch computation of all alignments."""
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

    executor = submitit.AutoExecutor(
        folder=Path(hydra_config.runtime.output_dir) / "logs"
    )
    executor.update_parameters(
        # slurm_mem_per_gpu=30,
        # slurm_partition="parietal,gpu",
        # slurm_partition="gpu",
        slurm_job_name="init_latent",
        slurm_time="01:00:00",
        # JZ config for computing latents (clip_vision_cls, sd_autokl)
        slurm_account="nry@v100",
        slurm_partition="gpu_p13",
        cpus_per_task=10,
        gpus_per_node=2,
        # JZ config for computing latents (clip_vision_latents, vdvae)
        # slurm_account="nry@v100",
        # slurm_partition="gpu_p4",
        # cpus_per_task=40,
        # gpus_per_node=2,
        # JZ config for saving precomputed latents
        # slurm_account="nry@cpu",
        # cpus_per_task=20,
    )

    def init_latent_wrapper(args):
        """Wrap plot_individual_surf to be used with submitit."""
        (latent_type, subject) = args

        return init_latent(
            latent_type,
            subject,
        )

    args_map = list(
        product(
            # latent type
            [
                "clip_vision_cls",
                # "sd_autokl",
                # "clip_vision_latents",
                # "vdvae_encoder_31l_latents",
            ],
            # subject
            [4, 6],
        )
    )

    jobs = executor.map_array(init_latent_wrapper, args_map)

    monitor_jobs(jobs, logger=logger, poll_frequency=120)


if __name__ == "__main__":
    launch_jobs()
