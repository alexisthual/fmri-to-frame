#!/usr/bin/env python
# coding: utf-8
"""Generate alignment figures."""

# %%
from itertools import combinations, product
from pathlib import Path

import hydra
import omegaconf
import submitit
from hydra.core.hydra_config import HydraConfig

from fmri2frame.scripts.figures_alignments import plot_alignment_maps
from fmri2frame.scripts.utils import get_logger, monitor_jobs


# %%
# Experiment parameters

all_alignments_path = Path("/lustre/fsn1/projects/rech/nry/uul79xi/inter-species/alignments")

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
eps = 100
# fmt: off
rhos = [
    # 1e7, 2e7, 3e7, 4e7, 5e7, 6e7, 7e7, 8e7, 9e7,
    # 1e8, 2e8, 3e8, 4e8, 5e8, 6e8, 7e8, 8e8, 9e8,
    # 1e9,
    1e7, 3e7,
    1e8, 3e8,
    1e9, 3e9,
]
# fmt: on
reg_modes = [
    "joint",
    "independent",
]
divergence = "l2"
solver = "mm"

args_map = list(
    product(
        alphas,
        rhos,
        reg_modes,
    )
)


# %%
def generate_figure_wrapper(args):
    """Compute alignment between two subjects."""
    (alpha, rho, reg_mode) = args

    output_prefix = (
        f"mk-1-2-3_mm_"
        f"div-{divergence}_alpha-{alpha}_rho-{rho:.1e}_eps-{eps}_reg-{reg_mode}"
    )
    alignments_path = all_alignments_path / output_prefix
    output_path = Path("/lustre/fsn1/projects/rech/nry/uul79xi/inter-species/figures/alignments")

    return plot_alignment_maps(
        alignments_path,
        output_path,
        output_prefix
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
        slurm_job_name="generate_figure",
        # slurm_time="00:15:00",
        slurm_time="00:20:00",
        # JZ config
        # slurm_account="nry@cpu",
        # slurm_partition="cpu_p1",
        slurm_account="nry@v100",
        slurm_partition="gpu_p13",
        cpus_per_task=4,
        gpus_per_node=1,
    )

    # Launch jobs
    jobs = executor.map_array(generate_figure_wrapper, args_map)
    monitor_jobs(jobs, logger=logger, poll_frequency=120)


if __name__ == "__main__":
    launch_jobs()
