#!/usr/bin/env python
# coding: utf-8

# %%
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import omegaconf
import submitit
from hydra.core.hydra_config import HydraConfig

from fmri2frame.scripts.eval import evaluate_brain_decoder
from fmri2frame.scripts.utils import get_logger, monitor_jobs

# %%
# Experiment parameters

# 1. Evaluate brain decoders on individual IBC subjects on clips-train

# subjects = [4, 6, 8, 9, 11, 12, 14, 15]
# dataset_ids = ["ibc_clips_seg-valid"]
# dataset_path = "/gpfsstore/rech/nry/uul79xi/data/ibc"
# lag = 2
# window_size = 2

# pretrained_models = SimpleNamespace(
#     **{
#         "vdvae": "/gpfsstore/rech/nry/uul79xi/data/vdvae",
#         "vd": "/gpfsstore/rech/nry/uul79xi/data",
#         "sd": "/gpfsstore/rech/nry/uul79xi/data/stable_diffusion",
#     }
# )
# cache = "/gpfsscratch/rech/nry/uul79xi/cache"

# latent_types = [
#     "clip_vision_cls",
#     # "sd_autokl",
#     # "clip_vision_latents",
#     # "vdvae_encoder_31l_latents",
# ]

# args_map = list(
#     product(
#         subjects,
#         list(
#             zip(
#                 subjects,
#                 np.tile(dataset_ids, (len(subjects), 1)).tolist(),
#                 np.tile(dataset_path, len(subjects)).tolist(),
#                 np.tile(lag, len(subjects)).tolist(),
#                 np.tile(window_size, len(subjects)).tolist(),
#                 np.tile(False, len(subjects)).tolist(),  # subject_is_macaque
#             )
#         ),
#         latent_types,
#     )
# )

# exps_path = Path("/gpfsscratch/rech/nry/uul79xi/inter-species")
# # decoders_path = exps_path / "decoders_single-subject" / "clips-train"
# decoders_path = exps_path / "decoders_multi-subject" / "clips-train"
# alignments_path = exps_path / "alignments" / "clips-train_mm"
# # output_path = exps_path / "evaluations" / "clips-train_mm_single"
# output_path = exps_path / "evaluations" / "clips-train_mm_multi"
# output_path.mkdir(parents=True, exist_ok=True)


# 2. Evaluate brain decoders on IBC + Leuven

human_subjects = [4, 6, 8, 9, 11, 12, 14, 15]
human_dataset_ids = [
    # "ibc_mk_seg-3",
    "ibc_mk_seg-4",
    # "ibc_mk_seg-5",
]
human_dataset_path = "/gpfsstore/rech/nry/uul79xi/data/ibc"
human_lag = 2
human_window_size = 2

macaque_subjects = ["Luce", "Jack"]
macaque_dataset_ids = [
    # "leuven_mk_seg-3",
    "leuven_mk_seg-4",
    # "leuven_mk_seg-5",
]
macaque_dataset_path = "/gpfsstore/rech/nry/uul79xi/data/leuven"
macaque_lag = 2
macaque_window_size = 2

pretrained_models = SimpleNamespace(
    **{
        "vdvae": "/gpfsstore/rech/nry/uul79xi/data/vdvae",
        "vd": "/gpfsstore/rech/nry/uul79xi/data",
        "sd": "/gpfsstore/rech/nry/uul79xi/data/stable_diffusion",
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
        human_subjects,
        list(
            zip(
                human_subjects,
                np.tile(human_dataset_ids, (len(human_subjects), 1)).tolist(),
                np.tile(human_dataset_path, len(human_subjects)).tolist(),
                np.tile(human_lag, len(human_subjects)).tolist(),
                np.tile(human_window_size, len(human_subjects)).tolist(),
                np.tile(False, len(human_subjects)).tolist(),  # subject_is_macaque
            )
        )
        + list(
            zip(
                macaque_subjects,
                np.tile(macaque_dataset_ids, (len(macaque_subjects), 1)).tolist(),
                np.tile(macaque_dataset_path, len(macaque_subjects)).tolist(),
                np.tile(macaque_lag, len(macaque_subjects)).tolist(),
                np.tile(macaque_window_size, len(macaque_subjects)).tolist(),
                np.tile(True, len(macaque_subjects)).tolist(),  # subject_is_macaque
            )
        ),
        latent_types,
    )
)

exps_path = Path("/gpfsscratch/rech/nry/uul79xi/inter-species")
decoders_path = (
    exps_path
    / "decoders_multi-subject"
    / "clips-train-valid_mk-1-2"
    / "clips-train-valid_mk-1-2_mm"
)
alignments_path = exps_path / "alignments" / "mk-1-2_mm"
output_path = (
    exps_path
    / "evaluations"
    / "multi-subject"
    / "clips-train-valid_mk-1-2"
    / "clips-train-valid_mk-1-2_mm"
    / "mk-4"
)
# decoders_path = exps_path / "decoders_single-subject" / "clips-train-valid_mk-1-2"
# alignments_path = exps_path / "alignments" / "mk-1-2_mm"
# output_path = exps_path / "evaluations" / "clips-train-valid_mk-1-2" / "mk-4"
output_path.mkdir(parents=True, exist_ok=True)


# %%
def eval_brain_decoder_wrapper(args):
    """Evaluate brain decoder."""
    (
        reference_subject,
        (
            leftout_subject,
            dataset_ids,
            dataset_path,
            lag,
            window_size,
            leftout_is_macaque,
        ),
        latent_type,
    ) = args
    print(f"Evaluate decoder {reference_subject} {leftout_subject} {latent_type}")

    left_mapping_path = None
    right_mapping_path = None
    invert_mapping = None
    selected_indices_left_path = None
    selected_indices_right_path = None

    if leftout_is_macaque:
        exp_name = f"{leftout_subject}_sub-{reference_subject:02d}"
        exp_name_invert = f"sub-{reference_subject:02d}_{leftout_subject}"
    else:
        exp_name = f"sub-{leftout_subject:02d}_sub-{reference_subject:02d}"
        exp_name_invert = f"sub-{reference_subject:02d}_sub-{leftout_subject:02d}"

    if leftout_subject != reference_subject:
        invert_mapping = False
        left_mapping_path = alignments_path / f"{exp_name}_left.pkl"
        right_mapping_path = alignments_path / f"{exp_name}_right.pkl"
        if leftout_is_macaque:
            selected_indices_left_path = (
                alignments_path / f"{exp_name}_selected_indices_left_source.npy"
            )
            selected_indices_right_path = (
                alignments_path / f"{exp_name}_selected_indices_right_source.npy"
            )

        if not left_mapping_path.exists():
            invert_mapping = True
            left_mapping_path = alignments_path / f"{exp_name_invert}_left.pkl"
            right_mapping_path = alignments_path / f"{exp_name_invert}_right.pkl"
            if leftout_is_macaque:
                selected_indices_left_path = (
                    alignments_path
                    / f"{exp_name_invert}_selected_indices_left_source.npy"
                )
                selected_indices_right_path = (
                    alignments_path
                    / f"{exp_name_invert}_selected_indices_right_source.npy"
                )

            if not left_mapping_path.exists():
                print(left_mapping_path)
                print(right_mapping_path)
                raise Exception(
                    "There is no mapping between subjects "
                    f"{leftout_subject} and {reference_subject}"
                )

    if leftout_is_macaque:
        output_name = f"sub-{reference_subject:02d}_{latent_type}_{leftout_subject}"
    else:
        output_name = (
            f"sub-{reference_subject:02d}_{latent_type}_sub-{leftout_subject:02d}"
        )

    evaluate_brain_decoder(
        decoder_path=decoders_path / f"sub-{reference_subject:02d}_{latent_type}.pkl",
        dataset_ids=dataset_ids,
        dataset_path=dataset_path,
        subject=leftout_subject,
        subject_is_macaque=leftout_is_macaque,
        lag=lag,
        window_size=window_size,
        latent_type=latent_type,
        pretrained_models_path=pretrained_models,
        cache=cache,
        left_mapping_path=left_mapping_path,
        right_mapping_path=right_mapping_path,
        invert_mapping=invert_mapping,
        selected_indices_left_path=selected_indices_left_path,
        selected_indices_right_path=selected_indices_right_path,
        output_name=output_name,
        output_path=output_path,
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
        slurm_job_name="eval_decoder",
        slurm_time="00:10:00",
        # JZ config
        slurm_account="nry@v100",
        slurm_partition="gpu_p13",
        # cpus_per_task=10,  # ok for clips-valid
        cpus_per_task=10,
        gpus_per_node=1,
    )

    # Launch jobs
    jobs = executor.map_array(eval_brain_decoder_wrapper, args_map)
    monitor_jobs(jobs, logger=logger, poll_frequency=120)


# %%
if __name__ == "__main__":
    launch_jobs()
