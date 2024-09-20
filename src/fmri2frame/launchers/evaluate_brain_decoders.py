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

from fmri2frame.scripts.evaluate_brain_decoder import evaluate_brain_decoder
from fmri2frame.scripts.utils import get_logger, monitor_jobs

# %%
# Experiment parameters


# 1.1. Evaluate brain decoders (single) on individual IBC subjects on runs of clips-valid

# eval_subjects = [4, 6, 8, 9, 11, 12, 14, 15]
# train_subjects = [4, 6, 8, 9, 11, 12, 14, 15]
# # train_subjects_str = f"{'-'.join([f'{s:02d}' for s in train_subjects])}"

# dataset_ids = ["ibc_clips_seg-valid-dedup"]
# dataset_path = "/lustre/fsn1/projects/rech/nry/uul79xi/store/datasets/ibc"
# lag = 2
# window_size = 2

# pretrained_models = SimpleNamespace(
#     **{
#         "clip": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/clip",
#         "sd": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/stable_diffusion",
#         "vd": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/versatile_diffusion",
#         "vdvae": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/vdvae",
#     }
# )
# cache = "/lustre/fsn1/projects/rech/nry/uul79xi/cache"

# latent_types = [
#     "clip_vision_cls",
#     # "sd_autokl",
#     # "clip_vision_latents",
#     # "vdvae_encoder_31l_latents",
# ]

# args_map = list(
#     product(
#         # reference / training subjects
#         train_subjects,
#         list(
#             zip(
#                 eval_subjects,
#                 np.tile(dataset_ids, (len(eval_subjects), 1)).tolist(),
#                 np.tile(dataset_path, len(eval_subjects)).tolist(),
#                 np.tile(lag, len(eval_subjects)).tolist(),
#                 np.tile(window_size, len(eval_subjects)).tolist(),
#                 np.tile(False, len(eval_subjects)).tolist(),  # subject_is_macaque
#             )
#         ),
#         latent_types,
#     )
# )

# # train_subjects = [4, 6, 8, 9, 11, 12, 14, 15]
# # train_subjects_str = f"{'-'.join([f'{s:02d}' for s in train_subjects])}"

# # Paths
# exps_path = Path("/lustre/fsn1/projects/rech/nry/uul79xi/inter-species")
# decoders_path = (
#     exps_path
#     / "decoders"
#     # type of decoder
#     / "single-subject"
#     # / "multi-subject"
#     / "contrastive"
#     # / "fused_alpha-0.99"
#     / "clips-train"
#     # / "sub-04_clip_vision_cls"
# )
# decoder_is_contrastive = True
# checkpoint = 10

# macaques_alignments_path = None
# humans_alignments_path = (
#     exps_path
#     / "alignments"
#     # / "clips-train_mm_alpha-0.5"
#     / "clips-train_mm_div-kl_alpha-0.5_rho-1.0e+00_eps-0.0001_reg-joint"
# )

# output_path = (
#     exps_path
#     / "evaluations"
#     / "single-subject"
#     / "contrastive"
#     # / "fused_alpha-0.99"
#     / "clips-train"
#     # / "clips-train_mm_alpha-0.5"
#     / "clips-valid-dedup"
# )
# output_path.mkdir(parents=True, exist_ok=True)


# 1.2. Evaluate brain decoders (multi) on individual IBC subjects on runs of clips-valid

# ref_subjects = [4, 6, 8, 9, 11, 12, 14, 15]
# eval_subjects = [4, 6, 8, 9, 11, 12, 14, 15]

# dataset_ids = ["ibc_clips_seg-valid-dedup"]
# dataset_path = "/lustre/fsn1/projects/rech/nry/uul79xi/store/datasets/ibc"
# lag = 2
# window_size = 2

# pretrained_models = SimpleNamespace(
#     **{
#         "clip": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/clip",
#         "sd": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/stable_diffusion",
#         "vd": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/versatile_diffusion",
#         "vdvae": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/vdvae",
#     }
# )
# cache = "/lustre/fsn1/projects/rech/nry/uul79xi/cache"

# latent_types = [
#     "clip_vision_cls",
#     # "sd_autokl",
#     # "clip_vision_latents",
#     # "vdvae_encoder_31l_latents",
# ]

# train_subjects = [4, 6, 8, 9, 11, 12, 14, 15]
# train_subjects_str = f"{'-'.join([f'{s:02d}' for s in train_subjects])}"

# args_map = list(
#     product(
#         # reference subject
#         ref_subjects,
#         list(
#             zip(
#                 eval_subjects,
#                 np.tile(dataset_ids, (len(eval_subjects), 1)).tolist(),
#                 np.tile(dataset_path, len(eval_subjects)).tolist(),
#                 np.tile(lag, len(eval_subjects)).tolist(),
#                 np.tile(window_size, len(eval_subjects)).tolist(),
#                 np.tile(False, len(eval_subjects)).tolist(),  # subject_is_macaque
#             )
#         ),
#         latent_types,
#     )
# )

# # Paths
# exps_path = Path("/lustre/fsn1/projects/rech/nry/uul79xi/inter-species")
# decoders_path = (
#     exps_path
#     / "decoders"
#     # type of decoder
#     / "multi-subject"
#     / "contrastive"
#     / "clips-train"
#     / "clips-train_mm_alpha-0.5"
# )
# decoder_is_contrastive = True
# checkpoint = 5

# alignments_path = exps_path / "alignments" / "clips-train_mm_alpha-0.5"

# output_path = (
#     exps_path
#     / "evaluations"
#     / "multi-subject"
#     / "contrastive"
#     # / "fused_alpha-0.99"
#     / "clips-train"
#     / "clips-train_mm_alpha-0.5"
#     / "clips-valid-dedup"
# )
# output_path.mkdir(parents=True, exist_ok=True)


# 1.3. Evaluate brain decoders (multi) on individual IBC subjects on runs of clips-valid

# ref_subjects = [4, 6, 8, 9, 11, 12, 14, 15]
# eval_subjects = [4, 6, 8, 9, 11, 12, 14, 15]

# dataset_ids = ["ibc_clips_seg-valid-dedup"]
# dataset_path = "/lustre/fsn1/projects/rech/nry/uul79xi/store/datasets/ibc"
# lag = 2
# window_size = 2

# pretrained_models = SimpleNamespace(
#     **{
#         "clip": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/clip",
#         "sd": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/stable_diffusion",
#         "vd": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/versatile_diffusion",
#         "vdvae": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/vdvae",
#     }
# )
# cache = "/lustre/fsn1/projects/rech/nry/uul79xi/cache"

# latent_types = [
#     "clip_vision_cls",
#     # "sd_autokl",
#     # "clip_vision_latents",
#     # "vdvae_encoder_31l_latents",
# ]

# all_training_subjects = [4, 6, 8, 9, 11, 12, 14, 15]
# # train_subjects = [4, 6, 8, 9, 11, 12, 14, 15]
# # train_subjects_str = f"{'-'.join([f'{s:02d}' for s in train_subjects])}"

# args_map = list(
#     product(
#         # reference subject
#         ref_subjects,
#         list(
#             zip(
#                 eval_subjects,
#                 np.tile(dataset_ids, (len(eval_subjects), 1)).tolist(),
#                 np.tile(dataset_path, len(eval_subjects)).tolist(),
#                 np.tile(lag, len(eval_subjects)).tolist(),
#                 np.tile(window_size, len(eval_subjects)).tolist(),
#                 np.tile(False, len(eval_subjects)).tolist(),  # subject_is_macaque
#             )
#         ),
#         latent_types,
#     )
# )

# # Paths
# exps_path = Path("/lustre/fsn1/projects/rech/nry/uul79xi/inter-species")
# decoders_path = (
#     exps_path
#     / "decoders"
#     # type of decoder
#     / "multi-subject"
#     / "contrastive"
#     / "clips-train"
#     / "clips-train_mm_alpha-0.5"
# )
# decoder_is_contrastive = True
# checkpoint = 10

# alignments_path = exps_path / "alignments" / "clips-train_mm_alpha-0.5"

# output_path = (
#     exps_path
#     / "evaluations"
#     / "multi-subject"
#     / "contrastive"
#     # / "fused_alpha-0.99"
#     / "clips-train"
#     / "clips-train_mm_alpha-0.5"
#     / "clips-valid-dedup"
# )
# output_path.mkdir(parents=True, exist_ok=True)


# 2.1 Evaluate brain decoders (multi) on IBC + Leuven

human_subjects = [4, 6, 8, 9, 11, 12, 14, 15]
# human_subjects = [4, 6]
human_dataset_ids = [
    # "ibc_mk_seg-3",
    "ibc_mk_seg-4",
    # "ibc_mk_seg-5",
]
human_dataset_path = "/lustre/fsn1/projects/rech/nry/uul79xi/store/datasets/ibc"
human_lag = 2
human_window_size = 2

macaque_subjects = ["Luce", "Jack"]
macaque_dataset_ids = [
    # "leuven_mk_seg-3",
    "leuven_mk_seg-4",
    # "leuven_mk_seg-5",
]
macaque_dataset_path = "/lustre/fsn1/projects/rech/nry/uul79xi/store/datasets/leuven"
macaque_lag = 2
macaque_window_size = 2

pretrained_models = SimpleNamespace(
    **{
        "clip": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/clip",
        "sd": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/stable_diffusion",
        "vd": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/versatile_diffusion",
        "vdvae": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/vdvae",
    }
)
cache = "/lustre/fsn1/projects/rech/nry/uul79xi/cache"

latent_types = [
    "clip_vision_cls",
    # "sd_autokl",
    # "clip_vision_latents",
    # "vdvae_encoder_31l_latents",
]

checkpoint = 5
train_subjects = [4, 6, 8, 9, 11, 12, 14, 15]
train_subjects_str = f"{'-'.join([f'{s:02d}' for s in train_subjects])}"

args_map = list(
    product(
        # training subject or reference subject
        human_subjects,
        # evaluated subjects
        list(
            zip(
                human_subjects,
                np.tile(human_dataset_ids, (len(human_subjects), 1)).tolist(),
                np.tile(human_dataset_path, len(human_subjects)).tolist(),
                np.tile(human_lag, len(human_subjects)).tolist(),
                np.tile(human_window_size, len(human_subjects)).tolist(),
                np.tile(False, len(human_subjects)).tolist(),  # subject_is_macaque
            )
        ) + 
        list(
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

exps_path = Path("/lustre/fsn1/projects/rech/nry/uul79xi/inter-species")
decoder_is_contrastive = True
decoders_path = (
    exps_path
    / "decoders"
    / "multi-subject"
    / "contrastive"
    # / "clips-train-valid_mk-1-2-3-4"
    / "clips-train-valid-mk-1-2-3"
    / "clips-train_mm_alpha-0.5"
)
humans_alignments_path = (
    exps_path / "alignments" / "mk-1-2-3_mm_div-kl_alpha-0.5_rho-1.0e+00_eps-0.0001_reg-joint"
    # exps_path / "alignments" / "mk-1-2_mm_div-kl_alpha-0.5_rho-1_eps-0.0001"
    # exps_path / "alignments" / "clips-train_mm_alpha-0.5"
)
# macaques_alignments_path = exps_path / "alignments" / "mk-1-2_mm_alpha-0.5"
# macaques_alignments_path = exps_path / "alignments" / "mk-1-2_mm_div-l2_alpha-0.5_rho-100000000.0_eps-100"
macaques_alignments_path = (
    exps_path / "alignments" / "mk-1-2-3_mm_div-l2_alpha-0.5_rho-1.0e+09_eps-100_reg-joint"
    # exps_path / "alignments" / "mk-1-2_mm_div-l2_alpha-0.5_rho-1000000000.0_eps-100"
)
output_path = (
    exps_path
    / "evaluations"
    # / "single-subject"
    / "multi-subject"
    / "contrastive"
    # traning data
    / "clips-train-valid-mk-1-2-3"
    # alignment data used for fitting the decoder
    / "clips-train_mm_alpha-0.5"
    # alignment data used for testing
    / "mk-1-2-3_mm_alpha-0.5"
    # / "clips-train_mm_alpha-0.5"
    # test data
    / "mk-4"
)
# decoders_path = exps_path / "decoders_single-subject" / "clips-train-valid_mk-1-2"
# alignments_path = exps_path / "alignments" / "mk-1-2_mm"
# output_path = exps_path / "evaluations" / "clips-train-valid_mk-1-2" / "mk-4"
output_path.mkdir(parents=True, exist_ok=True)


# 2.2 Evaluate brain decoders (single) on IBC / Leuven

# human_subjects = [4, 6, 8, 9, 11, 12, 14, 15]
# # human_subjects = [4, 6]
# human_dataset_ids = [
#     # "ibc_mk_seg-3",
#     "ibc_mk_seg-4",
#     # "ibc_mk_seg-5",
# ]
# human_dataset_path = "/lustre/fsn1/projects/rech/nry/uul79xi/store/datasets/ibc"
# human_lag = 2
# human_window_size = 2

# macaque_subjects = ["Luce", "Jack"]
# macaque_dataset_ids = [
#     # "leuven_mk_seg-3",
#     "leuven_mk_seg-4",
#     # "leuven_mk_seg-5",
# ]
# macaque_dataset_path = "/lustre/fsn1/projects/rech/nry/uul79xi/store/datasets/leuven"
# macaque_lag = 2
# macaque_window_size = 2

# pretrained_models = SimpleNamespace(
#     **{
#         "clip": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/clip",
#         "sd": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/stable_diffusion",
#         "vd": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/versatile_diffusion",
#         "vdvae": "/lustre/fsn1/projects/rech/nry/uul79xi/store/models/vdvae",
#     }
# )
# cache = "/lustre/fsn1/projects/rech/nry/uul79xi/cache"

# latent_types = [
#     "clip_vision_cls",
#     # "sd_autokl",
#     # "clip_vision_latents",
#     # "vdvae_encoder_31l_latents",
# ]

# checkpoint = 5

# args_map = list(
#     product(
#         # training subject or reference subject
#         human_subjects,
#         # evaluated subjects
#         list(
#             zip(
#                 human_subjects,
#                 np.tile(human_dataset_ids, (len(human_subjects), 1)).tolist(),
#                 np.tile(human_dataset_path, len(human_subjects)).tolist(),
#                 np.tile(human_lag, len(human_subjects)).tolist(),
#                 np.tile(human_window_size, len(human_subjects)).tolist(),
#                 np.tile(False, len(human_subjects)).tolist(),  # subject_is_macaque
#             )
#         ),
#         list(
#             zip(
#                 macaque_subjects,
#                 np.tile(macaque_dataset_ids, (len(macaque_subjects), 1)).tolist(),
#                 np.tile(macaque_dataset_path, len(macaque_subjects)).tolist(),
#                 np.tile(macaque_lag, len(macaque_subjects)).tolist(),
#                 np.tile(macaque_window_size, len(macaque_subjects)).tolist(),
#                 np.tile(True, len(macaque_subjects)).tolist(),  # subject_is_macaque
#             )
#         ),
#         latent_types,
#     )
# )

# exps_path = Path("/lustre/fsn1/projects/rech/nry/uul79xi/inter-species")
# decoder_is_contrastive = True
# decoders_path = (
#     exps_path
#     / "decoders"
#     / "single-subject"
#     / "contrastive"
#     / "clips-train-valid_mk-1-2-3"
# )
# humans_alignments_path = (
#     exps_path / "alignments" / "mk-1-2-3_mm_div-kl_alpha-0.5_rho-1.0e+00_eps-0.0001_reg-joint"
#     # exps_path / "alignments" / "mk-1-2_mm_div-kl_alpha-0.5_rho-1_eps-0.0001"
#     # exps_path / "alignments" / "clips-train_mm_alpha-0.5"
# )
# # macaques_alignments_path = exps_path / "alignments" / "mk-1-2_mm_alpha-0.5"
# # macaques_alignments_path = exps_path / "alignments" / "mk-1-2_mm_div-l2_alpha-0.5_rho-100000000.0_eps-100"
# macaques_alignments_path = (
#     exps_path / "alignments" / "mk-1-2-3_mm_div-l2_alpha-0.5_rho-1.0e+09_eps-100_reg-joint"
#     # exps_path / "alignments" / "mk-1-2_mm_div-l2_alpha-0.5_rho-1000000000.0_eps-100"
# )
# output_path = (
#     exps_path
#     / "evaluations"
#     / "single-subject"
#     # / "multi-subject"
#     / "contrastive"
#     # traning data
#     / "clips-train-valid-mk-1-2-3"
#     # alignment data used for fitting the decoder
#     # / "clips-train_mm_alpha-0.5"
#     # alignment data used for testing
#     / "mk-1-2-3_mm_alpha-0.5"
#     # / "clips-train_mm_alpha-0.5"
#     # test data
#     / "mk-4"
# )
# # decoders_path = exps_path / "decoders_single-subject" / "clips-train-valid_mk-1-2"
# # alignments_path = exps_path / "alignments" / "mk-1-2_mm"
# # output_path = exps_path / "evaluations" / "clips-train-valid_mk-1-2" / "mk-4"
# output_path.mkdir(parents=True, exist_ok=True)


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
        if leftout_is_macaque:
            left_mapping_path = macaques_alignments_path / f"{exp_name}_left.pkl"
            right_mapping_path = macaques_alignments_path / f"{exp_name}_right.pkl"
            selected_indices_left_path = (
                macaques_alignments_path
                / f"{exp_name}_selected_indices_left_source.npy"
            )
            selected_indices_right_path = (
                macaques_alignments_path
                / f"{exp_name}_selected_indices_right_source.npy"
            )
        else:
            left_mapping_path = humans_alignments_path / f"{exp_name}_left.pkl"
            right_mapping_path = humans_alignments_path / f"{exp_name}_right.pkl"

        if not left_mapping_path.exists():
            invert_mapping = True
            if leftout_is_macaque:
                left_mapping_path = (
                    macaques_alignments_path / f"{exp_name_invert}_left.pkl"
                )
                right_mapping_path = (
                    macaques_alignments_path / f"{exp_name_invert}_right.pkl"
                )
                selected_indices_left_path = (
                    macaques_alignments_path
                    / f"{exp_name_invert}_selected_indices_left_source.npy"
                )
                selected_indices_right_path = (
                    macaques_alignments_path
                    / f"{exp_name_invert}_selected_indices_right_source.npy"
                )
            else:
                left_mapping_path = (
                    humans_alignments_path / f"{exp_name_invert}_left.pkl"
                )
                right_mapping_path = (
                    humans_alignments_path / f"{exp_name_invert}_right.pkl"
                )

            if not left_mapping_path.exists():
                print(left_mapping_path)
                print(right_mapping_path)
                raise Exception(
                    "There is no mapping between subjects "
                    f"{leftout_subject} and {reference_subject}"
                )

    # train_subjects = all_training_subjects
    # train_subjects.remove(leftout_subject)
    # train_subjects_str = f"{'-'.join([f'{s:02d}' for s in train_subjects])}"
    # train_subjects_str = f"{reference_subject:02d}"

    if leftout_is_macaque:
        output_name = (
            # f"ref-{reference_subject:02d}_train-{train_subjects_str}_{latent_type}"
            f"sub-{reference_subject:02d}_{latent_type}"
            f"_test-{leftout_subject}"
        )
    else:
        output_name = (
            # f"ref-{reference_subject:02d}_train-{train_subjects_str}_{latent_type}"
            f"sub-{reference_subject:02d}_{latent_type}"
            f"_test-sub-{leftout_subject:02d}"
        )

    if decoder_is_contrastive:
        decoder_path = (
            decoders_path
            # multi-subject decoders
            / f"ref-{reference_subject:02d}_train-{train_subjects_str}_{latent_type}"
            # single-subject decoders
            # / f"sub-{reference_subject:02d}_{latent_type}"
            / f"checkpoint_{checkpoint:03d}.pt"
        )
    else:
        decoder_path = decoders_path / f"sub-{reference_subject:02d}_{latent_type}.pkl"

    evaluate_brain_decoder(
        decoder_path=decoder_path,
        decoder_is_contrastive=decoder_is_contrastive,
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
        should_generate_captions=True,
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
        slurm_time="00:20:00",
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
