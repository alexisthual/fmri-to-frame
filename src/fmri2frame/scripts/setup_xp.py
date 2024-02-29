import logging

from fmri2frame.scripts.compute_latents import (
    compute_latents,
    compute_augmented_latents,
)
from fmri2frame.scripts.data import (
    FmriDatasetBase,
    LatentsDataModuleBase,
    load_fmridataset,
)


logger = logging.getLogger(__name__)


def setup_xp(
    dataset_id,
    dataset_path,
    subject,
    n_train_examples,
    n_valid_examples,
    n_test_examples,
    pretrained_models_path,
    latent_types,
    generation_seed=0,
    n_augmentations=None,
    batch_size=None,
    shuffle_labels=None,
    cache=None,
    num_workers=0,
    # Arguments specific to brain volumes aggreagation
    window_size=1,
    lag=0,
    agg="mean",
    support=None,
) -> LatentsDataModuleBase:
    """Load models and extract latent representations of stimuli.

    Loading of models and latent extraction is all done in the same function
    so that it can be cached with joblib.Memory.

    Returns
    -------
    latents_datamodule: LatentsDataModuleBase
    """
    logger.info("Preparing data...")

    # Build data module containing:
    # - brain features
    # - latent representations for stimuli features

    # 1. Load beta coefficients
    fmri_dataset = load_fmridataset(
        dataset_id,
        dataset_path,
        subject,
    )

    betas = fmri_dataset.betas

    # 2. Compute or load latent representations of stimuli
    all_latents = dict()
    for latent_type in latent_types:
        if latent_type in ["clip_vision_latents", "clip_text_latents"]:
            model_path = pretrained_models_path.vd
        elif latent_type in ["vdvae_encoder_31l_latents"]:
            model_path = pretrained_models_path.vdvae
        elif latent_type in ["clip_vision_cls"]:
            model_path = pretrained_models_path.clip
        elif latent_type in ["sd_autokl"]:
            model_path = pretrained_models_path.sd
        else:
            raise NotImplementedError()

        # This function uses joblib's cache internally.
        # Changing its arguments will invalidate already cached results,
        # so one should refrain from adding / removing args of this function.
        if n_augmentations is None:
            latents, metadata = compute_latents(
                dataset_id,
                dataset_path,
                subject,
                latent_type,
                model_path=model_path,
                seed=generation_seed,
                batch_size=batch_size,
                cache=cache,
            )
        else:
            latents, metadata = compute_augmented_latents(
                dataset_id,
                dataset_path,
                subject,
                latent_type,
                model_path=model_path,
                seed=generation_seed,
                batch_size=batch_size,
                n_augmentations=n_augmentations,
                cache=cache,
            )

        all_latents[latent_type] = latents

    # 3. Split data in train / valid / test
    n_samples = betas.shape[0]

    # Split that respects the original order of examples
    if n_train_examples is None:
        n_train_examples = n_samples
    train_indices = range(0, n_train_examples)

    if n_valid_examples is None:
        n_valid_examples = n_samples - n_train_examples
    valid_indices = range(
        n_train_examples,
        n_train_examples + n_valid_examples,
    )

    if n_test_examples is None:
        n_test_examples = n_samples - n_train_examples - n_valid_examples
    test_indices = range(
        n_train_examples + n_valid_examples,
        n_train_examples + n_valid_examples + n_test_examples,
    )

    datasets = []
    for indices in [train_indices, valid_indices, test_indices]:
        datasets.append(
            FmriDatasetBase(
                betas[indices],
                {
                    latent_type: all_latents[latent_type][indices]
                    for latent_type in latent_types
                },
            )
        )

    # 4. Build latent data module
    latents_datamodule = LatentsDataModuleBase(
        *datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        # Shuffle data in training batches, to ensure IID distribution
        shuffle_train=True,
    )

    # 5. Aggregate brain / stimuli features
    latents_datamodule.process_features(
        window_size=window_size,
        lag=lag,
        agg=agg,
        support=support,
        shuffle_labels=shuffle_labels,
    )

    return latents_datamodule
