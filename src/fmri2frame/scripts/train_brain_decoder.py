"""Util functions to train brain decoders."""

import pickle

import numpy as np
from fugw.utils import load_mapping
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from fmri2frame.scripts.setup_xp import setup_xp


def train_single_subject_brain_decoder(
    dataset_ids=None,
    dataset_path=None,
    subject=None,
    lag=0,
    window_size=1,
    latent_type=None,
    pretrained_models_path=None,
    cache=None,
    output_path=None,
):
    """Train brain decoder in one participant."""
    # Load data
    brain_features = []
    latent_features = []
    for dataset_id in dataset_ids:
        datamodule = setup_xp(
            dataset_id=dataset_id,
            dataset_path=dataset_path,
            subject=subject,
            n_train_examples=None,
            n_valid_examples=None,
            n_test_examples=None,
            pretrained_models_path=pretrained_models_path,
            latent_types=[latent_type],
            generation_seed=0,
            batch_size=32,
            window_size=window_size,
            lag=lag,
            agg="mean",
            support="fsaverage5",
            shuffle_labels=False,
            cache=cache,
        )
        brain_features.append(datamodule.train_data.betas)
        latent_features.append(datamodule.train_data.labels[latent_type])

    # Fit model
    ridge = Ridge(alpha=50000, fit_intercept=True)
    ridge.fit(
        SimpleImputer().fit_transform(
            StandardScaler().fit_transform(np.concatenate(brain_features))
        ),
        StandardScaler().fit_transform(np.concatenate(latent_features)),
    )

    # Save model
    with open(output_path, "wb") as f:
        pickle.dump(ridge, f)


def train_multi_subject_brain_decoder(
    dataset_ids=None,
    dataset_path=None,
    reference_subject=None,
    training_subjects=None,
    lag=0,
    window_size=1,
    latent_type=None,
    pretrained_models_path=None,
    cache=None,
    alignments_path=None,
    output_path=None,
):
    """Train brain decoder in one participant."""
    # Load data
    brain_features_all = []
    latent_features_all = []

    for subject in training_subjects:
        invert_mapping = None
        left_mapping_path = None
        right_mapping_path = None

        exp_name = f"sub-{subject:02d}_sub-{reference_subject:02d}"
        exp_name_invert = f"sub-{reference_subject:02d}_sub-{subject:02d}"

        if subject != reference_subject:
            invert_mapping = False
            left_mapping_path = alignments_path / f"{exp_name}_left.pkl"
            right_mapping_path = alignments_path / f"{exp_name}_right.pkl"

            if not left_mapping_path.exists():
                invert_mapping = True
                left_mapping_path = alignments_path / f"{exp_name_invert}_left.pkl"
                right_mapping_path = alignments_path / f"{exp_name_invert}_right.pkl"

                if not left_mapping_path.exists():
                    print(left_mapping_path)
                    print(right_mapping_path)
                    raise Exception(
                        "There is no mapping between subjects "
                        f"{subject} and {reference_subject}"
                    )

        if left_mapping_path is not None and right_mapping_path is not None:
            left_mapping = load_mapping(left_mapping_path)
            right_mapping = load_mapping(right_mapping_path)

        for dataset_id in dataset_ids:
            datamodule = setup_xp(
                dataset_id=dataset_id,
                dataset_path=dataset_path,
                subject=subject,
                n_train_examples=None,
                n_valid_examples=None,
                n_test_examples=None,
                pretrained_models_path=pretrained_models_path,
                latent_types=[latent_type],
                generation_seed=0,
                batch_size=32,
                window_size=window_size,
                lag=lag,
                agg="mean",
                support="fsaverage5",
                shuffle_labels=False,
                cache=cache,
            )

            brain_features = datamodule.train_data.betas

            # Align brain features
            if left_mapping_path is not None and right_mapping_path is not None:
                if not invert_mapping:
                    brain_features = np.concatenate(
                        [
                            left_mapping.transform(brain_features[:, :10242]),
                            right_mapping.transform(brain_features[:, 10242:]),
                        ],
                        axis=1,
                    )
                else:
                    brain_features = np.concatenate(
                        [
                            left_mapping.inverse_transform(brain_features[:, :10242]),
                            right_mapping.inverse_transform(brain_features[:, 10242:]),
                        ],
                        axis=1,
                    )

            brain_features_all.append(brain_features)
            latent_features_all.append(datamodule.train_data.labels[latent_type])

    # Fit model
    ridge = Ridge(alpha=50000, fit_intercept=True)
    ridge.fit(
        SimpleImputer().fit_transform(
            StandardScaler().fit_transform(np.concatenate(brain_features_all))
        ),
        StandardScaler().fit_transform(np.concatenate(latent_features_all)),
    )

    # Save model
    with open(output_path, "wb") as f:
        pickle.dump(ridge, f)
