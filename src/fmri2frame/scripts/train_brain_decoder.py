"""Util functions to train brain decoders."""

import pickle

import numpy as np
from fmri2frame.scripts.setup_xp import setup_xp
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def train_brain_decoder(
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
            StandardScaler().fit_transform(
                np.concatenate(brain_features)
            )
        ),
        StandardScaler().fit_transform(
            np.concatenate(latent_features)
        ),
    )

    # Save model
    with open(output_path, "wb") as f:
        pickle.dump(ridge, f)
