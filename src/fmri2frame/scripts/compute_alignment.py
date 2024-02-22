"""Compute FUGW alignment between two subjects."""

import numpy as np
import yaml
from fmri2frame.scripts.setup_xp import setup_xp
from fugw.mappings import FUGW
from fugw.utils import save_mapping


def compute_alignment(
    source_datasets=None,
    source_dataset_path=None,
    source_subject=None,
    source_lag=0,
    source_window_size=1,
    target_datasets=None,
    target_dataset_path=None,
    target_subject=None,
    target_lag=0,
    target_window_size=1,
    alpha=0.5,
    eps=1,
    rho=1e-4,
    reg_mode="joint",
    divergence="kl",
    solver="mm",
    output_name=None,
    output_path=None,
):
    """Compute FUGW alignment between two subjects."""
    output_path.mkdir(parents=True, exist_ok=True)

    config = {
        "source_datasets": source_datasets,
        "source_dataset_path": source_dataset_path,
        "source_subject": source_subject,
        "source_lag": source_lag,
        "source_window_size": source_window_size,
        "target_datasets": target_datasets,
        "target_dataset_path": target_dataset_path,
        "target_subject": target_subject,
        "target_lag": target_lag,
        "target_window_size": target_window_size,
        "alpha": alpha,
        "rho": rho,
        "eps": eps,
        "reg_mode": reg_mode,
        "divergence": divergence,
        "solver": solver,
    }

    # Export parameters in config file
    with open(output_path / f"{output_name}_config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)

    # Load source and target features
    source_features_all = np.concatenate(
        [
            setup_xp(
                dataset_id=source_dataset,
                dataset_path=source_dataset_path,
                subject=source_subject,
                n_train_examples=None,
                n_valid_examples=None,
                n_test_examples=None,
                pretrained_models_path={},
                latent_types=[],
                generation_seed=0,
                batch_size=32,
                window_size=source_window_size,
                lag=source_lag,
                agg="mean",
                support="fsaverage5",
                shuffle_labels=False,
            ).train_data.betas
            for source_dataset in source_datasets
        ]
    )

    target_features_all = np.concatenate(
        [
            setup_xp(
                dataset_id=target_dataset,
                dataset_path=target_dataset_path,
                subject=target_subject,
                n_train_examples=None,
                n_valid_examples=None,
                n_test_examples=None,
                pretrained_models_path={},
                latent_types=[],
                generation_seed=0,
                batch_size=32,
                window_size=target_window_size,
                lag=target_lag,
                agg="mean",
                support="fsaverage5",
                shuffle_labels=False,
            ).train_data.betas
            for target_dataset in target_datasets
        ]
    )

    # Align left hemispheres
    source_features = source_features_all[:, :10242]
    target_features = target_features_all[:, :10242]

    with open(
        "/gpfsstore/rech/nry/uul79xi/outputs/_063_compute_fs5_geometry/fsaverage5_pial_left_geometry.npy",
        "rb",
    ) as f:
        source_geometry = np.load(f)
    target_geometry = np.copy(source_geometry)

    source_features_normalized = source_features / np.linalg.norm(
        source_features, axis=0
    )
    target_features_normalized = target_features / np.linalg.norm(
        target_features, axis=0
    )
    source_geometry_normalized = source_geometry / np.max(source_geometry)
    target_geometry_normalized = target_geometry / np.max(target_geometry)

    mapping = FUGW(
        alpha=alpha,
        rho=rho,
        eps=eps,
        reg_mode=reg_mode,
        divergence=divergence,
    )

    _ = mapping.fit(
        source_features_normalized,
        target_features_normalized,
        source_geometry=source_geometry_normalized,
        target_geometry=target_geometry_normalized,
        solver=solver,
        solver_params={
            "nits_bcd": 10,
        },
        verbose=True,
        device="cuda",
    )

    save_mapping(
        mapping,
        str(output_path / f"{output_name}_left.pkl"),
    )

    # Align right hemispheres
    source_features = source_features_all[:, 10242:]
    target_features = target_features_all[:, 10242:]

    with open(
        "/gpfsstore/rech/nry/uul79xi/outputs/_063_compute_fs5_geometry/fsaverage5_pial_right_geometry.npy",
        "rb",
    ) as f:
        source_geometry = np.load(f)
    target_geometry = np.copy(source_geometry)

    source_features_normalized = source_features / np.linalg.norm(
        source_features, axis=0
    )
    target_features_normalized = target_features / np.linalg.norm(
        target_features, axis=0
    )
    source_geometry_normalized = source_geometry / np.max(source_geometry)
    target_geometry_normalized = target_geometry / np.max(target_geometry)

    mapping = FUGW(
        alpha=alpha,
        rho=rho,
        eps=eps,
        reg_mode=reg_mode,
        divergence=divergence,
    )

    _ = mapping.fit(
        source_features_normalized,
        target_features_normalized,
        source_geometry=source_geometry_normalized,
        target_geometry=target_geometry_normalized,
        solver=solver,
        solver_params={
            "nits_bcd": 10,
        },
        verbose=True,
        device="cuda",
    )

    save_mapping(
        mapping,
        str(output_path / f"{output_name}_right.pkl"),
    )
