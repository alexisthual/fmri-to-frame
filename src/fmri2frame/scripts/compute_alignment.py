"""Compute FUGW alignment between two subjects."""

from pathlib import Path

import numpy as np
import yaml
from fugw.mappings import FUGW
from fugw.scripts import coarse_to_fine
from fugw.utils import save_mapping
from nilearn import surface
from scipy.spatial.distance import cdist

from fmri2frame.scripts.setup_xp import setup_xp

fsaverage_geometry_path = Path(
    "/gpfsstore/rech/nry/uul79xi/outputs/_063_compute_fs5_geometry/"
)
macaque_geometry_path = Path(
    "/gpfsstore/rech/nry/uul79xi/outputs/_197_leuven_geometries"
)
macaque_surfaces_path = Path("/gpfsstore/rech/nry/uul79xi/data/leuven/Surfaces")


def get_alignment_path(source_subject, target_subject, alignments_path=None):
    exp_name = f"sub-{source_subject:02d}_sub-{target_subject:02d}"
    exp_name_invert = f"sub-{target_subject:02d}_sub-{source_subject:02d}"

    invert_mapping = None
    left_mapping_path = None
    right_mapping_path = None

    if source_subject != target_subject:
        invert_mapping = False
        left_mapping_path = alignments_path / f"{exp_name}_left.pkl"
        right_mapping_path = alignments_path / f"{exp_name}_right.pkl"

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
                    f"{source_subject} and {target_subject}"
                )

    return invert_mapping, left_mapping_path, right_mapping_path


def get_n_vertices_left(subject):
    meshes_path = macaque_surfaces_path / str(subject)

    coordinates_left, _ = surface.load_surf_mesh(meshes_path / "surf" / "lh.pial")
    n_vertices_left = coordinates_left.shape[0]

    return n_vertices_left


def subsample_macaque_mesh(subject, hemi):
    """Sub-sample mesh."""
    meshes_path = macaque_surfaces_path / str(subject)

    with open(
        macaque_geometry_path / f"{subject}_pial_{hemi}_geometry_embeddings.npy",
        "rb",
    ) as f:
        embeddings = np.load(f)

    coordinates, triangles = surface.load_surf_mesh(
        meshes_path / "surf" / f"{'lh' if hemi == 'left' else 'rh'}.pial"
    )

    np.random.seed(0)
    selected_indices = coarse_to_fine.sample_mesh_uniformly(
        coordinates,
        triangles,
        embeddings=embeddings,
        n_samples=10242,
    )

    return selected_indices


def get_features_and_geometry(
    subject,
    hemi,
    features_all,
    subject_is_macaque,
    postfix=None,
    output_name=None,
    output_path=None,
):
    """Load features and geometry for given subject and hemi."""
    if subject_is_macaque:
        # Sub-sample mesh
        selected_indices = subsample_macaque_mesh(subject, hemi)
        with open(
            output_path / f"{output_name}_selected_indices_{hemi}{postfix}.npy", "wb"
        ) as f:
            np.save(f, selected_indices)

        # Sub-select features according to mesh sub-sampling
        if hemi == "left":
            features = features_all[:, selected_indices]
        else:
            n_vertices_left = get_n_vertices_left(subject)
            features = features_all[:, n_vertices_left + selected_indices]

        # Take opposite of brain features when subject is macaque
        # because they were scanned using MION instead of BOLD
        features = -1 * features

        # Compute geometry according to mesh sub-sampling
        with open(
            macaque_geometry_path / f"{subject}_pial_{hemi}_geometry_embeddings.npy",
            "rb",
        ) as f:
            embeddings = np.load(f)

        geometry = cdist(
            embeddings[selected_indices, :],
            embeddings[selected_indices, :],
        )
    else:
        if hemi == "left":
            features = features_all[:, :10242]
        else:
            features = features_all[:, 10242:]

        with open(
            fsaverage_geometry_path / f"fsaverage5_pial_{hemi}_geometry.npy",
            "rb",
        ) as f:
            geometry = np.load(f)

    return features, geometry


def compute_alignment(
    source_datasets=None,
    source_dataset_path=None,
    source_subject=None,
    source_lag=0,
    source_window_size=1,
    source_is_macaque=False,
    target_datasets=None,
    target_dataset_path=None,
    target_subject=None,
    target_lag=0,
    target_window_size=1,
    target_is_macaque=False,
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
    config = {
        "source_datasets": source_datasets,
        "source_dataset_path": source_dataset_path,
        "source_subject": source_subject,
        "source_lag": source_lag,
        "source_window_size": source_window_size,
        "source_is_macaque": source_is_macaque,
        "target_datasets": target_datasets,
        "target_dataset_path": target_dataset_path,
        "target_subject": target_subject,
        "target_lag": target_lag,
        "target_window_size": target_window_size,
        "target_is_macaque": target_is_macaque,
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
    source_features_all = []
    target_features_all = []
    for source_dataset, target_dataset in zip(source_datasets, target_datasets):
        source_features = setup_xp(
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

        target_features = setup_xp(
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

        # Ensure that source and target
        # have the same number of brain volumes
        n_volumes = min(source_features.shape[0], target_features.shape[0])

        source_features_all.append(source_features[:n_volumes])
        target_features_all.append(target_features[:n_volumes])

    source_features_all = np.concatenate(source_features_all)
    target_features_all = np.concatenate(target_features_all)

    # 1. Align left hemispheres
    source_features, source_geometry = get_features_and_geometry(
        source_subject,
        "left",
        source_features_all,
        source_is_macaque,
        postfix="_source",
        output_name=output_name,
        output_path=output_path,
    )

    target_features, target_geometry = get_features_and_geometry(
        target_subject,
        "left",
        target_features_all,
        target_is_macaque,
        postfix="_target",
        output_name=output_name,
        output_path=output_path,
    )

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

    # 2. Align right hemispheres
    source_features, source_geometry = get_features_and_geometry(
        source_subject,
        "right",
        source_features_all,
        source_is_macaque,
        postfix="_source",
        output_name=output_name,
        output_path=output_path,
    )

    target_features, target_geometry = get_features_and_geometry(
        target_subject,
        "right",
        target_features_all,
        target_is_macaque,
        postfix="_target",
        output_name=output_name,
        output_path=output_path,
    )

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
