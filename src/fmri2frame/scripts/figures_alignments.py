import colorsys
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
# import seaborn as sns
from fugw.utils import load_mapping
from nilearn import datasets, plotting, surface
from sklearn.neighbors import NearestNeighbors


def mix_colormaps(fg, bg):
    """Mixes foreground and background arrays of RGBA colors.

    Parameters
    ----------
    fg : numpy.ndarray
        Array of shape (n, 4), foreground RGBA colors
        represented as floats in [0, 1]
    bg : numpy.ndarray
        Array of shape (n, 4), background RGBA colors
        represented as floats in [0, 1]

    Returns
    -------
    mix : numpy.ndarray
        Array of shape (n, 4), mixed colors
        represented as floats in [0, 1]
    """
    # Adapted from https://stackoverflow.com/questions/726549/algorithm-for-additive-color-mixing-for-rgb-values/727339#727339 # noqa: E501
    if fg.shape != bg.shape:
        raise ValueError(
            "Trying to mix colormaps with different shapes: " f"{fg.shape}, {bg.shape}"
        )

    mix = np.empty_like(fg)

    mix[:, 3] = 1 - (1 - fg[:, 3]) * (1 - bg[:, 3])

    for color_index in range(0, 3):
        mix[:, color_index] = (
            fg[:, color_index] * fg[:, 3]
            + bg[:, color_index] * bg[:, 3] * (1 - fg[:, 3])
        ) / mix[:, 3]

    return mix


def _get_camera_view_from_elevation_and_azimut(view, is_macaque=False):
    """Compute plotly camera parameters from elevation and azimut."""
    elev, azim = view
    # The radius is useful only when using a "perspective" projection,
    # otherwise, if projection is "orthographic",
    # one should tweak the "aspectratio" to emulate zoom
    if is_macaque:
        r = 1.5
    else:
        r = 1.8
    # The camera position and orientation is set by three 3d vectors,
    # whose coordinates are independent of the plotted data.
    return {
        # Where the camera should look at
        # (it should always be looking at the center of the scene)
        "center": {"x": 0, "y": 0, "z": 0},
        # Where the camera should be located
        "eye": {
            "x": (
                r
                * math.cos(azim / 360 * 2 * math.pi)
                * math.cos(elev / 360 * 2 * math.pi)
            ),
            "y": (
                r
                * math.sin(azim / 360 * 2 * math.pi)
                * math.cos(elev / 360 * 2 * math.pi)
            ),
            "z": r * math.sin(elev / 360 * 2 * math.pi),
        },
        # How the camera should be rotated.
        # It is determined by a 3d vector indicating which direction
        # should look up in the generated plot
        "up": {
            "x": math.sin(elev / 360 * 2 * math.pi)
            * math.cos(azim / 360 * 2 * math.pi + math.pi),
            "y": math.sin(elev / 360 * 2 * math.pi)
            * math.sin(azim / 360 * 2 * math.pi + math.pi),
            "z": math.cos(elev / 360 * 2 * math.pi),
        },
        "projection": {"type": "perspective"},
        # "projection": {"type": "orthographic"},
    }


def plotly_mesh_3d(mesh, vertexcolor, bg_map=None):
    coords, triangles = surface.load_surf_mesh(mesh)
    x, y, z = coords.T
    i, j, k = triangles.T
    if bg_map is not None:
        bg_colors = plt.get_cmap("Greys")(bg_map)
        # print(bg_colors.shape)
        # print(vertexcolor.shape)
        surf_colors = mix_colormaps(
            # vertexcolor,
            np.hstack(
                (
                    vertexcolor,
                    0.7 * np.ones([vertexcolor.shape[0], 1], vertexcolor.dtype),
                )
            ),
            np.hstack(
                (
                    bg_colors[:, :3],
                    0.5 * np.ones([bg_colors.shape[0], 1], bg_colors.dtype),
                )
            ),
            # bg_colors
        )
        # print(bg_colors[0:10])
        # surf_colors = bg_colors
    else:
        surf_colors = vertexcolor
    mesh_3d = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, vertexcolor=surf_colors)

    return mesh_3d


AXIS_CONFIG = {
    "showgrid": False,
    "showline": False,
    "ticks": "",
    "title": "",
    "showticklabels": False,
    "zeroline": False,
    "showspikes": False,
    "spikesides": False,
    "showbackground": False,
}

CAMERAS = {
    "left": {
        "eye": {"x": -1.5, "y": 0, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
    },
    "right": {
        "eye": {"x": 1.5, "y": 0, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
    },
    "top": {
        "eye": {"x": 0, "y": 0, "z": 1.5},
        "up": {"x": 0, "y": 1, "z": 0},
        "center": {"x": 0, "y": 0, "z": 0},
    },
    "bottom": {
        "eye": {"x": 0, "y": 0, "z": -1.5},
        "up": {"x": 0, "y": 1, "z": 0},
        "center": {"x": 0, "y": 0, "z": 0},
    },
    "front": {
        "eye": {"x": 0, "y": 1.5, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
    },
    "back": {
        "eye": {"x": 0, "y": -1.5, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
    },
}

LAYOUT = {
    # "paper_bgcolor": "rgba(100,100,100,10)",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "hovermode": False,
    # "margin": {"l": 20, "r": 20, "b": 20, "t": 20, "pad": 0},
    "margin": {"l": 0, "r": 0, "b": 0, "t": 0, "pad": 0},
    "width": 800,
    "height": 800,
    "title": {
        "font": {
            "size": 25,
        },
        "yref": "paper",
        "y": 0.95,
    },
}


def plot_surf_color_map(
    mesh, surf_color_map, hemi="left", view="lateral", bg_map=None, is_macaque=False
):
    fig = go.Figure(data=[plotly_mesh_3d(mesh, surf_color_map, bg_map=bg_map)])
    fig = fig.update_layout(
        scene_camera=_get_camera_view_from_elevation_and_azimut(
            view, is_macaque=is_macaque
        ),
        scene={f"{dim}axis": AXIS_CONFIG for dim in ("x", "y", "z")},
        **LAYOUT,
    )
    return fig


def get_bg_map(curv_surf, bg_max=1, bg_min=0.25):
    bg_map = np.sign(curv_surf)
    bg_map = (bg_map + 1) / 2
    bg_map = bg_map * (bg_max - bg_min) + bg_min
    return bg_map


def interpolate_map(surf_map, indices):
    return surf_map[indices[:, 0]]


def plot_alignment_maps(alignments_path, output_path, output_prefix):
    luce_meshes = Path("/lustre/fsstor/projects/rech/nry/uul79xi/datasets/leuven/Surfaces/Luce/surf")
    luce_flat_left_path = luce_meshes / "lh.flat.surf.gii"
    # luce_pial_left_path = luce_meshes / "lh.pial"
    luce_curv_left_path = luce_meshes / "lh.curv"
    luce_flat_right_path = luce_meshes / "rh.flat.surf.gii"
    # luce_pial_right_path = luce_meshes / "rh.pial"
    luce_curv_right_path = luce_meshes / "rh.curv"

    glasser_path = Path("/lustre/fsstor/projects/rech/nry/uul79xi/datasets/glasser_atlas")
    leuven_embeddings_path = Path(
        "/lustre/fsstor/projects/rech/nry/uul79xi/outputs/_197_leuven_geometries"
    )
    fsaverage_embeddings_path = Path(
        "/lustre/fsstor/projects/rech/nry/uul79xi/outputs/_203_fsaverage_geometries"
    )

    luce_flat_left = surface.load_surf_mesh(luce_flat_left_path)
    # luce_pial_left = surface.load_surf_mesh(luce_pial_left_path)
    luce_curv_left = surface.load_surf_data(luce_curv_left_path)
    luce_flat_right = surface.load_surf_mesh(luce_flat_right_path)
    # luce_pial_right = surface.load_surf_mesh(luce_pial_right_path)
    luce_curv_right = surface.load_surf_data(luce_curv_right_path)

    luce_pial_left_embeddings = np.load(
        leuven_embeddings_path / "Luce_pial_left_geometry_embeddings.npy"
    )
    luce_pial_right_embeddings = np.load(
        leuven_embeddings_path / "Luce_pial_right_geometry_embeddings.npy"
    )

    # fs5 = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    fs7 = datasets.fetch_surf_fsaverage(mesh="fsaverage7")
    fs7_pial_left_embeddings = np.load(
        fsaverage_embeddings_path / "fsaverage7_pial_left_geometry_embeddings.npy"
    )
    fs7_pial_right_embeddings = np.load(
        fsaverage_embeddings_path / "fsaverage7_pial_right_geometry_embeddings.npy"
    )
    bg_map_left_luce = get_bg_map(luce_curv_left)
    bg_map_right_luce = get_bg_map(luce_curv_right)
    bg_map_left_human = get_bg_map(surface.load_surf_data(fs7["curv_left"]))
    bg_map_right_human = get_bg_map(surface.load_surf_data(fs7["curv_right"]))

    glasser_left = surface.load_surf_data(glasser_path / "lh.HCP-MMP1.annot")
    glasser_right = surface.load_surf_data(glasser_path / "rh.HCP-MMP1.annot")

    new_colors = np.stack(
        [
            colorsys.hls_to_rgb(i / glasser_left.max(), 0.55, 0.8)
            for i in range(glasser_left.max() + 1)
        ]
    )
    np.random.seed(3)
    # np.random.seed(9)
    # np.random.seed(17)
    colors_permutation = np.random.permutation(new_colors.shape[0])
    new_colors = new_colors[colors_permutation]

    colors_left = new_colors[glasser_left][:10242]
    colors_right = new_colors[glasser_right][:10242]

    knn_human = NearestNeighbors(n_neighbors=2, algorithm="ball_tree")
    knn_human.fit(fs7_pial_left_embeddings[:10242])
    distances_left_human, indices_left_human = knn_human.kneighbors(
        fs7_pial_left_embeddings
    )
    knn_human.fit(fs7_pial_right_embeddings[:10242])
    distances_right_human, indices_right_human = knn_human.kneighbors(
        fs7_pial_right_embeddings
    )

    alignment_left_path = alignments_path / "Luce_sub-04_left.pkl"
    alignment_right_path = alignments_path / "Luce_sub-04_right.pkl"
    alignment_left = load_mapping(alignment_left_path)
    alignment_right = load_mapping(alignment_right_path)
    selected_indices_left = np.load(
        alignments_path / "Luce_sub-04_selected_indices_left_source.npy"
    )
    selected_indices_right = np.load(
        alignments_path / "Luce_sub-04_selected_indices_right_source.npy"
    )

    knn_luce = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", n_jobs=4)

    knn_luce.fit(luce_pial_left_embeddings[selected_indices_left])
    distances_left_luce, indices_left_luce = knn_luce.kneighbors(
        luce_pial_left_embeddings
    )

    knn_luce.fit(luce_pial_right_embeddings[selected_indices_right])
    distances_right_luce, indices_right_luce = knn_luce.kneighbors(
        luce_pial_right_embeddings
    )

    # Atlas on source
    transported_colors_left = alignment_left.inverse_transform(colors_left.T).T
    transported_colors_right = alignment_right.inverse_transform(colors_right.T).T

    luce_colors_left = np.zeros((luce_flat_left[0].shape[0], 3))
    luce_colors_left[selected_indices_left] = transported_colors_left

    luce_colors_right = np.zeros((luce_flat_right[0].shape[0], 3))
    luce_colors_right[selected_indices_right] = transported_colors_right

    fig = plot_surf_color_map(
        luce_flat_left,
        interpolate_map(transported_colors_left, indices_left_luce),
        view=(90, 0),
        bg_map=bg_map_left_luce,
        is_macaque=True,
    )
    fig.write_image(output_path / f"{output_prefix}_source_atlas_left.png")

    fig = plot_surf_color_map(
        luce_flat_right,
        interpolate_map(transported_colors_right, indices_right_luce),
        view=(90, 0),
        bg_map=bg_map_right_luce,
        is_macaque=True,
    )
    fig.write_image(output_path / f"{output_prefix}_source_atlas_right.png")

    # Source mass
    source_mass_left = (
        alignment_left.pi.sum(axis=1).numpy() * alignment_left.pi.shape[0]
    )
    source_mass_right = (
        alignment_right.pi.sum(axis=1).numpy() * alignment_right.pi.shape[0]
    )
    vmin_luce = min(source_mass_left.min(), source_mass_right.min())
    vmax_luce = max(source_mass_left.max(), source_mass_right.max())

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    plotting.plot_surf(
        luce_flat_left,
        interpolate_map(source_mass_left, indices_left_luce),
        # hemi="left",
        view=(90, 0),
        bg_map=bg_map_left_luce,
        bg_on_data=True,
        cmap="rainbow",
        colorbar=True,
        cbar_tick_format="%.3g",
        vmin=vmin_luce,
        vmax=vmax_luce,
        axes=ax,
    )
    plt.savefig(output_path / f"{output_prefix}_source_mass_left.png", transparent=True)
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    plotting.plot_surf(
        luce_flat_right,
        interpolate_map(source_mass_right, indices_right_luce),
        # hemi="left",
        view=(90, 0),
        bg_map=bg_map_right_luce,
        bg_on_data=True,
        cmap="rainbow",
        colorbar=True,
        cbar_tick_format="%.3g",
        vmin=vmin_luce,
        vmax=vmax_luce,
        axes=ax,
    )
    plt.savefig(
        output_path / f"{output_prefix}_source_mass_right.png", transparent=True
    )
    plt.close()

    # Target mass
    target_mass_left = (
        alignment_left.pi.sum(axis=0).numpy() * alignment_left.pi.shape[1]
    )
    target_mass_right = (
        alignment_right.pi.sum(axis=0).numpy() * alignment_right.pi.shape[1]
    )

    vmin_human = min(target_mass_left.min(), target_mass_right.min())
    vmax_human = max(target_mass_left.max(), target_mass_right.max())

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    plotting.plot_surf(
        fs7["flat_left"],
        interpolate_map(target_mass_left, indices_left_human),
        # hemi="left",
        view=(90, 270),
        bg_map=bg_map_left_human,
        bg_on_data=True,
        cmap="rainbow",
        colorbar=True,
        cbar_tick_format="%.3g",
        vmin=vmin_human,
        vmax=vmax_human,
        axes=ax,
    )
    plt.savefig(output_path / f"{output_prefix}_target_mass_left.png", transparent=True)
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    plotting.plot_surf(
        fs7["flat_right"],
        interpolate_map(target_mass_right, indices_right_human),
        # hemi="left",
        view=(90, 270),
        bg_map=bg_map_right_human,
        bg_on_data=True,
        cmap="rainbow",
        colorbar=True,
        cbar_tick_format="%.3g",
        vmin=vmin_human,
        vmax=vmax_human,
        axes=ax,
    )
    plt.savefig(
        output_path / f"{output_prefix}_target_mass_right.png", transparent=True
    )
    plt.close()
