"""Project brain volumes from Wen 2017 to fsaverage."""

# %%
from pathlib import Path

import numpy as np
from nilearn import datasets, image, surface
from tqdm.contrib.concurrent import process_map


# %%
def project_run(subject, segment, run, output_path=Path(".")):
    subject_data_path = (
        Path("/storage/store2/data/wen2017")
        / subject_folder[subject]
        / "video_fmri_dataset"
    )

    img_path = (
        subject_data_path
        / f"subject{subject}"
        / "fmri"
        / segment
        / "mni"
        / f"{segment}_{run}_mni.nii.gz"
    )

    if not img_path.exists():
        print(
            f"WARNING: Missing file for subject {subject}, {segment}, "
            f"run {run}"
        )
        return
    else:
        img = image.load_img(img_path)

        surf_map_left = surface.vol_to_surf(img, fs6.pial_left)
        surf_map_right = surface.vol_to_surf(img, fs6.pial_right)
        surf_map = np.concatenate([surf_map_left, surf_map_right])

        filename = f"subject{subject}-{segment}-{run}.npy"

        np.save(output_path / filename, surf_map)


def project_run_wrapper(args):
    return project_run(*args)


# %%
if __name__ == "__main__":
    # %%
    n_jobs = 10

    subjects = [1, 2, 3]

    subject_folder = {
        1: "10_4231_R7X63K3M",
        2: "10_4231_R7NS0S1F",
        3: "10_4231_R7J101BV",
    }

    videos = {
        "seg": [f"seg{i}" for i in range(1, 19)],
        "test": [f"test{i}" for i in range(1, 6)],
    }

    n_runs = {
        "seg": 2,
        "test": 10,
    }

    # %%
    output_path = Path("/storage/store2/data/wen2017") / "preproc"
    output_path.mkdir(exist_ok=True, parents=True)

    # %%
    fs6 = datasets.fetch_surf_fsaverage(mesh="fsaverage6")

    # %%
    process_map(
        project_run_wrapper,
        [
            [
                1,
                "test1",
                9,
                output_path,
            ]
        ],
        max_workers=n_jobs,
    )

    # for video_type in videos:
    #     print(f"Processing {video_type} for all subjects...")
    #     process_map(
    #         project_run_wrapper,
    #         list(
    #             product(
    #                 subjects,
    #                 videos[video_type],
    #                 range(1, n_runs[video_type] + 1),
    #                 [output_path],
    #             )
    #         ),
    #         max_workers=n_jobs,
    #     )
