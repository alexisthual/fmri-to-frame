"""Preprocess brain volumes and video stimuli from Wen 2017."""

# %%
import typing as tp
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from nilearn.glm.first_level import make_first_level_design_matrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# %%
def load_video_as_tensor(video_path: str, mod: int = None) -> torch.Tensor:
    """Load video file in torch tensor.

    Parameters
    ----------
    video_path: str
        Path to the video
    mod: int, optional
        If not None, only frames congruent to 0 modulo mod will be selected

    Returns
    -------
    frames: torch.Tensor of size (frames, height, width, channels)
    """

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    frames = []
    i = 0
    while cap.isOpened():
        # Read a frame
        ret, frame = cap.read()

        if mod is not None and i % mod != 0:
            i += 1
            continue

        # Check if a frame was read successfully
        if ret:
            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Transform the frame into a tensor
            tensor = transforms.ToTensor()(frame_rgb)

            # Append the tensor to the list
            frames.append(tensor)
        else:
            break

        i += 1

    # Close the video capture object
    cap.release()

    # Stack the frames to create a tensor with shape
    # (frames, channels, height, width)
    video_tensor = torch.stack(frames)

    # Change dimensions to (frames, height, width, channels)
    video_tensor = video_tensor.permute(0, 2, 3, 1)

    # Convert to 255 RGB unit array
    video_tensor = (255 * video_tensor).to(torch.uint8)

    return video_tensor


# %%
def align_brain_and_video_features(
    brain_tensor: torch.Tensor,
    video_tensor: torch.Tensor,
    acquisition_delay: int = 0,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """Temporally align image tensor with the brain tensor.

    Parameters
    ----------
    acquisition_delay: int, optional
        Duration between start of video playing and start of acquisition,
        in TRs.
    """
    # 1. Associate brain features contained in brain volume t
    # with the first frame of the sequence of frames shown at time
    # t + acquisition_delay.
    # Load first video frame of each tr
    image_tensor = video_tensor[: (video_duration // tr)]
    image_tensor = image_tensor[acquisition_delay:]

    # 2. Repeat the last frame such that both arrays
    # have the same temporal length
    n_diff = brain_tensor.shape[0] - image_tensor.shape[0]
    if n_diff > 0:
        last_image = image_tensor[-1]
        image_tensor = F.pad(image_tensor, (0, 0, 0, 0, 0, 0, 0, n_diff))
        image_tensor[-n_diff:] = last_image.repeat((n_diff, 1, 1, 1))

    assert brain_tensor.shape[0] == image_tensor.shape[0]

    return brain_tensor, image_tensor


# %%
def build_subject_data(
    subject: str = None, force: bool = False
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    (output_path / subject).mkdir(exist_ok=True)
    bold_path = output_path / subject / "bolds.npy"
    image_path = output_path / subject / "images.npy"

    if bold_path.exists() and image_path.exists():
        if not force:
            return

    brain_tensors = []
    image_tensors = []

    print(f"Parsing data for {subject}")
    for video_type in videos.keys():
        for video_index in videos[video_type]:
            video_path = (
                p
                / "10_4231_R71Z42KK"
                / "video_fmri_dataset"
                / "stimuli"
                / f"{video_index}.mp4"
            )
            print(f"Parsing video {video_type} {video_index}")
            video_tensor = load_video_as_tensor(
                str(video_path), mod=video_fps * tr
            )

            # Group runs in batches
            if isinstance(runs_avg[video_type], int):
                batch_size = runs_avg[video_type]
            elif runs_avg[video_type] == "all":
                batch_size = n_runs[video_type]
            else:
                batch_size = 1
            batch_start = list(range(1, n_runs[video_type] + 1, batch_size))

            for fmri_run_start in batch_start:
                runs = range(fmri_run_start, fmri_run_start + batch_size)
                print(f"Averaging runs {list(runs)}")

                brain_tensors_run = []
                for fmri_run in runs:
                    brain_features_path = (
                        p
                        / "preproc"
                        / f"{subject}-{video_index}-{fmri_run}.npy"
                    )

                    brain_features = np.load(
                        brain_features_path
                    )  # shape (n_vertices, n_volumes)

                    # 1. Detrend each voxel (ie regress linear shift with time)
                    if detrend:
                        model = LinearRegression(fit_intercept=True)

                        t_r = 2
                        design_matrix = make_first_level_design_matrix(
                            np.linspace(
                                0,
                                brain_features.shape[1] * t_r,
                                brain_features.shape[1],
                            ),
                            drift_model="cosine",
                        )
                        model.fit(design_matrix.values, brain_features.T)

                        brain_features = (
                            brain_features
                            - (design_matrix.values @ model.coef_.T).T
                        )

                    # 2. Standardize each voxel (ie center reduce timeseries)
                    brain_tensor_run_standardized = torch.from_numpy(
                        StandardScaler().fit_transform(brain_features.T)
                    )
                    brain_tensors_run.append(brain_tensor_run_standardized)

                brain_tensor = torch.mean(
                    torch.stack(brain_tensors_run), dim=0
                )

                # Compute how much brain features should be delayed
                # to be temporally aligned with video features.
                # TODO: handle delay for subject3, test2, run9
                # Current status: it's not clear what this file represents
                delay = (
                    0
                    if f"{video_type}{video_index}"
                    in [f"test{i}" for i in range(2, 6)]
                    else 1
                )
                brain_tensor, image_tensor = align_brain_and_video_features(
                    brain_tensor,
                    video_tensor,
                    acquisition_delay=delay,
                )
                brain_tensors.append(brain_tensor)
                image_tensors.append(image_tensor)

    bold_array = torch.cat(brain_tensors).numpy()
    image_array = torch.cat(image_tensors).numpy()

    print(f"Saving data for {subject}...")
    np.save(bold_path, bold_array)
    np.save(image_path, image_array)


# %%
if __name__ == "__main__":
    # %%
    subjects = [f"subject{i}" for i in range(1, 4)]

    videos = {
        "seg": [f"seg{i}" for i in range(1, 19)],
        "test": [f"test{i}" for i in range(1, 6)],
    }

    # How many runs per video should be taken
    n_runs = {
        "seg": 2,
        "test": 10,
    }

    # How big should the groups of runs
    # which are averaged together be
    runs_avg = {
        "seg": 1,
        "test": "all",
    }

    detrend = True

    # %%
    tr = 2  # In seconds
    video_duration = 8 * 60  # In seconds
    video_fps = 30

    # %%
    p = Path("/storage/store2/data/wen2017")
    dname = (
        f"{'detrend_' if detrend else ''}"
        f"seg_{n_runs['seg']}_{runs_avg['seg']}_"
        f"test_{n_runs['test']}_{runs_avg['test']}"
    )
    output_path = p / dname / "preproc"
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"Generating dataset to {output_path}")

    # %%
    for subject in subjects:
        build_subject_data(subject, force=True)
