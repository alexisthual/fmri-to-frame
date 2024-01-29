import re
import typing as tp
from dataclasses import dataclass, field
from functools import partial
from operator import itemgetter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.io as spio
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate


def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_keys(d):
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        """
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def generate_slices(config, n_discard=10):
    """
    Generate run slices from a config file.

    Parameters
    ----------
    config: dict
        Config file as a dict.
    n_discard: int
        Number of volumes to discard at the beginning and end of each run.

    Returns
    -------
    slices: list
        List of slices corresponding to each run.
    """
    slices = []
    acc = 0
    for session in config["sessions"]:
        for run in config["sessions"][session]["runs"]:
            n_trs = config["sessions"][session]["runs"][run]["n_trs"]
            slices.append(slice(acc + n_discard, acc + n_trs - n_discard))
            acc += n_trs

    return slices


@dataclass
class ImageDatasetBase(Dataset):
    """Base Dataset class for handling features.

    Attributes
    ----------
    labels: dict
        Dict containing np.ndarrays of size (n_volumes, n_features)
    metadata: dict, optional
    """

    labels: tp.Dict[str, np.ndarray]
    metadata: tp.Optional[tp.Dict] = field(default_factory=dict)

    def __post_init__(
        self,
    ):
        assert hasattr(self, "labels")
        if self.labels is not None:
            len_ = len(list(self.labels.values())[0])
            assert all([len(v) == len_ for v in self.labels.values()])

        if self.metadata is not None:
            assert all([len(v) == len_ for v in self.metadata.values()])
            assert set(self.labels.keys()).isdisjoint(self.metadata.keys())

    def __getitem__(self, idx):
        return {
            **{label: self.labels[label][idx] for label in self.labels},
            **{
                metadata_name: self.metadata[metadata_name][idx]
                for metadata_name in self.metadata
            },
        }

    def __len__(self):
        if self.labels is not None:
            return len(list(self.labels.values())[0])
        else:
            return 0


class ThingsDataset(ImageDatasetBase):
    """Torch Dataset for the THINGS image dataset."""

    def __init__(self, data_path, **kwargs):
        images_path = Path(data_path) / "images_nkeep-2.h5"
        self.labels = {
            "images": h5py.File(images_path, "r")["images"],
        }

        super().__init__(self.labels)


@dataclass
class FmriDatasetBase(Dataset):
    """Base Dataset class for handling matched brain & other features.

    Attributes
    ----------
    betas: np.ndarray of size (n_volumes, n_voxels)
    labels: dict
        Dict containing np.ndarrays of size (n_volumes, n_features)
    metadata: dict, optional
    """

    betas: np.ndarray
    labels: tp.Dict[str, np.ndarray]
    metadata: tp.Optional[tp.Dict] = field(default_factory=dict)

    def __post_init__(
        self,
    ):
        assert hasattr(self, "betas")
        assert hasattr(self, "labels")
        len_ = len(self.betas)
        # print(len_)
        # print([len(v) for v in self.labels.values()])
        assert all([len(v) == len_ for v in self.labels.values()])
        if self.metadata is not None:
            assert all([len(v) == len_ for v in self.metadata.values()])
            assert set(self.labels.keys()).isdisjoint(self.metadata.keys())

    def __getitem__(self, idx):
        return {
            "betas": self.betas[idx],
            **{label: self.labels[label][idx] for label in self.labels},
            **{
                metadata_name: self.metadata[metadata_name][idx]
                for metadata_name in self.metadata
            },
        }

    def __len__(self):
        return len(self.betas)


class IBCClipsSingleFmriDataset(FmriDatasetBase):
    """Torch Dataset for the fMRI Individual Brain Charting Dataset."""

    def __init__(self, data_path, subject):
        # Load BOLD maps.
        # The underlying folder structure is specific to IBC
        bolds_path = (
            Path(data_path) / "clips" / "bolds" / f"sub-{subject:02d}.h5"
        )
        self.brain_features = h5py.File(bolds_path, "r")["brain_features"][:]

        images_path = (
            Path(data_path) / "clips" / "stimuli" / "images_nkeep-1.h5"
        )
        self.labels = {
            "images": h5py.File(images_path, "r")["images"],
        }

        super().__init__(self.brain_features, self.labels)


class IBCClipsFmriDataset(FmriDatasetBase):
    """Torch Dataset for the fMRI Individual Brain Charting Dataset."""

    def __init__(self, data_path, subject):
        # Load BOLD maps.
        # The underlying folder structure is specific to IBC
        bolds_path = (
            Path(data_path) / "clips" / "bolds" / f"sub-{subject:02d}.h5"
        )
        self.brain_features = h5py.File(bolds_path, "r")["brain_features"][:]

        images_path = (
            Path(data_path) / "clips" / "stimuli" / "images_nkeep-4.h5"
        )
        self.labels = {
            "images": h5py.File(images_path, "r")["images"],
        }

        super().__init__(self.brain_features, self.labels)


class IBCGBUFmriDataset(FmriDatasetBase):
    """Torch Dataset for the fMRI Individual Brain Charting Dataset."""

    def __init__(self, data_path, subject):
        # Load BOLD maps.
        # The underlying folder structure is specific to IBC
        bolds_path = (
            Path(data_path) / "gbu" / "bolds" / f"sub-{subject:02d}.h5"
        )
        self.brain_features = h5py.File(bolds_path, "r")["brain_features"][:]

        images_path = Path(data_path) / "gbu" / "stimuli" / "images_nkeep-4.h5"
        self.labels = {
            "images": h5py.File(images_path, "r")["images"],
        }

        super().__init__(self.brain_features, self.labels)


class IBCRaidersFmriDataset(FmriDatasetBase):
    """Torch Dataset for the fMRI Individual Brain Charting Dataset."""

    def __init__(self, data_path, subject):
        # Load BOLD maps.
        # The underlying folder structure is specific to IBC
        bolds_path = (
            Path(data_path) / "raiders" / "bolds" / f"sub-{subject:02d}.h5"
        )
        self.brain_features = h5py.File(bolds_path, "r")["brain_features"][:]

        images_path = (
            Path(data_path) / "raiders" / "stimuli" / "images_nkeep-4.h5"
        )
        self.labels = {
            "images": h5py.File(images_path, "r")["images"],
        }

        super().__init__(self.brain_features, self.labels)


class IBCMonkeyKingdomFmriDataset(FmriDatasetBase):
    """Torch Dataset for the fMRI Individual Brain Charting Dataset."""

    def __init__(self, data_path, subject):
        # Load BOLD maps.
        # The underlying folder structure is specific to IBC
        bolds_path = Path(data_path) / "mk" / "bolds" / f"sub-{subject:02d}.h5"
        self.brain_features = h5py.File(bolds_path, "r")["brain_features"][:]

        images_path = (
            Path(data_path)
            / "mk"
            / "stimuli"
            / f"images_sub-{subject:02d}_nkeep-4.h5"
        )
        self.labels = {
            "images": h5py.File(images_path, "r")["images"],
        }

        super().__init__(self.brain_features, self.labels)


class IBCMonkeyKingdomMionFmriDataset(FmriDatasetBase):
    """Torch Dataset for the fMRI Individual Brain Charting Dataset."""

    def __init__(self, data_path, subject):
        # Load BOLD maps.
        # The underlying folder structure is specific to IBC
        bolds_path = (
            Path(data_path) / "mk" / "bolds" / f"sub-{subject:02d}_mion.h5"
        )
        self.brain_features = h5py.File(bolds_path, "r")["brain_features"][:]

        with open(
            Path(data_path)
            / "mk"
            / "bolds"
            / f"sub-{subject:02d}_config.yaml",
            "r",
        ) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        images_path = (
            Path(data_path)
            / "mk"
            / "stimuli"
            / f"images_sub-{subject:02d}_nkeep-4.h5"
        )
        all_images = h5py.File(images_path, "r")["images"][:]
        slices = generate_slices(config)
        self.labels = {
            "images": np.vstack(
                itemgetter(*slices)(all_images)
            ),
        }

        super().__init__(self.brain_features, self.labels)


class LeuvenGBUFmriDataset(FmriDatasetBase):
    """Torch Dataset for the fMRI Individual Brain Charting Dataset."""

    def __init__(self, data_path, subject, segment=None):
        # Load MION maps.
        # The underlying folder structure is specific to Leuven
        if segment is None or segment == "":
            bolds_path = Path(data_path) / "gbu" / "mion" / f"{subject}.h5"
        else:
            bolds_path = (
                Path(data_path)
                / "gbu"
                / "mion"
                / f"{subject}_seg-{segment}.h5"
            )
        self.brain_features = h5py.File(bolds_path, "r")["brain_features"][:]

        # These files are from the IBC dataset
        images_path = Path(data_path) / "gbu" / "stimuli" / "images_nkeep-4.h5"
        # These indices are determined from corresponding IBC runs' length
        start_index = {
            "1": 265 + 244,
            "2": 265 + 244 + 304,
            "3": 265 + 244 + 304 + 304,
        }
        seg_length = 300

        self.labels = {
            "images": h5py.File(images_path, "r")["images"][
                start_index[segment] : start_index[segment] + seg_length
            ],
        }

        super().__init__(self.brain_features, self.labels)


class LeuvenMonkeyKingdomFmriDataset(FmriDatasetBase):
    """Torch Dataset for the fMRI Leuven Monkey Kingdom data."""

    def __init__(self, data_path, subject, segment=None):
        # Load MION maps.
        # The underlying folder structure is specific to Leuven
        if segment is None or segment == "":
            bolds_path = Path(data_path) / "mk" / "mion" / f"{subject}.h5"
        else:
            bolds_path = (
                Path(data_path) / "mk" / "mion" / f"{subject}_seg-{segment}.h5"
            )
        self.brain_features = h5py.File(bolds_path, "r")["brain_features"][:]

        # These files are from the IBC dataset
        images_path = Path(data_path) / "mk" / "stimuli" / "images_nkeep-4.h5"
        start_index = 465 * (int(segment) - 1)
        seg_length = 465

        self.labels = {
            "images": h5py.File(images_path, "r")["images"][
                start_index : start_index + seg_length
            ],
        }

        super().__init__(self.brain_features, self.labels)


class LeuvenMonkeyKingdomHRFFmriDataset(FmriDatasetBase):
    """Torch Dataset for the fMRI Leuven Monkey Kingdom data."""

    def __init__(self, data_path, subject, segment=None):
        # Load MION maps.
        # The underlying folder structure is specific to Leuven
        bolds_path = (
            Path(data_path) / "mk" / "mion" / f"{subject}_seg-{segment}_hrf.h5"
        )
        self.brain_features = h5py.File(bolds_path, "r")["brain_features"][:]

        # These files are from the IBC dataset
        images_path = Path(data_path) / "mk" / "stimuli" / "images_nkeep-4.h5"
        start_index = 465 * (int(segment) - 1) + 10
        seg_length = 445

        self.labels = {
            "images": h5py.File(images_path, "r")["images"][
                start_index : start_index + seg_length
            ],
        }

        super().__init__(self.brain_features, self.labels)


class NSDFmriDataset(FmriDatasetBase):
    """Torch Dataset for the fMRI Natural Scene Dataset."""

    def __init__(self, data_path, subject):
        label_keys = ["images", "captions"]

        # Load pre-computed beta maps.
        # The underlying folder structure is specific to NSD
        self.data_path = Path(data_path) / f"subj{subject:02d}"
        self.betas = np.load(self.data_path / "betas.npy")
        self.labels = {
            key: np.load(self.data_path / f"{key}.npy") for key in label_keys
        }

        super().__init__(self.betas, self.labels)


class NSDUnsharedFmriDataset(FmriDatasetBase):
    """Torch Dataset for the fMRI Natural Scene Dataset."""

    def __init__(self, data_path, subject):
        # Load pre-computed beta maps.
        # The underlying folder structure is specific to NSD
        self.data_path = Path(data_path) / f"sub-{subject:02d}"
        self.betas = np.load(
            self.data_path / "unshared_stim_brain_features.npy"
        )
        self.labels = {"images": np.load(self.data_path / "unshared_stim.npy")}

        super().__init__(self.betas, self.labels)


class NSDSharedFmriDataset(FmriDatasetBase):
    """Torch Dataset for the fMRI Natural Scene Dataset."""

    def __init__(self, data_path, subject):
        # Load pre-computed beta maps.
        # The underlying folder structure is specific to NSD
        self.data_path = Path(data_path) / f"sub-{subject:02d}"
        self.betas = np.load(self.data_path / "shared_stim_brain_features.npy")
        self.labels = {"images": np.load(self.data_path / "shared_stim.npy")}

        super().__init__(self.betas, self.labels)


class Wen2017FmriDataset(FmriDatasetBase):
    """Torch Dataset for the Wen2017 dataset."""

    def __init__(self, data_path, subject):
        label_keys = ["images"]

        # Load BOLD maps directly.
        # The underlying folder structure is specific to Wen2017
        self.data_path = Path(data_path) / "preproc" / f"subject{subject}"
        self.betas = np.load(self.data_path / "bolds.npy")
        self.labels = {
            key: np.load(self.data_path / f"{key}.npy") for key in label_keys
        }

        super().__init__(self.betas, self.labels)


class Zhou2023FmriDataset(FmriDatasetBase):
    """Torch Dataset for Zhou 2023 dataset."""

    def __init__(
        self,
        data_path,
        subject,
    ):
        # get the data path
        sub = f"sub-{subject:02d}"
        self.data_path = Path(f"{data_path}_{sub}")

        # load the data (bold is in shape [VOXEL, T] so transpose to [T, VOXEL])
        # NOTE: shape is usually [97679, 1872] for EPI
        self.betas = np.load(f"{self.data_path}_bold.npy").T

        # load the images
        label_keys = ["images"]

        # load the image labels (in shape [T, X, Y, CH])
        self.labels = {
            key: np.load(f"{self.data_path}_{key}.npy") for key in label_keys
        }
        assert self.labels["images"].shape[0] == self.betas.shape[0]

        # load the events
        events = pd.read_csv(f"{self.data_path}_videos.tsv", sep="\t")
        events = events[["stim_file", "onset", "duration"]]
        assert len(events) > 0

        # replicate the elements
        rows = []
        np_pos = 0
        TR = 2.0
        err = 0.1  # allowed deviation in seconds

        # iterate through all brain-image positions
        for pos in range(0, self.betas.shape[0]):
            # compute current position in seconds
            sec_pos = pos * TR

            # find next valid row
            row = None
            if np_pos < len(events):
                row = events.iloc[np_pos]
                while sec_pos > (row["onset"] + row["duration"] + err):
                    np_pos += 1
                    if np_pos >= len(events):
                        row = None
                        break
                    row = events.iloc[np_pos]

            # check current row
            if (
                row is not None
                and row["onset"] < sec_pos + err
                and row["onset"] + row["duration"] > sec_pos - err
            ):
                rows.append([row["stim_file"]])
            else:
                rows.append([None])

        # convert to numpy
        videos = np.array(rows)
        assert (
            videos.shape[0] == self.betas.shape[0]
        ), f"Mismatch {videos.shape} to {self.betas.shape}\n{videos}"
        self.labels["videos"] = videos

        super().__init__(self.betas, self.labels)


def load_fmridataset(
    dataset_id,
    dataset_path,
    subject,
):
    """Load brain features and stimuli."""
    # Load brain features and stimuli features,
    if dataset_id == "ibc_clips":
        dataset = IBCClipsFmriDataset(
            subject=subject,
            data_path=dataset_path,
        )
    elif dataset_id == "ibc_clips_single":
        dataset = IBCClipsSingleFmriDataset(
            subject=subject,
            data_path=dataset_path,
        )
    elif dataset_id == "ibc_gbu":
        dataset = IBCGBUFmriDataset(
            subject=subject,
            data_path=dataset_path,
        )
    elif dataset_id == "ibc_raiders":
        dataset = IBCRaidersFmriDataset(
            subject=subject,
            data_path=dataset_path,
        )
    elif dataset_id == "ibc_mk":
        dataset = IBCMonkeyKingdomFmriDataset(
            subject=subject,
            data_path=dataset_path,
        )
    elif dataset_id == "ibc_mk_mion":
        dataset = IBCMonkeyKingdomMionFmriDataset(
            subject=subject,
            data_path=dataset_path,
        )
    elif dataset_id.startswith("leuven_gbu"):
        segment = re.search(r".*seg-([0-9]*).*", dataset_id).group(1)
        dataset = LeuvenGBUFmriDataset(
            subject=subject,
            data_path=dataset_path,
            segment=segment,
        )
    elif dataset_id.startswith("leuven_mk"):
        segment = re.search(r".*seg-([0-9]*).*", dataset_id).group(1)
        dataset = LeuvenMonkeyKingdomFmriDataset(
            subject=subject,
            data_path=dataset_path,
            segment=segment,
        )
    elif dataset_id == "nsd":
        dataset = NSDFmriDataset(
            subject=subject,
            data_path=dataset_path,
        )
    elif dataset_id == "nsd_unshared":
        dataset = NSDUnsharedFmriDataset(
            subject=subject,
            data_path=dataset_path,
        )
    elif dataset_id == "nsd_shared":
        dataset = NSDSharedFmriDataset(
            subject=subject,
            data_path=dataset_path,
        )
    elif dataset_id.startswith("wen2017"):
        dataset = Wen2017FmriDataset(
            subject=subject,
            data_path=dataset_path,
        )
    elif dataset_id.startswith("zhou2023"):
        dataset = Zhou2023FmriDataset(
            subject=subject,
            data_path=dataset_path,
        )
    elif dataset_id == "things":
        dataset = ThingsDataset(
            data_path=dataset_path,
        )
    else:
        raise NotImplementedError()

    return dataset


def get_collate_fn(key):
    def collate_fn(key, batch):
        if key in ["captions", "videos"]:
            return np.array([np.array(d[key]) for d in batch], dtype=object)
        else:
            return torch.stack([torch.tensor(d[key]) for d in batch])

    return partial(collate_fn, key)


class TorchStandardScaler:
    def __init__(self, axis: int = 0):
        self.axis = axis

    def fit(self, x: torch.Tensor):
        self.mean = x.mean(axis=self.axis, keepdims=True)
        self.std = x.std(axis=self.axis, keepdims=True)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.fit(x).transform(x)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class LatentsDataModuleBase(pl.LightningDataModule):
    """PL DataModule to load brain data associated with latent stimuli data."""

    def __init__(
        self,
        train_data: FmriDatasetBase,
        valid_data: FmriDatasetBase,
        test_data: FmriDatasetBase,
        num_workers: int = 0,
        batch_size: int = 32,
        # Whether to shuffle training batches to ensure IID distribution
        shuffle_train: bool = False,
    ):
        super().__init__()
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train

        self.labels = self.train_data.labels

    def prepare_data(self, normalize_betas=False, normalize_latents=False):
        """Preprocess the dataset."""

        # Fit standard scalers on training set
        self.scaler = dict()
        self.scaler["betas"] = TorchStandardScaler().fit(self.train_data.betas)
        for label in self.labels:
            self.scaler[label] = TorchStandardScaler().fit(
                self.train_data.labels[label]
            )

        # Z-score normalize inputs of every split
        for split in [self.train_data, self.valid_data, self.test_data]:
            if split.betas == [] or split.betas.shape[0] == 0:
                continue

            if normalize_betas:
                split.betas = self.scaler["betas"].transform(split.betas)
            if normalize_latents:
                for label in self.labels:
                    split.labels[label] = self.scaler[label].transform(
                        split.labels[label]
                    )

    def process_features(
        self,
        window_size: tp.Optional[int] = 1,
        lag: tp.Optional[int] = 0,
        agg: tp.Optional[str] = "mean",
        support: tp.Optional[str] = None,
        # Whether to shuffle labels, useful for control experiments
        shuffle_labels: bool = False,
    ):
        """
        See conf/bd_config.yaml for more details about parameters like
        window_size, lag, etc used to aggregate / align brain and stimuli features.
        """
        for dataset in [self.train_data, self.valid_data, self.test_data]:
            if dataset.betas == [] or dataset.betas.shape[0] == 0:
                continue

            if support == "fsaverage5":
                # Let m be the number of vertices of fsaverage_n.
                # The first m vertices of fsaverage_{n+1} correspond
                # to that of fsaverage_n.
                # Moreover, both hemispheres have the same number of vertices.
                n_voxels_fs6 = 40962
                n_voxels_fs5 = 10242
                # If current support is fsaverage6, keep only fsaverage5 vertices
                if dataset.betas.shape[1] == 2 * n_voxels_fs6:
                    dataset.betas = np.hstack(
                        [
                            dataset.betas[:, :n_voxels_fs5],
                            dataset.betas[
                                :, n_voxels_fs6 : n_voxels_fs6 + n_voxels_fs5
                            ],
                        ]
                    )
                else:
                    pass
            elif support is not None:
                raise NotImplementedError()

            # Aggregate fMRI volumes according to window_size
            if window_size is None:
                window_size = 1
            assert window_size > 0

            if agg is None:
                agg = "mean"
            assert agg in ["mean", "stack"]

            betas_view = np.lib.stride_tricks.sliding_window_view(
                dataset.betas, window_size, axis=0
            )
            if agg == "mean":
                # Converts shape (time, voxels, window_size)
                # into (time, voxels)
                dataset.betas = np.mean(betas_view, axis=-1)
            elif agg == "stack":
                # Converts shape (time, voxels, window_size)
                # into (time, voxels * window_size)
                dataset.betas = betas_view.reshape(betas_view.shape[0], -1)

            if lag is None:
                lag = 0
            assert lag >= 0
            # Shift fMRI volumes according to lag
            dataset.betas = dataset.betas[lag:]

            # Discard last images (due to lag) and
            # only select images for which we could stack enough fMRI volumes
            if lag + window_size == 1:
                # If lag + window_size is 1, then arr[:-0] won't work
                pass
            else:
                dataset.labels = {
                    k: v[: -(lag + window_size - 1)]
                    for k, v in dataset.labels.items()
                }

            for latents_type in dataset.labels.keys():
                assert (
                    dataset.betas.shape[0]
                    == dataset.labels[latents_type].shape[0]
                )

            if shuffle_labels:
                generator = torch.Generator()
                generator.manual_seed(2023)
                shuffled_indices = torch.randperm(
                    dataset.betas.shape[0], generator=generator
                )

                for label_type in dataset.labels.keys():
                    dataset.labels[label_type] = dataset.labels[label_type][
                        shuffled_indices
                    ]

    def collate_fn(self, batch):
        """Apply scaling to the inputs."""
        batch = default_collate(batch)

        result = dict()
        for key in batch.keys():
            if key in self.labels:
                result[key] = self.scaler[key](batch[key])
            else:
                # metadata is not fitted
                result[key] = batch[key]
        return result

    def setup(self, stage: str):
        pass  # Done in __init__

    def train_dataloader(self, normalize_latents=False, shuffle=None):
        collate_fn = self.collate_fn if normalize_latents else None
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            shuffle=shuffle
            if shuffle is not None
            else (self.shuffle_train if len(self.train_data) > 0 else False),
        )

    def val_dataloader(self, normalize_latents=False):
        collate_fn = self.collate_fn if normalize_latents else None
        return DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self, normalize_latents=False):
        collate_fn = self.collate_fn if normalize_latents else None
        # batch_size = len(self.test_data)  # XXX To parametrize
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def predict_dataloder(self):
        return self.test_dataloader()
