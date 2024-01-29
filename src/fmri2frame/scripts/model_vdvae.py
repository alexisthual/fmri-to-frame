import sys
import typing as tp
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

from fmri2frame.scripts.utils import convert_to_PIL

# Imported from openai/vdvae
# https://github.com/openai/vdvae
lib_path = Path("/storage/store2/work/athual/repo/vdvae")
assert lib_path.exists()
sys.path.append(str(lib_path))
from vae import VAE
from utils import maybe_download


# Copied from openai/vdvae/train_helpers.py
# because some dependencies are missing / hard to install
@contextmanager
def first_rank_first(local_rank, mpi_size):
    if mpi_size > 1 and local_rank > 0:
        dist.barrier()

    try:
        yield
    finally:
        if mpi_size > 1 and local_rank == 0:
            dist.barrier()


def distributed_maybe_download(path, local_rank, mpi_size):
    if not path.startswith("gs://"):
        return path
    filename = path[5:].replace("/", "-")
    with first_rank_first(local_rank, mpi_size):
        fp = maybe_download(path, filename)
    return fp


def restore_params(model, path, local_rank, mpi_size, map_ddp=True, map_cpu=False):
    state_dict = torch.load(
        distributed_maybe_download(path, local_rank, mpi_size),
        map_location="cpu" if map_cpu else None,
    )
    if map_ddp:
        new_state_dict = {}
        l = len("module.")
        for k in state_dict:
            if k.startswith("module."):
                new_state_dict[k[l:]] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]
        state_dict = new_state_dict
    model.load_state_dict(state_dict)


# Copied from ozcelikfu/brain-diffuser/vdvae/model_utils.py
def set_up_data(H):
    shift_loss = -127.5
    scale_loss = 1.0 / 127.5

    H.image_size = 64
    H.image_channels = 3
    shift = -115.92961967
    scale = 1.0 / 69.37404

    shift = torch.tensor([shift]).cuda().view(1, 1, 1, 1)
    scale = torch.tensor([scale]).cuda().view(1, 1, 1, 1)
    shift_loss = torch.tensor([shift_loss]).cuda().view(1, 1, 1, 1)
    scale_loss = torch.tensor([scale_loss]).cuda().view(1, 1, 1, 1)

    def preprocess_func(x):
        nonlocal shift
        nonlocal scale
        nonlocal shift_loss
        nonlocal scale_loss
        inp = x.cuda(non_blocking=True).float()
        out = inp.clone()
        inp.add_(shift).mul_(scale)
        out.add_(shift_loss).mul_(scale_loss)
        return inp, out

    return H, preprocess_func


def load_vaes(H, logprint=None):
    ema_vae = VAE(H)
    if H.restore_ema_path is not None:
        print(f"Restoring ema vae from {H.restore_ema_path}")
        restore_params(
            ema_vae,
            H.restore_ema_path,
            map_cpu=True,
            local_rank=H.local_rank,
            mpi_size=H.mpi_size,
        )
    else:
        raise NotImplementedError()

    ema_vae.requires_grad_(False)
    ema_vae = ema_vae.cuda(H.local_rank)

    return ema_vae


# New classes
class vdvae_batch_generator(Dataset):
    def __init__(self, images):
        self.images = images

    def __getitem__(self, idx):
        return VdvaeModel.prepare_image(self.images[idx])

    def __len__(self):
        return len(self.images)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class VdvaeModel:
    def __init__(self, path):
        H = {
            "image_size": 64,
            "image_channels": 3,
            "seed": 0,
            "port": 29500,
            "save_dir": "./saved_models/test",
            "data_root": "./",
            "desc": "test",
            "hparam_sets": "imagenet64",
            "restore_path": str(Path(path) / "imagenet64-iter-1600000-model.th"),
            "restore_ema_path": str(
                Path(path) / "imagenet64-iter-1600000-model-ema.th"
            ),
            "restore_log_path": str(Path(path) / "imagenet64-iter-1600000-log.jsonl"),
            "restore_optimizer_path": str(
                Path(path) / "imagenet64-iter-1600000-opt.th"
            ),
            "dataset": "imagenet64",
            "ema_rate": 0.999,
            "enc_blocks": "64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5",
            "dec_blocks": "1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12",
            "zdim": 16,
            "width": 512,
            "custom_width_str": "",
            "bottleneck_multiple": 0.25,
            "no_bias_above": 64,
            "scale_encblock": False,
            "test_eval": True,
            "warmup_iters": 100,
            "num_mixtures": 10,
            "grad_clip": 220.0,
            "skip_threshold": 380.0,
            "lr": 0.00015,
            "lr_prior": 0.00015,
            "wd": 0.01,
            "wd_prior": 0.0,
            "num_epochs": 10000,
            "n_batch": 4,
            "adam_beta1": 0.9,
            "adam_beta2": 0.9,
            "temperature": 1.0,
            "iters_per_ckpt": 25000,
            "iters_per_print": 1000,
            "iters_per_save": 10000,
            "iters_per_images": 10000,
            "epochs_per_eval": 1,
            "epochs_per_probe": None,
            "epochs_per_eval_save": 1,
            "num_images_visualize": 8,
            "num_variables_visualize": 6,
            "num_temperatures_visualize": 3,
            "mpi_size": 1,
            "local_rank": 0,
            "rank": 0,
            "logdir": "./saved_models/test/log",
        }

        H = dotdict(H)
        self.H, self.preprocess_fn = set_up_data(H)
        self.vae = load_vaes(H)

    def get_vdvae_latent_from_images(
        self,
        images: tp.List,
        batch_size=30,
        save_to_path: tp.Optional[str] = None,
        seed: int = 0,
    ) -> np.ndarray:
        """
        Args
            - images: a list of N RGB PIL.Image or corresponding np.ndarray (h,w,3)
            - vdvae: an instantiated VDVAE model
            - save_to_path: if not None, a .npy will be created with the resulting vdvae latents stored at this path
        Return
            - an array of shape (N,91168) made of the vdvae latents of the input N images
        """
        torch.manual_seed(seed)
        prepared_images = convert_to_PIL(images)
        img_ds = vdvae_batch_generator(prepared_images)
        img_dl = DataLoader(img_ds, batch_size, shuffle=False)
        latents = []
        for batch in tqdm(img_dl):
            batch_latents = self.get_vdvae_latent(batch)
            latents.append(batch_latents)

        all_latents = torch.concat(latents).cpu().numpy()
        if save_to_path is not None:
            np.save(save_to_path, all_latents)
        return all_latents

    def get_vdvae_latent(
        self,
        tensor_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            - tensor_images: a tensor of shape (bs, h, 64, 64) of images as tensors with values in [0,255]
        Return:
            - a tensor of shape (bs, 91168) made of the 31st vdvae's encoder latents of the input `tensor_images`
        """
        num_latents = 31
        x = tensor_images
        data_input, _ = self.preprocess_fn(x)
        with torch.no_grad():
            activations = self.vae.encoder.forward(data_input)
            _, stats = self.vae.decoder.forward(activations, get_latents=True)
            batch_latent = []
            for i in range(num_latents):
                batch_latent.append(stats[i]["z"].view(len(data_input), -1))
        return torch.hstack(batch_latent)

    def _latent_transformation(self, latents, ref):
        # dimensions of the first 31 layers of the encoder
        layer_dims = np.array(
            [
                2**4,
                2**4,
                2**8,
                2**8,
                2**8,
                2**8,
                2**10,
                2**10,
                2**10,
                2**10,
                2**10,
                2**10,
                2**10,
                2**10,
                2**12,
                2**12,
                2**12,
                2**12,
                2**12,
                2**12,
                2**12,
                2**12,
                2**12,
                2**12,
                2**12,
                2**12,
                2**12,
                2**12,
                2**12,
                2**12,
                2**14,
            ]
        )
        transformed_latents = []
        for i in range(31):
            # latents has shape (n_test, dim_latent)
            t_lat = latents[:, layer_dims[:i].sum() : layer_dims[: i + 1].sum()]
            c, h, w = ref[i]["z"].shape[1:]  # this provides a shape layer_dim x h x w
            transformed_latents.append(t_lat.reshape(len(latents), c, h, w))
        return transformed_latents

    def _sample_from_higher_latents(self, latents, sample_ids):
        # latents = 31 latents, each of shape corresponding to the shape they had in the VDVAE
        # Each latents[i] is a tensor of shape (n_test, h_layer, *stuff)
        # sample_ids are the indices of this batch
        # latents[0] is the total size of the dataset, so we don't overflow the last batch
        sample_ids = [id for id in sample_ids if id < len(latents[0])]
        # layers_num = the total size of the test set
        layers_num = len(latents)
        sample_latents = []
        for i in range(layers_num):
            sample_latents.append(torch.tensor(latents[i][sample_ids]).float().cuda())
        # This is a list of size 31, each tensor inside is the
        # values for this layer for the batch passed as `sample_ids`. So it's sample as in 'echantillon'
        return sample_latents

    def reconstruct_from_vdvae_latents(
        self,
        vdvae_latents: np.ndarray,
        batch_size=30,
        save_to_path: tp.Optional[str] = None,
        seed: int = 0,
    ) -> tp.List:
        """
        Args
        - vae_latents: an array of shape (N, dim_vae_latents)
        - images: a list of images corresponding to the vae_latents
        - save_to_path: if not None, save the images to this path (with their index in `vae_latents` as names)
        Return
        - a list of PIL.Image
        """
        images = np.zeros((len(vdvae_latents), 64, 64, 3)).astype(np.uint8)
        torch.manual_seed(seed)
        prepared_images = convert_to_PIL(images)
        img_ds = vdvae_batch_generator(prepared_images)
        img_dl = DataLoader(img_ds, batch_size, shuffle=False)
        latents = []
        for i, x in tqdm(enumerate(img_dl)):
            data_input, _ = self.preprocess_fn(x)
            with torch.no_grad():
                activations = self.vae.encoder.forward(data_input)
                _, stats = self.vae.decoder.forward(activations, get_latents=True)
                # recons = ema_vae.decoder.out_net.sample(px_z)
                batch_latent = []
                for i in range(31):
                    batch_latent.append(
                        stats[i]["z"].cpu().numpy().reshape(len(data_input), -1)
                    )
                latents.append(np.hstack(batch_latent))
        latents = np.concatenate(latents)

        idx = range(len(prepared_images))
        # basically a reshape
        input_latent = self._latent_transformation(vdvae_latents[idx], stats)
        ll_results = []
        for i in range(int(np.ceil(len(prepared_images) / batch_size))):
            samp = self._sample_from_higher_latents(
                input_latent, range(i * batch_size, (i + 1) * batch_size)
            )
            # len(samp[0]) is just the number of layers
            px_z = self.vae.decoder.forward_manual_latents(len(samp[0]), samp, t=None)
            sample_from_latent = self.vae.decoder.out_net.sample(px_z)
            # sample_from_latent is a batch of batch_size RGB images 64x64
            for j in trange(len(sample_from_latent)):
                im = sample_from_latent[j]
                im = Image.fromarray(im)
                im = im.resize((512, 512), resample=3)
                if save_to_path is not None:
                    im.save(Path(save_to_path) / f"{i*batch_size+j}.png")
                ll_results.append(im)
        return ll_results

    @staticmethod
    def prepare_images(images):
        images = convert_to_PIL(images)
        return [VdvaeModel.prepare_image(img) for img in images]

    @staticmethod
    def prepare_image(image):
        image = T.functional.resize(image, (64, 64))
        return torch.tensor(np.array(image)).float()
