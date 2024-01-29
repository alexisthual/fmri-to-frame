import io
import logging
import typing as tp
from pathlib import Path

import clip
import numpy as np
import PIL
import torch
from joblib import Memory

# From comp-vis/stable-diffusion
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from fmri2frame.scripts.data import get_collate_fn, load_fmridataset
from fmri2frame.scripts.model_diffusion import VersatileDiffusionModel
from fmri2frame.scripts.model_vdvae import VdvaeModel

logger = logging.getLogger(__name__)


def stack_betas(dataloader: torch.utils.data.DataLoader) -> np.ndarray:
    """Stack brain features of a given dataloader."""
    betas = [batch["betas"] for batch in dataloader]

    # Check that lists are not empty
    # (validation set can be empty for instance)
    if len(betas) > 0:
        # Concatenate batches
        betas = torch.cat(betas).numpy()

    return betas


def compute_latents(
    dataset_id,
    dataset_path,
    subject,
    latent_type,
    model_path=None,
    seed=0,
    batch_size=32,
    cache=None,
):
    """Compute latent representations of stimuli."""
    if latent_type == "vdvae_encoder_31l_latents":
        compute_specific_latents = compute_vae_latents
    elif latent_type == "clip_vision_latents":
        compute_specific_latents = compute_clip_vision_latents
    elif latent_type == "clip_text_latents":
        compute_specific_latents = compute_clip_text_latents
    elif latent_type == "clip_vision_cls":
        compute_specific_latents = compute_clip_vision_cls
    elif latent_type == "sd_autokl":
        compute_specific_latents = compute_sd_autokl
    else:
        raise NotImplementedError()

    if cache is not None:
        memory = Memory(cache, verbose=100)
        compute_specific_latents_fn = memory.cache(
            compute_specific_latents, ignore=["batch_size"]
        )
    else:
        compute_specific_latents_fn = compute_specific_latents

    latents, metadata = compute_specific_latents_fn(
        dataset_id,
        dataset_path,
        subject,
        model_path=model_path,
        seed=seed,
        batch_size=batch_size,
    )

    return latents, metadata


# VDVAE MODEL


def compute_vae_latents(
    dataset_id,
    dataset_path,
    subject,
    model_path=None,
    seed=0,
    batch_size=32,
):
    torch.manual_seed(seed)

    fmri_dataset = load_fmridataset(
        dataset_id,
        dataset_path,
        subject,
    )
    model = load_vae_model(model_path)
    stimulus_key = "images"

    dataloader = DataLoader(
        fmri_dataset,
        batch_size=batch_size,
        collate_fn=get_collate_fn(stimulus_key),
    )

    latents = extract_vae_latents(dataloader, model, seed)
    metadata = None

    return latents, metadata


def load_vae_model(pretrained_models_path: tp.Union[str, Path]) -> VdvaeModel:
    """Load VDVAE model"""
    logger.info("Loading VAE model...")
    vae = VdvaeModel(pretrained_models_path)
    return vae


def extract_vae_latents(
    dataloader: torch.utils.data.DataLoader,
    vae: VdvaeModel,
    seed: int = 0,
) -> np.ndarray:
    """Compute VAE latent representations for each stimulus."""
    torch.manual_seed(seed)

    vdvae_encoder_31l_latents = []

    for batch in tqdm(dataloader):
        if len(batch.shape) == 5:
            for tr in batch:
                tr_images_for_vdvae = torch.stack(
                    VdvaeModel.prepare_images(tr)
                )
                tr_vdvae_latents = vae.get_vdvae_latent(tr_images_for_vdvae)
                vdvae_encoder_31l_latents.append(
                    np.mean(tr_vdvae_latents.cpu().numpy(), axis=0)
                )
        elif len(batch.shape) == 4:
            batch_images_for_vdvae = torch.stack(
                VdvaeModel.prepare_images(batch)
            )
            batch_vdvae_latents = vae.get_vdvae_latent(batch_images_for_vdvae)
            vdvae_encoder_31l_latents.extend(batch_vdvae_latents.cpu().numpy())

    # Check that lists are not empty
    # (validation set can be empty for instance)
    if len(vdvae_encoder_31l_latents) > 0:
        # Concatenate batches
        vdvae_encoder_31l_latents = np.stack(vdvae_encoder_31l_latents)

        # Flatten latents for each sample
        n_samples = vdvae_encoder_31l_latents.shape[0]
        vdvae_encoder_31l_latents = vdvae_encoder_31l_latents.reshape(
            n_samples, -1
        )

    return vdvae_encoder_31l_latents


# VERSATILE DIFFUSION CLIP VISION 256 AND TEXT


def compute_clip_vision_latents(
    dataset_id,
    dataset_path,
    subject,
    model_path=None,
    seed=0,
    batch_size=32,
):
    torch.manual_seed(seed)

    fmri_dataset = load_fmridataset(
        dataset_id,
        dataset_path,
        subject,
    )
    model = load_versatile_diffusion_model(model_path)
    stimulus_key = "images"

    dataloader = DataLoader(
        fmri_dataset,
        batch_size=batch_size,
        collate_fn=get_collate_fn(stimulus_key),
    )

    latents = extract_clip_vision_latents(dataloader, model, seed)
    metadata = None

    return latents, metadata


def load_versatile_diffusion_model(
    pretrained_models_path: tp.Union[str, Path],
) -> VersatileDiffusionModel:
    """Load Versatile diffusion."""
    logger.info("Loading VersatileDiffusion model...")
    diffusion = VersatileDiffusionModel(path=pretrained_models_path)
    return diffusion


def extract_clip_vision_latents(
    dataloader: torch.utils.data.DataLoader,
    diffusion: VersatileDiffusionModel,
    generation_seed: int,
) -> np.ndarray:
    """Compute VAE latent representations for each stimulus."""
    torch.manual_seed(generation_seed)

    clip_vision_latents = []

    for batch in tqdm(dataloader):
        if len(batch.shape) == 5:
            for tr in batch:
                tr_latents = []
                for frame in tr:
                    frame_latents = (
                        diffusion.get_clip_image(
                            VersatileDiffusionModel.prepare_images([frame])
                        )
                        .cpu()
                        .numpy()
                    )

                    tr_latents.append(frame_latents)
                clip_vision_latents.append(
                    np.mean(np.stack(tr_latents), axis=0)
                )
        elif len(batch.shape) == 4:
            for frame in batch:
                batch_images_for_clip = torch.stack(
                    VersatileDiffusionModel.prepare_images([frame])
                )
                batch_clip_vision_latents = (
                    diffusion.get_clip_image(batch_images_for_clip)
                    .cpu()
                    .numpy()
                )
                clip_vision_latents.append(batch_clip_vision_latents)

    # Check that lists are not empty
    # (validation set can be empty for instance)
    if len(clip_vision_latents) > 0:
        # Concatenate batches
        clip_vision_latents = np.stack(clip_vision_latents)

        # Flatten latents for each sample
        n_samples = clip_vision_latents.shape[0]
        clip_vision_latents = clip_vision_latents.reshape(n_samples, -1)

    return clip_vision_latents


def compute_clip_text_latents(
    dataset_id,
    dataset_path,
    subject,
    model_path=None,
    seed=0,
    batch_size=32,
):
    torch.manual_seed(seed)

    fmri_dataset = load_fmridataset(dataset_id, dataset_path, subject)
    model = load_versatile_diffusion_model(model_path)
    stimulus_key = "captions"

    dataloader = DataLoader(
        fmri_dataset,
        batch_size=batch_size,
        collate_fn=get_collate_fn(stimulus_key),
    )

    latents = extract_clip_text_latents(dataloader, model, seed)
    metadata = None

    return latents, metadata


def extract_clip_text_latents(
    dataloader: torch.utils.data.DataLoader,
    diffusion: VersatileDiffusionModel,
    generation_seed: int,
) -> np.ndarray:
    """Compute VAE latent representations for each stimulus."""
    torch.manual_seed(generation_seed)

    clip_text_latents = []

    for batch in tqdm(dataloader):
        batch_clip_text_latents = diffusion.get_clip_text(batch)
        clip_text_latents.append(batch_clip_text_latents.cpu())

    # Check that lists are not empty
    # (validation set can be empty for instance)
    if len(clip_text_latents) > 0:
        # Concatenate batches
        clip_text_latents = torch.cat(clip_text_latents)

        # Flatten latents for each sample
        n_samples = clip_text_latents.shape[0]
        clip_text_latents = clip_text_latents.numpy().reshape(n_samples, -1)

    return clip_text_latents


# CLIP VISION CLS


def compute_clip_vision_cls(
    dataset_id,
    dataset_path,
    subject,
    model_path=None,
    seed=0,
    batch_size=32,
):
    torch.manual_seed(seed)

    fmri_dataset = load_fmridataset(dataset_id, dataset_path, subject)
    model, clip_preprocess = clip.load("ViT-L/14", device="cuda", jit=False)
    stimulus_key = "images"

    dataloader = DataLoader(
        fmri_dataset,
        batch_size=batch_size,
        collate_fn=get_collate_fn(stimulus_key),
    )

    latents = extract_clip_vision_cls(
        dataloader,
        model,
        clip_preprocess,
    )
    metadata = None

    return latents, metadata


@torch.no_grad()
def extract_clip_vision_cls(
    dataloader: torch.utils.data.DataLoader,
    model,
    clip_preprocess,
    resolution: int = 256,
    device: str = "cuda",
) -> np.ndarray:
    """Compute Clip vision CLS latents."""
    latents = []

    model = model.to(device).float()

    def process_frame(frame):
        img = Image.fromarray(frame.astype(np.uint8))
        res = image_processor(
            img, resolution=resolution, preprocess=clip_preprocess
        )
        img = res["x_clip"].unsqueeze(0).to(device).float()

        clip_latents = model.encode_image(img)
        clip_latents /= clip_latents.norm(dim=-1, keepdim=True)

        return clip_latents.squeeze(0).detach().cpu().numpy()

    for batch in tqdm(dataloader):
        if len(batch.shape) == 5:
            for tr in batch:
                tr_latents = []
                for frame in tr:
                    tr_latents.append(process_frame(frame.numpy()))
                # Compute average latent represention for each TR
                latents.append(np.mean(np.stack(tr_latents), axis=0))
        elif len(batch.shape) == 4:
            for frame in batch:
                latents.append(process_frame(frame.numpy()))

    if len(latents) > 0:
        # Concatenate batches
        latents = np.stack(latents)

    return latents


# STABLE DIFFUSION AUTOKL


def compute_sd_autokl(
    dataset_id,
    dataset_path,
    subject,
    model_path=None,
    seed=0,
    batch_size=32,
):
    torch.manual_seed(seed)

    fmri_dataset = load_fmridataset(dataset_id, dataset_path, subject)
    _, clip_preprocess = clip.load("ViT-L/14", device="cuda", jit=False)
    model = load_stablediffusion_model(model_path)
    stimulus_key = "images"

    dataloader = DataLoader(
        fmri_dataset,
        batch_size=batch_size,
        collate_fn=get_collate_fn(stimulus_key),
    )

    latents = extract_sd_autokl(dataloader, model, clip_preprocess)
    metadata = None

    return latents, metadata


def load_stablediffusion_model(
    pretrained_models_path: tp.Union[str, Path],
) -> any:
    """Load pretrained stable diffusion."""
    logger.info("Loading stable-diffusion model...")

    p = Path(pretrained_models_path)
    model = instantiate_from_config(
        OmegaConf.load(p / "v1-inference.yaml").model
    )
    model.load_state_dict(
        torch.load(p / "sd-v1-3.ckpt", map_location="cpu")["state_dict"],
        strict=False,
    )

    return model


@torch.no_grad()
def extract_sd_autokl(
    dataloader: torch.utils.data.DataLoader,
    model,
    clip_preprocess,
    resolution: int = 256,
    device: str = "cuda",
) -> np.ndarray:
    """Compute stable diffusion autokl latents."""
    latents = []

    model.first_stage_model = model.first_stage_model.to(device)

    for batch in tqdm(dataloader):
        if len(batch.shape) == 5:
            for tr in batch:
                tr_latents = []
                for frame in tr:
                    img = Image.fromarray(frame.numpy().astype(np.uint8))
                    res = image_processor(
                        img, resolution=resolution, preprocess=clip_preprocess
                    )
                    img = res["x"].unsqueeze(0).to(device).float()

                    autokl_latents = model.get_first_stage_encoding(
                        model.encode_first_stage(img)
                    )
                    tr_latents.append(autokl_latents.detach().cpu().numpy())
                latents.append(np.mean(np.stack(tr_latents), axis=0))
        elif len(batch.shape) == 4:
            for frame in batch:
                img = Image.fromarray(frame.numpy().astype(np.uint8))
                res = image_processor(
                    img, resolution=resolution, preprocess=clip_preprocess
                )
                img = res["x"].unsqueeze(0).to(device).float()

                autokl_latents = model.get_first_stage_encoding(
                    model.encode_first_stage(img)
                )
                latents.append(autokl_latents.detach().cpu().numpy())

    # Check that lists are not empty
    # (validation set can be empty for instance)
    if len(latents) > 0:
        # Concatenate batches
        latents = np.stack(latents)

        # Flatten latents for each sample
        n_samples = latents.shape[0]
        latents = latents.reshape(n_samples, -1)

    return latents


def image_processor(
    data, resolution, preprocess, skip_crop=False, caption_preprocess=None
):
    if isinstance(data, PIL.Image.Image):
        pil_image = data
    else:
        with io.BytesIO(data) as stream:
            pil_image = PIL.Image.open(stream)
            pil_image.load()

    pil_image = pil_image.convert("RGB")

    # remove padding
    # arr = np.array(pil_image)
    # pad_rows = (255 - arr).sum(axis=1).sum(axis=1) != 0
    # pad_cols = (255 - arr).sum(axis=0).sum(axis=1) != 0
    # bbox = (pad_cols.argmax(), pad_rows.argmax(), len(pad_cols) - pad_cols[::-1].argmax(), len(pad_rows) - pad_rows[::-1].argmax())
    # pil_image = pil_image.crop(bbox) if (bbox[0] != bbox[2] and bbox[1] != bbox[3]) else pil_image

    pil_image_clip = pil_image.copy()

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * resolution:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=PIL.Image.BOX
        )
    scale = resolution / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=PIL.Image.BICUBIC,
    )
    arr = np.array(pil_image)
    if not skip_crop:
        crop_y = (arr.shape[0] - resolution) // 2
        crop_x = (arr.shape[1] - resolution) // 2
        arr = arr[crop_y : crop_y + resolution, crop_x : crop_x + resolution]
    arr = arr.astype(np.float32) / 127.5 - 1

    clip_resolution = preprocess.transforms[0].size
    while min(*pil_image_clip.size) >= 2 * clip_resolution:
        pil_image_clip = pil_image_clip.resize(
            tuple(x // 2 for x in pil_image_clip.size), resample=PIL.Image.BOX
        )
    scale = clip_resolution / min(*pil_image_clip.size)
    pil_image_clip = pil_image_clip.resize(
        tuple(round(x * scale) for x in pil_image_clip.size),
        resample=PIL.Image.BICUBIC,
    )
    arr_clip = preprocess(pil_image_clip)

    result = {
        "x": torch.from_numpy(arr).permute([2, 0, 1]),
        "x_clip": arr_clip,
    }

    if caption_preprocess is not None:
        result["x_caption"] = caption_preprocess(pil_image_clip)

    return result
