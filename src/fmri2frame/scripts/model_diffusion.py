import sys
import typing as tp
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL.Image import Image as pil_image_type
from torch.utils.data import DataLoader

from fmri2frame.scripts.utils import convert_to_PIL, regularize_image

# Imported from Versatile-Diffusion
# https://github.com/SHI-Labs/Versatile-Diffusion
# lib_path = Path("/storage/store2/work/athual/repo/Versatile-Diffusion")
lib_path = Path("/gpfswork/rech/nry/uul79xi/repo/Versatile-Diffusion")
assert lib_path.exists()
sys.path.append(str(lib_path))
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD


class ClipImageBatchGenerator(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images

    def __getitem__(self, idx):
        return VersatileDiffusionModel.prepare_image(self.images[idx])

    def __len__(self):
        return len(self.images)


class VersatileDiffusionModel:
    def __init__(self, path="."):
        cfgm_name = "vd_noema"
        self.sampler = DDIMSampler_VD
        # self.pth = (
        #     "versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth"
        # )
        self.cfgm = model_cfg_bank(path)(cfgm_name)
        self.net = get_model()(self.cfgm)
        self.sd = torch.load(Path(path) / "pretrained/vd-four-flow-v1-0-fp16-deprecated.pth", map_location="cpu")
        self.net.load_state_dict(self.sd, strict=False)

        self.net.clip.cuda(0)
        self.net.autokl.cuda(0)

        self.sampler = self.sampler(self.net)
        self.batch_size = 1
        self.strength = 0.75
        self.mixing = 0.4

        self.n_samples = 1
        self.ddim_steps = 50
        self.ddim_eta = 0
        self.scale = 7.5
        self.xtype = "image"
        self.ctype = "prompt"
        self.net.autokl.half()

    def _normalize_clip_image_for_versatile(
        self, clip_images: tp.Union[torch.tensor, np.ndarray]
    ):
        clip_images_flattened = clip_images.reshape(len(clip_images), -1)
        clip_images_reshaped = clip_images.reshape(-1, 257, 768)
        return (
            clip_images_flattened / clip_images_reshaped[:, 0].norm(dim=1, keepdim=True)
        ).reshape(-1, 257, 768)

    def reconstruct_from_clip_and_vdvae_lowlevel(
        self,
        vdvae_low_level_recons: tp.List[pil_image_type],
        clip_image: np.ndarray,
        clip_text: np.ndarray = None,
        normalize_clip_image_for_versatile=True,
        save_to_path: tp.Optional[str] = None,
        image_names: tp.Optional[tp.List[str]] = None,
        seed: int = 0,
    ) -> tp.List[pil_image_type]:
        """ """
        torch.manual_seed(seed)
        assert clip_image.shape[0] == len(vdvae_low_level_recons)
        if clip_text is not None:
            assert clip_text.shape[0] == clip_image.shape[0]

        clip_image = torch.tensor(clip_image).half().cuda(1)
        if clip_text is not None:
            clip_text = torch.tensor(clip_text).half().cuda(1)

        if normalize_clip_image_for_versatile:
            clip_image = self._normalize_clip_image_for_versatile(clip_image)

        recons = []
        for im_id in range(len(vdvae_low_level_recons)):
            zim = vdvae_low_level_recons[im_id]

            zim = regularize_image(zim)
            zin = zim * 2 - 1
            zin = zin.unsqueeze(0).cuda(0).half()

            init_latent = self.net.autokl_encode(zin)

            self.sampler.make_schedule(
                ddim_num_steps=self.ddim_steps,
                ddim_eta=self.ddim_eta,
                verbose=False,
            )
            # strength=0.75
            assert (
                0.0 <= self.strength <= 1.0
            ), "can only work with strength in [0.0, 1.0]"
            t_enc = int(self.strength * self.ddim_steps)
            device = "cuda:0"
            z_enc = self.sampler.stochastic_encode(
                init_latent, torch.tensor([t_enc]).to(device)
            )

            dummy = ""
            utx = self.net.clip_encode_text(dummy)
            utx = utx.cuda(1).half()

            dummy = torch.zeros((1, 3, 224, 224)).cuda(0)
            uim = self.net.clip_encode_vision(dummy)
            uim = uim.cuda(1).half()

            z_enc = z_enc.cuda(1)

            cim = clip_image[im_id].unsqueeze(0)
            if clip_text is not None:
                ctx = clip_text[im_id].unsqueeze(0)

            self.sampler.model.model.diffusion_model.device = "cuda:1"
            self.sampler.model.model.diffusion_model.half().cuda(1)

            if clip_text is not None:
                z = self.sampler.decode_dc(
                    x_latent=z_enc,
                    first_conditioning=[uim, cim],
                    second_conditioning=[utx, ctx],
                    t_start=t_enc,
                    unconditional_guidance_scale=self.scale,
                    xtype="image",
                    first_ctype="vision",
                    second_ctype="prompt",
                    mixed_ratio=(1 - self.mixing),
                )
            else:
                z = self.sampler.decode(
                    x_latent=z_enc,
                    cond=cim,
                    t_start=t_enc,
                    unconditional_guidance_scale=self.scale,
                    unconditional_conditioning=uim,
                    xtype="image",
                    ctype="vision",
                )

            z = z.cuda(0).half()
            x = self.net.autokl_decode(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = [T.ToPILImage()(xi) for xi in x]

            if save_to_path is not None:
                x[0].save(
                    Path(save_to_path)
                    / f"{im_id if image_names is None else image_names[im_id]}.png"
                )
            recons.append(x[0])
        return recons

    def reconstruct_from_clip_and_autokl(
        self,
        autokl_latents: np.ndarray,
        clip_image: np.ndarray,
        clip_text: np.ndarray,
        normalize_clip_image_for_versatile=True,
        save_to_path: tp.Optional[str] = None,
        image_names: tp.Optional[tp.List[str]] = None,
        seed: int = 0,
    ) -> tp.List[pil_image_type]:
        torch.manual_seed(seed)
        assert clip_text.shape[0] == clip_image.shape[0] == len(autokl_latents)

        clip_text = torch.tensor(clip_text).half().cuda(1)
        clip_image = torch.tensor(clip_image).half().cuda(1)

        if normalize_clip_image_for_versatile:
            clip_image = self._normalize_clip_image_for_versatile(clip_image)

        recons = []
        autokl_latents = torch.from_numpy(autokl_latents)
        for im_id in range(len(autokl_latents)):
            init_latent = autokl_latents[im_id].cuda(0).half()

            self.sampler.make_schedule(
                ddim_num_steps=self.ddim_steps,
                ddim_eta=self.ddim_eta,
                verbose=False,
            )
            # strength=0.75
            assert (
                0.0 <= self.strength <= 1.0
            ), "can only work with strength in [0.0, 1.0]"
            t_enc = int(self.strength * self.ddim_steps)
            device = "cuda:0"
            z_enc = self.sampler.stochastic_encode(
                init_latent, torch.tensor([t_enc]).to(device)
            )

            dummy = ""
            utx = self.net.clip_encode_text(dummy)
            utx = utx.cuda(1).half()

            dummy = torch.zeros((1, 3, 224, 224)).cuda(0)
            uim = self.net.clip_encode_vision(dummy)
            uim = uim.cuda(1).half()

            z_enc = z_enc.cuda(1)

            cim = clip_image[im_id].unsqueeze(0)
            ctx = clip_text[im_id].unsqueeze(0)

            self.sampler.model.model.diffusion_model.device = "cuda:1"
            self.sampler.model.model.diffusion_model.half().cuda(1)

            z = self.sampler.decode_dc(
                x_latent=z_enc,
                first_conditioning=[uim, cim],
                second_conditioning=[utx, ctx],
                t_start=t_enc,
                unconditional_guidance_scale=self.scale,
                xtype="image",
                first_ctype="vision",
                second_ctype="prompt",
                mixed_ratio=(1 - self.mixing),
            )

            z = z.cuda(0).half()
            x = self.net.autokl_decode(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = [T.ToPILImage()(xi) for xi in x]

            if save_to_path is not None:
                x[0].save(
                    Path(save_to_path)
                    / f"{im_id if image_names is None else image_names[im_id]}.png"
                )
            recons.append(x[0])
        return recons

    def get_clip_image(self, tensor_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - tensor_images: a tensor of shape (bs, c, h, w) corresponding to images with values scaled in [-1,1]
        Return:
            - a tensor of shape (bs, 257, 768) corresponding to the CLIP-image (vision) embeddings of the input `tensor_images`
        """
        return self.net.clip_encode_vision(tensor_images)

    def get_clip_image_from_images(
        self,
        pil_images: tp.List,
        batch_size=1,
        save_to_path: tp.Optional[str] = None,
    ) -> np.ndarray:
        """
        Args
            - images: a list of N RGB PIL.Image or corresponding np.ndarray (h,w,c)
            - save_to_path: if not None, a .npy will be created with the result at this path
        Return
            - an array of shape (N,257,768) made of the clip image embeddings of the input N images
        """
        # Fixme: allow to take a list or an array of images as input
        # make a list of corresponding PIL.Image images
        prepared_images = convert_to_PIL(pil_images)

        img_dataset = ClipImageBatchGenerator(prepared_images)
        # return self.get_clip_image_from_tensors(img_dataset)

        img_dl = DataLoader(img_dataset, batch_size, shuffle=False)
        num_embed, num_features, num_images = 257, 768, len(prepared_images)

        clip_img = np.zeros((num_images, num_embed, num_features))

        with torch.no_grad():
            for i, cin in enumerate(img_dl):
                c = self.get_clip_image(cin)
                # the norm of the CLS token is 1
                clip_img[i] = c[0].cpu().numpy()

        if save_to_path is not None:
            np.save(save_to_path, clip_img)
        return clip_img

    def get_clip_text(
        self,
        captions: np.array,
    ) -> torch.Tensor:
        """
        Args
            - captions: a string array of shape (bs, *), each line containing captions for a same image
        Return
            - A tensor of CLIP-text embeddings of shape (bs,77,768),
            corresponding to the mean CLIP-text embeddings of all captions for a single image
        """
        clip_txt = torch.zeros((len(captions), 77, 768))
        with torch.no_grad():
            for i, annots in enumerate(captions):
                cin = list(annots[annots != ""])
                # The norm of the CLS token is not necessarily 1
                clip_txt[i] = self.net.clip_encode_text(cin).mean(0)
        return clip_txt

    def get_clip_text_from_captions_list(
        self,
        captions: tp.List[tp.List[str]],
        save_to_path: tp.Optional[str] = None,
    ) -> np.ndarray:
        """
        Args
            - images: a list of N lists of (str) captions
            - save_to_path: if not None, a .npy will be created with the result at this path
        Return
            - an array of shape (N,77,768) made of the clip text embeddings of the input N images
        """
        captions = np.array([np.array(caps) for caps in captions], dtype=object)
        clip_txt = self.get_clip_text(captions).cpu().numpy()
        if save_to_path is not None:
            np.save(save_to_path, clip_txt)
        return clip_txt

    def get_autokl_latents_from_images(
        self,
        images: tp.List,
        use_half_precision: bool = True,
        device: str = "cuda:0",
        save_to_path: tp.Optional[str] = None,
    ) -> np.ndarray:
        """
        Extract Versatile diffusion's Autoencoder latent for each input (PIL) image
        - images: an iterator of images in the form of paths / PIL / arrays (that will be converted to PIL images)

        Returns an array of Versatile diffusion's Autoencoder latents (each of shape (4,64,64))
        """
        images = convert_to_PIL(images)

        latents = []
        for im in images:
            zim = im.resize((512, 512), resample=3)
            zim = T.ToTensor()(zim)
            zin = zim * 2 - 1
            zin = zin.unsqueeze(0)
            latents.append(
                self.get_autokl_latents(
                    zin, use_half_precision=use_half_precision, device=device
                )
            )
        latents = torch.stack(latents).cpu().numpy()
        if save_to_path is not None:
            np.save(save_to_path, latents)
        return latents

    def get_autokl_latents(
        self,
        tensor_images: torch.Tensor,
        use_half_precision: bool = True,
        device: str = "cuda:0",
    ) -> torch.Tensor:
        """
        Args
            - tensor_images: a tensor of shape (bs, 3, 512, 512) corresponding to bs 512x512 RGB images,
          with values scaled in (-1,1)
        Return
            - a tensor of shape (bs, 4, 64, 64) of Versatile diffusion's Autoencoder bottleneck latents
        """
        if use_half_precision:
            tensor_images = tensor_images.half()
        return self.net.autokl_encode(tensor_images.to(device))

    @staticmethod
    def prepare_images(images):
        images = convert_to_PIL(images)
        return [VersatileDiffusionModel.prepare_image(img) for img in images]

    @staticmethod
    def prepare_image(image):
        image = T.functional.resize(image, (512, 512))
        image = T.functional.to_tensor(
            image
        ).float()  # each component is between 0 and 1
        return image * 2 - 1  # each component is between -1 and 1
