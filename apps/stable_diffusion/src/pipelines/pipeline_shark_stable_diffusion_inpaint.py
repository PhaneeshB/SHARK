import torch
from tqdm.auto import tqdm
import numpy as np
from random import randint
from PIL import Image
from transformers import CLIPTokenizer
from typing import Union
from shark.shark_inference import SharkInference
from diffusers import (
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from apps.stable_diffusion.src.schedulers import SharkEulerDiscreteScheduler
from apps.stable_diffusion.src.pipelines.pipeline_shark_stable_diffusion_utils import (
    StableDiffusionPipeline,
)
import scipy
import skimage


def edge_pad(img, mask):
    record = {}
    kernel = [[1] * 3 for _ in range(3)]
    nmask = mask.copy()
    nmask[nmask > 0] = 1
    res = scipy.signal.convolve2d(
        nmask, kernel, mode="same", boundary="fill", fillvalue=1
    )
    res[nmask < 1] = 0
    res[res == 9] = 0
    res[res > 0] = 1
    ylst, xlst = res.nonzero()
    queue = [(y, x) for y, x in zip(ylst, xlst)]
    # bfs here
    cnt = res.astype(np.float32)
    acc = img.astype(np.float32)
    step = 1
    h = acc.shape[0]
    w = acc.shape[1]
    offset = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        target = []
        for y, x in queue:
            val = acc[y][x]
            for yo, xo in offset:
                yn = y + yo
                xn = x + xo
                if 0 <= yn < h and 0 <= xn < w and nmask[yn][xn] < 1:
                    if record.get((yn, xn), step) == step:
                        acc[yn][xn] = acc[yn][xn] * cnt[yn][xn] + val
                        cnt[yn][xn] += 1
                        acc[yn][xn] /= cnt[yn][xn]
                        if (yn, xn) not in record:
                            record[(yn, xn)] = step
                            target.append((yn, xn))
        step += 1
        queue = target
    img = acc.astype(np.uint8)
    return img, mask


def prepare_mask_and_masked_image(image):
    width, height = image.size
    sel_buffer = np.array(image)
    img = sel_buffer[:, :, 0:3]
    mask = sel_buffer[:, :, -1]

    img, mask = edge_pad(img, mask)
    mask = 255 - mask
    mask = skimage.measure.block_reduce(mask, (8, 8), np.max)
    mask = mask.repeat(8, axis=0).repeat(8, axis=1)
    init_image = Image.fromarray(img)
    mask_image = Image.fromarray(mask)
    input_image = init_image.resize(
        (width, height), resample=Image.Resampling.LANCZOS
    )
    mask_image = mask_image.resize((width, height))

    image = input_image
    mask = mask_image

    # preprocess image
    if isinstance(image, (Image.Image, np.ndarray)):
        image = [image]

    if isinstance(image, list) and isinstance(image[0], Image.Image):
        image = [np.array(i.convert("RGB"))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
    elif isinstance(image, list) and isinstance(image[0], np.ndarray):
        image = np.concatenate([i[None, :] for i in image], axis=0)

    image = image.transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    # preprocess mask
    if isinstance(mask, (Image.Image, np.ndarray)):
        mask = [mask]

    if isinstance(mask, list) and isinstance(mask[0], Image.Image):
        mask = np.concatenate(
            [np.array(m.convert("L"))[None, None, :] for m in mask], axis=0
        )
        mask = mask.astype(np.float32) / 255.0
    elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
        mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image


class InpaintPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae_encode: SharkInference,
        vae: SharkInference,
        text_encoder: SharkInference,
        tokenizer: CLIPTokenizer,
        unet: SharkInference,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
            SharkEulerDiscreteScheduler,
        ],
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler)
        self.vae_encode = vae_encode

    def prepare_latents(
        self,
        batch_size,
        height,
        width,
        generator,
        num_inference_steps,
        dtype,
    ):
        latents = torch.randn(
            (
                batch_size,
                4,
                height // 8,
                width // 8,
            ),
            generator=generator,
            dtype=torch.float32,
        ).to(dtype)

        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.is_scale_input_called = True
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        batch_size,
        height,
        width,
        dtype,
    ):
        mask = torch.nn.functional.interpolate(
            mask, size=(height // 8, width // 8)
        )
        mask = mask.to(dtype)

        masked_image = masked_image.to(dtype)
        masked_image_latents = self.vae_encode("forward", (masked_image,))
        masked_image_latents = torch.from_numpy(masked_image_latents)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(
                batch_size // masked_image_latents.shape[0], 1, 1, 1
            )
        return mask, masked_image_latents

    def generate_images(
        self,
        prompts,
        neg_prompts,
        image,
        batch_size,
        height,
        width,
        num_inference_steps,
        guidance_scale,
        seed,
        max_length,
        dtype,
        use_base_vae,
        cpu_scheduling,
    ):
        # prompts and negative prompts must be a list.
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(neg_prompts, str):
            neg_prompts = [neg_prompts]

        prompts = prompts * batch_size
        neg_prompts = neg_prompts * batch_size

        # seed generator to create the inital latent noise. Also handle out of range seeds.
        uint32_info = np.iinfo(np.uint32)
        uint32_min, uint32_max = uint32_info.min, uint32_info.max
        if seed < uint32_min or seed >= uint32_max:
            seed = randint(uint32_min, uint32_max)
        generator = torch.manual_seed(seed)

        # Get initial latents
        init_latents = self.prepare_latents(
            batch_size=batch_size,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_inference_steps,
            dtype=dtype,
        )

        # Get text embeddings from prompts
        text_embeddings = self.encode_prompts(prompts, neg_prompts, max_length)

        # guidance scale as a float32 tensor.
        guidance_scale = torch.tensor(guidance_scale).to(torch.float32)

        # Preprocess mask and image
        mask, masked_image = prepare_mask_and_masked_image(image)

        # Prepare mask latent variables
        mask, masked_image_latents = self.prepare_mask_latents(
            mask=mask,
            masked_image=masked_image,
            batch_size=batch_size,
            height=height,
            width=width,
            dtype=dtype,
        )

        # Get Image latents
        latents = self.produce_img_latents(
            latents=init_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale,
            total_timesteps=self.scheduler.timesteps,
            dtype=dtype,
            cpu_scheduling=cpu_scheduling,
            mask=mask,
            masked_image_latents=masked_image_latents,
        )

        # Img latents -> PIL images
        all_imgs = []
        for i in tqdm(range(0, latents.shape[0], batch_size)):
            imgs = self.decode_latents(
                latents=latents[i : i + batch_size],
                use_base_vae=use_base_vae,
                cpu_scheduling=cpu_scheduling,
            )
            all_imgs.extend(imgs)

        return all_imgs
