import torch
import time
import numpy as np
from tqdm.auto import tqdm
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
from apps.stable_diffusion.src.utils import controlnet_hint_conversion


class StencilPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        controlnet: SharkInference,
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
        self.controlnet = controlnet

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

    def generate_images(
        self,
        prompts,
        neg_prompts,
        image,
        batch_size,
        height,
        width,
        num_inference_steps,
        strength,
        guidance_scale,
        seed,
        max_length,
        dtype,
        use_base_vae,
        cpu_scheduling,
        use_stencil,
    ):
        # Control Embedding check & conversion
        # TODO: 1. Change `num_images_per_prompt`.
        controlnet_hint_og, controlnet_hint = controlnet_hint_conversion(
            image, use_stencil, height, width, dtype, num_images_per_prompt=1
        )
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

        # Get text embeddings from prompts
        text_embeddings = self.encode_prompts(prompts, neg_prompts, max_length)

        # guidance scale as a float32 tensor.
        guidance_scale = torch.tensor(guidance_scale).to(torch.float32)

        # Prepare initial latent.
        init_latents = self.prepare_latents(
            batch_size=batch_size,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_inference_steps,
            dtype=dtype,
        )
        final_timesteps = self.scheduler.timesteps

        # Get Image latents
        latents = self.produce_stencil_latents(
            latents=init_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale,
            total_timesteps=final_timesteps,
            dtype=dtype,
            cpu_scheduling=cpu_scheduling,
            controlnet_hint=controlnet_hint,
            controlnet=self.controlnet,
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
        all_imgs.append(controlnet_hint_og)

        return all_imgs
