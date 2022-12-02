from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTextModel
from utils import compile_through_fx
from stable_args import args
import torch

BATCH_SIZE = len(args.prompts)

model_config = {
    "v2": "stabilityai/stable-diffusion-2",
    "v1.4": "CompVis/stable-diffusion-v1-4",
}

model_input = {
    "v2": {
        "clip": (torch.randint(1, 2, (1, 77)),),
        "vae": (torch.randn(1, 4, 96, 96),),
        "unet": (
            torch.randn(1, 4, 96, 96),  # latents
            torch.tensor([1]).to(torch.float32),  # timestep
            torch.randn(2, 77, 1024),  # embedding
            torch.tensor(1).to(torch.float32),  # guidance_scale
        ),
    },
    "v1.4": {
        "clip": (torch.randint(1, 2, (1, 77)),),
        "vae": (torch.randn(1, 4, 64, 64),),
        "unet": (
            torch.randn(1, 4, 64, 64),
            torch.tensor([1]).to(torch.float32),  # timestep
            torch.randn(2, 77, 768),
            torch.tensor(1).to(torch.float32),
        ),
    },
}


def get_clip_mlir(model_name="clip_text", extra_args=[]):
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14"
    )
    if args.version == "v2":
        text_encoder = CLIPTextModel.from_pretrained(
            model_config[args.version], subfolder="text_encoder"
        )

    class CLIPText(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = text_encoder

        def forward(self, input):
            return self.text_encoder(input)[0]

    clip_model = CLIPText()
    shark_clip = compile_through_fx(
        clip_model,
        model_input[args.version]["clip"],
        model_name=model_name,
        extra_args=extra_args,
    )
    return shark_clip


def get_vae_mlir(model_name="vae", extra_args=[]):
    class VaeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = AutoencoderKL.from_pretrained(
                model_config[args.version],
                subfolder="vae",
                revision="fp16" if args.precision == "fp16" else "main",
            )

        def forward(self, input):
            x = self.vae.decode(input, return_dict=False)[0]
            return (x / 2 + 0.5).clamp(0, 1)

    vae = VaeModel()
    if args.precision == "fp16":
        vae = vae.half().cuda()
        inputs = tuple(
            [inputs.half().cuda() for inputs in model_input[args.version]["vae"]]
        )
    else:
        inputs = model_input[args.version]["vae"]
    shark_vae = compile_through_fx(
        vae,
        inputs,
        model_name=model_name,
        extra_args=extra_args,
    )
    return shark_vae


def get_unet_mlir(model_name="unet", extra_args=[]):
    class UnetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unet = UNet2DConditionModel.from_pretrained(
                model_config[args.version],
                subfolder="unet",
                revision="fp16" if args.precision == "fp16" else "main",
            )
            self.in_channels = self.unet.in_channels
            self.train(False)
            if args.enable_attention_slice:
                if args.attention_slice_size == 0:
                    if isinstance(self.unet.config.attention_head_dim, int):
                        attention_slice_size = self.unet.config.attention_head_dim // 2
                    else:
                        attention_slice_size = min(self.unet.config.attention_head_dim)
                print(f"[DEBUG] num heads = {self.unet.config.attention_head_dim}   || slice size = {attention_slice_size}")
                self.unet.set_attention_slice(attention_slice_size)

        def forward(self, latent, timestep, text_embedding, guidance_scale):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latents = torch.cat([latent] * 2)
            unet_out = self.unet.forward(
                latents, timestep, text_embedding, return_dict=False
            )[0]
            noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            return noise_pred

    unet = UnetModel()
    if args.precision == "fp16":
        unet = unet.half().cuda()
        inputs = tuple(
            [
                inputs.half().cuda() if len(inputs.shape) != 0 else inputs
                for inputs in model_input[args.version]["unet"]
            ]
        )
    else:
        inputs = model_input[args.version]["unet"]


    shark_unet = compile_through_fx(
        unet,
        inputs,
        model_name=model_name,
        extra_args=extra_args,
    )
    return shark_unet
