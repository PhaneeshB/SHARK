import torch
from opt_params import get_vae
from PIL import Image
from diffusers import AutoencoderKL
from shark.shark_inference import SharkInference
from stable_args import args
from shark.shark_importer import import_with_fx
import os

YOUR_TOKEN = "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"

def _compile_module(shark_module, model_name, extra_args=[]):
    if args.load_vmfb or args.save_vmfb:
        extended_name = "{}_{}".format(model_name, args.device)
        vmfb_path = os.path.join(os.getcwd(), extended_name + ".vmfb")

        if args.load_vmfb and os.path.isfile(vmfb_path) and not args.save_vmfb:
            print("Loading flatbuffer from {}".format(vmfb_path))
            shark_module.load_module(vmfb_path)
        else:
            if args.save_vmfb:
                print("Saving to {}".format(vmfb_path))
            else:
                print(f"No vmfb found. Compiling and saving to {vmfb_path}")
            path = shark_module.save_module(
                os.getcwd(), extended_name, extra_args
            )
            shark_module.load_module(path)
    else:
        shark_module.compile(extra_args)
    return shark_module

# Converts the torch-module into shark_module.
def compile_through_fx(model, inputs, model_name, extra_args=[]):

    mlir_module, func_name = import_with_fx(model, inputs)

    shark_module = SharkInference(
        mlir_module,
        func_name,
        device=args.device,
        mlir_dialect="linalg",
    )

    return _compile_module(shark_module, model_name, extra_args)

def get_vae32(model_name="vae_fp32"):
    class VaeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = AutoencoderKL.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="vae",
                use_auth_token=YOUR_TOKEN,
            )

        def forward(self, input):
            x = self.vae.decode(input, return_dict=False)[0]
            return (x / 2 + 0.5).clamp(0, 1)

    vae = VaeModel()
    vae_input = torch.rand(1, 4, 64, 64)
    iree_flags = [
        # "--iree-flow-enable-conv-nchw-to-nhwc-transform",
        "--iree-flow-enable-padding-linalg-ops",
        "--iree-flow-linalg-ops-padding-size=16",
    ]
    shark_vae = compile_through_fx(
        vae,
        (vae_input,),
        model_name=model_name,
        extra_args = iree_flags,
    )
    return shark_vae


if __name__ == "__main__":

    # latents = torch.load("unet_latents_tensor.pt")
    latents = torch.rand(1, 4, 64, 64)
    # print(f"saved tensor: size = {latents.shape}")
    latents_numpy = latents.detach().numpy()
    vae = get_vae32("vae_fp32")
    print(f"get vae completed")
    image = vae.forward((latents_numpy,))
    print(f"vae forward done")
    image = torch.from_numpy(image)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")

    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].show()
    pil_images[0].save(f"vae_test_image_1.jpg")
