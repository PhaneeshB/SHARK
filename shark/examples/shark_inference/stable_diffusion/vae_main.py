from transformers import CLIPTextModel, CLIPTokenizer
import torch
from PIL import Image
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm
import numpy as np
from stable_args import args
from utils import get_shark_model, set_iree_runtime_flags
from opt_params import get_unet, get_vae, get_clip
import time


# Helper function to profile the vulkan device.
def start_profiling(file_path="foo.rdc", profiling_mode="queue"):
    if args.vulkan_debug_utils and "vulkan" in args.device:
        import iree

        print(f"Profiling and saving to {file_path}.")
        vulkan_device = iree.runtime.get_device(args.device)
        vulkan_device.begin_profiling(mode=profiling_mode, file_path=file_path)
        return vulkan_device
    return None


def end_profiling(device):
    if device:
        return device.end_profiling()


if __name__ == "__main__":

    dtype = torch.float32 if args.precision == "fp32" else torch.half

    prompt = args.prompts
    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion

    num_inference_steps = args.steps  # Number of denoising steps

    batch_size = len(prompt)

    set_iree_runtime_flags()

    start = time.time()

    # scale and decode the image latents with vae
    # latents = 1 / 0.18215 * latents
    # torch.save(latents, "unet_latents_tensor.pt")
    # print(f"saved tensor: size = {latents.shape}")
    latents = torch.rand(1, 4, 64, 64)
    latents_numpy = latents.detach().numpy()
    vae = get_vae()
    image = vae.forward((latents_numpy,))
    image = torch.from_numpy(image)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")

    print("Total image generation runtime (s): {}".format(time.time() - start))

    pil_images = [Image.fromarray(image) for image in images]
    for i in range(batch_size):
        pil_images[i].save(f"{args.prompts[i]}_{i}.jpg")
