import sys
from model_wrappers import (
    get_vae_mlir,
    get_vae_encode_mlir,
    get_unet_mlir,
    get_clip_mlir,
)
from stable_args import args
from utils import get_shark_model

BATCH_SIZE = len(args.prompts)
if BATCH_SIZE != 1:
    sys.exit("Only batch size 1 is supported.")

def _get_mlir_model_name(model):
    date = "15dec"
    version = args.version.replace('.', '-')
    precision = args.precision if model != "clip" else "fp32"
    prompt_max_len = f"prompt_len_{args.max_length}"
    name = '_'.join([model, date, version, precision, prompt_max_len])
    return name

def get_unet():
    iree_flags = []
    if len(args.iree_vulkan_target_triple) > 0:
        iree_flags.append(
            f"-iree-vulkan-target-triple={args.iree_vulkan_target_triple}"
        )
    # Tuned model is present for `fp16` precision.
    if args.precision == "fp16":
        if args.use_tuned:
            bucket = "gs://shark_tank/vivian"
            if args.version == "v1.4":
                model_name = "unet_1dec_fp16_tuned"
            if args.version == "v2.1base":
                model_name = "unet2base_8dec_fp16_tuned"
            return get_shark_model(bucket, model_name, iree_flags)
        else:
            bucket = "gs://shark_tank/stable_diffusion"
            model_name = "unet_8dec_fp16"
            if args.version == "v2.1base":
                model_name = "unet2base_8dec_fp16"
            if args.version == "v2.1":
                model_name = "unet2_14dec_fp16"
            iree_flags += [
                "--iree-flow-enable-padding-linalg-ops",
                "--iree-flow-linalg-ops-padding-size=32",
                "--iree-flow-enable-conv-img2col-transform",
            ]
            if args.import_mlir:
                model_name = _get_mlir_model_name("unet")
                return get_unet_mlir(model_name, iree_flags)
            return get_shark_model(bucket, model_name, iree_flags)

    # Tuned model is not present for `fp32` case.
    if args.precision == "fp32":
        bucket = "gs://shark_tank/stable_diffusion"
        model_name = "unet_1dec_fp32"
        iree_flags += [
            "--iree-flow-enable-conv-nchw-to-nhwc-transform",
            "--iree-flow-enable-padding-linalg-ops",
            "--iree-flow-linalg-ops-padding-size=16",
        ]
        if args.import_mlir:
            model_name = _get_mlir_model_name("unet")
            return get_unet_mlir(model_name, iree_flags)
        return get_shark_model(bucket, model_name, iree_flags)

    if args.precision == "int8":
        bucket = "gs://shark_tank/prashant_nod"
        model_name = "unet_int8"
        iree_flags += [
            "--iree-flow-enable-padding-linalg-ops",
            "--iree-flow-linalg-ops-padding-size=32",
        ]
        sys.exit("int8 model is currently in maintenance.")
        # # TODO: Pass iree_flags to the exported model.
        # if args.import_mlir:
        # sys.exit(
        # "--import_mlir is not supported for the int8 model, try --no-import_mlir flag."
        # )
        # return get_shark_model(bucket, model_name, iree_flags)


def get_vae():
    iree_flags = []
    if len(args.iree_vulkan_target_triple) > 0:
        iree_flags.append(
            f"-iree-vulkan-target-triple={args.iree_vulkan_target_triple}"
        )
    if args.precision in ["fp16", "int8"]:
        bucket = "gs://shark_tank/stable_diffusion"
        model_name = "vae_8dec_fp16"
        if args.version == "v2.1base":
            model_name = "vae2base_8dec_fp16"
        if args.version == "v2.1":
            model_name = "vae2_14dec_fp16"
        iree_flags += [
            "--iree-flow-enable-padding-linalg-ops",
            "--iree-flow-linalg-ops-padding-size=32",
            "--iree-flow-enable-conv-img2col-transform",
        ]
        if args.import_mlir:
            model_name = _get_mlir_model_name("vae")
            return get_vae_mlir(model_name, iree_flags)
        return get_shark_model(bucket, model_name, iree_flags)

    if args.precision == "fp32":
        bucket = "gs://shark_tank/stable_diffusion"
        model_name = "vae_1dec_fp32"
        iree_flags += [
            "--iree-flow-enable-conv-nchw-to-nhwc-transform",
            "--iree-flow-enable-padding-linalg-ops",
            "--iree-flow-linalg-ops-padding-size=16",
        ]
        if args.import_mlir:
            model_name = _get_mlir_model_name("vae")
            return get_vae_mlir(model_name, iree_flags)
        return get_shark_model(bucket, model_name, iree_flags)


def get_vae_encode():
    iree_flags = []
    if len(args.iree_vulkan_target_triple) > 0:
        iree_flags.append(
            f"-iree-vulkan-target-triple={args.iree_vulkan_target_triple}"
        )
    if args.precision in ["fp16", "int8"]:
        bucket = "gs://shark_tank/stable_diffusion"
        model_name = "vae_encode_1dec_fp16"
        if args.version == "v2":
            model_name = "vae2_encode_29nov_fp16"
        iree_flags += [
            "--iree-flow-enable-conv-nchw-to-nhwc-transform",
            "--iree-flow-enable-padding-linalg-ops",
            "--iree-flow-linalg-ops-padding-size=32",
        ]
        if args.import_mlir:
            model_name = _get_mlir_model_name("vae_encode")
            return get_vae_encode_mlir(model_name, iree_flags)
        return get_shark_model(bucket, model_name, iree_flags)

    if args.precision == "fp32":
        bucket = "gs://shark_tank/stable_diffusion"
        model_name = "vae_encode_1dec_fp32"
        iree_flags += [
            "--iree-flow-enable-conv-nchw-to-nhwc-transform",
            "--iree-flow-enable-padding-linalg-ops",
            "--iree-flow-linalg-ops-padding-size=16",
        ]
        if args.import_mlir:
            model_name = _get_mlir_model_name("vae_encode")
            return get_vae_mlir(model_name, iree_flags)
        return get_shark_model(bucket, model_name, iree_flags)


def get_clip():
    iree_flags = []
    if len(args.iree_vulkan_target_triple) > 0:
        iree_flags.append(
            f"-iree-vulkan-target-triple={args.iree_vulkan_target_triple}"
        )
    bucket = "gs://shark_tank/stable_diffusion"
    model_name = "clip_8dec_fp32"
    if args.version == "v2.1base":
        model_name = "clip2base_8dec_fp32"
    if args.version == "v2.1":
        model_name = "clip2_14dec_fp32"
    iree_flags += [
        "--iree-flow-linalg-ops-padding-size=16",
        "--iree-flow-enable-padding-linalg-ops",
    ]
    if args.import_mlir:
        model_name = _get_mlir_model_name("clip")
        return get_clip_mlir(model_name, iree_flags)
    return get_shark_model(bucket, model_name, iree_flags)
