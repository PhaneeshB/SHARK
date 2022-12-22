import os

import torch
from shark.shark_inference import SharkInference
from stable_args import args
from shark.shark_importer import import_with_fx
from shark.iree_utils.vulkan_utils import (
    set_iree_vulkan_runtime_flags,
    get_vulkan_target_triple,
)


def _compile_module(shark_module, model_name, extra_args=[]):
    if args.load_vmfb or args.save_vmfb:
        device = (
            args.device
            if "://" not in args.device
            else "-".join(args.device.split("://"))
        )
        extended_name = "{}_{}".format(model_name, device)
        vmfb_path = os.path.join(os.getcwd(), extended_name + ".vmfb")
        if args.load_vmfb and os.path.isfile(vmfb_path) and not args.save_vmfb:
            print(f"loading existing vmfb from: {vmfb_path}")
            shark_module.load_module(vmfb_path, extra_args=extra_args)
        else:
            if args.save_vmfb:
                print("Saving to {}".format(vmfb_path))
            else:
                print(
                    "No vmfb found. Compiling and saving to {}".format(
                        vmfb_path
                    )
                )
            path = shark_module.save_module(
                os.getcwd(), extended_name, extra_args
            )
            shark_module.load_module(path, extra_args=extra_args)
    else:
        shark_module.compile(extra_args)
    return shark_module


# Downloads the model from shark_tank and returns the shark_module.
def get_shark_model(tank_url, model_name, extra_args=[]):
    from shark.shark_downloader import download_model
    from shark.parser import shark_args

    # Set local shark_tank cache directory.
    shark_args.local_tank_cache = args.local_tank_cache

    mlir_model, func_name, inputs, golden_out = download_model(
        model_name,
        tank_url=tank_url,
        frontend="torch",
    )
    shark_module = SharkInference(
        mlir_model, func_name, device=args.device, mlir_dialect="linalg"
    )
    return _compile_module(shark_module, model_name, extra_args)


# Converts the torch-module into a shark_module.
def compile_through_fx(model, inputs, model_name, extra_args=[]):

    mlir_module, func_name = import_with_fx(model, inputs, debug=True, model_name=model_name)

    shark_module = SharkInference(
        mlir_module,
        func_name,
        device=args.device,
        mlir_dialect="linalg",
    )

    return _compile_module(shark_module, model_name, extra_args)


def set_iree_runtime_flags():

    vulkan_runtime_flags = [
        f"--vulkan_large_heap_block_size={args.vulkan_large_heap_block_size}",
        f"--vulkan_validation_layers={'true' if args.vulkan_validation_layers else 'false'}",
    ]
    if args.enable_rgp:
        vulkan_runtime_flags += [
            f"--enable_rgp=true",
            f"--vulkan_debug_utils=true",
        ]
    if "vulkan" in args.device:
        set_iree_vulkan_runtime_flags(flags=vulkan_runtime_flags)

    return


def set_init_device_flags():
    def get_all_devices(driver_name):
        """
        Inputs: driver_name
        Returns a list of all the available devices for a given driver sorted by
        the iree path names of the device as in --list_devices option in iree.
        Set `full_dict` flag to True to get a dict
        with `path`, `name` and `device_id` for all devices
        """
        from iree.runtime import get_driver

        driver = get_driver(driver_name)
        device_list_src = driver.query_available_devices()
        device_list_src.sort(key=lambda d: d["path"])
        return device_list_src

    def get_device_mapping(driver, key_combination=3):
        """This method ensures consistent device ordering when choosing
        specific devices for execution
        Args:
            driver (str): execution driver (vulkan, cuda, rocm, etc)
            key_combination (int, optional): choice for mapping value for device name.
            1 : path
            2 : name
            3 : (name, path)
            Defaults to 3.
        Returns:
            dict: map to possible device names user can input mapped to desired combination of name/path.
        """
        from shark.iree_utils._common import iree_device_map

        driver = iree_device_map(driver)
        device_list = get_all_devices(driver)
        device_map = dict()

        def get_output_value(dev_dict):
            if key_combination == 1:
                return f"{driver}://{dev_dict['path']}"
            if key_combination == 2:
                return dev_dict["name"]
            if key_combination == 3:
                return (dev_dict["name"], f"{driver}://{dev_dict['path']}")

        # mapping driver name to default device (driver://0)
        device_map[f"{driver}"] = get_output_value(device_list[0])
        for i, device in enumerate(device_list):
            # mapping with index
            device_map[f"{driver}://{i}"] = get_output_value(device)
            # mapping with full path
            device_map[f"{driver}://{device['path']}"] = get_output_value(
                device
            )
        return device_map

    def map_device_to_name_path(device, key_combination=3):
        """Gives the appropriate device data (supported name/path) for user selected execution device
        Args:
            device (str): user
            key_combination (int, optional): choice for mapping value for device name.
            1 : path
            2 : name
            3 : (name, path)
            Defaults to 3.
        Raises:
            ValueError:
        Returns:
            str / tuple: returns the mapping str or tuple of mapping str for the device depending on key_combination value
        """
        driver = device.split("://")[0]
        device_map = get_device_mapping(driver, key_combination)
        try:
            device_mapping = device_map[device]
        except KeyError:
            raise ValueError(f"Device '{device}' is not a valid device.")
        return device_mapping

    if "vulkan" in args.device:
        name, args.device = map_device_to_name_path(args.device)
        triple = get_vulkan_target_triple(name)
        print(f"Found device {name}. Using target triple {triple}")
        # set triple flag to avoid multiple calls to get_vulkan_triple_flag
        if args.iree_vulkan_target_triple == "" and triple is not None:
            args.iree_vulkan_target_triple = triple

        # use tuned models only in the case of rdna3 cards.
        if not args.iree_vulkan_target_triple:
            if triple is not None and "rdna3" not in triple:
                args.use_tuned = False
        elif "rdna3" not in args.iree_vulkan_target_triple:
            args.use_tuned = False

        if args.use_tuned:
            print("Using tuned models for rdna3 card")
    else:
        if args.use_tuned:
            print("Tuned models not currently supported for device")
            args.use_tuned = False
