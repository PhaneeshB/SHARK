# Copyright 2020 The Nod Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Common utilities to be shared by iree utilities.

import os
import sys
import subprocess
from iree.runtime import get_driver, get_device


def run_cmd(cmd):
    """
    Inputs: cli command string.
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        result_str = result.stdout.decode()
        return result_str
    except Exception:
        sys.exit("Exiting program due to error running:", cmd)


def get_all_devices(driver_name):
    """
    Inputs: driver_name
    Returns a list of all the available devices for a given driver sorted by
    the iree path names of the device as in --list_devices option in iree.
    Set `full_dict` flag to True to get a dict
    with `path`, `name` and `device_id` for all devices
    """
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

    driver = iree_device_map(driver)
    device_list = get_all_devices(driver)
    device_map = dict()
    def get_output_value(dev_dict):
        if key_combination == 1:
            return f"{driver}://{dev_dict['path']}"
        if key_combination == 2:
            return dev_dict['name']
        if key_combination == 3:
            return (dev_dict['name'], f"{driver}://{dev_dict['path']}")

    # mapping driver name to default device (driver://0)
    device_map[f"{driver}"] = get_output_value(device_list[0])
    for i, device in enumerate(device_list):
        # mapping with index
        device_map[f"{driver}://{i}"] = get_output_value(device)
        # mapping with full path
        device_map[f"{driver}://{device['path']}"] = get_output_value(device)
    return device_map


def map_device_to_name_path(device, key_combination=3):
    """ Gives the appropriate device data (supported name/path) for user selected execution device 

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


def iree_device_map(device):
    uri_parts = device.split("://", 1)
    if len(uri_parts) == 1:
        return _IREE_DEVICE_MAP[uri_parts[0]]
    else:
        return f"{_IREE_DEVICE_MAP[uri_parts[0]]}://{uri_parts[1]}"


def get_supported_device_list():
    return list(_IREE_DEVICE_MAP.keys())


_IREE_DEVICE_MAP = {
    "cpu": "local-task",
    "cuda": "cuda",
    "vulkan": "vulkan",
    "metal": "vulkan",
    "rocm": "rocm",
    "intel-gpu": "level_zero",
}


def iree_target_map(device):
    if "://" in device:
        device = device.split("://")[0]
    return _IREE_TARGET_MAP[device]


_IREE_TARGET_MAP = {
    "cpu": "llvm-cpu",
    "cuda": "cuda",
    "vulkan": "vulkan",
    "metal": "vulkan",
    "rocm": "rocm",
    "intel-gpu": "opencl-spirv",
}


# Finds whether the required drivers are installed for the given device.
def check_device_drivers(device):
    """Checks necessary drivers present for gpu and vulkan devices"""
    if "://" in device:
        device = device.split("://")[0]

    if device == "cuda":
        try:
            subprocess.check_output("nvidia-smi")
        except Exception:
            return True
    elif device in ["metal", "vulkan"]:
        try:
            subprocess.check_output("vulkaninfo")
        except Exception:
            return True
    elif device in ["intel-gpu"]:
        try:
            subprocess.check_output(["dpkg", "-L", "intel-level-zero-gpu"])
            return False
        except Exception:
            return True
    elif device == "cpu":
        return False
    elif device == "rocm":
        try:
            subprocess.check_output("rocminfo")
        except Exception:
            return True
    # Unknown device.
    else:
        return True

    return False


# Installation info for the missing device drivers.
def device_driver_info(device):
    if device == "cuda":
        return "nvidia-smi not found, please install the required drivers from https://www.nvidia.in/Download/index.aspx?lang=en-in"
    elif device in ["metal", "vulkan"]:
        return "vulkaninfo not found, Install from https://vulkan.lunarg.com/sdk/home or your distribution"
    elif device == "rocm":
        return "rocm info not found. Please install rocm"
    else:
        return f"{device} is not supported."
