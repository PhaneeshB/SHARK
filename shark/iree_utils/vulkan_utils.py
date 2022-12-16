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

# All the iree_vulkan related functionalities go here.

from os import linesep
from shark.iree_utils._common import map_device_to_name_path
from sys import platform


def get_os_name():
    if platform.startswith("linux"):
        return "linux"
    elif platform == "darwin":
        return "macos"
    elif platform == "win32":
        return "windows"
    else:
        print("Cannot detect OS type, defaulting to linux.")
        return "linux"


def get_vulkan_triple_flag(device="vulkan", device_name=None, extra_args=[]):
    for flag in extra_args:
        if "-iree-vulkan-target-triple=" in flag:
            triple = flag.split("=")[1]
            print(f"Found device: {device_name}. Target Triple: {triple}")
            return None

    system_os = get_os_name()
    vulkan_device = (
        device_name
        if device_name is not None
        else map_device_to_name_path(device, key_combination=2)
    )
    triple = None
    # Apple Targets
    if all(x in vulkan_device for x in ("Apple", "M1")):
        triple = "m1-moltenvk-macos"
    elif all(x in vulkan_device for x in ("Apple", "M2")):
        triple = "m1-moltenvk-macos"
    # Nvidia Targets
    elif all(x in vulkan_device for x in ("RTX", "2080")):
        print(
            f"Found {vulkan_device} Device. Using turing-rtx2080-{system_os}"
        )
        return f"-iree-vulkan-target-triple=turing-rtx2080-{system_os}"
    elif all(x in vulkan_device for x in ("A100", "SXM4")):
        triple = f"ampere-rtx3080-{system_os}"
    elif all(x in vulkan_device for x in ("RTX", "3090")):
        triple = f"ampere-rtx3090-{system_os}"
    elif all(x in vulkan_device for x in ("RTX", "4090")):
        triple = f"ampere-rtx3090-{system_os}"
    elif all(x in vulkan_device for x in ("RTX", "4000")):
        triple = f"turing-rtx4000-{system_os}"
    elif all(x in vulkan_device for x in ("RTX", "5000")):
        triple = f"turing-rtx5000-{system_os}"
    elif all(x in vulkan_device for x in ("RTX", "6000")):
        triple = f"turing-rtx6000-{system_os}"
    elif all(x in vulkan_device for x in ("RTX", "8000")):
        triple = f"turing-rtx8000-{system_os}"
    # Amd Targets
    elif all(x in vulkan_device for x in ("AMD", "7900")):
        triple = f"rdna3-7900-{system_os}"
    elif any(x in vulkan_device for x in ("AMD", "Radeon")):
        triple = f"rdna2-unknown-{system_os}"
    else:
        print(
            """Optimized kernel for your target device is not added yet.
            Contact SHARK Admin on discord[https://discord.com/invite/RUqY2h2s9u]
            or pull up an issue."""
        )
        print(f"Target : {vulkan_device}")
        return None

    print(f"Found device: {vulkan_device}. Target Triple: {triple}")
    return f"-iree-vulkan-target-triple={triple}"


def get_iree_vulkan_args(device, extra_args=[]):
    vulkan_flag = []
    vulkan_triple_flag = get_vulkan_triple_flag(
        device=device, extra_args=extra_args
    )
    if vulkan_triple_flag is not None:
        vulkan_flag.append(vulkan_triple_flag)
    return vulkan_flag


def set_iree_vulkan_runtime_flags(flags):
    import iree.runtime as ireert

    for flag in flags:
        ireert.flags.parse_flags(flag)
    return
