# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import platform

import torch
from dynamicemb_extensions import HostVMMTensor, VMMTensor


class ExtendableBuffer(abc.ABC):
    """Base for host/device extendable buffers. Use is_device_buffer to distinguish
    device (HBM) from host; tensor.is_cuda can be True for host memory registered
    to CUDA address space and is not reliable for this."""

    @abc.abstractmethod
    def capacity(self) -> int:
        capacity: int
        return capacity

    @abc.abstractmethod
    def extend(self, capacity) -> None:
        pass

    @abc.abstractmethod
    def tensor(self) -> torch.Tensor:
        tensor: torch.Tensor
        return tensor

    def is_device_buffer(self) -> bool:
        """True if storage is GPU device memory (HBM); False if host memory.
        Prefer this over tensor().is_cuda when host may be CUDA-registered."""
        return False


class DeviceExtendableBuffer(ExtendableBuffer):
    def __init__(self, capacity, dtype, device: torch.device = None):
        device_id = device.index if device is not None else torch.cuda.current_device()

        self._capacity = capacity
        self._dtype = dtype
        self._device = device_id

        self.vmm_tensor = VMMTensor(capacity, dtype, device_id)

    def is_device_buffer(self) -> bool:
        return True

    def extend(self, capacity) -> None:
        torch.cuda.synchronize()
        self.vmm_tensor.extend(capacity)

    def tensor(self) -> torch.Tensor:
        return self.vmm_tensor.data()

    def capacity(self) -> int:
        return self.vmm_tensor.data().numel()


class HostExtendableBuffer(ExtendableBuffer):
    def __init__(self, capacity, dtype, device):
        if platform.system() != "Linux":
            raise RuntimeError("Only support extendable host buffer on Linux platform.")
        device_id = device.index if device is not None else torch.cuda.current_device()

        self._capacity = capacity
        self._dtype = dtype
        self._device = device_id

        self.vmm_tensor = HostVMMTensor(capacity, dtype, device_id)

    def extend(self, capacity) -> None:
        torch.cuda.synchronize()
        self.vmm_tensor.extend(capacity)

    def tensor(self) -> torch.Tensor:
        return self.vmm_tensor.data()

    def capacity(self) -> int:
        return self.vmm_tensor.data().numel()
