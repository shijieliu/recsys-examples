/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved. # SPDX-License-Identifier: Apache-2.0
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
******************************************************************************/

#pragma once

#include "device_count.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <optional>

namespace dyn_emb {

// The host chooses an upper bound inside the physical buffer capacity.
// A device count may shorten that prefix without a host read.
inline int64_t counted_host_limit(std::optional<int64_t> host_num_valid,
                                  int64_t capacity) {
  return std::clamp(host_num_valid.value_or(capacity), int64_t{0}, capacity);
}

// A count can be a one-element view such as table_offsets[T:T+1].
inline const int64_t *counted_num_valid_ptr(const at::Tensor &t,
                                           const at::Device &device) {
  TORCH_CHECK(t.is_cuda() && t.device() == device,
              "num_valid must be a CUDA tensor on the device of the data");
  TORCH_CHECK(t.scalar_type() == at::kLong && t.numel() == 1,
              "num_valid must be a one-element int64 tensor");
  return t.data_ptr<int64_t>();
}

inline const int64_t *
counted_num_valid_ptr(const std::optional<at::Tensor> &num_valid,
                      const at::Device &device) {
  return num_valid ? counted_num_valid_ptr(*num_valid, device) : nullptr;
}

// Blocks of a grid-stride launch that wants `blocks` of them (to cover its
// threads, or one per row): the same number, but capped the way the vec4
// kernels cap theirs (4 waves of resident blocks), so that an empty batch costs
// one cheap launch, not a block per capacity entry.
inline int counted_grid(int64_t blocks, int block_size) {
  const cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
  // The hardware keeps at most 32 blocks of a kernel resident on one SM (a
  // block-per-row kernel can have a block of a few threads).
  const int64_t blocks_per_sm = std::min(
      32, std::max(1, prop->maxThreadsPerMultiProcessor /
                          std::max(1, block_size)));
  const int64_t max_grid =
      static_cast<int64_t>(prop->multiProcessorCount) * blocks_per_sm * 4;
  return static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>(blocks, max_grid)));
}

} // namespace dyn_emb
