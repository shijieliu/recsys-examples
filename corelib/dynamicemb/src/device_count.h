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
#include <cstdint>
#include <cuda_runtime.h>
#include <optional>

namespace dyn_emb {

// Per-key arrays are allocated at a capacity the host knows (the number of
// input keys); the number of live leading entries is a device tensor
// `num_valid`: int64, exactly one element (any view with numel() == 1, e.g.
// table_offsets[T:T+1]). Entries from n = clamp(*num_valid, 0, capacity) on are
// undefined memory that no kernel reads or writes. Without a count (nullptr)
// every entry is live. Kernels read n with counted_limit (device_count.cuh).

// Checks an optional `num_valid` tensor and returns the device pointer the
// kernels read, or nullptr when there is none. Takes c10::optional and
// std::optional alike.
template <typename OptionalTensor>
inline const int64_t *counted_num_valid_ptr(const OptionalTensor &num_valid,
                                            const at::Device &device) {
  if (!num_valid.has_value() || !num_valid.value().defined()) {
    return nullptr;
  }
  const at::Tensor &t = num_valid.value();
  TORCH_CHECK(t.is_cuda() && t.device() == device,
              "num_valid must be a CUDA tensor on the device of the data");
  TORCH_CHECK(t.scalar_type() == at::kLong && t.numel() == 1,
              "num_valid must be a one-element int64 tensor");
  return t.data_ptr<int64_t>();
}

} // namespace dyn_emb
