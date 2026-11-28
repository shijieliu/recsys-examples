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

#include <cstdint>
#include <cuda_runtime.h>

namespace dyn_emb {

// Largest offset index with offsets[i] <= row. Equal offsets skip empty segments.
__device__ __forceinline__ int binary_search_upper_bound(const int64_t *arr,
                                                         int n, int64_t val, int64_t stride = 1) {
  int lo = 0, hi = n;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (arr[mid * stride] <= val) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo - 1;
}

} // namespace dyn_emb
