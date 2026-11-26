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
// Requester-local work for embedding bags. Rows travel over NVLink and IB
// exactly once per requester-unique key, irrespective of its bag membership.
#pragma once

#include "hier_output_dist/common/device.cuh"

namespace dyn_emb {
namespace hier_output_dist {
namespace pooled {

constexpr int kBlock = 256;

// A bag is feature-major in the KJT and batch-major in the pooled tensor.
__device__ __forceinline__ int64_t batch_major(int64_t bag, int batch, int features) {
  return (bag % batch) * features + bag / batch;
}

// Sum a strided subsequence of one column vector in fp32. Four independent
// gather loads per thread hide some of the indirect read latency.
template <typename RowT>
__device__ __forceinline__ void sum_entries(const int4 *rows, const int64_t *reverse,
                                            const float *weights, int64_t begin, int64_t end,
                                            int stride, int vecs, int col, float *acc) {
  using V = Vec16<RowT>;
  for (int64_t i = begin; i < end; i += 4 * stride) {
    int4 values[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int64_t occurrence = i + j * stride;
      if (occurrence < end) values[j] = rows[reverse[occurrence] * vecs + col];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int64_t occurrence = i + j * stride;
      if (occurrence >= end) break;
      if (weights == nullptr) {
        V::add(acc, values[j]);
      } else {
        float x[V::kValues];
        V::get(x, values[j]);
        const float weight = weights[occurrence];
#pragma unroll
        for (int c = 0; c < V::kValues; ++c) acc[c] += x[c] * weight;
      }
    }
  }
}

// For short bags a lane owns an output vector, including its empty-bag zero.
template <typename RowT, typename Offset>
__global__ void sum_bags(const int4 *rows, const int64_t *reverse, const Offset *offsets,
                         const float *weights, int batch, int features, int vecs, int4 *out) {
  using V = Vec16<RowT>;
  const int64_t count = (int64_t)batch * features * vecs;
  for (int64_t item = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; item < count;
       item += (int64_t)blockDim.x * gridDim.x) {
    const int64_t bag = item / vecs;
    const int col = item % vecs;
    float acc[V::kValues]{};
    sum_entries<RowT>(rows, reverse, weights, offsets[bag], offsets[bag + 1], 1, vecs, col, acc);
    out[batch_major(bag, batch, features) * vecs + col] = V::pack(acc);
  }
}

// For longer bags a CTA shares a tile of at most 32 vectors. Groups of lanes
// sum interleaved entries of the same bag, coalescing adjacent vectors of a
// row, then the first group combines their partials in a fixed order.
template <typename RowT, typename Offset>
__global__ void sum_long_bags(const int4 *rows, const int64_t *reverse, const Offset *offsets,
                              const float *weights, int batch, int features, int vecs, int4 *out) {
  using V = Vec16<RowT>;
  __shared__ float partial[kBlock][V::kValues];
  const int width = min(vecs, 32), groups = kBlock / width, tiles = (vecs + width - 1) / width;
  const int group = threadIdx.x / width, lane = threadIdx.x % width;
  for (int64_t tile = blockIdx.x; tile < (int64_t)batch * features * tiles; tile += gridDim.x) {
    const int64_t bag = tile / tiles;
    const int col = (tile % tiles) * width + lane;
    float acc[V::kValues]{};
    if (group < groups && col < vecs)
      sum_entries<RowT>(rows, reverse, weights, (int64_t)offsets[bag] + group, offsets[bag + 1],
                        groups, vecs, col, acc);
#pragma unroll
    for (int c = 0; c < V::kValues; ++c) partial[threadIdx.x][c] = acc[c];
    __syncthreads();
    if (group == 0 && col < vecs) {
      for (int g = 1; g < groups; ++g) {
#pragma unroll
        for (int c = 0; c < V::kValues; ++c) acc[c] += partial[g * width + lane][c];
      }
      out[batch_major(bag, batch, features) * vecs + col] = V::pack(acc);
    }
    __syncthreads();
  }
}

// Find the nonempty bag containing an original occurrence, skipping runs of
// empty bags. The planner bounds its occurrence indices to signed int32.
template <typename Offset>
__device__ __forceinline__ int bag_of(const Offset *offsets, int bags, int occurrence) {
  int lo = 0, hi = bags;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (offsets[mid + 1] <= occurrence)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

// Substitute bag-gradient indices for occurrence-gradient indices in the
// backward plan. Its light and heavy sums can then read the pooled gradient
// directly, with the original multiplicity of every key in every bag.
template <typename Offset>
__global__ void bag_indices(const int32_t *occurrences, int64_t count, const Offset *offsets,
                            int batch, int features, int32_t *indices,
                            const float *weights, float *sorted_weights) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += (int64_t)blockDim.x * gridDim.x) {
    const int occurrence = occurrences[i];
    const int bag = bag_of(offsets, batch * features, occurrence);
    if (sorted_weights) sorted_weights[i] = weights[occurrence];
    indices[i] = (int32_t)batch_major(bag, batch, features);
  }
}

}  // namespace pooled
}  // namespace hier_output_dist
}  // namespace dyn_emb
