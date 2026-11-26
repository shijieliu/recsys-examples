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

#include "device_count.h"
#include "lookup_backward.h"

namespace dyn_emb {

struct SgdCacheTarget {
  const int64_t *rows;
  const int64_t *slots;
  const int64_t *table_ids;
  const int64_t *tables;
  const int64_t *bucket_offsets;
  int64_t bucket_capacity;
  const int64_t *overflow_offsets;
  int64_t overflow_capacity;
  int64_t main_capacity;
  int64_t pin_count;
  int32_t *pins;
  int64_t capacity;
  const int64_t *count;
  float lr;
};

__device__ __forceinline__ float4 load_cache_sgd_vec(const float *p) {
  if ((reinterpret_cast<uintptr_t>(p) & 15) == 0)
    return *reinterpret_cast<const float4 *>(p);
  float4 v;
  memcpy(&v, p, sizeof(v));
  return v;
}

__device__ __forceinline__ void add_cache_sgd_vec(float4 &a, float4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

__device__ __forceinline__ int64_t resolve_cache_sgd_pin(
    SgdCacheTarget target, int64_t uid, int64_t tid) {
  const auto slot = target.slots[uid];
  if (slot < 0) return -1;
  const auto overflow = target.overflow_offsets[tid];
  const auto pin = slot < overflow ? target.bucket_offsets[tid] * target.bucket_capacity + slot
      : target.main_capacity + tid * target.overflow_capacity + slot - overflow;
  return pin < target.pin_count ? pin : -1;
}

__device__ __forceinline__ void apply_cache_sgd_vec(float *dest, float4 g, float lr) {
  auto w = load_cache_sgd_vec(dest);
  w.x += -lr * g.x; w.y += -lr * g.y;
  w.z += -lr * g.z; w.w += -lr * g.w;
  if ((reinterpret_cast<uintptr_t>(dest) & 15) == 0)
    *reinterpret_cast<float4 *>(dest) = w;
  else
    memcpy(dest, &w, sizeof(w));
}

// One CTA owns a contiguous 32-occurrence slice of the stable radix sort. A
// complete group updates the resident row and releases its pin here. A group
// crossing a slice boundary has at most one outgoing partial and one incoming
// tail per slice; the second launch touches only those exceptional groups.
template <typename IndexT>
__global__ void mark_cache_gradient_groups_tiled(
    const IndexT *ids, const GradInfo<IndexT> *infos, const float *grads,
    int64_t n, SgdCacheTarget target, IndexT *tails, float4 *partials) {
  const int t = threadIdx.x, lane = t % 16;
  const int64_t begin = int64_t(blockIdx.x) * 32;
  const int width = n - begin < 32 ? static_cast<int>(n - begin) : 32;
  const int64_t count = counted_limit(target.count, target.capacity);
  __shared__ IndexT input_rows[32];
  __shared__ int head[32], mode[32];
  __shared__ float *dest[32];
  int64_t pin = -1;
  if (t < 32) {
    const auto i = begin + t;
    const bool live = t < width;
    const auto uid = live ? ids[i] : IndexT{0};
    if (live) input_rows[t] = infos[i].grad_index;
    const bool begins = live && (t == 0 || ids[i - 1] != uid);
    const uint32_t heads = __ballot_sync(0xffffffffu, begins);
    mode[t] = 0;
    if (live && uid < count && (t + 1 == width || ids[i + 1] != uid)) {
      const uint32_t lower = 0xffffffffu >> (31 - t);
      const int first = 31 - __clz(heads & lower);
      const bool left = first == 0 && begin > 0 && uid == ids[begin - 1];
      const bool right = t + 1 == width && i + 1 < n && uid == ids[i + 1];
      head[t] = first;
      mode[t] = right ? 2 : left ? 3 : 1;
      if (!left && !right) {
        const auto tid = target.table_ids[uid], row = target.rows[uid];
        dest[t] = row < 0 ? nullptr : reinterpret_cast<float *>(target.tables[tid]) + row * 64;
        pin = resolve_cache_sgd_pin(target, uid, tid);
      }
    }
    if (t == 0) {
      const bool tail = begin > 0 && uid < count && uid == ids[begin - 1] &&
          (begin + width == n || uid != ids[begin + width]);
      tails[blockIdx.x] = tail ? uid : static_cast<IndexT>(-1);
    }
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int pos = 8 * j + t / 16, m = mode[pos];
    if (m == 0) continue;
    float4 sum{};
    for (int k = head[pos]; k <= pos; ++k)
      add_cache_sgd_vec(sum, load_cache_sgd_vec(grads + int64_t(input_rows[k]) * 64 + lane * 4));
    if (m == 1) {
      if (dest[pos]) apply_cache_sgd_vec(dest[pos] + lane * 4, sum, target.lr);
    } else {
      // The outgoing partial occupies the first vector, the closing tail the second.
      partials[(int64_t(blockIdx.x) * 2 + (m == 3)) * 16 + lane] = sum;
    }
  }
  __threadfence();
  __syncthreads();
  if (pin >= 0) ::atomicAdd(target.pins + pin, -1);
}

template <typename IndexT>
__global__ void reduce_cache_groups_and_unpin_tiled(
    const IndexT *ids, const IndexT *tails, const float4 *partials,
    int64_t num_chunks, SgdCacheTarget target, int window) {
  const int lane = threadIdx.x % 16;
  const int64_t chunk = int64_t(blockIdx.x) * 8 + threadIdx.x / 16;
  if (chunk >= num_chunks) return;
  const IndexT uid = tails[chunk];
  if (uid == static_cast<IndexT>(-1)) return;
  const int64_t tid = target.table_ids[uid], row = target.rows[uid];
  if (row >= 0) {
    // A single boundary accounts for most split groups. For a hot group,
    // find its first outgoing partial in logarithmic time.
    int64_t first = chunk - 1;
    if (first > 0 && ids[first * 32 - 1] == uid) {
      int64_t lower = 0, upper = first - 1;
      while (lower < upper) {
        const auto mid = (lower + upper) / 2;
        if (ids[(mid + 1) * 32 - 1] < uid) lower = mid + 1;
        else upper = mid;
      }
      first = lower;
    }
    float4 sum = partials[(chunk * 2 + 1) * 16 + lane];
    const auto window_width = int64_t(window);
    for (auto i = first; i < chunk;) {
      const int64_t next_window = (i / window_width + 1) * window_width;
      const int64_t end = next_window < chunk ? next_window : chunk;
      float4 acc{};
      for (; i < end; ++i) add_cache_sgd_vec(acc, partials[i * 32 + lane]);
      add_cache_sgd_vec(sum, acc);
    }
    auto *dest = reinterpret_cast<float *>(target.tables[tid]) + row * 64 + lane * 4;
    apply_cache_sgd_vec(dest, sum, target.lr);
  }
  __threadfence();
  __syncwarp(0xffffu << (threadIdx.x & 16));
  if (lane == 0) {
    const auto pin = resolve_cache_sgd_pin(target, uid, tid);
    if (pin >= 0) ::atomicAdd(target.pins + pin, -1);
  }
}

} // namespace dyn_emb
