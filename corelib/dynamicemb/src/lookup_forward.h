/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef LOOKUP_FORWARD_H
#define LOOKUP_FORWARD_H
#include "lookup_kernel.cuh"
#include "pooled_layout.cuh"
#include "utils.h"

namespace dyn_emb {

// A pooled cache read goes from input to unique row to the pinned table slot.
// The flat key/value tables keep their native per-table optimizer-state stride.
struct PooledCacheRows {
  const int64_t *table_ptrs = nullptr;
  const int64_t *value_dims = nullptr;
  const int64_t *slots = nullptr;
  const int64_t *table_ids = nullptr;
};

// One group processes each bag in input order. The layout identifies its output
// columns. Source strides include any optimizer state appended to the embedding.
template <typename SrcType, typename DstType, typename IndexType>
struct ForwardMultiToOneFMLayoutDesc {
  using SrcT = SrcType;
  using DstT = DstType;

  HOST_DEVICE_INLINE int get_offset(int i) { return offset_ptr[i]; }
  HOST_DEVICE_INLINE int get_vec_length(int i) {
    return layout.col_width(layout.feature_of_input(i));
  }
  HOST_DEVICE_INLINE int get_average_pooling_factor(int i) {
    int pooling_factor = static_cast<int>(offset_ptr[i + 1] - offset_ptr[i]);
    return pooling_mode == PoolingMode::kMean ? pooling_factor : 1;
  }
  HOST_DEVICE_INLINE float get_weight(int i) {
    // nullptr => unweighted pooling (identical to the old path).
    return weights_ptr ? weights_ptr[i] : 1.0f;
  }
  HOST_DEVICE_INLINE const SrcType *get_src_ptr(int i) {
    const auto idx = inverse_idx_ptr[i];
    if (cache.table_ptrs) {
      const auto slot = cache.slots[idx];
      if (slot < 0) return nullptr;
      const auto tid = cache.table_ids[idx];
      return reinterpret_cast<const SrcType *>(cache.table_ptrs[tid]) + slot * cache.value_dims[tid];
    }
    return src_ptr + int64_t(src_stride) * idx;
  }
  HOST_DEVICE_INLINE DstType *get_dst_ptr(int i) {
    const int f = layout.feature_of_input(i);
    const int b = layout.sample_of_input(i);
    return dst_ptr + (int64_t)b * layout.total_D + layout.col_begin(f);
  }

  PoolingMode pooling_mode;
  PooledLayout layout;
  int src_stride; // source row stride; exceeds layout.max_D when optimizer
                  // states are appended to each row
  const IndexType *__restrict__ offset_ptr;
  // torch.unique's inverse mapping: for input position i, the row of the
  // deduplicated table that position reads.  Most of the codebase still spells
  // this "reverse"; it is the same array.
  const IndexType *__restrict__ inverse_idx_ptr;
  const SrcType *__restrict__ src_ptr;
  DstType *dst_ptr;
  const float *__restrict__ weights_ptr; // nullptr -> unweighted
  PooledCacheRows cache;
};

// Resolve a group of independent row addresses before consuming any embedding.
// Cache rows require two indirect reads (unique row -> slot -> value pointer).
// Broadcasting the resolved pointer lets those reads overlap across the bag.
template <int Dim, int Group, typename Index>
__global__ void pool_cache_rows_kernel(ForwardMultiToOneFMLayoutDesc<float, float, Index> desc) {
  constexpr int vectors = Dim / (Group * 4);
  const int bag = (blockIdx.x * blockDim.x + threadIdx.x) / Group;
  if (bag >= desc.layout.num_bags()) return;
  const int lane = threadIdx.x % Group;
  const unsigned mask = Group == 32 ? 0xffffffffu : (0xffffu << (threadIdx.x & 16));
  const int start = desc.get_offset(bag);
  const int end = desc.get_offset(bag + 1);
  Vec4T<float> sum[vectors];
  // Four-key bags are common. Expose all four independent cache reads to
  // the scheduler before the dependent sum, which preserves input order.
  if constexpr (Dim == 64) {
    if (end - start == 4 && !desc.weights_ptr) {
      const auto ptr = lane < 4 ? desc.get_src_ptr(start + lane) : nullptr;
      const auto addr = reinterpret_cast<unsigned long long>(ptr);
      if (__all_sync(mask, lane >= 4 || (addr && (addr & 15u) == 0))) {
        float4 values[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const auto row = __shfl_sync(mask, addr, j, Group);
          values[j] = reinterpret_cast<const float4 *>(row)[lane];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          sum[0].val.x += values[j].x;
          sum[0].val.y += values[j].y;
          sum[0].val.z += values[j].z;
          sum[0].val.w += values[j].w;
        }
        if (desc.pooling_mode == PoolingMode::kMean) {
          sum[0].val.x *= 0.25f;
          sum[0].val.y *= 0.25f;
          sum[0].val.z *= 0.25f;
          sum[0].val.w *= 0.25f;
        }
        sum[0].store(desc.get_dst_ptr(bag) + lane * 4, 4);
        return;
      }
    }
  }
  for (int first = start; first < end; first += Group) {
    const int length = min(Group, end - first);
    const auto ptr = lane < length ? desc.get_src_ptr(first + lane) : nullptr;
    const auto addr = reinterpret_cast<unsigned long long>(ptr);
    const float weight = lane < length ? desc.get_weight(first + lane) : 0.0f;
    for (int j = 0; j < length; ++j) {
      const auto row = __shfl_sync(mask, addr, j, Group);
      const float scale = __shfl_sync(mask, weight, j, Group);
      if (!row) continue;
#pragma unroll
      for (int v = 0; v < vectors; ++v) {
        const auto src = reinterpret_cast<const float *>(row) + 4 * (v * Group + lane);
        Vec4T<float> val;
        if ((row & 15u) == 0) val.load(src, 4);
        else val.val = make_float4(src[0], src[1], src[2], src[3]);
        sum[v].accumulate_multiply(val, scale);
      }
    }
  }
  float *dst = desc.get_dst_ptr(bag);
#pragma unroll
  for (int v = 0; v < vectors; ++v) {
    if (desc.pooling_mode == PoolingMode::kMean && end > start) {
      auto &a = sum[v].val;
      a.x /= end - start;
      a.y /= end - start;
      a.z /= end - start;
      a.w /= end - start;
    }
    sum[v].store(dst + 4 * (v * Group + lane), 4);
  }
}

// Unified pooled gather (scatter-combine).  ``layout`` carries the shapes, the
// bag numbering and the copy width; see pooled_layout.cuh.  ``src_stride`` is
// the source row stride, which exceeds the copy width when optimizer state is
// appended to each row.  ``IndexType`` is shared by the offsets and the inverse
// indices, which the descriptor requires to have the same width.
template <typename SrcType, typename DstType, typename IndexType>
void scatter_combine(const SrcType *src_ptr, DstType *dst_ptr,
                     const IndexType *offset_ptr,
                     const IndexType *inverse_idx_ptr, PoolingMode pooling_mode,
                     const PooledLayout &layout, int src_stride,
                     bool feature_dims_vec4, cudaStream_t stream,
                     const float *weights_ptr = nullptr, PooledCacheRows cache = {}) {
  ForwardMultiToOneFMLayoutDesc<SrcType, DstType, IndexType> desc{
      pooling_mode, layout,  src_stride,  offset_ptr, inverse_idx_ptr,
      src_ptr,      dst_ptr, weights_ptr, cache};
  if constexpr (std::is_same_v<SrcType, float> && std::is_same_v<DstType, float>) {
    if (cache.table_ptrs && !layout.mixed_D() && layout.num_bags() &&
        (reinterpret_cast<uintptr_t>(dst_ptr) % alignof(float4)) == 0) {
      constexpr int threads = 128;
      const int bags = layout.num_bags();
      if (layout.max_D == 64) {
        pool_cache_rows_kernel<64, 16><<<(bags + 7) / 8, threads, 0, stream>>>(desc);
      } else if (layout.max_D == 256) {
        pool_cache_rows_kernel<256, 32><<<(bags + 3) / 4, threads, 0, stream>>>(desc);
      } else if (layout.max_D == 512) {
        pool_cache_rows_kernel<512, 32><<<(bags + 3) / 4, threads, 0, stream>>>(desc);
      } else {
        copy_multi_to_one(desc, feature_dims_vec4, stream);
        return;
      }
      DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
  }
  copy_multi_to_one(desc, feature_dims_vec4, stream);
}

// Source and destination row maps are optional. A null map is the identity.
struct RowMapping {
  int64_t host_limit, input_rows, output_rows;
  const int64_t *num_valid;
  const int64_t *src64, *dst64;
  const int32_t *src32, *dst32;

  DEVICE_INLINE bool resolve(int64_t i, int64_t &src, int64_t &dst) const {
    src = src64 ? src64[i] : (src32 ? src32[i] : i);
    dst = dst64 ? dst64[i] : (dst32 ? dst32[i] : i);
    return src >= 0 && src < input_rows && dst >= 0 && dst < output_rows;
  }
};

void scatter_fused(const void *src, void *dst, const RowMapping &rows,
                   int64_t width, int64_t src_stride, int64_t dst_stride,
                   size_t raw_element_bytes, DataType src_type,
                   DataType dst_type, bool vec4_aligned, int num_sms,
                   cudaStream_t stream);

void get_new_length_and_offsets(uint64_t *d_unique_offsets,
                                int64_t *d_table_offsets_in_feature,
                                int table_num, int64_t new_lengths_size,
                                int local_batch_size, DataType length_type,
                                DataType offset_type, void *new_offsets,
                                void *new_lenghths, cudaStream_t stream);

} // namespace dyn_emb
#endif // LOOKUP_FORWARD_H
