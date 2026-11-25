// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace dyn_emb {

// Give short rows a fraction of a warp. Two independent vector accesses per
// lane keep more rows in flight for the usual 256- and 512-byte cases.
inline int cache_copy_lane_shift(int64_t row_bytes) {
  return row_bytes <= 256 ? 3 : row_bytes <= 512 ? 4 : 5;
}

template <int Stripes>
__device__ __forceinline__ void copy_cache_stripes(
    const uint4 *src, uint4 *dst, int lane, int lanes) {
  uint4 values[Stripes];
#pragma unroll
  for (int j = 0; j < Stripes; ++j)
    values[j] = src[lane + j * lanes];
#pragma unroll
  for (int j = 0; j < Stripes; ++j)
    dst[lane + j * lanes] = values[j];
}

template <typename Word>
__device__ __forceinline__ void copy_cache_region(
    const Word *src, Word *dst, int64_t words, int lane, int lanes) {
  constexpr int words_per_vector = sizeof(uint4) / sizeof(Word);
  const auto alignment = reinterpret_cast<uintptr_t>(src) |
                         reinterpret_cast<uintptr_t>(dst);
  if (!(alignment & (alignof(uint4) - 1)) && words >= words_per_vector) {
    const auto *in = reinterpret_cast<const uint4 *>(src);
    auto *out = reinterpret_cast<uint4 *>(dst);
    const int stripe = lanes * words_per_vector;
    if (words == stripe) {
      copy_cache_stripes<1>(in, out, lane, lanes);
      return;
    }
    if (words == 2 * stripe) {
      copy_cache_stripes<2>(in, out, lane, lanes);
      return;
    }
    if (words == 4 * stripe) {
      copy_cache_stripes<4>(in, out, lane, lanes);
      return;
    }
    const int64_t n = words / words_per_vector;
    for (int64_t i = lane; i < n; i += lanes)
      out[i] = in[i];
    const int64_t copied = n * words_per_vector;
    words -= copied;
    src += copied;
    dst += copied;
  }
  for (int64_t i = lane; i < words; i += lanes)
    dst[i] = src[i];
}

// Four independent 256-byte vectors per half warp. Issue the aligned
// loads before waiting on any one transfer. Plain tensor inputs may also
// provide an unaligned FP8 row; its raw bytes still follow the scalar path.
struct CacheTransfer256 {
  const uint4 *src;
  uint4 *dst;
  uint4 data;

  __device__ bool aligned() const {
    return !((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 15);
  }
};

__device__ __forceinline__ void copy_cache_transfers_256(CacheTransfer256 (&rows)[4],
                                                         int lane) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    auto &row = rows[j];
    if (row.src && row.dst && row.aligned())
      asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" :
                   "=r"(row.data.x), "=r"(row.data.y), "=r"(row.data.z), "=r"(row.data.w) :
                   "l"(row.src + lane));
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const auto &row = rows[j];
    if (!row.src || !row.dst) continue;
    if (row.aligned())
      row.dst[lane] = row.data;
    else
      copy_cache_region(reinterpret_cast<const uint8_t *>(row.src),
                        reinterpret_cast<uint8_t *>(row.dst), 256, lane, 16);
  }
}

template <typename Word>
__device__ __forceinline__ void copy_cache_row(
    const Word *src, Word *dst, int64_t width, int64_t emb,
    int64_t src_state, int64_t dst_state, int lane, int lanes) {
  // Common fixed-dimension tables are contiguous across both regions.
  if (width == emb || (src_state == emb && dst_state == emb)) {
    copy_cache_region(src, dst, width, lane, lanes);
  } else {
    copy_cache_region(src, dst, emb, lane, lanes);
    copy_cache_region(src + src_state, dst + dst_state, width - emb, lane, lanes);
  }
}

} // namespace dyn_emb
