// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace dyn_emb {

// Each lane can transfer 16 bytes at once. A group covers a short row in one
// instruction, while the rest of its warp handles independent rows.
inline int cache_copy_lane_shift(int64_t row_bytes) {
  return row_bytes <= 128 ? 3 : row_bytes <= 256 ? 4 : 5;
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
    // A short aligned row fits in one vector access per lane.
    if (words == lanes * words_per_vector) {
      out[lane] = in[lane];
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
