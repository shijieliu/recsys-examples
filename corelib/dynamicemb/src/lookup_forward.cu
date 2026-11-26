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

#include "check.h"
#include "lookup_forward.h"
#include "lookup_kernel.cuh"

namespace dyn_emb {

template <typename SrcType, typename DstType, bool UseVec4>
struct ForwardSequenceFusedCopyDesc {
  using SrcT = SrcType;
  using DstT = DstType;
  static constexpr bool kUseVec4 = UseVec4;
  RowMapping rows;
  const SrcT *src;
  DstT *dst;
  int64_t src_stride, dst_stride, width;
};

template <typename Src, typename Dst, bool UseVec4>
void launch_row_copy(const void *src, void *dst, const RowMapping &rows,
                     int64_t width, int64_t src_stride, int64_t dst_stride,
                     int num_sms, cudaStream_t stream) {
  ForwardSequenceFusedCopyDesc<Src, Dst, UseVec4> desc{
      rows, static_cast<const Src *>(src), static_cast<Dst *>(dst),
      src_stride, dst_stride, width};
  copy_one_to_one(desc, num_sms, stream);
}

void scatter_fused(const void *src, void *dst, const RowMapping &rows,
                   int64_t width, int64_t src_stride, int64_t dst_stride,
                   size_t raw_element_bytes, DataType src_type,
                   DataType dst_type, bool vec4_aligned, int num_sms,
                   cudaStream_t stream) {
  // Same dtype: copy the original bits, including FP8 weights and optimizer
  // fields packed in their rows. Use the widest word aligned on every row.
  if (raw_element_bytes) {
    const int64_t bytes = width * raw_element_bytes;
    const int64_t src_pitch = src_stride * raw_element_bytes;
    const int64_t dst_pitch = dst_stride * raw_element_bytes;
    const uintptr_t alignment = reinterpret_cast<uintptr_t>(src) |
                                reinterpret_cast<uintptr_t>(dst) | bytes |
                                src_pitch | dst_pitch;
    auto copy = [&](auto word) {
      using Word = decltype(word);
      launch_row_copy<Word, Word, false>(
          src, dst, rows, bytes / sizeof(Word), src_pitch / sizeof(Word),
          dst_pitch / sizeof(Word), num_sms, stream);
    };
    if (!(alignment & 15)) copy(uint4{});
    else if (!(alignment & 7)) copy(uint64_t{});
    else if (!(alignment & 3)) copy(uint32_t{});
    else if (!(alignment & 1)) copy(uint16_t{});
    else copy(uint8_t{});
    return;
  }
  DISPATCH_VALUE_DATATYPE_FUNCTION(src_type, src_t, [&] {
    DISPATCH_VALUE_DATATYPE_FUNCTION(dst_type, dst_t, [&] {
      DISPATCH_BOOLEAN(vec4_aligned && width > 0 && width <= 256, UseVec4, [&] {
        launch_row_copy<src_t, dst_t, UseVec4>(src, dst, rows, width, src_stride,
                                               dst_stride, num_sms, stream);
      });
    });
  });
}

void get_new_length_and_offsets(uint64_t *d_unique_offsets,
                                int64_t *d_table_offsets_in_feature,
                                int table_num, int64_t new_lengths_size,
                                int local_batch_size, DataType length_type,
                                DataType offset_type, void *new_offsets,
                                void *new_lenghths, cudaStream_t stream) {

  int block_size = 256;
  int grid_size = (new_lengths_size + block_size - 1) / block_size;
  DISPATCH_OFFSET_INT_TYPE(offset_type, offset_t, [&] {
    DISPATCH_OFFSET_INT_TYPE(length_type, length_t, [&] {
      get_new_length_and_offsets_kernel<offset_t, length_t>
          <<<grid_size, block_size, 0, stream>>>(
              d_unique_offsets, d_table_offsets_in_feature, table_num,
              new_lengths_size, local_batch_size,
              reinterpret_cast<offset_t *>(new_offsets),
              reinterpret_cast<length_t *>(new_lenghths));
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace dyn_emb
