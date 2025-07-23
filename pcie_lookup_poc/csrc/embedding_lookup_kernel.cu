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
#
# Implementation based on FlashInfer library.
#
******************************************************************************/

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <driver_types.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#define CASE_TYPE_USING_HINT(enum_type, type, HINT, ...) \
  case (enum_type): {                                    \
    using HINT = type;                                   \
    __VA_ARGS__();                                       \
    break;                                               \
  }

#define DISPATCH_INTEGRAL_FUNCTION(DATA_TYPE, HINT, ...)                    \
  switch (DATA_TYPE) {                                                      \
    CASE_TYPE_USING_HINT(at::ScalarType::Long, int64_t, HINT, __VA_ARGS__)  \
    CASE_TYPE_USING_HINT(at::ScalarType::Int, int32_t, HINT, __VA_ARGS__)   \
    default:                                                                \
      TORCH_CHECK(false, "DISPATCH_INTEGRAL_FUNCTION do not support type"); \
  }

#define DISPATCH_FLOAT_AND_HALF_FUNCTION(DATA_TYPE, HINT, ...)              \
  switch (DATA_TYPE) {                                                      \
    CASE_TYPE_USING_HINT(at::ScalarType::Float, float, HINT, __VA_ARGS__)   \
    CASE_TYPE_USING_HINT(at::ScalarType::Half, at::Half, HINT, __VA_ARGS__) \
    CASE_TYPE_USING_HINT(at::ScalarType::BFloat16, at::BFloat16, HINT,      \
                         __VA_ARGS__)                                       \
    default:                                                                \
      TORCH_CHECK(false,                                                    \
                  "DISPATCH_FLOAT_AND_HALF_FUNCTION do not support type");  \
  }

template <typename T, int kVecWidth>
struct VecType {};

struct float4_type {
  using type = float4;
};

struct float2_type {
  using type = float2;
};

struct float_type {
  using type = float;
};

template <>
struct VecType<float, 4> : public float4_type {};

template <>
struct VecType<float, 2> : public float2_type {};

template <>
struct VecType<float, 1> : public float_type {};

template <>
struct VecType<at::Half, 8> : public float4_type {};

template <>
struct VecType<at::Half, 4> : public float2_type {};

template <>
struct VecType<at::Half, 2> : public float_type {};

template <>
struct VecType<at::Half, 1> {
  using type = at::Half;
};

template <>
struct VecType<at::BFloat16, 8> : public float4_type {};

template <>
struct VecType<at::BFloat16, 4> : public float2_type {};

template <>
struct VecType<at::BFloat16, 2> : public float_type {};

template <>
struct VecType<at::BFloat16, 1> {
  using type = at::BFloat16;
};

template <typename IndexType,
          typename EmbeddingType,
          typename OutputType,
          int kVecWidth,
          int kThreadsPerWG,
          int kUnrollFactor>
__global__ __launch_bounds__(1024, 1) void embedding_lookup_kernel(
    const IndexType* indices,
    const EmbeddingType* embedding_table,
    int64_t num_indices,
    int64_t num_embeddings,
    int64_t embedding_dim,
    int num_wgs,
    OutputType* output) {
  using EmbeddingVecType = typename VecType<EmbeddingType, kVecWidth>::type;
  using OutputVecType = typename VecType<OutputType, kVecWidth>::type;
  static_assert(sizeof(EmbeddingVecType) / sizeof(EmbeddingType) ==
                    sizeof(OutputVecType) / sizeof(OutputType),
                "EmbeddingVecType and OutputVecType must have the same width");

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int wg_id = tid / kThreadsPerWG;
  int lane_id = tid % kThreadsPerWG;

  for (int64_t i = wg_id; i < num_indices; i += num_wgs) {
#pragma unroll kUnrollFactor
    for (int64_t l = lane_id * kVecWidth; l < embedding_dim;
         l += kThreadsPerWG * kVecWidth) {
      // EmbeddingVecType embedding_vec =
      //     *reinterpret_cast<const EmbeddingVecType*>(
      //         &embedding_table[indices[i] * embedding_dim + l]);
      EmbeddingVecType embedding_vec =
          __ldcv(reinterpret_cast<const EmbeddingVecType*>(
              &embedding_table[indices[i] * embedding_dim + l]));
      OutputVecType* output_vec =
          reinterpret_cast<OutputVecType*>(&output[i * embedding_dim + l]);
      // *output_vec = embedding_vec;
      __stwt(output_vec, embedding_vec);
    }
  }
}

// using mbarrier_t = uint64_t;
// namespace cde = cuda::device::experimental;

// template <typename T, int kNumStages, int kElemPerStage>
// struct SmemType {
//   alignas(128) T data[kNumStages][kElemPerStage];

//   mbarrier_t read_mbarriers[kNumStages];
//   mbarrier_t write_mbarriers[kNumStages];

//   __device__ __forceinline__ void init_barrier(int tid) {
//     if (tid > 0)
//       return;
//     for (int i = 0; i < kNumStages; i++) {
//       cuda::ptx::mbarrier_init(&read_mbarriers[i], 1);
//       cuda::ptx::mbarrier_init(&write_mbarriers[i], 1);
//     }
//     cuda::ptx::fence() cde::fence_proxy_async_shared_cta();
//   }

//   __device__ __forceinline__ void wait_data_ready_for_read(int stage,
//                                                            int parity) {
//     while (!cuda::ptx::mbarrier_try_wait_parity(
//         cuda::ptx::sem_acquire, cuda::ptx::scope_cta,
//         reinterpret_cast<uint64_t*>(&read_mbarriers[stage]), parity)) {
//     }
//   }

//   __device__ __forceinline__ void wait_data_ready_for_write(int stage,
//                                                             int parity) {
//     while (!cuda::ptx::mbarrier_try_wait_parity(
//         cuda::ptx::sem_acquire, cuda::ptx::scope_cta,
//         reinterpret_cast<uint64_t*>(&write_mbarriers[stage]), parity)) {
//     }
//   }
// };

// template <typename IndexType,
//           typename EmbeddingType,
//           typename OutputType,
//           int kNumStages,
//           int kEmbeddingDim,
//           int kVecWidth,
//           int kThreadsPerWG,
//           int kUnrollFactor>
// __global__ __launch_bounds__(1024, 1) void embedding_lookup_v2_kernel(
//     const IndexType* indices,
//     const EmbeddingType* embedding_table,
//     int64_t num_indices,
//     int64_t num_embeddings,
//     int64_t embedding_dim,
//     int num_wgs,
//     OutputType* output) {
//   __shared__ SmemType smem_data;

//   int parity = 1;

//   cp_async_bulk();
//   using EmbeddingVecType = typename VecType<EmbeddingType, kVecWidth>::type;
//   using OutputVecType = typename VecType<OutputType, kVecWidth>::type;
//   static_assert(sizeof(EmbeddingVecType) / sizeof(EmbeddingType) ==
//                     sizeof(OutputVecType) / sizeof(OutputType),
//                 "EmbeddingVecType and OutputVecType must have the same width");

//   int tid = blockIdx.x * blockDim.x + threadIdx.x;
//   int wg_id = tid / kThreadsPerWG;
//   int lane_id = tid % kThreadsPerWG;

//   for (int64_t i = wg_id; i < num_indices; i += num_wgs) {
// #pragma unroll kUnrollFactor
//     for (int64_t l = lane_id * kVecWidth; l < embedding_dim;
//          l += kThreadsPerWG * kVecWidth) {
//       // EmbeddingVecType embedding_vec =
//       //     *reinterpret_cast<const EmbeddingVecType*>(
//       //         &embedding_table[indices[i] * embedding_dim + l]);
//       EmbeddingVecType embedding_vec =
//           __ldcv(reinterpret_cast<const EmbeddingVecType*>(
//               &embedding_table[indices[i] * embedding_dim + l]));
//       OutputVecType* output_vec =
//           reinterpret_cast<OutputVecType*>(&output[i * embedding_dim + l]);
//       // *output_vec = embedding_vec;
//       __stwt(output_vec, embedding_vec);
//     }
//   }
// }

void embedding_lookup_cuda(at::Tensor indices,
                           at::Tensor embedding_table,
                           at::Tensor output,
                           int num_sms,
                           int max_threads_per_sm) {
  auto device = indices.device();

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(indices.dim() == 1, "indices must be a 1D tensor");
  TORCH_CHECK(embedding_table.dim() == 2,
              "embedding_table must be a 2D tensor");
  TORCH_CHECK(output.dim() == 2, "output must be a 2D tensor");
  int64_t num_indices = indices.size(0);
  int64_t num_embeddings = embedding_table.size(0);
  int64_t embedding_dim = embedding_table.size(1);
  if (embedding_dim % 128 != 0) {
    throw std::runtime_error("embedding_dim must be divisible by 128");
  }

  DISPATCH_INTEGRAL_FUNCTION(indices.scalar_type(), IndexType, [&] {
    // if (embedding_table.scalar_type() == at::ScalarType::Float &&
    //     output.scalar_type() == at::ScalarType::Float) {
    //   embedding_lookup_kernel<IndexType, float, float4, float4, float, 1024,
    //   1>
    //       <<<grid, block, 0, stream>>>(
    //           indices.data_ptr<IndexType>(),
    //           embedding_table.data_ptr<float>(), output.data_ptr<float>(),
    //           num_indices, num_embeddings, embedding_dim, num_wgs,
    //           output.data_ptr<float>());
    // }

    using EmbeddingType = float;
    using OutputType = float;

    constexpr int kThreadsPerWG = 32;
    constexpr int kUnrollFactor = 1;
    constexpr int kVecWidth = 4;
    auto kernel =
        embedding_lookup_kernel<IndexType, EmbeddingType, OutputType, kVecWidth,
                                kThreadsPerWG, kUnrollFactor>;
    dim3 block(1024);
    dim3 grid((std::min(num_indices * kThreadsPerWG,
                        static_cast<int64_t>(max_threads_per_sm * num_sms)) +
               block.x - 1) /
              block.x);

    int num_wgs = grid.x * block.x / kThreadsPerWG;
    kernel<<<grid, block, 0, stream>>>(
        indices.data_ptr<IndexType>(),
        embedding_table.data_ptr<EmbeddingType>(), indices.size(0),
        embedding_table.size(0), embedding_table.size(1), num_wgs,
        output.data_ptr<OutputType>());
    // DISPATCH_FLOAT_AND_HALF_FUNCTION(
    //     embedding_table.scalar_type(), EmbeddingType, [&] {
    //       DISPATCH_FLOAT_AND_HALF_FUNCTION(
    //           output.scalar_type(), OutputType, [&] {
    //             if (embedding_dim <= 512 && embedding_dim % 128 == 0) {
    //               if (output.scalar_type() == embedding_table.scalar_type()
    //               &&
    //                   embedding_table.scalar_type() == at::ScalarType::Float)
    //                   {
    //               } else if (output.scalar_type() ==
    //                              embedding_table.scalar_type() &&
    //                          embedding_table.scalar_type() ==
    //                              at::ScalarType::Half) {
    //                 constexpr int kVecWidth = 8;
    //               }

    //             } else {
    //               throw std::runtime_error(
    //                   "embedding_dim is greater than 1024");
    //             }
    //           });
    //     });
  });

  //   //   DISPATCH_INTEGRAL_FUNCTION(indices.scalar_type(), IndexType, [&] {
  //   //     DISPATCH_FLOAT_AND_HALF_FUNCTION(
  //   //         embedding_table.scalar_type(), EmbeddingType, [&] {
  //   //           auto indices_ptr = indices.data_ptr<IndexType>();
  //   //           auto embedding_table_ptr =
  //   //           embedding_table.data_ptr<EmbeddingType>(); auto output_ptr =
  //   //           output.data_ptr<OutputType>();
  //   //         });
  //   //   });
}
