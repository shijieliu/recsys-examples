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
#include "device_count.h"
#include "optimizer.h"
#include "optimizer_kernel.cuh"
#include "torch_utils.h"
#include <c10/cuda/CUDAGuard.h>
#include "utils.h"
#include <cmath>
#include <functional>
#include <optional>
#include <type_traits>

namespace dyn_emb {

constexpr int MULTIPLIER = 4;
constexpr int WARPSIZE = 32;
constexpr int OPTIMIZER_BLOCKSIZE_VEC = 64;
constexpr int OPTIMIZER_BLOCKSIZE = 1024;

template <typename GradType, typename WeightType, typename IndexType,
          typename OptimizerType>
void launch_update_kernel_for_flat_table(
    GradType *grads, int64_t *table_ptrs, IndexType *indices,
    int64_t *table_ids, int64_t *table_value_dims, int64_t *table_emb_dims,
    OptimizerType opt, int64_t const ev_nums, uint32_t const grad_stride,
    uint32_t const max_emb_dim, bool all_dims_vec4, int device_id,
    const int64_t *num_valid,
    std::function<float(int)> smem_size_f = [](int block_size) { return 0; }) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto &device_prop = DeviceProp::getDeviceProp(device_id);
  if (all_dims_vec4) {
    const int max_grid_size =
        device_prop.num_sms *
        (device_prop.max_thread_per_sm / OPTIMIZER_BLOCKSIZE_VEC);
    const int warp_per_block = OPTIMIZER_BLOCKSIZE_VEC / WARPSIZE;

    int grid_size = 0;
    if (ev_nums / warp_per_block < max_grid_size) {
      grid_size = (ev_nums - 1) / warp_per_block + 1;
    } else if (ev_nums / warp_per_block > max_grid_size * MULTIPLIER) {
      grid_size = max_grid_size * MULTIPLIER;
    } else {
      grid_size = max_grid_size;
    }

    auto kernel =
        update4_with_index_flat_table_kernel<GradType, WeightType, IndexType,
                                             OptimizerType>;
    kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
        ev_nums, grad_stride, grads, table_ptrs, indices, table_ids,
        table_value_dims, table_emb_dims, opt, num_valid);
  } else {
    int block_size =
        max_emb_dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : max_emb_dim;
    const int grid_size = counted_grid(ev_nums, block_size);

    auto kernel = update_with_index_flat_table_kernel<GradType, WeightType,
                                                      IndexType, OptimizerType>;
    kernel<<<grid_size, block_size, smem_size_f(block_size), stream>>>(
        ev_nums, grad_stride, grads, table_ptrs, indices, table_ids,
        table_value_dims, table_emb_dims, opt, num_valid);
  }
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

// One CTA updates 32 short rows. Metadata is read consecutively and shared by
// the 16 lanes that cover every 64-float row. Every writer orders its value
// stores before the block releases the associated cache pins.
__global__ void sgd_update_and_unpin_cache_kernel(
    const float *grads, const int64_t *value_rows, const int64_t *slots,
    const int64_t *tids, const int64_t *ptrs, const int64_t *bucket_offsets,
    int64_t bucket_capacity, const int64_t *overflow_offsets, int64_t overflow_capacity,
    int64_t main_capacity, int64_t pin_count, int32_t *pins, int64_t bound, const int64_t *num_valid,
    float lr) {
  constexpr int block = 128, tile = 32, width = 64;
  const int t = threadIdx.x;
  const auto n = counted_limit(num_valid, bound);
  const int64_t begin = int64_t(blockIdx.x) * tile;
  if (begin >= n) return;
  __shared__ float *dest[tile];
  int64_t pin = -1;
  if (t < tile) {
    const int64_t i = begin + t;
    dest[t] = nullptr;
    if (i < n) {
      const int64_t row = value_rows[i], slot = slots[i], tid = tids[i];
      if (row >= 0) dest[t] = reinterpret_cast<float *>(ptrs[tid]) + row * width;
      if (slot >= 0) {
        const int64_t overflow = overflow_offsets[tid];
        pin = slot < overflow ? bucket_offsets[tid] * bucket_capacity + slot
                              : main_capacity + tid * overflow_capacity + slot - overflow;
      }
    }
  }
  __syncthreads();
  float4 weight[4];
  float4 gradient[4];
  float *outputs[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int vector = j * block + t, row = vector / 16, col = (vector % 16) * 4;
    outputs[j] = dest[row] ? dest[row] + col : nullptr;
    if (!outputs[j]) continue;
    const auto *src = grads + (begin + row) * width + col;
    if (!(reinterpret_cast<uintptr_t>(src) & 15))
      gradient[j] = *reinterpret_cast<const float4 *>(src);
    else
      memcpy(&gradient[j], src, sizeof(float4));
    if (!(reinterpret_cast<uintptr_t>(outputs[j]) & 15))
      weight[j] = *reinterpret_cast<float4 *>(outputs[j]);
    else
      memcpy(&weight[j], outputs[j], sizeof(float4));
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    if (!outputs[j]) continue;
    auto &w = weight[j];
    const auto g = gradient[j];
    w.x += g.x * -lr;
    w.y += g.y * -lr;
    w.z += g.z * -lr;
    w.w += g.w * -lr;
    if (!(reinterpret_cast<uintptr_t>(outputs[j]) & 15))
      *reinterpret_cast<float4 *>(outputs[j]) = w;
    else
      memcpy(outputs[j], &w, sizeof(float4));
  }
  __threadfence();
  __syncthreads();
  if (pin >= 0 && pin < pin_count) ::atomicAdd(pins + pin, -1);
}

void sgd_update_and_unpin_cache(
    at::Tensor grads, at::Tensor rows, at::Tensor slots, at::Tensor tids, at::Tensor ptrs,
    at::Tensor bucket_offsets, int64_t bucket_capacity, at::Tensor overflow_offsets,
    int64_t overflow_capacity, at::Tensor pins, at::Tensor num_valid, float lr) {
  TORCH_CHECK(grads.is_cuda() && grads.is_contiguous() && grads.scalar_type() == at::kFloat &&
              grads.dim() == 2 && grads.size(1) == 64, "expected contiguous 64-column FP32 CUDA gradients");
  const auto n = grads.size(0), tables = ptrs.numel();
  const auto check = [&](const at::Tensor &t, int64_t len, at::ScalarType type) {
    TORCH_CHECK(t.is_contiguous() && t.device() == grads.device() && t.dim() == 1 &&
                t.numel() == len && t.scalar_type() == type, "invalid cache SGD metadata");
  };
  check(rows,n,at::kLong); check(slots,n,at::kLong); check(tids,n,at::kLong);
  check(ptrs,tables,at::kLong); check(bucket_offsets,tables+1,at::kLong);
  check(overflow_offsets,tables,at::kLong); check(pins,pins.numel(),at::kInt);
  TORCH_CHECK(tables > 0 && bucket_capacity > 0 && overflow_capacity > 0 &&
              pins.numel() >= tables * overflow_capacity, "invalid cache capacities");
  const c10::cuda::CUDAGuard guard(grads.device());
  const auto *count = counted_num_valid_ptr(num_valid, grads.device());
  if (n) {
    sgd_update_and_unpin_cache_kernel<<<(n + 31) / 32,128,0,at::cuda::getCurrentCUDAStream()>>>(
        grads.data_ptr<float>(), rows.data_ptr<int64_t>(), slots.data_ptr<int64_t>(),
        tids.data_ptr<int64_t>(), ptrs.data_ptr<int64_t>(), bucket_offsets.data_ptr<int64_t>(),
        bucket_capacity, overflow_offsets.data_ptr<int64_t>(), overflow_capacity,
        pins.numel() - tables * overflow_capacity, pins.numel(), pins.data_ptr<int32_t>(), n, count, lr);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void sgd_update_for_flat_table(at::Tensor grads, at::Tensor indices,
                               at::Tensor table_ptrs, at::Tensor table_ids,
                               at::Tensor table_value_dims,
                               at::Tensor table_emb_dims, int64_t max_emb_dim,
                               bool all_dims_vec4, float const lr,
                               int64_t table_dtype,
                               const std::optional<at::Tensor> &num_valid,
                               std::optional<int64_t> host_num_valid) {
  const int64_t *d_num_valid = counted_num_valid_ptr(num_valid, grads.device());
  const int64_t ev_nums = counted_host_limit(host_num_valid, grads.size(0));
  uint32_t grad_stride = grads.size(1);
  if (ev_nums == 0)
    return;
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");

  uint32_t max_emb_dim_u32 = static_cast<uint32_t>(max_emb_dim);

  auto grad_type = get_data_type(grads);
  auto val_type = static_cast<DataType>(table_dtype);
  auto index_type = get_data_type(indices);
  int device_id = grads.device().index();

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      DISPATCH_OFFSET_INT_TYPE(index_type, i_t, [&] {
        auto grad_ptr = get_pointer<g_t>(grads);
        auto table_ptrs_ptr = get_pointer<int64_t>(table_ptrs);
        auto index_ptr = get_pointer<i_t>(indices);
        auto tid_ptr = get_pointer<int64_t>(table_ids);
        auto tvd_ptr = get_pointer<int64_t>(table_value_dims);
        auto ted_ptr = get_pointer<int64_t>(table_emb_dims);

        SGDVecOptimizer<g_t, w_t> opt{lr};

        launch_update_kernel_for_flat_table<g_t, w_t, i_t, decltype(opt)>(
            grad_ptr, table_ptrs_ptr, index_ptr, tid_ptr, tvd_ptr, ted_ptr, opt,
            ev_nums, grad_stride, max_emb_dim_u32, all_dims_vec4, device_id,
            d_num_valid);
      });
    });
  });
}

void adam_update_for_flat_table(at::Tensor grads, at::Tensor indices,
                                at::Tensor table_ptrs, at::Tensor table_ids,
                                at::Tensor table_value_dims,
                                at::Tensor table_emb_dims, const float lr,
                                const float beta1, const float beta2,
                                const float eps, const float weight_decay,
                                const uint32_t iter_num, int64_t max_emb_dim,
                                bool all_dims_vec4, int64_t table_dtype,
                                const std::optional<at::Tensor> &num_valid,
                                std::optional<int64_t> host_num_valid) {
  const int64_t *d_num_valid = counted_num_valid_ptr(num_valid, grads.device());
  const int64_t ev_nums = counted_host_limit(host_num_valid, grads.size(0));
  uint32_t grad_stride = grads.size(1);
  if (ev_nums == 0)
    return;
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");

  uint32_t max_emb_dim_u32 = static_cast<uint32_t>(max_emb_dim);

  auto grad_type = get_data_type(grads);
  auto val_type = static_cast<DataType>(table_dtype);
  auto index_type = get_data_type(indices);
  int device_id = grads.device().index();

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      DISPATCH_OFFSET_INT_TYPE(index_type, i_t, [&] {
        auto grad_ptr = get_pointer<g_t>(grads);
        auto table_ptrs_ptr = get_pointer<int64_t>(table_ptrs);
        auto index_ptr = get_pointer<i_t>(indices);
        auto tid_ptr = get_pointer<int64_t>(table_ids);
        auto tvd_ptr = get_pointer<int64_t>(table_value_dims);
        auto ted_ptr = get_pointer<int64_t>(table_emb_dims);

        AdamVecOptimizer<g_t, w_t> opt{lr,  beta1,        beta2,
                                       eps, weight_decay, iter_num};

        launch_update_kernel_for_flat_table<g_t, w_t, i_t, decltype(opt)>(
            grad_ptr, table_ptrs_ptr, index_ptr, tid_ptr, tvd_ptr, ted_ptr, opt,
            ev_nums, grad_stride, max_emb_dim_u32, all_dims_vec4, device_id,
            d_num_valid);
      });
    });
  });
}

void adagrad_update_for_flat_table(at::Tensor grads, at::Tensor indices,
                                   at::Tensor table_ptrs, at::Tensor table_ids,
                                   at::Tensor table_value_dims,
                                   at::Tensor table_emb_dims, const float lr,
                                   const float eps, int64_t max_emb_dim,
                                   bool all_dims_vec4, int64_t table_dtype,
                                   const std::optional<at::Tensor> &num_valid,
                                   std::optional<int64_t> host_num_valid) {
  const int64_t *d_num_valid = counted_num_valid_ptr(num_valid, grads.device());
  const int64_t ev_nums = counted_host_limit(host_num_valid, grads.size(0));
  uint32_t grad_stride = grads.size(1);
  if (ev_nums == 0)
    return;
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");

  uint32_t max_emb_dim_u32 = static_cast<uint32_t>(max_emb_dim);

  auto grad_type = get_data_type(grads);
  auto val_type = static_cast<DataType>(table_dtype);
  auto index_type = get_data_type(indices);
  int device_id = grads.device().index();

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      DISPATCH_OFFSET_INT_TYPE(index_type, i_t, [&] {
        auto grad_ptr = get_pointer<g_t>(grads);
        auto table_ptrs_ptr = get_pointer<int64_t>(table_ptrs);
        auto index_ptr = get_pointer<i_t>(indices);
        auto tid_ptr = get_pointer<int64_t>(table_ids);
        auto tvd_ptr = get_pointer<int64_t>(table_value_dims);
        auto ted_ptr = get_pointer<int64_t>(table_emb_dims);

        AdaGradVecOptimizer<g_t, w_t> opt{lr, eps};

        launch_update_kernel_for_flat_table<g_t, w_t, i_t, decltype(opt)>(
            grad_ptr, table_ptrs_ptr, index_ptr, tid_ptr, tvd_ptr, ted_ptr, opt,
            ev_nums, grad_stride, max_emb_dim_u32, all_dims_vec4, device_id,
            d_num_valid);
      });
    });
  });
}

// Calls fn with a value of the row-wise Adagrad accumulator's type: the table's
// own type, or float. No other pairing has a kernel.
template <typename w_t, typename Fn>
void dispatch_rowwise_adagrad_state_type(DataType state_type,
                                         DataType val_type, Fn &&fn) {
  TORCH_CHECK(state_type == val_type || state_type == DataType::Float32,
              "row-wise Adagrad keeps its accumulator in the table's dtype or "
              "in float32");
  if constexpr (std::is_same_v<w_t, dyn_fp8_t>) {
    TORCH_CHECK(state_type == DataType::Float32,
                "row-wise Adagrad on an fp8 table keeps its accumulator in "
                "float32");
    fn(float{});
  } else {
    if (state_type == val_type) {
      fn(w_t{});
    } else {
      fn(float{});
    }
  }
}

void rowwise_adagrad_for_flat_table(
    at::Tensor grads, at::Tensor indices, at::Tensor table_ptrs,
    at::Tensor table_ids, at::Tensor table_value_dims,
    at::Tensor table_emb_dims, const float lr, const float eps,
    int64_t max_emb_dim, bool all_dims_vec4, int64_t table_dtype,
    int64_t state_dtype, const std::optional<at::Tensor> &num_valid,
    std::optional<int64_t> host_num_valid) {
  const int64_t *d_num_valid = counted_num_valid_ptr(num_valid, grads.device());
  const int64_t ev_nums = counted_host_limit(host_num_valid, grads.size(0));
  uint32_t grad_stride = grads.size(1);
  if (ev_nums == 0)
    return;
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");

  uint32_t max_emb_dim_u32 = static_cast<uint32_t>(max_emb_dim);

  auto grad_type = get_data_type(grads);
  auto val_type = static_cast<DataType>(table_dtype);
  auto state_type = static_cast<DataType>(state_dtype);
  auto index_type = get_data_type(indices);
  int device_id = grads.device().index();

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_VALUE_DATATYPE_FUNCTION(val_type, w_t, [&] {
      dispatch_rowwise_adagrad_state_type<w_t>(
          state_type, val_type, [&](auto state_tag) {
            using s_t = decltype(state_tag);
            DISPATCH_OFFSET_INT_TYPE(index_type, i_t, [&] {
              auto grad_ptr = get_pointer<g_t>(grads);
              auto table_ptrs_ptr = get_pointer<int64_t>(table_ptrs);
              auto index_ptr = get_pointer<i_t>(indices);
              auto tid_ptr = get_pointer<int64_t>(table_ids);
              auto tvd_ptr = get_pointer<int64_t>(table_value_dims);
              auto ted_ptr = get_pointer<int64_t>(table_emb_dims);

              RowWiseAdaGradVecOptimizer<g_t, w_t, s_t> opt{lr, eps};

              launch_update_kernel_for_flat_table<g_t, w_t, i_t,
                                                  decltype(opt)>(
                  grad_ptr, table_ptrs_ptr, index_ptr, tid_ptr, tvd_ptr,
                  ted_ptr, opt, ev_nums, grad_stride, max_emb_dim_u32,
                  all_dims_vec4, device_id, d_num_valid,
                  [](int block_size) { return block_size * sizeof(float); });
            });
          });
    });
  });
}

// The accumulator is raised to -learning_rate_power, so -0.5 asks for a square
// root -- the exponent the paper fixes, and by far the common choice. Test for
// it here so the kernel can call sqrtf instead of a general powf. Exact
// comparison is right: -0.5 is representable, and a value merely near it is a
// different exponent the caller meant to ask for.
namespace {
inline bool ftrl_uses_sqrt(float learning_rate_power) {
  return learning_rate_power == -0.5f;
}
} // namespace

void ftrl_update_for_flat_table(at::Tensor grads, at::Tensor indices,
                                at::Tensor table_ptrs, at::Tensor table_ids,
                                at::Tensor table_value_dims,
                                at::Tensor table_emb_dims, const float lr,
                                const float learning_rate_power,
                                const float beta, const float l1_reg,
                                const float l2_reg, int64_t max_emb_dim,
                                bool all_dims_vec4, int64_t table_dtype,
                                const float group_lasso_reg,
                                const std::optional<at::Tensor> &num_valid,
                                std::optional<int64_t> host_num_valid) {
  const int64_t *d_num_valid = counted_num_valid_ptr(num_valid, grads.device());
  const int64_t ev_nums = counted_host_limit(host_num_valid, grads.size(0));
  uint32_t grad_stride = grads.size(1);
  if (ev_nums == 0)
    return;
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(lr > 0.0f, "FTRL learning rate must be positive, got ", lr);
  TORCH_CHECK(group_lasso_reg >= 0.0f, "FTRL group_lasso_reg must be non-negative, got ",
              group_lasso_reg);

  uint32_t max_emb_dim_u32 = static_cast<uint32_t>(max_emb_dim);
  const bool use_sqrt = ftrl_uses_sqrt(learning_rate_power);
  const bool use_group = group_lasso_reg > 0.0f;

  auto grad_type = get_data_type(grads);
  auto val_type = static_cast<DataType>(table_dtype);
  auto index_type = get_data_type(indices);
  int device_id = grads.device().index();

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      DISPATCH_OFFSET_INT_TYPE(index_type, i_t, [&] {
        auto grad_ptr = get_pointer<g_t>(grads);
        auto table_ptrs_ptr = get_pointer<int64_t>(table_ptrs);
        auto index_ptr = get_pointer<i_t>(indices);
        auto tid_ptr = get_pointer<int64_t>(table_ids);
        auto tvd_ptr = get_pointer<int64_t>(table_value_dims);
        auto ted_ptr = get_pointer<int64_t>(table_emb_dims);

        FTRLVecOptimizer<g_t, w_t> opt{lr,     learning_rate_power,
                                       use_sqrt, beta,
                                       l1_reg, l2_reg,
                                       group_lasso_reg};

        // Only the group-sparse path reduces across the block, and it needs
        // one float of shared memory per thread to do it; plain FTRL asks for
        // none, as before.
        launch_update_kernel_for_flat_table<g_t, w_t, i_t, decltype(opt)>(
            grad_ptr, table_ptrs_ptr, index_ptr, tid_ptr, tvd_ptr, ted_ptr, opt,
            ev_nums, grad_stride, max_emb_dim_u32, all_dims_vec4, device_id,
            d_num_valid, [use_group](int block_size) {
              return use_group ? block_size * sizeof(float) : 0;
            });
      });
    });
  });
}

template <typename GradType, typename WeightType, typename OptimizerType>
void launch_update_kernel_for_padded_buffer(
    GradType *grads, WeightType *values, OptimizerType opt,
    int64_t const num_rows, uint32_t const grad_stride,
    uint32_t const value_stride, uint32_t const emb_dim, bool all_dims_vec4,
    int device_id, int64_t const *table_ids, int64_t const *table_emb_dims,
    const int64_t *num_valid,
    std::function<float(int)> smem_size_f = [](int block_size) { return 0; }) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto &device_prop = DeviceProp::getDeviceProp(device_id);
  if (all_dims_vec4) {
    const int max_grid_size =
        device_prop.num_sms *
        (device_prop.max_thread_per_sm / OPTIMIZER_BLOCKSIZE_VEC);
    const int warp_per_block = OPTIMIZER_BLOCKSIZE_VEC / WARPSIZE;
    int grid_size = 0;
    if (num_rows / warp_per_block < max_grid_size) {
      grid_size = (num_rows - 1) / warp_per_block + 1;
    } else if (num_rows / warp_per_block > max_grid_size * MULTIPLIER) {
      grid_size = max_grid_size * MULTIPLIER;
    } else {
      grid_size = max_grid_size;
    }
    update4_padded_buffer_kernel<GradType, WeightType, OptimizerType>
        <<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            num_rows, grad_stride, value_stride, emb_dim, grads, values,
            table_ids, table_emb_dims, opt, num_valid);
  } else {
    int block_size =
        emb_dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : emb_dim;
    const int grid_size = counted_grid(num_rows, block_size);
    update_padded_buffer_kernel<GradType, WeightType, OptimizerType>
        <<<grid_size, block_size, smem_size_f(block_size), stream>>>(
            num_rows, grad_stride, value_stride, emb_dim, grads, values,
            table_ids, table_emb_dims, opt, num_valid);
  }
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void sgd_update_for_padded_buffer(at::Tensor grads, at::Tensor values,
                                  at::Tensor table_ids,
                                  at::Tensor table_emb_dims, int64_t emb_dim,
                                  int64_t value_dim, bool all_dims_vec4,
                                  float lr,
                                  const std::optional<at::Tensor> &num_valid,
                                  std::optional<int64_t> host_num_valid) {
  const int64_t *d_num_valid = counted_num_valid_ptr(num_valid, grads.device());
  const int64_t num_rows = counted_host_limit(host_num_valid, grads.size(0));
  uint32_t grad_stride = grads.size(1);
  if (num_rows == 0)
    return;
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK(table_ids.is_cuda(), "table_ids must be a CUDA tensor");
  TORCH_CHECK(table_emb_dims.is_cuda(),
              "table_emb_dims must be a CUDA tensor");
  uint32_t emb_dim_u32 = static_cast<uint32_t>(emb_dim);
  uint32_t value_stride = static_cast<uint32_t>(value_dim);
  auto grad_type = get_data_type(grads);
  auto val_type = get_data_type(values);
  int device_id = grads.device().index();
  auto tid_ptr = get_pointer<int64_t>(table_ids);
  auto ted_ptr = get_pointer<int64_t>(table_emb_dims);
  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      SGDVecOptimizer<g_t, w_t> opt{lr};
      launch_update_kernel_for_padded_buffer<g_t, w_t, decltype(opt)>(
          get_pointer<g_t>(grads), get_pointer<w_t>(values), opt, num_rows,
          grad_stride, value_stride, emb_dim_u32, all_dims_vec4, device_id,
          tid_ptr, ted_ptr, d_num_valid);
    });
  });
}

void adam_update_for_padded_buffer(at::Tensor grads, at::Tensor values,
                                   at::Tensor table_ids,
                                   at::Tensor table_emb_dims, int64_t emb_dim,
                                   int64_t value_dim, bool all_dims_vec4,
                                   float lr, float beta1, float beta2,
                                   float eps, float weight_decay,
                                   uint32_t iter_num,
                                   const std::optional<at::Tensor> &num_valid,
                                   std::optional<int64_t> host_num_valid) {
  const int64_t *d_num_valid = counted_num_valid_ptr(num_valid, grads.device());
  const int64_t num_rows = counted_host_limit(host_num_valid, grads.size(0));
  uint32_t grad_stride = grads.size(1);
  if (num_rows == 0)
    return;
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK(table_ids.is_cuda(), "table_ids must be a CUDA tensor");
  TORCH_CHECK(table_emb_dims.is_cuda(),
              "table_emb_dims must be a CUDA tensor");
  uint32_t emb_dim_u32 = static_cast<uint32_t>(emb_dim);
  uint32_t value_stride = static_cast<uint32_t>(value_dim);
  auto grad_type = get_data_type(grads);
  auto val_type = get_data_type(values);
  int device_id = grads.device().index();
  auto tid_ptr = get_pointer<int64_t>(table_ids);
  auto ted_ptr = get_pointer<int64_t>(table_emb_dims);
  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      AdamVecOptimizer<g_t, w_t> opt{lr,  beta1,        beta2,
                                     eps, weight_decay, iter_num};
      launch_update_kernel_for_padded_buffer<g_t, w_t, decltype(opt)>(
          get_pointer<g_t>(grads), get_pointer<w_t>(values), opt, num_rows,
          grad_stride, value_stride, emb_dim_u32, all_dims_vec4, device_id,
          tid_ptr, ted_ptr, d_num_valid);
    });
  });
}

void adagrad_update_for_padded_buffer(
    at::Tensor grads, at::Tensor values, at::Tensor table_ids,
    at::Tensor table_emb_dims, int64_t emb_dim, int64_t value_dim,
    bool all_dims_vec4, float lr, float eps,
    const std::optional<at::Tensor> &num_valid,
    std::optional<int64_t> host_num_valid) {
  const int64_t *d_num_valid = counted_num_valid_ptr(num_valid, grads.device());
  const int64_t num_rows = counted_host_limit(host_num_valid, grads.size(0));
  uint32_t grad_stride = grads.size(1);
  if (num_rows == 0)
    return;
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK(table_ids.is_cuda(), "table_ids must be a CUDA tensor");
  TORCH_CHECK(table_emb_dims.is_cuda(),
              "table_emb_dims must be a CUDA tensor");
  uint32_t emb_dim_u32 = static_cast<uint32_t>(emb_dim);
  uint32_t value_stride = static_cast<uint32_t>(value_dim);
  auto grad_type = get_data_type(grads);
  auto val_type = get_data_type(values);
  int device_id = grads.device().index();
  auto tid_ptr = get_pointer<int64_t>(table_ids);
  auto ted_ptr = get_pointer<int64_t>(table_emb_dims);
  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      AdaGradVecOptimizer<g_t, w_t> opt{lr, eps};
      launch_update_kernel_for_padded_buffer<g_t, w_t, decltype(opt)>(
          get_pointer<g_t>(grads), get_pointer<w_t>(values), opt, num_rows,
          grad_stride, value_stride, emb_dim_u32, all_dims_vec4, device_id,
          tid_ptr, ted_ptr, d_num_valid);
    });
  });
}

void rowwise_adagrad_for_padded_buffer(
    at::Tensor grads, at::Tensor values, at::Tensor table_ids,
    at::Tensor table_emb_dims, int64_t emb_dim, int64_t value_dim,
    bool all_dims_vec4, float lr, float eps, int64_t state_dtype,
    const std::optional<at::Tensor> &num_valid,
    std::optional<int64_t> host_num_valid) {
  const int64_t *d_num_valid = counted_num_valid_ptr(num_valid, grads.device());
  const int64_t num_rows = counted_host_limit(host_num_valid, grads.size(0));
  uint32_t grad_stride = grads.size(1);
  if (num_rows == 0)
    return;
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK(table_ids.is_cuda(), "table_ids must be a CUDA tensor");
  TORCH_CHECK(table_emb_dims.is_cuda(),
              "table_emb_dims must be a CUDA tensor");
  uint32_t emb_dim_u32 = static_cast<uint32_t>(emb_dim);
  uint32_t value_stride = static_cast<uint32_t>(value_dim);
  auto grad_type = get_data_type(grads);
  auto val_type = get_data_type(values);
  auto state_type = static_cast<DataType>(state_dtype);
  int device_id = grads.device().index();
  auto tid_ptr = get_pointer<int64_t>(table_ids);
  auto ted_ptr = get_pointer<int64_t>(table_emb_dims);
  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_VALUE_DATATYPE_FUNCTION(val_type, w_t, [&] {
      dispatch_rowwise_adagrad_state_type<w_t>(
          state_type, val_type, [&](auto state_tag) {
            using s_t = decltype(state_tag);
            RowWiseAdaGradVecOptimizer<g_t, w_t, s_t> opt{lr, eps};
            launch_update_kernel_for_padded_buffer<g_t, w_t, decltype(opt)>(
                get_pointer<g_t>(grads), get_pointer<w_t>(values), opt,
                num_rows, grad_stride, value_stride, emb_dim_u32,
                all_dims_vec4, device_id, tid_ptr, ted_ptr, d_num_valid,
                [](int block_size) { return block_size * sizeof(float); });
          });
    });
  });
}

void ftrl_update_for_padded_buffer(at::Tensor grads, at::Tensor values,
                                   at::Tensor table_ids,
                                   at::Tensor table_emb_dims, int64_t emb_dim,
                                   int64_t value_dim, bool all_dims_vec4,
                                   float lr, float learning_rate_power,
                                   float beta, float l1_reg, float l2_reg,
                                   float group_lasso_reg,
                                   const std::optional<at::Tensor> &num_valid,
                                   std::optional<int64_t> host_num_valid) {
  const int64_t *d_num_valid = counted_num_valid_ptr(num_valid, grads.device());
  const int64_t num_rows = counted_host_limit(host_num_valid, grads.size(0));
  uint32_t grad_stride = grads.size(1);
  if (num_rows == 0)
    return;
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK(table_ids.is_cuda(), "table_ids must be a CUDA tensor");
  TORCH_CHECK(table_emb_dims.is_cuda(),
              "table_emb_dims must be a CUDA tensor");
  TORCH_CHECK(lr > 0.0f, "FTRL learning rate must be positive, got ", lr);
  TORCH_CHECK(group_lasso_reg >= 0.0f, "FTRL group_lasso_reg must be non-negative, got ",
              group_lasso_reg);
  uint32_t emb_dim_u32 = static_cast<uint32_t>(emb_dim);
  uint32_t value_stride = static_cast<uint32_t>(value_dim);
  const bool use_sqrt = ftrl_uses_sqrt(learning_rate_power);
  const bool use_group = group_lasso_reg > 0.0f;
  auto grad_type = get_data_type(grads);
  auto val_type = get_data_type(values);
  int device_id = grads.device().index();
  auto tid_ptr = get_pointer<int64_t>(table_ids);
  auto ted_ptr = get_pointer<int64_t>(table_emb_dims);
  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      FTRLVecOptimizer<g_t, w_t> opt{lr,     learning_rate_power,
                                     use_sqrt, beta,
                                     l1_reg, l2_reg,
                                     group_lasso_reg};
      launch_update_kernel_for_padded_buffer<g_t, w_t, decltype(opt)>(
          get_pointer<g_t>(grads), get_pointer<w_t>(values), opt, num_rows,
          grad_stride, value_stride, emb_dim_u32, all_dims_vec4, device_id,
          tid_ptr, ted_ptr, d_num_valid, [use_group](int block_size) {
            return use_group ? block_size * sizeof(float) : 0;
          });
    });
  });
}

} // namespace dyn_emb

// PYTHON WRAP
// Every update reads the live gradient row count on the device.
void bind_optimizer_kernel_op(py::module &m) {
  m.def("sgd_update_and_unpin_cache", &dyn_emb::sgd_update_and_unpin_cache);
  m.def("sgd_update_for_flat_table", &dyn_emb::sgd_update_for_flat_table,
        "SGD optimizer for multi-table buffer via table_ptrs", py::arg("grads"),
        py::arg("indices"), py::arg("table_ptrs"), py::arg("table_ids"),
        py::arg("table_value_dims"), py::arg("table_emb_dims"),
        py::arg("max_emb_dim"), py::arg("all_dims_vec4"), py::arg("lr"),
        py::arg("table_dtype"), py::arg("num_valid") = py::none(),
        py::arg("host_num_valid") = py::none());

  m.def("adam_update_for_flat_table", &dyn_emb::adam_update_for_flat_table,
        "Adam optimizer for multi-table buffer via table_ptrs",
        py::arg("grads"), py::arg("indices"), py::arg("table_ptrs"),
        py::arg("table_ids"), py::arg("table_value_dims"),
        py::arg("table_emb_dims"), py::arg("lr"), py::arg("beta1"),
        py::arg("beta2"), py::arg("eps"), py::arg("weight_decay"),
        py::arg("iter_num"), py::arg("max_emb_dim"), py::arg("all_dims_vec4"),
        py::arg("table_dtype"), py::arg("num_valid") = py::none(),
        py::arg("host_num_valid") = py::none());

  m.def(
      "adagrad_update_for_flat_table", &dyn_emb::adagrad_update_for_flat_table,
      "Adagrad optimizer for multi-table buffer via table_ptrs",
      py::arg("grads"), py::arg("indices"), py::arg("table_ptrs"),
      py::arg("table_ids"), py::arg("table_value_dims"),
      py::arg("table_emb_dims"), py::arg("lr"), py::arg("eps"),
      py::arg("max_emb_dim"), py::arg("all_dims_vec4"), py::arg("table_dtype"),
      py::arg("num_valid") = py::none(), py::arg("host_num_valid") = py::none());

  m.def("rowwise_adagrad_for_flat_table",
        &dyn_emb::rowwise_adagrad_for_flat_table,
        "Row Wise Adagrad optimizer for multi-table buffer via table_ptrs",
        py::arg("grads"), py::arg("indices"), py::arg("table_ptrs"),
        py::arg("table_ids"), py::arg("table_value_dims"),
        py::arg("table_emb_dims"), py::arg("lr"), py::arg("eps"),
        py::arg("max_emb_dim"), py::arg("all_dims_vec4"),
        py::arg("table_dtype"), py::arg("state_dtype"),
        py::arg("num_valid") = py::none(), py::arg("host_num_valid") = py::none());

  m.def("ftrl_update_for_flat_table", &dyn_emb::ftrl_update_for_flat_table,
        "FTRL optimizer for multi-table buffer via table_ptrs", py::arg("grads"),
        py::arg("indices"), py::arg("table_ptrs"), py::arg("table_ids"),
        py::arg("table_value_dims"), py::arg("table_emb_dims"), py::arg("lr"),
        py::arg("learning_rate_power"), py::arg("beta"), py::arg("l1_reg"),
        py::arg("l2_reg"), py::arg("max_emb_dim"), py::arg("all_dims_vec4"),
        py::arg("table_dtype"), py::arg("group_lasso_reg") = 0.0f, py::kw_only(),
        py::arg("num_valid") = py::none(), py::arg("host_num_valid") = py::none());

  m.def("sgd_update_for_padded_buffer", &dyn_emb::sgd_update_for_padded_buffer,
        "SGD optimizer for contiguous padded buffer", py::arg("grads"),
        py::arg("values"), py::arg("table_ids"), py::arg("table_emb_dims"),
        py::arg("emb_dim"), py::arg("value_dim"), py::arg("all_dims_vec4"),
        py::arg("lr"), py::arg("num_valid") = py::none(),
        py::arg("host_num_valid") = py::none());

  m.def("adam_update_for_padded_buffer",
        &dyn_emb::adam_update_for_padded_buffer,
        "Adam optimizer for contiguous padded buffer", py::arg("grads"),
        py::arg("values"), py::arg("table_ids"), py::arg("table_emb_dims"),
        py::arg("emb_dim"), py::arg("value_dim"), py::arg("all_dims_vec4"),
        py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
        py::arg("weight_decay"), py::arg("iter_num"),
        py::arg("num_valid") = py::none(), py::arg("host_num_valid") = py::none());

  m.def("adagrad_update_for_padded_buffer",
        &dyn_emb::adagrad_update_for_padded_buffer,
        "Adagrad optimizer for contiguous padded buffer", py::arg("grads"),
        py::arg("values"), py::arg("table_ids"), py::arg("table_emb_dims"),
        py::arg("emb_dim"), py::arg("value_dim"), py::arg("all_dims_vec4"),
        py::arg("lr"), py::arg("eps"), py::arg("num_valid") = py::none(),
        py::arg("host_num_valid") = py::none());

  m.def("rowwise_adagrad_for_padded_buffer",
        &dyn_emb::rowwise_adagrad_for_padded_buffer,
        "Row Wise Adagrad optimizer for contiguous padded buffer",
        py::arg("grads"), py::arg("values"), py::arg("table_ids"),
        py::arg("table_emb_dims"), py::arg("emb_dim"), py::arg("value_dim"),
        py::arg("all_dims_vec4"), py::arg("lr"), py::arg("eps"),
        py::arg("state_dtype"), py::arg("num_valid") = py::none(),
        py::arg("host_num_valid") = py::none());

  m.def("ftrl_update_for_padded_buffer",
        &dyn_emb::ftrl_update_for_padded_buffer,
        "FTRL optimizer for contiguous padded buffer", py::arg("grads"),
        py::arg("values"), py::arg("table_ids"), py::arg("table_emb_dims"),
        py::arg("emb_dim"), py::arg("value_dim"), py::arg("all_dims_vec4"),
        py::arg("lr"), py::arg("learning_rate_power"), py::arg("beta"),
        py::arg("l1_reg"), py::arg("l2_reg"), py::arg("group_lasso_reg") = 0.0f, py::kw_only(),
        py::arg("num_valid") = py::none(), py::arg("host_num_valid") = py::none());
}
