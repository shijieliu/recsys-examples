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
// The output dist's kernels, fwd_kernel and bwd_kernel (forward.cuh,
// backward.cuh: this is the one translation unit that holds them), and the
// out channel that launches them, one per direction of a call on the
// caller's stream (output_dist.h); then the kernels' trace.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <nccl_device.h>

#include <algorithm>
#include <cstdint>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/config.h"
#include "hier_output_dist/output_dist/args.cuh"
#include "hier_output_dist/output_dist/backward.cuh"
#include "hier_output_dist/output_dist/forward.cuh"
#include "hier_output_dist/output_dist/output_dist.h"
#include "hier_output_dist/output_dist/pooling.cuh"

namespace dyn_emb {
namespace hier_output_dist {

namespace {

// The kernels of GIN backend `gin` (gin_context; NCCL_GIN_TYPE_NONE: one
// node).
template <bool kUnique>
const void *forward_kernel(ncclGinType_t gin) {
  switch (gin) {
    case NCCL_GIN_TYPE_PROXY:
      return (const void *)fwd_kernel<NCCL_GIN_TYPE_PROXY, kUnique>;
    case NCCL_GIN_TYPE_GDAKI:
      return (const void *)fwd_kernel<NCCL_GIN_TYPE_GDAKI, kUnique>;
    case NCCL_GIN_TYPE_GPI:
      return (const void *)fwd_kernel<NCCL_GIN_TYPE_GPI, kUnique>;
    default:
      return (const void *)fwd_kernel<NCCL_GIN_TYPE_NONE, kUnique>;
  }
}

template <typename RowT>
const void *backward_kernel_of(ncclGinType_t gin) {
  switch (gin) {
    case NCCL_GIN_TYPE_PROXY:
      return (const void *)bwd_kernel<RowT, NCCL_GIN_TYPE_PROXY>;
    case NCCL_GIN_TYPE_GDAKI:
      return (const void *)bwd_kernel<RowT, NCCL_GIN_TYPE_GDAKI>;
    case NCCL_GIN_TYPE_GPI:
      return (const void *)bwd_kernel<RowT, NCCL_GIN_TYPE_GPI>;
    default:
      return (const void *)bwd_kernel<RowT, NCCL_GIN_TYPE_NONE>;
  }
}

const void *backward_kernel(at::ScalarType dtype, ncclGinType_t gin) {
  TORCH_CHECK(dtype == at::kFloat || dtype == at::kBFloat16 || dtype == at::kHalf,
              "hier_output_dist: gradient rows must be fp32, bf16 or fp16, got ", dtype);
  return dtype == at::kFloat      ? backward_kernel_of<float>(gin)
         : dtype == at::kBFloat16 ? backward_kernel_of<__nv_bfloat16>(gin)
                                  : backward_kernel_of<__half>(gin);
}

// The heavy rows' sub-tasks of a backward whose gradient rows are of `dtype`
// (checked by backward_kernel).
const void *heavy_partial_kernel(at::ScalarType dtype) {
  return dtype == at::kFloat      ? (const void *)bwd_heavy_partial_kernel<float>
         : dtype == at::kBFloat16 ? (const void *)bwd_heavy_partial_kernel<__nv_bfloat16>
                                  : (const void *)bwd_heavy_partial_kernel<__half>;
}

// Pooled offsets describe the original KJT, before index dedup. The input
// dist already consumes that KJT, so these checks need no device-host sync.
int pooled_batch(const at::Tensor &offsets, const c10::optional<at::Tensor> &weights,
                 int device, int features, int64_t occurrences) {
  const int64_t bags = offsets.numel() - 1;
  TORCH_CHECK(offsets.is_cuda() && offsets.get_device() == device && offsets.is_contiguous() &&
                  offsets.dim() == 1 && (offsets.scalar_type() == at::kLong ||
                                        offsets.scalar_type() == at::kInt) &&
                  bags >= 0 && bags <= INT32_MAX && bags % features == 0,
              "hier_output_dist: expected contiguous CUDA offsets for each feature and sample");
  if (weights)
    TORCH_CHECK(weights->device() == offsets.device() && weights->is_contiguous() &&
                    weights->scalar_type() == at::kFloat && weights->dim() == 1 &&
                    weights->numel() == occurrences,
                "hier_output_dist: expected one contiguous fp32 CUDA weight per occurrence");
  return (int)(bags / features);
}

template <typename Launch>
void with_offsets(const at::Tensor &offsets, Launch launch) {
  if (offsets.scalar_type() == at::kLong)
    launch(offsets.data_ptr<int64_t>());
  else
    launch(offsets.data_ptr<int32_t>());
}

// Pass the row type as a tag to instantiate a kernel on that type.
template <typename Launch>
void with_row_type(at::ScalarType dtype, Launch launch) {
  if (dtype == at::kFloat)
    launch(float{});
  else if (dtype == at::kHalf)
    launch(__half{});
  else
    launch(__nv_bfloat16{});
}

int pooled_grid(int64_t items, int sms) {
  return (int)std::min<int64_t>((items + pooled::kBlock - 1) / pooled::kBlock, 16 * sms);
}

}  // namespace

// SMs left to NCCL's own kernels.
constexpr int kNcclReserve = 32;

OutChannel::OutChannel(const Config &config, const Sizes &sizes, ncclWindow_t window,
                       const ncclDevComm &dev, ncclGinType_t gin, const WindowLayout &layout,
                       int device)
    : sizes_(sizes),
      window_(window),
      dev_(dev),
      lookup_rows_(layout.rows.lookup_rows()),
      dim_(config.dim),
      dtype_(config.dtype),
      world_size_(config.world_size),
      device_(device),
      sms_(sms_of(device)),
      out_ctas_(config.out_ctas),
      reduce_ctas_(config.reduce_ctas),
      multi_node_(sizes.multi_node()),
      gin_(multi_node_ ? gin : NCCL_GIN_TYPE_NONE),
      trace_(device) {}

// DYNAMICEMB_HIER_OUT_CTAS if set. Else on one node 32 for the forward and 64
// for the backward: inside a node pulls saturate NVLink at about 32 CTAs, and
// pushes keep improving up to about 64. Across nodes 32 for both: InfiniBand
// bounds them under uniform keys, and under skew the hot owner's NVLink and
// the proxies' sums do (more CTAs crowd the backward's pushes). Every
// kernel that waits for peers must be resident at once: an out kernel leaves
// an SM to a comm kernel of an input dist on another stream, kNcclReserve to
// NCCL's own kernels, and one to spare. Across nodes the backward needs CTA 0
// (its issuer) and a reducing CTA at least.
int OutChannel::out_ctas(OutKernel which) const {
  const bool forward = which == OutKernel::kForward;
  const int want = out_ctas_ > 0 ? out_ctas_ : (forward || multi_node_ ? 32 : 64);
  const int least = multi_node_ && !forward ? 2 : 1;
  return std::max(least, std::min(want, sms_ - 2 - kNcclReserve));
}

int OutChannel::reducers(int ctas) const { return std::max(1, std::min(reduce_ctas_, ctas - 1)); }

// Storage that changes size only when a call needs more rows than it has or
// fewer than half of them (it then takes rows plus 1/32): batches of about
// the same size ask for one size, so the caching allocator hands each call a
// block an earlier output freed. resize_ keeps the storage and makes no view
// (autograd refuses in-place changes to a custom function's output that is a
// view).
at::Tensor OutChannel::output(int64_t rows) {
  if (rows > out_rows_ || 2 * rows < out_rows_) out_rows_ = rows + rows / 32;
  auto out =
      at::empty({out_rows_, dim_}, at::TensorOptions().device(at::kCUDA, device_).dtype(dtype_));
  return out.resize_({rows, dim_});
}

template <typename Args>
void OutChannel::launch(const void *kernel, OutKernel which, int ctas, Args &args) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_);
  int64_t seq = seq_ + 1, arrive0 = arrive_;
  args.trace = trace_.start_out(which, stream);
  void *params[] = {&window_, &dev_, &sizes_, &seq, &arrive0, &args};
  HIER_CUDA_CHECK(cudaLaunchCooperativeKernel(
      kernel, ctas, kThreads, params,
      out_shared_bytes(sizes_.topo.local_size, args.trace != nullptr), stream));
  trace_.launched(which);
  seq_ = seq;
  arrive_ += ctas;
}

at::Tensor OutChannel::forward(const FwdPlan &plan, const at::Tensor &lookup_out, bool unique) {
  c10::cuda::CUDAGuard guard(device_);
  TORCH_CHECK(
      lookup_out.numel() == 0 || (lookup_out.is_cuda() && lookup_out.data_ptr() == lookup_rows_),
      "hier_output_dist: the lookup output must be the rows lookup_rows() returned");
  at::Tensor out = output(unique ? plan.inv.numel() : plan.num_out());
  at::Tensor pulled_from_proxy;  // the kernel zeroes it
  if (multi_node_ && !unique)
    pulled_from_proxy = at::empty({sizes_.topo.local_size}, out.options().dtype(at::kLong));
  FwdArgs args{};
  args.plan = plan.view(sizes_);
  args.out = (char *)out.data_ptr();
  args.pulled_from_proxy =
      pulled_from_proxy.defined() ? pulled_from_proxy.data_ptr<int64_t>() : nullptr;
  launch(unique ? forward_kernel<true>(gin_) : forward_kernel<false>(gin_), OutKernel::kForward, out_ctas(OutKernel::kForward), args);
  return out;
}

at::Tensor OutChannel::forward_pooled(const FwdPlan &plan, const at::Tensor &lookup_out,
                                      const at::Tensor &offsets, const at::Tensor &reverse_indices,
                                      const c10::optional<at::Tensor> &weights) {
  c10::cuda::CUDAGuard guard(device_);
  const int features = sizes_.num_features, vecs = sizes_.row_vecs();
  const int batch = pooled_batch(offsets, weights, device_, features, plan.num_out());
  TORCH_CHECK(reverse_indices.device() == offsets.device() && reverse_indices.is_contiguous() &&
                  reverse_indices.dim() == 1 && reverse_indices.scalar_type() == at::kLong &&
                  reverse_indices.numel() == plan.num_out(),
              "hier_output_dist: expected the input dist's CUDA reverse indices");
  const at::Tensor rows = forward(plan, lookup_out, true);
  at::Tensor out = at::empty({batch, features * dim_}, rows.options());
  const int64_t bags = (int64_t)batch * features;
  if (bags > 0) {
    const bool long_bags = plan.num_out() >= 32 * bags;
    const int grid = long_bags ? (int)std::min<int64_t>(bags * ((vecs + 31) / 32), 16 * sms_)
                               : pooled_grid(bags * vecs, sms_);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_);
    with_offsets(offsets, [&](const auto *off) {
      with_row_type(dtype_, [&](auto tag) {
        const int4 *src = (const int4 *)rows.data_ptr();
        const int64_t *reverse = reverse_indices.data_ptr<int64_t>();
        const float *weight = weights ? weights->data_ptr<float>() : nullptr;
        int4 *dst = (int4 *)out.data_ptr();
        if (long_bags)
          pooled::sum_long_bags<decltype(tag)><<<grid, pooled::kBlock, 0, stream>>>(
              src, reverse, off, weight, batch, features, vecs, dst);
        else
          pooled::sum_bags<decltype(tag)><<<grid, pooled::kBlock, 0, stream>>>(
              src, reverse, off, weight, batch, features, vecs, dst);
      });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

void OutChannel::backward_pooled(BwdPlan plan, bool divide, at::Tensor grad,
                                 const at::Tensor &offsets, const c10::optional<at::Tensor> &weights) {
  c10::cuda::CUDAGuard guard(device_);
  const int features = sizes_.num_features;
  const int64_t occurrences = plan.fanout.exp_idx.numel();
  const int batch = pooled_batch(offsets, weights, device_, features, occurrences);
  TORCH_CHECK(grad.device() == offsets.device() && grad.scalar_type() == dtype_ &&
                  grad.dim() == 2 && grad.size(0) == batch && grad.size(1) == features * dim_,
              "hier_output_dist: pooled gradient must match the pooled output's device, dtype and shape");
  grad = grad.contiguous().view({(int64_t)batch * features, dim_});
  at::Tensor sorted_weights;
  if (occurrences) {
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_);
    at::Tensor indices = at::empty_like(plan.fanout.exp_idx);
    if (weights) sorted_weights = at::empty_like(*weights);
    with_offsets(offsets, [&](const auto *off) {
      pooled::bag_indices<<<pooled_grid(occurrences, sms_), pooled::kBlock, 0, stream>>>(
          plan.fanout.exp_idx.data_ptr<int32_t>(), occurrences, off, batch, features,
          indices.data_ptr<int32_t>(), weights ? weights->data_ptr<float>() : nullptr,
          sorted_weights.defined() ? sorted_weights.data_ptr<float>() : nullptr);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    plan.fanout.exp_idx = indices;
  }
  backward(plan, grad, divide, sorted_weights);
}

void OutChannel::backward(const BwdPlan &plan, at::Tensor grad, bool divide,
                          const at::Tensor &weights) {
  c10::cuda::CUDAGuard guard(device_);
  grad = grad.contiguous();
  if (grad.numel() > 0 && (uintptr_t)grad.data_ptr() % 16 != 0) grad = grad.clone();
  const void *kernel = backward_kernel(grad.scalar_type(), gin_);
  // The heavy rows' sub-tasks' fp32 sums, and across nodes the proxy's keys
  // reduced of each chunk of its stream (the kernel zeroes those).
  const auto options = grad.options();
  const at::Tensor partial = at::empty({plan.heavy_subs, grad.size(1)}, options.dtype(at::kFloat));
  at::Tensor ready;
  if (multi_node_) ready = at::empty({sizes_.max_chunks}, options.dtype(at::kLong));
  const int ctas = out_ctas(OutKernel::kBackward);
  BwdArgs args{};
  args.plan = plan.view(sizes_);
  args.grad = (const char *)grad.data_ptr();
  args.weights = weights.defined() ? weights.data_ptr<float>() : nullptr;
  args.partial = partial.data_ptr<float>();
  args.ready = ready.defined() ? ready.data_ptr<int64_t>() : nullptr;
  args.reducers = reducers(ctas);
  // TorchRec's owner divides by world_size, a CPU scalar: PyTorch multiplies
  // by the fp32 reciprocal.
  args.scale = divide ? 1.0f / (float)world_size_ : 1.0f;
  if (plan.heavy_subs > 0) {
    // The heavy rows' sums first, on the whole GPU: the sub-tasks' partials
    // (a group of lanes each), then each heavy row's sum (a CTA each).
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_);
    const int64_t per_cta = kThreads / heavy_group_lanes(sizes_.row_vecs());
    const int grid =
        (int)std::min<int64_t>((plan.heavy_subs + per_cta - 1) / per_cta, 4 * (int64_t)sms_);
    void *params[] = {&sizes_, &args};
    HIER_CUDA_CHECK(cudaLaunchKernel(heavy_partial_kernel(grad.scalar_type()), grid, kThreads,
                                     params, 0, stream));
    int dim = (int)grad.size(1);
    const int sum_grid = (int)std::min<int64_t>(plan.fanout.pos.numel(), 8 * (int64_t)sms_);
    void *sum_params[] = {&args, &dim};
    HIER_CUDA_CHECK(cudaLaunchKernel((const void *)bwd_heavy_sum_kernel, sum_grid, kSumThreads,
                                     sum_params, 0, stream));
  }
  launch(kernel, OutKernel::kBackward, ctas, args);
}

// ── the trace ──

Trace::Trace(int device) {
  if constexpr (kTrace) {
    out_ = at::zeros({2, kTraceRing, kTraceWords},
                     at::TensorOptions().device(at::kCUDA, device).dtype(at::kLong));
  }
}

// The earliest stamps start at UINT64_MAX, the rest at 0.
unsigned long long *Trace::start_out(OutKernel which, cudaStream_t stream) {
  if constexpr (kTrace) {
    const int direction = (int)which;
    const int64_t at = direction * kTraceRing + launches_[direction] % kTraceRing;
    int64_t *row = out_.data_ptr<int64_t>() + at * kTraceWords;
    HIER_CUDA_CHECK(cudaMemsetAsync(row, 0xff, kNumEarliest * sizeof(int64_t), stream));
    HIER_CUDA_CHECK(cudaMemsetAsync(row + kNumEarliest, 0,
                                    (kTraceWords - kNumEarliest) * sizeof(int64_t), stream));
    return (unsigned long long *)row;
  }
  return nullptr;
}

void Trace::launched(OutKernel which) {
  if constexpr (kTrace) ++launches_[(int)which];
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
