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

#include "hier_all2all.h"
#include "hier_all2all_kernel.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef DEMB_USE_GIN
#include <nccl.h>
#endif

namespace py = pybind11;

namespace hier_a2a {

// ---------------------------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------------------------

#define HIER_CUDA_CHECK(expr)                                                  \
  do {                                                                         \
    cudaError_t __err = (expr);                                                \
    TORCH_CHECK(__err == cudaSuccess, "CUDA error: ",                          \
                cudaGetErrorString(__err), " at ", __FILE__, ":", __LINE__);    \
  } while (0)

#ifdef DEMB_USE_GIN
#define HIER_NCCL_CHECK(expr)                                                  \
  do {                                                                         \
    ncclResult_t __res = (expr);                                               \
    TORCH_CHECK(__res == ncclSuccess, "NCCL error: ", ncclGetErrorString(__res),\
                " at ", __FILE__, ":", __LINE__);                               \
  } while (0)
#endif

// ---------------------------------------------------------------------------
// IPC Handle helpers
// ---------------------------------------------------------------------------

/// Allocate device memory via cudaMalloc (bypasses PyTorch caching allocator).
/// Required for cudaIpcGetMemHandle which needs a cudaMalloc-origin pointer.
uintptr_t ipc_cuda_malloc(int64_t size_bytes) {
  void *ptr = nullptr;
  HIER_CUDA_CHECK(cudaMalloc(&ptr, size_bytes));
  HIER_CUDA_CHECK(cudaMemset(ptr, 0, size_bytes));
  return reinterpret_cast<uintptr_t>(ptr);
}

void ipc_cuda_free(uintptr_t ptr) {
  HIER_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(ptr)));
}

std::vector<uint8_t> ipc_get_handle(uintptr_t dev_ptr) {
  cudaIpcMemHandle_t handle;
  HIER_CUDA_CHECK(
      cudaIpcGetMemHandle(&handle, reinterpret_cast<void *>(dev_ptr)));
  std::vector<uint8_t> bytes(sizeof(handle));
  std::memcpy(bytes.data(), &handle, sizeof(handle));
  return bytes;
}

uintptr_t ipc_open_handle(const std::vector<uint8_t> &handle_bytes) {
  TORCH_CHECK(handle_bytes.size() == sizeof(cudaIpcMemHandle_t),
              "Invalid IPC handle size");
  cudaIpcMemHandle_t handle;
  std::memcpy(&handle, handle_bytes.data(), sizeof(handle));
  void *ptr = nullptr;
  HIER_CUDA_CHECK(
      cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<uintptr_t>(ptr);
}

void ipc_close_handle(uintptr_t mapped_ptr) {
  HIER_CUDA_CHECK(cudaIpcCloseMemHandle(reinterpret_cast<void *>(mapped_ptr)));
}

std::vector<std::tuple<uintptr_t, int64_t>>
ipc_exchange_handles(torch::Tensor recv_buf, int64_t signal_pad_offset,
                     int local_world_size, int my_local_rank,
                     int64_t local_pg_handle) {
  // This is called from Python which orchestrates the all_gather of handles.
  // Here we just return the local info needed.
  std::vector<std::tuple<uintptr_t, int64_t>> result;
  // Placeholder: actual exchange is done in Python via dist.all_gather
  result.emplace_back(
      reinterpret_cast<uintptr_t>(recv_buf.data_ptr()), signal_pad_offset);
  return result;
}

// ---------------------------------------------------------------------------
// GIN initialization (NCCL >= 2.29 only)
// ---------------------------------------------------------------------------

#ifdef DEMB_USE_GIN

// Opaque GIN context stored as int64 handle
struct GINContext {
  ncclComm_t comm;
  ncclDevComm_t *dev_comm;
  void *gin_recv_buf;
  int64_t symmetric_size;
  int num_nodes;
  int local_world_size;
  std::vector<int> send_signal_ids;
  std::vector<int> recv_signal_ids;
};

static std::unordered_map<int64_t, GINContext *> g_gin_contexts;
static int64_t g_gin_handle_counter = 1;

int64_t gin_init(int64_t nccl_comm_handle, int num_nodes, int local_world_size,
                 int64_t max_gin_rows, int64_t D, int64_t elem_size) {
  auto *ctx = new GINContext();
  ctx->comm = reinterpret_cast<ncclComm_t>(nccl_comm_handle);
  ctx->num_nodes = num_nodes;
  ctx->local_world_size = local_world_size;

  // 1. Query communicator properties
  ncclCommProperties props;
  ncclResult_t res = ncclCommQueryProperties(ctx->comm, &props);
  if (res != ncclSuccess || !props.deviceApiSupport) {
    delete ctx;
    return -1;
  }

  // 2. Create ncclDevComm with GIN resources
  ncclDevCommRequirements req = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  req.gin = 1;
  req.ginSignals = num_nodes - 1; // one signal per remote same-rail sender

  res = ncclCommDevCommCreate(ctx->comm, &ctx->dev_comm, &req);
  if (res != ncclSuccess) {
    delete ctx;
    return -1;
  }

  // 3. Allocate symmetric GIN recv window
  ctx->symmetric_size = max_gin_rows * D * elem_size;
  res = ncclMemAlloc(&ctx->gin_recv_buf, ctx->symmetric_size);
  if (res != ncclSuccess) {
    ncclCommDevCommDestroy(ctx->comm, ctx->dev_comm);
    delete ctx;
    return -1;
  }

  // 4. Register GIN recv window
  // ncclCommWindowRegister is handled separately

  // 5. Assign signal IDs (deterministic based on topology)
  ctx->recv_signal_ids.resize(num_nodes - 1);
  ctx->send_signal_ids.resize(num_nodes - 1);
  // Populated by Python side based on topology

  int64_t handle = g_gin_handle_counter++;
  g_gin_contexts[handle] = ctx;
  return handle;
}

void gin_destroy(int64_t gin_context_handle) {
  auto it = g_gin_contexts.find(gin_context_handle);
  if (it == g_gin_contexts.end())
    return;
  GINContext *ctx = it->second;
  if (ctx->gin_recv_buf) {
    ncclMemFree(ctx->gin_recv_buf);
  }
  if (ctx->dev_comm) {
    ncclCommDevCommDestroy(ctx->comm, ctx->dev_comm);
  }
  delete ctx;
  g_gin_contexts.erase(it);
}

int64_t gin_alloc_recv_window(int64_t symmetric_size) {
  void *ptr = nullptr;
  ncclResult_t res = ncclMemAlloc(&ptr, symmetric_size);
  if (res != ncclSuccess)
    return 0;
  return reinterpret_cast<int64_t>(ptr);
}

void gin_free_recv_window(int64_t gin_recv_buf_ptr) {
  ncclMemFree(reinterpret_cast<void *>(gin_recv_buf_ptr));
}

int64_t gin_register_window(int64_t nccl_comm_handle, int64_t gin_recv_buf_ptr,
                            int64_t symmetric_size) {
  // Placeholder: actual ncclCommWindowRegister call
  return 0;
}

void gin_unregister_window(int64_t nccl_comm_handle, int64_t window_handle) {
  // Placeholder: actual ncclCommWindowDeregister call
}

#endif // DEMB_USE_GIN

// ---------------------------------------------------------------------------
// Scatter map construction (GPU)
// ---------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
build_scatter_map(torch::Tensor feature_offsets,  // [F+1]
                  torch::Tensor recatted_offsets, // [F+1]
                  torch::Tensor split_offsets,    // [W+1]
                  torch::Tensor feature_recat,    // [F]
                  int64_t total_rows, int my_node, int my_local_rank,
                  int local_world_size, int num_nodes,
                  torch::Tensor rank_to_node,  // [W] int32
                  torch::Tensor rank_to_local, // [W] int32
                  torch::optional<torch::Tensor> inv_unbucketize,
                  cudaStream_t stream) {
  auto device = feature_offsets.device();
  int64_t num_features = feature_offsets.size(0) - 1;
  int world_size = rank_to_node.size(0);

  // Allocate intermediate tensors
  auto gather_indices =
      torch::empty({total_rows}, torch::dtype(torch::kInt64).device(device));
  auto category =
      torch::zeros({total_rows}, torch::dtype(torch::kInt32).device(device));
  auto dest_id =
      torch::zeros({total_rows}, torch::dtype(torch::kInt32).device(device));
  auto dest_node_id =
      torch::zeros({total_rows}, torch::dtype(torch::kInt32).device(device));
  auto pos_in_dest =
      torch::zeros({total_rows}, torch::dtype(torch::kInt32).device(device));

  // Pass 1: Classify + gather
  const int block_size = 256;
  const int num_blocks = (total_rows + block_size - 1) / block_size;

  const int64_t *inv_unbuck_ptr =
      inv_unbucketize.has_value()
          ? inv_unbucketize.value().data_ptr<int64_t>()
          : nullptr;

  classify_and_gather_kernel<<<num_blocks, block_size, 0, stream>>>(
      recatted_offsets.data_ptr<int64_t>(),
      feature_offsets.data_ptr<int64_t>(),
      split_offsets.data_ptr<int64_t>(), feature_recat.data_ptr<int64_t>(),
      rank_to_node.data_ptr<int>(), rank_to_local.data_ptr<int>(), my_node,
      my_local_rank, local_world_size, num_nodes, total_rows, num_features,
      world_size, inv_unbuck_ptr, gather_indices.data_ptr<int64_t>(),
      category.data_ptr<int>(), dest_id.data_ptr<int>(),
      dest_node_id.data_ptr<int>(), pos_in_dest.data_ptr<int>());

  // Pass 2a: Histogram
  auto peer_counts = torch::zeros({local_world_size},
                                   torch::dtype(torch::kInt32).device(device));
  auto relay_counts =
      torch::zeros({local_world_size * num_nodes},
                   torch::dtype(torch::kInt32).device(device));
  auto inter_counts =
      torch::zeros({num_nodes}, torch::dtype(torch::kInt32).device(device));

  histogram_kernel<<<num_blocks, block_size, 0, stream>>>(
      category.data_ptr<int>(), dest_id.data_ptr<int>(),
      dest_node_id.data_ptr<int>(), total_rows, local_world_size, num_nodes,
      peer_counts.data_ptr<int>(), relay_counts.data_ptr<int>(),
      inter_counts.data_ptr<int>());

  HIER_CUDA_CHECK(cudaStreamSynchronize(stream));

  // Pass 2b: Compute CSR offsets on CPU from histogram
  auto peer_counts_cpu = peer_counts.cpu();
  auto relay_counts_cpu = relay_counts.cpu();
  auto inter_counts_cpu = inter_counts.cpu();

  int *pc = peer_counts_cpu.data_ptr<int>();
  int *rc = relay_counts_cpu.data_ptr<int>();
  int *ic = inter_counts_cpu.data_ptr<int>();

  // Peer CSR: [intra_rows | relay_dest_node_0 | relay_dest_node_1 | ...]
  // per local_rank
  std::vector<int64_t> peer_offsets_vec(local_world_size + 1, 0);
  std::vector<int64_t> peer_intra_counts_vec(local_world_size, 0);
  std::vector<int32_t> peer_relay_dest_counts_vec(local_world_size * num_nodes,
                                                   0);

  for (int lr = 0; lr < local_world_size; ++lr) {
    peer_intra_counts_vec[lr] = pc[lr];
    int64_t relay_total = 0;
    for (int n = 0; n < num_nodes; ++n) {
      int cnt = rc[lr * num_nodes + n];
      peer_relay_dest_counts_vec[lr * num_nodes + n] = cnt;
      relay_total += cnt;
    }
    peer_offsets_vec[lr + 1] =
        peer_offsets_vec[lr] + pc[lr] + relay_total;
  }

  // Compute relay sub-offsets within each peer's relay portion
  // relay_sub_offsets[lr * num_nodes + n] = offset of dest_node n's rows
  // within lr's relay sub-region
  std::vector<int64_t> relay_sub_offsets_vec(local_world_size * num_nodes, 0);
  for (int lr = 0; lr < local_world_size; ++lr) {
    int64_t running = 0;
    for (int n = 0; n < num_nodes; ++n) {
      relay_sub_offsets_vec[lr * num_nodes + n] = running;
      running += rc[lr * num_nodes + n];
    }
  }

  // Inter CSR offsets (only remote nodes)
  std::vector<int64_t> inter_offsets_vec(num_nodes + 1, 0);
  for (int n = 0; n < num_nodes; ++n) {
    inter_offsets_vec[n + 1] = inter_offsets_vec[n] + ic[n];
  }

  // Copy CSR structures to device
  auto peer_offsets = torch::tensor(peer_offsets_vec,
                                     torch::dtype(torch::kInt64).device(device));
  auto peer_intra_counts = torch::tensor(
      peer_intra_counts_vec, torch::dtype(torch::kInt64).device(device));
  auto peer_relay_dest_counts = torch::tensor(
      peer_relay_dest_counts_vec, torch::dtype(torch::kInt32).device(device));
  auto relay_sub_offsets = torch::tensor(
      relay_sub_offsets_vec, torch::dtype(torch::kInt64).device(device));
  auto inter_offsets = torch::tensor(inter_offsets_vec,
                                      torch::dtype(torch::kInt64).device(device));

  // Allocate output CSR arrays
  int64_t total_peer_rows = peer_offsets_vec.back();
  int64_t total_inter_rows = inter_offsets_vec.back();

  auto peer_gather_out = torch::empty(
      {total_peer_rows}, torch::dtype(torch::kInt64).device(device));
  auto inter_gather_out = torch::empty(
      {total_inter_rows}, torch::dtype(torch::kInt64).device(device));

  // Pass 2c: Scatter to CSR (deterministic using pos_in_dest)
  auto peer_intra_counts_i32 = peer_intra_counts.to(torch::kInt32);
  scatter_to_csr_kernel<<<num_blocks, block_size, 0, stream>>>(
      gather_indices.data_ptr<int64_t>(), category.data_ptr<int>(),
      dest_id.data_ptr<int>(), dest_node_id.data_ptr<int>(),
      pos_in_dest.data_ptr<int>(), total_rows,
      local_world_size, num_nodes, peer_offsets.data_ptr<int64_t>(),
      peer_intra_counts_i32.data_ptr<int>(),
      relay_sub_offsets.data_ptr<int64_t>(),
      inter_offsets.data_ptr<int64_t>(), peer_gather_out.data_ptr<int64_t>(),
      inter_gather_out.data_ptr<int64_t>());

  return std::make_tuple(peer_gather_out, peer_offsets, peer_intra_counts,
                         peer_relay_dest_counts, inter_gather_out,
                         inter_offsets);
}

// ---------------------------------------------------------------------------
// Backward row recat materialization
// ---------------------------------------------------------------------------

torch::Tensor materialize_backward_row_recat(
    torch::Tensor sparse_features_recat, torch::Tensor out_feature_offsets,
    torch::Tensor out_recatted_offsets, int64_t total_rows,
    cudaStream_t stream) {
  auto device = sparse_features_recat.device();
  int64_t num_features = sparse_features_recat.size(0);

  auto backward_row_recat =
      torch::empty({total_rows}, torch::dtype(torch::kInt64).device(device));

  const int block_size = 256;
  const int num_blocks = (total_rows + block_size - 1) / block_size;

  materialize_backward_row_recat_kernel<<<num_blocks, block_size, 0, stream>>>(
      sparse_features_recat.data_ptr<int64_t>(),
      out_feature_offsets.data_ptr<int64_t>(),
      out_recatted_offsets.data_ptr<int64_t>(), total_rows, num_features,
      backward_row_recat.data_ptr<int64_t>());

  return backward_row_recat;
}

// Persistent done_counters tensor for multi-CTA outcast signaling
static torch::Tensor g_done_counters;

static int32_t *get_done_counters(int local_world_size, torch::Device device) {
  if (!g_done_counters.defined() ||
      g_done_counters.size(0) < local_world_size ||
      g_done_counters.device() != device) {
    g_done_counters =
        torch::zeros({local_world_size}, torch::dtype(torch::kInt32).device(device));
  }
  return g_done_counters.data_ptr<int32_t>();
}
torch::Tensor hier_all2all_fwd_pipelined(
    torch::Tensor output_embs,         // [total_send_rows, D]
    torch::Tensor peer_gather_indices, // [total_peer_rows] int64
    torch::Tensor peer_offsets,        // [L+1] int64
    torch::Tensor output_splits_t,     // [W] int64 — raw, NOT prefix sum
    torch::Tensor rank_to_local_dev,   // [W] int32
    torch::Tensor unbucketize_permute, // [total_recv] int64 or empty
    int64_t max_rows_per_rank,
    torch::Tensor peer_slot_ptrs_dev,
    torch::Tensor peer_sig_ptrs_dev,
    int64_t my_ipc_buf_ptr,
    int64_t signal_pad_ptr,
    int my_local_rank, int local_world_size,
    int64_t total_output_rows, int64_t D,
    torch::Tensor device_flag,
    int32_t iter_id,
    torch::optional<torch::Tensor> output_buf) { // pre-allocated output (optional)

  TORCH_CHECK(output_embs.is_cuda(), "output_embs must be CUDA tensor");
  auto device = output_embs.device();
  auto stream = at::cuda::getCurrentCUDAStream(device.index());
  int W = output_splits_t.size(0);

  static int s_sm_count = 0;
  if (s_sm_count == 0) {
    HIER_CUDA_CHECK(cudaDeviceGetAttribute(
        &s_sm_count, cudaDevAttrMultiProcessorCount, device.index()));
  }

  auto *my_signals = reinterpret_cast<int32_t *>(signal_pad_ptr);
  auto *flag_ptr = device_flag.data_ptr<int32_t>();
  auto *done_counters_ptr = get_done_counters(local_world_size, device);

  return AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      output_embs.scalar_type(), "hier_all2all_fwd_pipelined", [&] {
        torch::Tensor final_output;
        if (output_buf.has_value() && output_buf->numel() >= total_output_rows * D) {
          final_output = output_buf->view({total_output_rows, D});
        } else {
          auto options =
              torch::dtype(torch::CppTypeToScalarType<scalar_t>::value)
                  .device(device);
          final_output = torch::empty({total_output_rows, D}, options);
        }

        const int row_bytes = D * sizeof(scalar_t);

        // ---- Compute shared memory layout ----
        const int prefix_bytes = (W + 1) * (int)sizeof(int64_t);
        const int r2l_bytes = W * (int)sizeof(int);
        const int meta_end = (prefix_bytes + r2l_bytes + 127) & ~127;

        // cp.async pipeline for large D outcast (producer CTAs only).
        // Consumer CTAs allocate the smem but don't use it.
        // Target 48 KB pipeline buffer (doesn't reduce CTA occupancy since
        // registers are the bottleneck with launch_bounds(256,1)).
        int batch_rows = 0;
        int smem_size = meta_end;
        if (row_bytes > kOutcastThreadCopyThreshold) {
          constexpr int kTargetPipeSmem = 48 * 1024;
          const int avail = kTargetPipeSmem;
          batch_rows = avail / (2 * row_bytes);
          batch_rows = (batch_rows / kWarpsPerCTA) * kWarpsPerCTA;
          batch_rows = std::max(batch_rows, kWarpsPerCTA);
          smem_size = meta_end + 2 * batch_rows * row_bytes;
        }

        // Tell the driver we need more than the default 48 KB smem
        static int s_smem_configured = 0;
        if (s_smem_configured < smem_size) {
          HIER_CUDA_CHECK(cudaFuncSetAttribute(
              (const void *)hier_a2a_fwd_pipelined_kernel<scalar_t>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
          s_smem_configured = smem_size;
        }

        // ---- Grid: producer CTAs + extra consumer CTAs ----
        // Producer CTAs: multi-CTA per peer for outcast
        int ctas_per_peer = std::max(1, s_sm_count / local_world_size);
        int outcast_ctas = local_world_size * ctas_per_peer;
        // Consumer CTAs: 4x SM count for gather parallelism
        // (matches the gather grid from the old two-kernel path)
        int gather_ctas = s_sm_count * 4;
        int grid = std::max(outcast_ctas, gather_ctas);

        // ---- Kernel args ----
        const scalar_t *arg_input = output_embs.data_ptr<scalar_t>();
        const int64_t *arg_gi =
            peer_gather_indices.template data_ptr<int64_t>();
        const int64_t *arg_off =
            peer_offsets.template data_ptr<int64_t>();
        const uintptr_t *arg_slots = reinterpret_cast<const uintptr_t *>(
            peer_slot_ptrs_dev.template data_ptr<int64_t>());
        const uintptr_t *arg_sigs = reinterpret_cast<const uintptr_t *>(
            peer_sig_ptrs_dev.template data_ptr<int64_t>());
        int64_t arg_D = D;
        int arg_cpp = ctas_per_peer;
        int32_t *arg_dc = done_counters_ptr;
        const scalar_t *arg_buf =
            reinterpret_cast<const scalar_t *>(my_ipc_buf_ptr);
        scalar_t *arg_out = final_output.template data_ptr<scalar_t>();
        const int64_t *arg_splits = output_splits_t.data_ptr<int64_t>();
        const int *arg_r2l = rank_to_local_dev.data_ptr<int>();
        const int64_t *arg_unbuck =
            (unbucketize_permute.numel() > 0)
                ? unbucketize_permute.data_ptr<int64_t>()
                : nullptr;
        int64_t arg_mrpr = max_rows_per_rank;
        int32_t *arg_sig = my_signals;
        int32_t *arg_flag = flag_ptr;
        int64_t arg_total = total_output_rows;
        int arg_W = W;
        int arg_lws = local_world_size;
        int arg_my_lr = my_local_rank;
        int32_t arg_iter_id = iter_id;
        int arg_batch = batch_rows;
        int arg_outcast_ctas = outcast_ctas;
        int64_t *arg_phase_clocks = nullptr; // debug timing disabled

        void *args[] = {&arg_input, &arg_gi,      &arg_off,   &arg_slots,
                        &arg_sigs,  &arg_D,       &arg_cpp,   &arg_dc,
                        &arg_buf,   &arg_out,     &arg_splits, &arg_r2l,
                        &arg_unbuck, &arg_mrpr,   &arg_sig,   &arg_flag,
                        &arg_total, &arg_W,       &arg_lws,   &arg_my_lr,
                        &arg_iter_id, &arg_batch, &arg_outcast_ctas,
                        &arg_phase_clocks};

        HIER_CUDA_CHECK(cudaLaunchKernel(
            (const void *)hier_a2a_fwd_pipelined_kernel<scalar_t>,
            dim3(grid), dim3(kThreadsPerCTA), args, smem_size, stream));

        return final_output;
      });
}
void bind_hier_all2all_op(py::module &m) {
  auto sub = m.def_submodule("hier_a2a", "Hierarchical All2All operations");

  sub.def("ipc_cuda_malloc", &ipc_cuda_malloc,
          "Allocate device memory via cudaMalloc (IPC-safe, bypasses PyTorch allocator)");
  sub.def("ipc_cuda_free", &ipc_cuda_free,
          "Free device memory allocated by ipc_cuda_malloc");
  sub.def("ipc_get_handle", &ipc_get_handle,
          "Get IPC memory handle for a device pointer");
  sub.def("ipc_open_handle", &ipc_open_handle,
          "Open an IPC memory handle from bytes");
  sub.def("ipc_close_handle", &ipc_close_handle,
          "Close a previously opened IPC memory handle");

  sub.def(
      "build_scatter_map",
      [](torch::Tensor feature_offsets, torch::Tensor recatted_offsets,
         torch::Tensor split_offsets, torch::Tensor feature_recat,
         int64_t total_rows, int my_node, int my_local_rank,
         int local_world_size, int num_nodes, torch::Tensor rank_to_node,
         torch::Tensor rank_to_local,
         torch::optional<torch::Tensor> inv_unbucketize) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        return build_scatter_map(
            feature_offsets, recatted_offsets, split_offsets, feature_recat,
            total_rows, my_node, my_local_rank, local_world_size, num_nodes,
            rank_to_node, rank_to_local, inv_unbucketize, stream);
      },
      "Build GPU scatter map (classify + compact)");

  sub.def(
      "materialize_backward_row_recat",
      [](torch::Tensor sparse_features_recat,
         torch::Tensor out_feature_offsets,
         torch::Tensor out_recatted_offsets, int64_t total_rows) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        return materialize_backward_row_recat(sparse_features_recat,
                                               out_feature_offsets,
                                               out_recatted_offsets,
                                               total_rows, stream);
      },
      "Materialize backward_row_recat on GPU");
  sub.def(
      "fwd_pipelined",
      [](torch::Tensor output_embs, torch::Tensor peer_gather_indices,
         torch::Tensor peer_offsets, torch::Tensor output_splits_t,
         torch::Tensor rank_to_local_dev, torch::Tensor unbucketize_permute,
         int64_t max_rows_per_rank, torch::Tensor peer_slot_ptrs_dev,
         torch::Tensor peer_sig_ptrs_dev, int64_t my_ipc_buf_ptr,
         int64_t signal_pad_ptr, int my_local_rank, int local_world_size,
         int64_t total_output_rows, int64_t D, torch::Tensor device_flag,
         int32_t iter_id,
         torch::optional<torch::Tensor> output_buf) {
        return hier_all2all_fwd_pipelined(
            output_embs, peer_gather_indices, peer_offsets, output_splits_t,
            rank_to_local_dev, unbucketize_permute, max_rows_per_rank,
            peer_slot_ptrs_dev, peer_sig_ptrs_dev, my_ipc_buf_ptr,
            signal_pad_ptr, my_local_rank, local_world_size,
            total_output_rows, D, device_flag, iter_id, output_buf);
      },
      "Pipelined fwd: cp.async double-buffered outcast + gather (1 kernel, no external cumsum)",
      py::arg("output_embs"), py::arg("peer_gather_indices"),
      py::arg("peer_offsets"), py::arg("output_splits_t"),
      py::arg("rank_to_local_dev"), py::arg("unbucketize_permute"),
      py::arg("max_rows_per_rank"), py::arg("peer_slot_ptrs_dev"),
      py::arg("peer_sig_ptrs_dev"), py::arg("my_ipc_buf_ptr"),
      py::arg("signal_pad_ptr"), py::arg("my_local_rank"),
      py::arg("local_world_size"), py::arg("total_output_rows"),
      py::arg("D"), py::arg("device_flag"), py::arg("iter_id"),
      py::arg("output_buf") = py::none());

  // Debug: pipelined forward with phase timing
  sub.def(
      "fwd_pipelined_timed",
      [](torch::Tensor output_embs, torch::Tensor peer_gather_indices,
         torch::Tensor peer_offsets, torch::Tensor output_splits_t,
         torch::Tensor rank_to_local_dev, torch::Tensor unbucketize_permute,
         int64_t max_rows_per_rank, torch::Tensor peer_slot_ptrs_dev,
         torch::Tensor peer_sig_ptrs_dev, int64_t my_ipc_buf_ptr,
         int64_t signal_pad_ptr, int my_local_rank, int local_world_size,
         int64_t total_output_rows, int64_t D, torch::Tensor device_flag,
         int32_t iter_id, torch::Tensor phase_clocks_tensor) {
        TORCH_CHECK(output_embs.is_cuda());
        auto device = output_embs.device();
        auto stream = at::cuda::getCurrentCUDAStream(device.index());
        int W = output_splits_t.size(0);

        static int s_sm = 0;
        if (s_sm == 0)
          cudaDeviceGetAttribute(&s_sm, cudaDevAttrMultiProcessorCount,
                                 device.index());

        auto *my_signals = reinterpret_cast<int32_t *>(signal_pad_ptr);
        auto *flag_ptr = device_flag.data_ptr<int32_t>();
        auto *done_counters_ptr = get_done_counters(local_world_size, device);

        return AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            output_embs.scalar_type(), "fwd_pipelined_timed", [&] {
              auto options =
                  torch::dtype(torch::CppTypeToScalarType<scalar_t>::value)
                      .device(device);
              auto final_output =
                  torch::empty({total_output_rows, D}, options);

              const int row_bytes = D * sizeof(scalar_t);
              const int prefix_bytes = (W + 1) * (int)sizeof(int64_t);
              const int r2l_bytes = W * (int)sizeof(int);
              const int meta_end = (prefix_bytes + r2l_bytes + 127) & ~127;
              constexpr int kTargetSmem = 164 * 1024;
              int batch_rows = (kTargetSmem - meta_end) / (2 * row_bytes);
              batch_rows = (batch_rows / kWarpsPerCTA) * kWarpsPerCTA;
              batch_rows = std::max(batch_rows, kWarpsPerCTA);
              const int smem_size = meta_end + 2 * batch_rows * row_bytes;

              static int s_smem_cfg = 0;
              if (s_smem_cfg < smem_size) {
                cudaFuncSetAttribute(
                    (const void *)hier_a2a_fwd_pipelined_kernel<scalar_t>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
                s_smem_cfg = smem_size;
              }

              int ctas_per_peer = std::max(1, s_sm / local_world_size);
              int grid = local_world_size * ctas_per_peer;

              const scalar_t *arg_input =
                  output_embs.data_ptr<scalar_t>();
              const int64_t *arg_gi =
                  peer_gather_indices.template data_ptr<int64_t>();
              const int64_t *arg_off =
                  peer_offsets.template data_ptr<int64_t>();
              const uintptr_t *arg_slots =
                  reinterpret_cast<const uintptr_t *>(
                      peer_slot_ptrs_dev.template data_ptr<int64_t>());
              const uintptr_t *arg_sigs =
                  reinterpret_cast<const uintptr_t *>(
                      peer_sig_ptrs_dev.template data_ptr<int64_t>());
              int64_t arg_D = D;
              int arg_cpp = ctas_per_peer;
              int32_t *arg_dc = done_counters_ptr;
              const scalar_t *arg_buf =
                  reinterpret_cast<const scalar_t *>(my_ipc_buf_ptr);
              scalar_t *arg_out =
                  final_output.template data_ptr<scalar_t>();
              const int64_t *arg_splits =
                  output_splits_t.data_ptr<int64_t>();
              const int *arg_r2l = rank_to_local_dev.data_ptr<int>();
              const int64_t *arg_unbuck =
                  (unbucketize_permute.numel() > 0)
                      ? unbucketize_permute.data_ptr<int64_t>()
                      : nullptr;
              int64_t arg_mrpr = max_rows_per_rank;
              int32_t *arg_sig = my_signals;
              int32_t *arg_flag = flag_ptr;
              int64_t arg_total = total_output_rows;
              int arg_W = W;
              int arg_lws = local_world_size;
              int arg_my_lr = my_local_rank;
              int32_t arg_iter_id = iter_id;
              int arg_batch = batch_rows;
              int arg_outcast_ctas = grid; // timed variant: all CTAs do outcast
              int64_t *arg_pc =
                  phase_clocks_tensor.data_ptr<int64_t>();

              void *args[] = {
                  &arg_input,  &arg_gi,     &arg_off,    &arg_slots,
                  &arg_sigs,   &arg_D,      &arg_cpp,    &arg_dc,
                  &arg_buf,    &arg_out,    &arg_splits, &arg_r2l,
                  &arg_unbuck, &arg_mrpr,   &arg_sig,    &arg_flag,
                  &arg_total,  &arg_W,      &arg_lws,    &arg_my_lr,
                  &arg_iter_id, &arg_batch, &arg_outcast_ctas, &arg_pc};

              cudaLaunchKernel(
                  (const void *)hier_a2a_fwd_pipelined_kernel<scalar_t>,
                  dim3(grid), dim3(kThreadsPerCTA), args, smem_size,
                  stream);

              return final_output;
            });
      },
      "Pipelined fwd with phase timing (debug)");
#ifdef DEMB_USE_GIN
  sub.def("gin_init", &gin_init, "Initialize NCCL GIN resources");
  sub.def("gin_destroy", &gin_destroy, "Destroy GIN resources");
  sub.def("gin_alloc_recv_window", &gin_alloc_recv_window,
          "Allocate symmetric GIN recv window");
  sub.def("gin_free_recv_window", &gin_free_recv_window,
          "Free GIN recv window");
  sub.def("gin_register_window", &gin_register_window,
          "Register GIN recv window");
  sub.def("gin_unregister_window", &gin_unregister_window,
          "Unregister GIN recv window");
#endif
}

} // namespace hier_a2a
