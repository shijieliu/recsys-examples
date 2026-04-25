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

#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>

// Conditional compilation for NCCL GIN (device API)
// Requires NCCL >= 2.29 at compile time.
// When DEMB_USE_GIN is not defined, GIN code is excluded and only
// baseline NCCL all2all is available.
#ifdef DEMB_USE_GIN
#include <nccl.h>
// NCCL GIN device API headers (NCCL >= 2.28.7)
#include <nccl_device.h>
#endif

namespace hier_a2a {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Threads per CTA
constexpr int kThreadsPerCTA = 256;

/// Warps per CTA
constexpr int kWarpsPerCTA = kThreadsPerCTA / 32;

/// Vectorized copy element size (float4 = 16 bytes)
constexpr int kVecBytes = 16;

// ---------------------------------------------------------------------------
// Host-side function declarations
// ---------------------------------------------------------------------------

/// Exchange IPC handles within a node. Returns a list of (peer_ptr, signal_offset)
/// for each local_rank peer.
std::vector<std::tuple<uintptr_t, int64_t>> ipc_exchange_handles(
    torch::Tensor recv_buf, int64_t signal_pad_offset, int local_world_size,
    int my_local_rank, int64_t local_pg_handle);

/// Open a single IPC handle from raw bytes.
uintptr_t ipc_open_handle(const std::vector<uint8_t> &handle_bytes);

/// Get IPC handle bytes for a device pointer.
std::vector<uint8_t> ipc_get_handle(uintptr_t dev_ptr);

/// Close a previously opened IPC handle.
void ipc_close_handle(uintptr_t mapped_ptr);

/// Build the GPU scatter map (classify + compact).
/// Returns: (peer_gather_indices, peer_offsets, peer_intra_counts,
///           peer_relay_dest_counts, inter_gather_indices, inter_peer_offsets)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
build_scatter_map(
    torch::Tensor feature_offsets,   // [F+1] int64
    torch::Tensor recatted_offsets,  // [F+1] int64
    torch::Tensor split_offsets,     // [W+1] int64
    torch::Tensor feature_recat,     // [F] int64
    int64_t total_rows,              // sum(input_splits)
    int my_node, int my_local_rank, int local_world_size, int num_nodes,
    torch::Tensor rank_to_node,  // [W] int32
    torch::Tensor rank_to_local, // [W] int32
    // For backward scatter map with fused inv_unbucketize:
    torch::optional<torch::Tensor> inv_unbucketize, // [total_rows] int64,
                                                      // optional
    cudaStream_t stream);

/// Materialize backward_row_recat on GPU.
torch::Tensor materialize_backward_row_recat(
    torch::Tensor sparse_features_recat, // [F] int64
    torch::Tensor out_feature_offsets,   // [F+1] int64
    torch::Tensor out_recatted_offsets,  // [F+1] int64
    int64_t total_rows, cudaStream_t stream);

#ifdef DEMB_USE_GIN
/// Initialize NCCL GIN resources. Returns opaque GINContext handle.
/// On failure returns -1 (caller falls back to NCCL all2all).
int64_t gin_init(int64_t nccl_comm_handle, int num_nodes, int local_world_size,
                 int64_t max_gin_rows, int64_t D, int64_t elem_size);

/// Destroy GIN resources.
void gin_destroy(int64_t gin_context_handle);

/// Allocate symmetric GIN recv window via ncclMemAlloc.
/// Returns device pointer as int64.
int64_t gin_alloc_recv_window(int64_t symmetric_size);

/// Free GIN recv window.
void gin_free_recv_window(int64_t gin_recv_buf_ptr);

/// Register GIN recv window. Returns window handle as int64.
int64_t gin_register_window(int64_t nccl_comm_handle, int64_t gin_recv_buf_ptr,
                            int64_t symmetric_size);

/// Unregister GIN recv window.
void gin_unregister_window(int64_t nccl_comm_handle, int64_t window_handle);
#endif

/// Pipelined fwd: cp.async double-buffered outcast + gather (single kernel).
/// Eliminates external cumsum/memcpy; computes prefix sum in smem.
torch::Tensor hier_all2all_fwd_pipelined(
    torch::Tensor output_embs, torch::Tensor peer_gather_indices,
    torch::Tensor peer_offsets, torch::Tensor output_splits_t,
    torch::Tensor rank_to_local_dev, torch::Tensor unbucketize_permute,
    int64_t max_rows_per_rank, torch::Tensor peer_slot_ptrs_dev,
    torch::Tensor peer_sig_ptrs_dev, int64_t my_ipc_buf_ptr,
    int64_t signal_pad_ptr, int my_local_rank, int local_world_size,
    int64_t total_output_rows, int64_t D, torch::Tensor device_flag,
    int32_t iter_id,
    torch::optional<torch::Tensor> output_buf = torch::nullopt);

/// Pybind11 binding function.
void bind_hier_all2all_op(pybind11::module &m);

} // namespace hier_a2a
