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

#include "hier_all2all.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef DEMB_USE_GIN
#include <nccl_device.h>
#endif

namespace hier_a2a {
// ---------------------------------------------------------------------------
// Inline helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ int get_warp_id() {
  return threadIdx.x / 32;
}

__device__ __forceinline__ int get_lane_id() {
  return threadIdx.x % 32;
}

/// Elect one thread in warp (lane 0)
__device__ __forceinline__ bool elect_one_sync() {
  return get_lane_id() == 0;
}

/// Upper bound search in sorted array (device)
__device__ __forceinline__ int64_t
upper_bound_device(const int64_t *arr, int64_t n, int64_t val) {
  int64_t lo = 0, hi = n;
  while (lo < hi) {
    int64_t mid = lo + (hi - lo) / 2;
    if (arr[mid] <= val) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// ---------------------------------------------------------------------------
// Warp-cooperative vectorized row copy (kVectorized path)
// ---------------------------------------------------------------------------

/// Copy one row of D elements from src to dst using warp-cooperative
/// float4/int4 vectorized loads+stores. No shared memory needed.
template <typename scalar_t>
__device__ __forceinline__ void
vectorized_copy_row(const scalar_t *__restrict__ src,
                    scalar_t *__restrict__ dst, int64_t D) {
  const int lane = get_lane_id();
  const int row_bytes = D * sizeof(scalar_t);
  const char *src_bytes = reinterpret_cast<const char *>(src);
  char *dst_bytes = reinterpret_cast<char *>(dst);

  // Each thread copies 16 bytes (float4) at a time, strided across the warp
  for (int byte_off = lane * kVecBytes; byte_off < row_bytes;
       byte_off += 32 * kVecBytes) {
    if (byte_off + kVecBytes <= row_bytes) {
      float4 val =
          *reinterpret_cast<const float4 *>(src_bytes + byte_off);
      *reinterpret_cast<float4 *>(dst_bytes + byte_off) = val;
    } else {
      // Handle tail bytes with byte-by-byte copy
      for (int b = byte_off; b < row_bytes && b < byte_off + kVecBytes; ++b) {
        dst_bytes[b] = src_bytes[b];
      }
    }
  }
}

/// Thread-level row copy for small D (avoids wasting warp lanes).
/// Each thread copies an entire row using sequential float4 loads/stores.
template <typename scalar_t>
__device__ __forceinline__ void
thread_copy_row(const scalar_t *__restrict__ src,
                scalar_t *__restrict__ dst, int64_t D) {
  const int row_bytes = D * sizeof(scalar_t);
  const char *src_bytes = reinterpret_cast<const char *>(src);
  char *dst_bytes = reinterpret_cast<char *>(dst);
  for (int off = 0; off < row_bytes; off += kVecBytes) {
    if (off + kVecBytes <= row_bytes) {
      *reinterpret_cast<float4 *>(dst_bytes + off) =
          *reinterpret_cast<const float4 *>(src_bytes + off);
    }
  }
}

/// Gather threshold: row_bytes <= this value uses thread-level copy for gather.
/// At 128 bytes/row (D=64 bf16, D=32 fp32), only 8/32 warp lanes are active
/// with warp-level copy. Thread-level uses all 256 threads per CTA.
constexpr int kThreadCopyThreshold = 128;

/// Outcast threshold: row_bytes <= this value uses thread-level copy for outcast.
/// Lower than kThreadCopyThreshold because outcast does scattered reads from
/// input_embs (via gather_indices). At 128 bytes/row, the cp.async pipeline
/// hides HBM latency better than thread-level sequential loads from random
/// addresses. At 64 bytes/row, the overhead of cp.async staging outweighs
/// the latency hiding benefit.
constexpr int kOutcastThreadCopyThreshold = 64;

// ---------------------------------------------------------------------------
// Signal helpers — Hopper-optimized (PTX st.release.sys / ld.acquire.sys)
// ---------------------------------------------------------------------------

/// Store-release a signal to a peer via system-scope store.
/// Uses st.release.sys (PTX): orders all prior writes (including NVLink
/// outcast data) before the signal becomes visible to the peer GPU.
/// Cheaper than atomicExch_system — a plain store suffices since only one
/// producer thread writes each signal slot.
__device__ __forceinline__ void signal_peer(int32_t *signal_ptr, int32_t val) {
  asm volatile("st.release.sys.b32 [%0], %1;\n"
               :
               : "l"(signal_ptr), "r"(val)
               : "memory");
}

/// Load-acquire a signal from a peer via system-scope load.
/// Uses ld.acquire.sys (PTX): ensures all data written before the signal
/// (on the remote GPU) is visible after this load returns the expected value.
/// Cheaper than atomicAdd_system(ptr, 0) — avoids read-modify-write overhead.
__device__ __forceinline__ int32_t load_signal(int32_t *signal_ptr) {
  int32_t val;
  asm volatile("ld.acquire.sys.b32 %0, [%1];\n"
               : "=r"(val)
               : "l"(signal_ptr)
               : "memory");
  return val;
}

/// Nanosleep hint for spin-wait backoff (Hopper, sm_90+).
/// Reduces NVLink polling pressure and power consumption during spin-wait.
__device__ __forceinline__ void nanosleep_backoff(unsigned ns) {
  asm volatile("nanosleep.u32 %0;\n" ::"r"(ns));
}

/// Spin-wait until signal >= expected value, with nanosleep backoff.
__device__ __forceinline__ void spin_wait_signal(int32_t *signal_ptr,
                                                  int32_t expected) {
  while (load_signal(signal_ptr) < expected) {
    nanosleep_backoff(64);
  }
}

// Keep backward-compatible name for existing call sites
__device__ __forceinline__ int32_t wait_signal(int32_t *signal_ptr) {
  return load_signal(signal_ptr);
}

// ---------------------------------------------------------------------------
// Scatter map classify + gather kernel (Pass 1)
// ---------------------------------------------------------------------------

// Category constants for scatter map classification
constexpr int kCatPeer = 0;  // intra-node or self (keyed by dest_local_rank)
constexpr int kCatInter = 1; // direct GIN put (same rail, inter-node)
constexpr int kCatRelay = 2; // cross-rail relay via NVLink peer

/// Pass 1: Classify each row and compute its gather index.
/// 1 thread per row in rank-major order.
__global__ void classify_and_gather_kernel(
    const int64_t *__restrict__ recatted_offsets, // [F+1]
    const int64_t *__restrict__ feature_offsets,  // [F+1]
    const int64_t *__restrict__ split_offsets,    // [W+1]
    const int64_t *__restrict__ feature_recat,    // [F]
    const int *__restrict__ rank_to_node,         // [W]
    const int *__restrict__ rank_to_local,        // [W]
    int my_node, int my_local_rank, int local_world_size, int num_nodes,
    int64_t total_rows, int64_t num_features, int world_size,
    // Optional: inv_unbucketize for backward scatter map
    const int64_t *__restrict__ inv_unbucketize, // NULL for forward
    // Outputs:
    int64_t *__restrict__ gather_indices,  // [total_rows]
    int *__restrict__ category,            // [total_rows]
    int *__restrict__ dest_id,             // [total_rows]
    int *__restrict__ dest_node_id,        // [total_rows] (only for relay)
    int *__restrict__ pos_in_dest          // [total_rows] deterministic position
) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total_rows)
    return;

  // Feature-to-row expansion (inline, consumes prefix sums)
  int64_t feat_idx = upper_bound_device(recatted_offsets, num_features + 1, i) - 1;
  int64_t row_within_feat = i - recatted_offsets[feat_idx];
  int64_t src_feat = feature_recat[feat_idx];
  int64_t base_gather = feature_offsets[src_feat] + row_within_feat;

  // Apply inv_unbucketize composition for backward
  int64_t gi = (inv_unbucketize != nullptr) ? inv_unbucketize[base_gather]
                                             : base_gather;
  gather_indices[i] = gi;

  // Classification
  int64_t dest_rank =
      upper_bound_device(split_offsets, world_size + 1, i) - 1;
  int dn = rank_to_node[dest_rank];
  int dl = rank_to_local[dest_rank];

  // Deterministic position within destination's CSR bucket
  pos_in_dest[i] = (int)(i - split_offsets[dest_rank]);

  if (dn == my_node) {
    // PEER (includes self when dl == my_local_rank)
    category[i] = kCatPeer;
    dest_id[i] = dl;
    dest_node_id[i] = dn;
  } else if (dl == my_local_rank) {
    // INTER (direct GIN put, same rail)
    category[i] = kCatInter;
    dest_id[i] = (int)dest_rank;
    dest_node_id[i] = dn;
  } else {
    // RELAY (cross-rail, via same-rail peer on my node)
    category[i] = kCatRelay;
    dest_id[i] = dl; // relay through peer with this local_rank
    dest_node_id[i] = dn;
  }
}

/// Pass 2a: Histogram counts per category bucket.
/// Atomically counts rows per (category, dest_id) pair.
__global__ void histogram_kernel(
    const int *__restrict__ category, const int *__restrict__ dest_id,
    const int *__restrict__ dest_node_id, int64_t total_rows,
    int local_world_size, int num_nodes,
    // Outputs:
    int *__restrict__ peer_counts,      // [local_world_size] intra+self counts
    int *__restrict__
        relay_counts, // [local_world_size * num_nodes] (lr, dest_node)
    int *__restrict__ inter_counts // [num_nodes] inter-node counts (excl my
                                    // node)
) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total_rows)
    return;

  int cat = category[i];
  int did = dest_id[i];

  if (cat == kCatPeer) {
    atomicAdd(&peer_counts[did], 1);
  } else if (cat == kCatInter) {
    // Map dest_rank to inter bucket (by dest_node)
    int dn = dest_node_id[i];
    atomicAdd(&inter_counts[dn], 1);
  } else { // kCatRelay
    int relay_lr = did;
    int dn = dest_node_id[i];
    atomicAdd(&relay_counts[relay_lr * num_nodes + dn], 1);
  }
}

/// Pass 2b + 2c: Build CSR from histogram (prefix sums + scatter).
/// This is a host-side orchestration kernel — see hier_all2all.cu for the
/// full implementation using CUB prefix sums + scatter kernel.

/// Scatter kernel: write gather_indices into CSR arrays using per-bucket
/// atomicAdd counters.
__global__ void scatter_to_csr_kernel(
    const int64_t *__restrict__ gather_indices, // [total_rows]
    const int *__restrict__ category,           // [total_rows]
    const int *__restrict__ dest_id,            // [total_rows]
    const int *__restrict__ dest_node_id,       // [total_rows]
    const int *__restrict__ pos_in_dest,        // [total_rows] deterministic pos
    int64_t total_rows, int local_world_size, int num_nodes,
    // CSR structure (pre-computed offsets from prefix sum):
    const int64_t *__restrict__ peer_csr_offsets,    // [local_world_size + 1]
    const int *__restrict__ peer_intra_counts_array, // [local_world_size]
    const int64_t *__restrict__
        relay_sub_offsets, // [local_world_size * num_nodes] within peer's relay
    const int64_t *__restrict__ inter_csr_offsets, // [num_inter + 1]
    // Outputs:
    int64_t *__restrict__ peer_gather_out, // [total_peer_rows]
    int64_t *__restrict__ inter_gather_out // [total_inter_rows]
) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total_rows)
    return;

  int cat = category[i];
  int did = dest_id[i];
  int64_t gi = gather_indices[i];
  int pos = pos_in_dest[i];

  if (cat == kCatPeer) {
    int64_t base = peer_csr_offsets[did];
    peer_gather_out[base + pos] = gi;
  } else if (cat == kCatInter) {
    int64_t base = inter_csr_offsets[dest_node_id[i]];
    inter_gather_out[base + pos] = gi;
  } else { // kCatRelay
    int relay_lr = did;
    int dn = dest_node_id[i];
    int64_t base = peer_csr_offsets[relay_lr] + peer_intra_counts_array[relay_lr];
    int64_t sub_off = relay_sub_offsets[relay_lr * num_nodes + dn];
    peer_gather_out[base + sub_off + pos] = gi;
  }
}

/// Materialize backward_row_recat on GPU.
/// 1 thread per row.
__global__ void materialize_backward_row_recat_kernel(
    const int64_t *__restrict__ sparse_features_recat, // [F]
    const int64_t *__restrict__ out_feature_offsets,    // [F+1]
    const int64_t *__restrict__ out_recatted_offsets,   // [F+1]
    int64_t total_rows, int64_t num_features,
    int64_t *__restrict__ backward_row_recat // [total_rows]
) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total_rows)
    return;

  int64_t feat_idx =
      upper_bound_device(out_feature_offsets, num_features + 1, i) - 1;
  int64_t row_within_feat = i - out_feature_offsets[feat_idx];
  int64_t src_feat = sparse_features_recat[feat_idx];
  backward_row_recat[i] = out_recatted_offsets[src_feat] + row_within_feat;
}

// ---------------------------------------------------------------------------
// cp.async helpers (sm_90a, used by pipelined kernel)
// ---------------------------------------------------------------------------

// The pipelined kernel and its helpers target sm_90a (H100/H200) only.
// Older architectures fall back to NCCL at the Python dispatch level.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

/// Convert generic pointer to 32-bit shared memory address for cp.async PTX.
__device__ __forceinline__ uint32_t __smem_addr(const void *ptr) {
  uint32_t addr;
  asm volatile("{ .reg .u64 u64addr;\n\t"
               "  cvta.to.shared.u64 u64addr, %1;\n\t"
               "  cvt.u32.u64 %0, u64addr;\n\t"
               "}"
               : "=r"(addr)
               : "l"(ptr));
  return addr;
}

/// Issue cp.async: 16 bytes from global to shared memory (cached at all levels).
__device__ __forceinline__ void cp_async_16(void *smem_dst,
                                             const void *global_src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(
                   __smem_addr(smem_dst)),
               "l"(global_src)
               : "memory");
}

/// Commit current cp.async group.
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

/// Wait until at most N async groups remain in-flight.
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

#endif // __CUDA_ARCH__ >= 900

// ---------------------------------------------------------------------------
// Pipelined fused kernel for H100/H200 (sm_90a)
// ---------------------------------------------------------------------------
// CTA-specialized producer-consumer with overlapped outcast and gather:
//
//   CTAs 0..outcast_ctas-1 (Producer): direct outcast (HBM → NVLink write),
//     then join gather after signaling.
//   CTAs outcast_ctas..grid-1 (Consumer): gather-only, start immediately
//     by waiting for peer signals. Run on different SMs than producers.
//
// NVLink writes (outcast OUT) and reads (gather IN) use opposite link
// directions and overlap bidirectionally on H100 NVSwitch.
// No cp.async pipeline — keeps smem minimal for maximum CTA count.
//
// Grid : outcast_ctas + gather_ctas (>= outcast_ctas).
// Block: kThreadsPerCTA (256) threads = 8 warps.
//
// Shared memory layout (dynamic, metadata only):
//   [0                   ) int64_t splits_prefix[W+1]
//   [prefix_bytes        ) int     r2l_cache[W]
// ---------------------------------------------------------------------------

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

template <typename scalar_t>
__global__ void __launch_bounds__(kThreadsPerCTA, 1)
    hier_a2a_fwd_pipelined_kernel(
        // ---- Outcast params ----
        const scalar_t *__restrict__ input_embs,
        const int64_t *__restrict__ peer_gather_indices,
        const int64_t *__restrict__ peer_offsets,
        const uintptr_t *__restrict__ peer_remote_slot_ptrs,
        const uintptr_t *__restrict__ peer_signal_ptrs,
        int64_t D, int ctas_per_peer,
        int32_t *__restrict__ done_counters,
        // ---- Gather params (inline permute) ----
        const scalar_t *__restrict__ ipc_recv_buf,
        scalar_t *__restrict__ final_output,
        const int64_t *__restrict__ output_splits, // [W] raw, NOT prefix sum
        const int *__restrict__ rank_to_local,     // [W]
        const int64_t *__restrict__ unbucketize_perm, // [total_recv] or NULL
        int64_t max_rows_per_rank,
        // ---- Sync ----
        int32_t *__restrict__ my_signals,
        int32_t *__restrict__ device_flag,
        // ---- Dims ----
        int64_t total_output_rows, int world_size,
        int local_world_size, int my_local_rank,
        int32_t iter_id, int batch_rows, // batch_rows unused (no cp.async)
        int outcast_ctas, // CTAs 0..outcast_ctas-1 do outcast; rest skip to gather
        int64_t *__restrict__ phase_clocks // [4] or NULL — debug timing
        ) {
  extern __shared__ char smem_raw[];
  constexpr int kNumWarps = kThreadsPerCTA / 32;
  const int warp_id = get_warp_id();
  const int lane_id = get_lane_id();
  const int row_bytes = D * sizeof(scalar_t);

  // ================================================================
  // Shared memory layout (metadata only — no pipeline buffer)
  // ================================================================
  int64_t *splits_prefix = reinterpret_cast<int64_t *>(smem_raw);
  const int prefix_bytes = (world_size + 1) * (int)sizeof(int64_t);
  const int r2l_bytes = world_size * (int)sizeof(int);
  const int meta_end = (prefix_bytes + r2l_bytes + 127) & ~127;
  int *r2l_cache = reinterpret_cast<int *>(smem_raw + prefix_bytes);

  // ---- Kernel start time ----
  if (phase_clocks && blockIdx.x == 0 && threadIdx.x == 0)
    phase_clocks[3] = clock64();

  // ================================================================
  // Prologue: compute prefix sum + cache topology in smem
  // ================================================================
  if (threadIdx.x == 0) {
    splits_prefix[0] = 0;
    for (int i = 0; i < world_size; ++i)
      splits_prefix[i + 1] = splits_prefix[i] + output_splits[i];
  }
  if ((int)threadIdx.x < world_size)
    r2l_cache[threadIdx.x] = rank_to_local[threadIdx.x];
  __syncthreads();

  // ================================================================
  // PHASE 1: OUTCAST (producer CTAs only)
  //
  // Consumer CTAs skip straight to gather.
  // Producer CTAs use all 8 warps for outcast, then join gather.
  //
  // Large D (row_bytes > kOutcastThreadCopyThreshold): cp.async double-buffered
  //   pipeline overlapping scattered HBM reads with NVLink writes.
  // Small D: direct thread_copy_row (1 row/thread, no smem needed).
  // ================================================================
  if ((int)blockIdx.x < outcast_ctas) {
    const int dest_lr = (int)blockIdx.x / ctas_per_peer;
    const int sub_cta = (int)blockIdx.x % ctas_per_peer;
    const int64_t off_start = peer_offsets[dest_lr];
    const int64_t total_peer_rows = peer_offsets[dest_lr + 1] - off_start;
    scalar_t *remote_slot =
        reinterpret_cast<scalar_t *>(peer_remote_slot_ptrs[dest_lr]);

    // Partition rows across sub-CTAs
    const int64_t rows_per_sub =
        (total_peer_rows + ctas_per_peer - 1) / ctas_per_peer;
    const int64_t my_start =
        min((int64_t)sub_cta * rows_per_sub, total_peer_rows);
    const int64_t my_end = min(my_start + rows_per_sub, total_peer_rows);
    const int64_t my_rows = my_end - my_start;

    if (row_bytes <= kOutcastThreadCopyThreshold) {
      // ---- Small D: direct thread-level copy (no smem) ----
      for (int64_t r = (int)threadIdx.x; r < my_rows; r += kThreadsPerCTA) {
        const int64_t gi = peer_gather_indices[off_start + my_start + r];
        thread_copy_row(input_embs + gi * D, remote_slot + (my_start + r) * D, D);
      }
    } else if (batch_rows > 0) {
      // ---- Large D: cp.async pipeline (HBM→smem→NVLink) ----
      // Pipeline buffer sits after metadata in smem (only producer CTAs use it).
      char *pipe = smem_raw + meta_end;
      const int64_t buf_stride = (int64_t)batch_rows * row_bytes;
      const int num_batches = (int)((my_rows + batch_rows - 1) / batch_rows);

      if (num_batches > 0) {
        // Prologue: issue cp.async for batch 0
        {
          const int b0_cnt = (int)min((int64_t)batch_rows, my_rows);
          for (int r = warp_id; r < b0_cnt; r += kNumWarps) {
            const int64_t gi =
                peer_gather_indices[off_start + my_start + r];
            const char *src =
                reinterpret_cast<const char *>(input_embs + gi * D);
            char *dst = pipe + (int64_t)r * row_bytes;
            for (int b = lane_id * kVecBytes; b < row_bytes;
                 b += 32 * kVecBytes)
              cp_async_16(dst + b, src + b);
          }
          cp_async_commit();
        }

        // Steady-state pipeline
        for (int batch = 0; batch < num_batches; ++batch) {
          const int cur_buf = batch & 1;
          const int cur_start = batch * batch_rows;
          const int cur_cnt =
              min(batch_rows, (int)(my_rows - cur_start));

          if (batch + 1 < num_batches) {
            const int nxt_start = (batch + 1) * batch_rows;
            const int nxt_cnt =
                min(batch_rows, (int)(my_rows - nxt_start));
            char *nxt_buf = pipe + (1 - cur_buf) * buf_stride;
            for (int r = warp_id; r < nxt_cnt; r += kNumWarps) {
              const int64_t gi =
                  peer_gather_indices[off_start + my_start + nxt_start + r];
              const char *src =
                  reinterpret_cast<const char *>(input_embs + gi * D);
              char *dst = nxt_buf + (int64_t)r * row_bytes;
              for (int b = lane_id * kVecBytes; b < row_bytes;
                   b += 32 * kVecBytes)
                cp_async_16(dst + b, src + b);
            }
            cp_async_commit();
            cp_async_wait<1>();
          } else {
            cp_async_wait<0>();
          }
          __syncthreads(); // smem visible to all warps in this CTA

          // Write current batch: smem → NVLink remote slot
          const char *cur_base = pipe + cur_buf * buf_stride;
          for (int r = warp_id; r < cur_cnt; r += kNumWarps) {
            const char *src = cur_base + (int64_t)r * row_bytes;
            char *dst = reinterpret_cast<char *>(
                remote_slot + (my_start + cur_start + r) * D);
            for (int b = lane_id * kVecBytes; b < row_bytes;
                 b += 32 * kVecBytes)
              *reinterpret_cast<float4 *>(dst + b) =
                  *reinterpret_cast<const float4 *>(src + b);
          }
          __syncthreads(); // writes done before buffer reuse
        }
      }
    } else {
      // ---- Fallback: direct warp-level copy ----
      for (int64_t r = warp_id; r < my_rows; r += kNumWarps) {
        const int64_t gi = peer_gather_indices[off_start + my_start + r];
        vectorized_copy_row(input_embs + gi * D, remote_slot + (my_start + r) * D, D);
      }
    }

    // Record outcast end time
    if (phase_clocks && blockIdx.x == 0 && threadIdx.x == 0)
      phase_clocks[0] = clock64();

    // Ensure all NVLink writes from this CTA are globally visible before
    // signaling. __syncthreads orders intra-CTA; __threadfence_system
    // flushes prior stores to peer GPU memory.
    __syncthreads();
    __threadfence_system();

    if (threadIdx.x == 0) {
      int finished = atomicAdd(&done_counters[dest_lr], 1) + 1;
      if (finished == ctas_per_peer) {
        done_counters[dest_lr] = 0;
        __threadfence();
        uintptr_t sig = peer_signal_ptrs[dest_lr];
        if (sig != 0)
          signal_peer(reinterpret_cast<int32_t *>(sig), iter_id);
        else
          signal_peer(&my_signals[my_local_rank], iter_id);
      }
    }
  } // end outcast (producer CTAs)

  // ================================================================
  // PHASE 2+3: FUSED WAIT + GATHER (ALL CTAs participate)
  //
  // Consumer CTAs arrive here immediately (overlap with producers).
  // Producer CTAs arrive after outcast + signal.
  // ================================================================
  {
    if (unbucketize_perm == nullptr) {
      // ---- Per-peer PARALLEL gather (no permute) ----
      // Assign CTA groups to peers: each peer gets gridDim.x/L CTAs.
      // All L peer groups wait+gather independently and simultaneously,
      // using all L NVLink links in parallel (vs sequential 1-at-a-time).
      const int ctas_per_lr = (int)gridDim.x / local_world_size;
      const int my_peer_lr = min((int)blockIdx.x / ctas_per_lr,
                                  local_world_size - 1);
      const int peer_cta_idx = (int)blockIdx.x - my_peer_lr * ctas_per_lr;

      // Wait for this specific peer's signal (independent per peer group)
      if (lane_id == 0) {
        spin_wait_signal(&my_signals[my_peer_lr], iter_id);
      }
      __syncwarp();

      // Gather only this peer's rows (all NVLink links active simultaneously)
      const int64_t ipc_base = (int64_t)my_peer_lr * max_rows_per_rank;

      if (row_bytes <= kThreadCopyThreshold) {
        // ---- Small D: thread-level copy (1 thread per row) ----
        // At D=64 bf16 (128B/row), warp-level copy uses only 8/32 lanes.
        // Thread-level: all 256 threads active, each copying a full row.
        // With 66 CTAs/peer (528/8), that's 16896 threads vs 528 warps.
        const int threads_per_peer = ctas_per_lr * kThreadsPerCTA;
        const int my_thread_in_peer = peer_cta_idx * kThreadsPerCTA + (int)threadIdx.x;
        for (int src_rank = my_peer_lr; src_rank < world_size;
             src_rank += local_world_size) {
          const int64_t row_start = splits_prefix[src_rank];
          const int64_t row_end = splits_prefix[src_rank + 1];
          const int64_t peer_rows = row_end - row_start;
          if (peer_rows <= 0) continue;

          for (int64_t idx = my_thread_in_peer; idx < peer_rows;
               idx += threads_per_peer) {
            thread_copy_row(
                ipc_recv_buf + (ipc_base + idx) * D,
                final_output + (row_start + idx) * D, D);
          }
        }
      } else {
        // ---- Large D: warp-level vectorized copy ----
        const int warps_per_peer = ctas_per_lr * kNumWarps;
        const int my_warp_in_peer = peer_cta_idx * kNumWarps + warp_id;
        for (int src_rank = my_peer_lr; src_rank < world_size;
             src_rank += local_world_size) {
          const int64_t row_start = splits_prefix[src_rank];
          const int64_t row_end = splits_prefix[src_rank + 1];
          const int64_t peer_rows = row_end - row_start;
          if (peer_rows <= 0) continue;

          for (int64_t idx = my_warp_in_peer; idx < peer_rows;
               idx += warps_per_peer) {
            vectorized_copy_row(
                ipc_recv_buf + (ipc_base + idx) * D,
                final_output + (row_start + idx) * D, D);
          }
        }
      }
    } else {
      // ---- Per-CTA signal wait + full gather with unbucketize ----
      if (threadIdx.x == 0) {
        for (int s = 0; s < local_world_size; ++s)
          spin_wait_signal(&my_signals[s], iter_id);
      }
      __syncthreads();

      // Gather with unbucketize: scattered reads across all peers.
      if (row_bytes <= kThreadCopyThreshold) {
        // ---- Small D: thread-level copy (1 thread per row) ----
        const int global_tid = (int)blockIdx.x * kThreadsPerCTA + (int)threadIdx.x;
        const int total_threads = (int)gridDim.x * kThreadsPerCTA;
        for (int64_t i = global_tid; i < total_output_rows;
             i += total_threads) {
          const int64_t rm_i = unbucketize_perm[i];
          const int64_t s =
              upper_bound_device(splits_prefix, world_size + 1, rm_i) - 1;
          const int64_t j = rm_i - splits_prefix[s];
          const int lr = r2l_cache[s];
          const int64_t ipc_pos = (int64_t)lr * max_rows_per_rank + j;

          thread_copy_row(ipc_recv_buf + ipc_pos * D,
                          final_output + i * D, D);
        }
      } else {
        // ---- Large D: warp-level vectorized copy ----
        const int global_warp = (int)blockIdx.x * kNumWarps + warp_id;
        const int total_warps = (int)gridDim.x * kNumWarps;
        for (int64_t i = global_warp; i < total_output_rows;
             i += total_warps) {
          const int64_t rm_i = unbucketize_perm[i];
          const int64_t s =
              upper_bound_device(splits_prefix, world_size + 1, rm_i) - 1;
          const int64_t j = rm_i - splits_prefix[s];
          const int lr = r2l_cache[s];
          const int64_t ipc_pos = (int64_t)lr * max_rows_per_rank + j;

          vectorized_copy_row(ipc_recv_buf + ipc_pos * D,
                              final_output + i * D, D);
        }
      }
    }
  }

  // ---- Kernel end time ----
  if (phase_clocks && blockIdx.x == 0 && threadIdx.x == 0) {
    phase_clocks[1] = clock64();
    phase_clocks[2] = clock64();
  }
}

#else // !(__CUDA_ARCH__ >= 900)

// Non-Hopper stub: the symbol must exist for the linker, but should never run.
template <typename scalar_t>
__global__ void __launch_bounds__(kThreadsPerCTA, 1)
    hier_a2a_fwd_pipelined_kernel(
        const scalar_t *__restrict__ input_embs,
        const int64_t *__restrict__ peer_gather_indices,
        const int64_t *__restrict__ peer_offsets,
        const uintptr_t *__restrict__ peer_remote_slot_ptrs,
        const uintptr_t *__restrict__ peer_signal_ptrs,
        int64_t D, int ctas_per_peer,
        int32_t *__restrict__ done_counters,
        const scalar_t *__restrict__ ipc_recv_buf,
        scalar_t *__restrict__ final_output,
        const int64_t *__restrict__ output_splits,
        const int *__restrict__ rank_to_local,
        const int64_t *__restrict__ unbucketize_perm,
        int64_t max_rows_per_rank,
        int32_t *__restrict__ my_signals,
        int32_t *__restrict__ device_flag,
        int64_t total_output_rows, int world_size,
        int local_world_size, int my_local_rank,
        int32_t iter_id, int batch_rows,
        int outcast_ctas,
        int64_t *__restrict__ phase_clocks) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    asm("trap;\n");
}

#endif // __CUDA_ARCH__ >= 900

} // namespace hier_a2a
