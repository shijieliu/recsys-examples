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
// The comm kernels of the in channel (the first wait, on the op's side
// stream): the input dist's only custom kernels, each one exchange with the
// node peers or the rail peers. The planner's steps (plan/) run between them
// as ATen ops on the same stream, in the order of InChannel::start:
//   node_count_exchange   the node barrier of the call, and the node's
//                         counts as this rank needs them;
//   node_key_push         this rank's keys into its targets' node_keys over
//                         NVLink;
//   node_done             tells each target its keys are there, and waits
//                         until every requester of the node told it;
//   rail_count_exchange   (across nodes) its count message to each rail
//                         peer, and theirs;
//   rail_key_exchange     (across nodes) its key records to each rail owner
//                         and its cell starts to each rail proxy, and theirs;
//   node_status_exchange  (across nodes) its rail's needs and cell starts to
//                         its node, and every node peer's;
//   peer_gather           (across nodes) words of the node peers' remote_map
//                         or proxy header.
// A kernel that waits runs on one CTA and waits only for the same kernel (the
// same seq) of its node or rail peers; the others never wait. Its threads (or
// lanes) take the peers in turn, so any number of them. Memory order:
// what peers wrote is read only after an acquire of their flag or signal, by
// the comm kernel that acquired it (with ld.global.cg) or by a later kernel
// on the stream (peer_gather with ld.global.cg, the planner's ATen steps with
// plain loads). What this rank writes for its peers (its own stores, or
// ATen's earlier on the stream) is ordered before the flag it releases or the
// put it makes: by fence.sc.sys, or in node_count_exchange by the release
// itself after the CTA's barrier.
#pragma once

#include <nccl_device.h>

#include <cstdint>

#include "hier_output_dist/common/channel_args.h"
#include "hier_output_dist/common/constants.h"
#include "hier_output_dist/common/ctrl.h"
#include "hier_output_dist/common/gin_signals.h"
#include "hier_output_dist/common/status.h"
#include "hier_output_dist/common/windows.h"
#include "hier_output_dist/device/ctrl_words.cuh"
#include "hier_output_dist/device/memory_order.cuh"
#include "hier_output_dist/device/node_sync.cuh"
#include "hier_output_dist/device/rail.cuh"
#include "hier_output_dist/device/spin.cuh"
#include "hier_output_dist/device/sym_window.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// The argument of a comm kernel that waits: the part both channels share,
// and the in channel at the call's sequence number (its node barrier word
// in_started), which NodeBarrier and PeerFlags read.
struct CommArgs : ChannelStatic {
  Channel ch;
};

// 1 CTA: this rank's count vector (plan/node.h) into its in_meta slot of the
// call, the node barrier, then the node's count vectors as this rank needs
// them: their sum (totals) and that of the ranks before it (before), both
// [words], and each requester's counts for this rank (mine [local_size]
// [num_features + 1]: its keys for its remote part, then for each feature
// of its own part). Passing the barrier also means every peer finished its
// previous call on its side stream, whose steps read what this call writes
// into their windows.
__global__ void node_count_exchange_kernel(CommArgs args, const int64_t *my_counts, int64_t *totals,
                                           int64_t *before, int64_t *mine) {
  const int local_size = args.topo.local_size, num_features = args.num_features,
            me = args.topo.local_rank, tid = threadIdx.x;
  const int words = local_size * (num_features + 1);
  const int slot = args.ctrl_at.in_meta[args.ch.seq & 1];
  const CtrlWords own = CtrlWords::of(args, me);
  for (int i = tid; i < words; i += blockDim.x) *own.word(slot + i) = (uint64_t)my_counts[i];
  __syncthreads();
  const NodeBarrier<CommArgs> barrier(args, args.ch.started);
  if (tid == 0) barrier.publish(own);
  for (int peer = tid; peer < local_size; peer += blockDim.x)
    barrier.wait(peer, "in-channel node barrier");
  __syncthreads();
  auto count_of = [&](int peer, int word) {
    return (int64_t)__ldcg((const unsigned long long *)CtrlWords::of(args, peer).word(slot + word));
  };
  for (int i = tid; i < words; i += blockDim.x) {
    int64_t total = 0, earlier = 0;
    for (int peer = 0; peer < local_size; ++peer) {
      const int64_t count = count_of(peer, i);
      total += count;
      if (peer < me) earlier += count;
    }
    totals[i] = total;
    before[i] = earlier;
  }
  for (int i = tid; i < local_size * (num_features + 1); i += blockDim.x) {
    const int peer = i / (num_features + 1), column = i % (num_features + 1);
    mine[i] = count_of(peer, column == 0 ? me : local_size + me * num_features + column - 1);
  }
}

// Every CTA, a thread per row: row `row` of part 1 (row < *rows_to_push) goes
// to destination_base[cell] + row of node_keys (and node_weights) of its
// target, the owner's local rank; a destination from remote_base on is an
// occurrence of the target's remote part, whose cell tag (owner node *
// num_features + feature) goes to remote_cells. Consecutive rows of a cell
// go to consecutive destinations. Each CTA's stores are fenced at system
// scope before it ends.
__global__ void __launch_bounds__(kThreads)
    node_key_push_kernel(ChannelStatic channel, const int64_t *keys, const float *weights,
                         const int32_t *row_cell, const int64_t *destination_base,
                         const int64_t *rows_to_push, int64_t remote_base) {
  const int local_size = channel.topo.local_size, num_features = channel.num_features;
  const int64_t rows = *rows_to_push;
  for (int64_t row = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; row < rows;
       row += (int64_t)gridDim.x * blockDim.x) {
    const int cell = row_cell[row];
    const int owner = cell / num_features, target = owner % local_size;
    const int64_t destination = destination_base[cell] + row;
    SymRegion<kNodeKeys, int64_t>(channel.win).peer(target)[destination] = keys[row];
    if (weights != nullptr)
      SymRegion<kNodeWeights, float>(channel.win).peer(target)[destination] = weights[row];
    if (destination >= remote_base)
      SymRegion<kRemoteCells, int32_t>(channel.win).peer(target)[destination - remote_base] =
          owner / local_size * num_features + cell % num_features;
  }
  __syncthreads();
  if (threadIdx.x == 0) fence_sc_sys();
}

// 1 warp: each thread, after a system fence of its own, tells its node peers
// (peer = thread, thread + 32, ...) that this rank's keys are in their
// node_keys, then waits until each of them told this rank.
__global__ void node_done_kernel(CommArgs args) {
  const int me = args.topo.local_rank;
  const PeerFlags<CommArgs> written(args, args.ctrl_at.in_written);
  fence_sc_sys();
  for (int peer = threadIdx.x; peer < args.topo.local_size; peer += blockDim.x)
    written.tell(peer, me);
  for (int peer = threadIdx.x; peer < args.topo.local_size; peer += blockDim.x)
    written.wait(me, peer, "pushed keys");
}

// 1 CTA (8 warps): this rank's count messages (msg_out_words words,
// common/ctrl.h) into its count_msg_out, and its own message (the common
// part and the part for its own node) into its row of the call's
// count_msgs slot. Then warp 1's lanes, each taking rail peers in turn, put
// each peer its message in two puts into this rank's row of the peer's slot:
// the common part, then the part for the peer's node, which signals (the
// signal also covers the first put). Warp 0's lanes wait for the rail peers'
// messages, and every peer's lands in messages [num_nodes][msg_len]. The
// slots go by in_seq parity: a peer's message of call seq + 2 needs this
// rank's of seq + 1, which it sends once call seq ended here.
__global__ void rail_count_exchange_kernel(CommArgs args, const int64_t *message,
                                           int64_t *messages) {
  const int num_nodes = args.topo.num_nodes, node = args.topo.node,
            num_features = args.num_features, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int common = msg_common(num_nodes), length = msg_len(num_nodes, num_features),
            stride = args.ctrl_at.msg_words;
  const CtrlWords mine = CtrlWords::of(args, args.topo.local_rank);
  const uint64_t slot = args.ctrl_at.count_msgs + (uint64_t)(args.ch.seq & 1) * num_nodes * stride;
  int64_t *out = mine.count_msg_out();
  int64_t *land = (int64_t *)mine.word(slot);
  for (int i = tid; i < msg_out_words(num_nodes, num_features); i += blockDim.x)
    out[i] = message[i];
  for (int i = tid; i < length; i += blockDim.x)
    land[node * stride + i] = message[i < common ? i : i + node * num_features];
  __syncthreads();
  ncclGin gin(args.dev, kCtxIn);
  const ncclTeam rail = ncclTeamRail(args.dev);
  if (warp == 1) {
    const uint64_t ctrl = args.win.off[kCtrl], row = slot + (uint64_t)node * stride;
    const uint64_t out_at = (uint64_t)args.ctrl_at.count_msg_out;
    const ncclWindow_t win = args.win.handle[kCtrl];
    const RailMessage counts(gin, rail, sig_count(node));
    fence_sc_sys();
    for (int peer_node = lane; peer_node < num_nodes; peer_node += 32) {
      if (peer_node == node) continue;
      gin.put(rail, peer_node, win, ctrl + 8 * row, win, ctrl + 8 * out_at, (size_t)common * 8,
              ncclGin_None{}, ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
              cuda::thread_scope_system, cuda::thread_scope_system);
      counts.put(peer_node, win, ctrl + 8 * (row + common),
                 ctrl + 8 * (out_at + common + (uint64_t)peer_node * num_features),
                 (size_t)num_features * 8);
    }
  }
  if (warp == 0) {
    Spin spin(args.timeout_ns);
    for (int peer_node = lane; peer_node < num_nodes; peer_node += 32)
      while (peer_node != node && (int64_t)gin.readSignal(sig_count(peer_node)) < args.ch.seq)
        spin.pause("count message", peer_node, args.ch.seq, args.topo.rank);
  }
  __syncthreads();
  for (int i = tid; i < num_nodes * length; i += blockDim.x)
    messages[i] =
        (int64_t)__ldcg((const unsigned long long *)land + (i / length) * stride + i % length);
  // count_msg_out may be written again once the NIC has read it.
  if (warp == 1) gin.flush(ncclCoopThread{});
}

// 1 CTA (2 warps): warp 1's lanes, each taking rail peers in turn, send each
// peer o this call's key stream: as proxy, this rank's slice of key_stage
// (bytes [put_src[o], put_src[o] + put_bytes[o])) into the owner's rail_keys
// at put_dst[o], unless put_skip[o] (the owner overflows lookup_keys); then
// as owner, where that proxy's cells start in its KJT (row o of cell_stage)
// into the proxy's rail_cells, which signals keys[node]. Its system fence
// orders the planner's writes of key_stage and cell_stage before the puts.
// Warp 0 waits for the other nodes' key streams. A proxy puts only after it
// has the owner's count message of the call, which the owner sends once its
// previous call ended.
__global__ void rail_key_exchange_kernel(CommArgs args, const int64_t *put_src,
                                         const int64_t *put_dst, const int64_t *put_bytes,
                                         const bool *put_skip) {
  const int num_nodes = args.topo.num_nodes, node = args.topo.node,
            num_features = args.num_features, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  ncclGin gin(args.dev, kCtxIn);
  const ncclTeam rail = ncclTeamRail(args.dev);
  if (warp == 1) {
    const RailMessage key_stream(gin, rail, sig_keys(num_nodes, node));
    const ncclWindow_t win = args.win.handle[kKeyStage];
    const uint64_t cells_bytes = 8 * (uint64_t)num_features;
    fence_sc_sys();
    for (int owner_node = lane; owner_node < num_nodes; owner_node += 32) {
      if (owner_node == node) continue;
      if (!put_skip[owner_node])
        gin.put(rail, owner_node, win, args.win.off[kRailKeys] + put_dst[owner_node], win,
                args.win.off[kKeyStage] + put_src[owner_node], (size_t)put_bytes[owner_node],
                ncclGin_None{}, ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
                cuda::thread_scope_system, cuda::thread_scope_system);
      key_stream.put(owner_node, win, args.win.off[kRailCells] + node * cells_bytes,
                     args.win.off[kCellStage] + owner_node * cells_bytes, cells_bytes);
    }
  }
  if (warp == 0) {
    Spin spin(args.timeout_ns);
    for (int peer_node = lane; peer_node < num_nodes; peer_node += 32)
      while (peer_node != node &&
             (int64_t)gin.readSignal(sig_keys(num_nodes, peer_node)) < args.ch.seq)
        spin.pause("key stream", peer_node, args.ch.seq, args.topo.rank);
  }
  // key_stage and cell_stage may be written again once the NIC has read them.
  if (warp == 1) gin.flush(ncclCoopThread{});
}

// 1 CTA: this rank's publish vector (its rail's kNumNeeds needs, then its
// cell starts [requester][feature]) into its in_done needs and cell_start
// words; after a system fence, which also orders the planner's writes of
// remote_map and the proxy header before it, in_done gets seq. Once every
// node peer's has it, every peer's needs land in peer_needs [local_size]
// [kNumNeeds] and where each one's KJT holds this requester's cells in
// owner_cell_start [local_size][num_features].
__global__ void node_status_exchange_kernel(CommArgs args, const int64_t *publish,
                                            int64_t *peer_needs, int64_t *owner_cell_start) {
  const int local_size = args.topo.local_size, num_features = args.num_features,
            me = args.topo.local_rank, tid = threadIdx.x;
  const CtrlWords mine = CtrlWords::of(args, me);
  for (int i = tid; i < kNumNeeds + local_size * num_features; i += blockDim.x) {
    if (i < kNumNeeds)
      *mine.in_done_need(i) = (uint64_t)publish[i];
    else
      mine.cell_start()[i - kNumNeeds] = publish[i];
  }
  __syncthreads();
  const NodeBarrier<CommArgs> status(args, args.ctrl_at.in_done);
  if (tid == 0) {
    fence_sc_sys();
    status.publish(mine);
  }
  for (int peer = tid; peer < local_size; peer += blockDim.x) status.wait(peer, "node's statuses");
  __syncthreads();
  auto word_of = [&](int peer, int word) {
    return (int64_t)__ldcg((const unsigned long long *)CtrlWords::of(args, peer).word(word));
  };
  for (int i = tid; i < local_size * kNumNeeds; i += blockDim.x)
    peer_needs[i] = word_of(i / kNumNeeds, args.ctrl_at.in_done + 1 + i % kNumNeeds);
  for (int i = tid; i < local_size * num_features; i += blockDim.x)
    owner_cell_start[i] =
        word_of(i / num_features, args.ctrl_at.cell_start + me * num_features + i % num_features);
}

// Every CTA, a thread per word: word k of item i is int32 word
// first_word[i] + k * word_stride of node peer peer[i]'s region (none when
// peer[i] < 0), into out[k][i]. The peers published the region before the
// in_done this rank's node_status_exchange acquired earlier on the stream.
__global__ void __launch_bounds__(kThreads)
    peer_gather_kernel(ChannelStatic channel, Region region, const int32_t *peer,
                       const int64_t *first_word, int64_t items, int words, int64_t word_stride,
                       int32_t *out) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < items * words;
       i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t item = i % items, word = i / items;
    const int32_t from = peer[item];
    if (from >= 0)
      out[i] = __ldcg((const int32_t *)ncclGetLsaPointer(channel.win.handle[region],
                                                         channel.win.off[region], from) +
                      first_word[item] + word * word_stride);
  }
}

// Signals and counters of every GIN context start at 0: `bad` counts those
// that do not.
__global__ void gin_zero_kernel(ncclDevComm dev, int signals, int counters, int *bad) {
  ncclGin gin(dev, blockIdx.x);
  for (int signal = threadIdx.x; signal < signals; signal += blockDim.x)
    if (gin.readSignal(signal) != 0) atomicAdd(bad, 1);
  for (int counter = threadIdx.x; counter < counters; counter += blockDim.x)
    if (gin.readCounter(counter) != 0) atomicAdd(bad, 1);
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
