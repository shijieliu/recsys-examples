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
// The call's tables as a requester's warps read them.
#pragma once

#include <cstdint>

#include "hier_output_dist/common/formats.h"
#include "hier_output_dist/common/windows.h"
#include "hier_output_dist/device/ctrl_words.cuh"
#include "hier_output_dist/device/search.cuh"
#include "hier_output_dist/device/sym_window.cuh"
#include "hier_output_dist/output/args.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// What a requester's warps read of the call, from global memory: the block
// starts, each same-node proxy's rounds and the entries of two tables, its
// round table (where the rows of each owner node's keys sit in its
// proxy_rows) and this rank's push table there (ReqEntry; the next entry
// holds the ends); and the regions of the node peers, through their LSA
// pointers. The one word per proxy a CTA keeps, half_seen, is in its shared
// memory. Every thread of a warp makes the same one.
struct OutTables {
  const OutArgs &args;
  int max_rounds;  // the most rounds of the node's proxies (1 on one node)
  bool staged;     // index dedup's heavy rows at a proxy of more than one round

  __device__ explicit OutTables(const OutArgs &out_args) : args(out_args), max_rounds(1) {
    if (args.topo.num_nodes > 1)
      for (int proxy = 0; proxy < args.topo.local_size; ++proxy)
        max_rounds = max(max_rounds, rounds(proxy));
    staged = args.plan.fanout.exp_off != nullptr && max_rounds > 1 && args.plan.fanout.count[1] > 0;
  }
  // A same-node proxy's rounds.
  __device__ __forceinline__ int rounds(int proxy) const {
    return (int)args.plan.meta[args.meta_at.proxy_rounds + proxy];
  }
  // Node peers' regions and words.
  __device__ __forceinline__ char *lookup_rows(int peer) const {
    return SymRegion<kLookupRows>(args.win).peer(peer);
  }
  __device__ __forceinline__ char *proxy_rows(int proxy) const {
    return SymRegion<kProxyRows>(args.win).peer(proxy);
  }
  __device__ __forceinline__ uint64_t *push_flags(int proxy) const {
    return SymRegion<kPushFlags, uint64_t>(args.win).peer(proxy);
  }
  __device__ __forceinline__ const uint64_t *landed_rows(int proxy, int owner_node) const {
    return CtrlWords::of(args, proxy).landed_rows(owner_node);
  }
  // This rank's two done words at a proxy.
  __device__ __forceinline__ uint64_t *done(int proxy) const {
    return CtrlWords::of(args, proxy).done(args.topo.local_rank);
  }
  __device__ __forceinline__ const uint64_t *half_free(int proxy) const {
    return CtrlWords::of(args, proxy).half_free();
  }
  // The most of a proxy's half_free word that this CTA's warps saw.
  __device__ __forceinline__ uint64_t *half_seen(int proxy) const {
    return (uint64_t *)out_shared() + proxy;
  }
  // This rank's push order at a proxy in a round (positions of push_order)
  // and the next round's entry, which holds the ends.
  __device__ __forceinline__ void req_of(int proxy, int rnd, ReqEntry &entry,
                                         ReqEntry &next) const {
    const ReqEntry *push_table = (const ReqEntry *)(args.plan.meta + args.meta_at.push_tables) +
                                 proxy * (args.win.rounds_cap + 1) + rnd;
    entry = push_table[0];
    next = push_table[1];
  }
  // Where an owner node's node-unique keys of a round sit in a proxy's
  // proxy_rows.
  __device__ __forceinline__ RoundPiece piece_of(int proxy, int rnd, int owner_node) const {
    return ((const RoundPiece *)(args.plan.meta + args.meta_at.proxy_round_tables))
        [((int64_t)proxy * args.win.rounds_cap + rnd) * args.topo.num_nodes + owner_node];
  }
  // The owner of one of this rank's rows (its bucketized rows by owner).
  __device__ __forceinline__ int owner_of(int64_t row) const {
    return seg_search(args.plan.block_start, args.topo.world_size, row);
  }
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
