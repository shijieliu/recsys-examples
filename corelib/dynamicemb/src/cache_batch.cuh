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

#pragma once

#include "cache_deterministic.h"
#include "table_operation/table.cuh"

namespace dyn_emb {

// Tuples cross the binding as plain tensors and scalars. Prepare and exchange
// use the same physical maps, with independent scores for lookup and insertion.
using CacheOptionalTensor = std::optional<at::Tensor>;
using CacheScoreArg = std::tuple<CacheOptionalTensor, ScorePolicyType>;
// Main buffer, bucket offsets/capacity, pins, overflow buffer/capacity/output
// offsets, score word count, optional global hashing capacities.
using CacheLookupMap =
    std::tuple<at::Tensor, at::Tensor, int64_t, at::Tensor, at::Tensor,
               int64_t, at::Tensor, int64_t, CacheOptionalTensor>;
// Main buffer, bucket offsets/capacity, slot-to-row map, score word count,
// optional global hashing capacities.
using BackingLookupMap =
    std::tuple<at::Tensor, at::Tensor, int64_t, CacheOptionalTensor,
               int64_t, CacheOptionalTensor>;
// Per-table value-buffer pointers, embedding and value widths, maximum
// embedding and value widths, and the dtype of the stored words. Uniform
// tables use the two scalar widths and omit both per-table width tensors.
using CacheValueLayout =
    std::tuple<at::Tensor, CacheOptionalTensor, CacheOptionalTensor, int64_t, int64_t,
               at::ScalarType>;

// Byte addresses and a layout tag for each selected row. Zero is a missing
// address. Tag zero means a contiguous full-width row; tag t+1 identifies a
// narrower table. Exchange also asks for each dirty cache slot's source.
struct BackingValueAddresses {
  at::Tensor backing;
  CacheOptionalTensor layout_tags, sources;
};
BackingValueAddresses lookup_backing_values(
    const BackingLookupMap &map, const CacheValueLayout &values, at::Tensor keys,
    at::Tensor tids, const CacheScoreArg &score, const CacheOptionalTensor &num_valid = {},
    const CacheOptionalTensor &needed = {}, const CacheOptionalTensor &source_ptrs = {},
    const CacheOptionalTensor &source_slots = {});

std::optional<EvictionResult> try_ranked_cache_reservation(
    const CacheLookupMap &map, at::Tensor sizes, at::Tensor overflow_sizes,
    at::Tensor keys, at::Tensor tids, at::Tensor count, at::Tensor backing_count,
    at::Tensor status);

std::tuple<at::Tensor, at::Tensor, std::vector<CacheOptionalTensor>>
flagged_compact(at::Tensor flags, std::vector<CacheOptionalTensor> inputs,
                const CacheOptionalTensor &num_valid, bool invert,
                std::optional<int64_t> host_num_valid);

std::tuple<at::Tensor, at::Tensor, std::vector<CacheOptionalTensor>>
compact_missing_values(at::Tensor addresses, std::vector<CacheOptionalTensor> fields,
                       const CacheOptionalTensor &num_valid);

} // namespace dyn_emb
