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

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <tuple>

namespace dyn_emb {

/**
 * @brief Segmented unique operation that deduplicates keys per table.
 *
 * Sorts keys within each table segment using CUB DeviceSegmentedRadixSort,
 * then uses adjacent-element comparison + scan to produce unique keys,
 * reverse_indices, frequencies, and table offsets. No hash tables, no atomics
 * for dedup, no spin-waits. Sorted output benefits downstream kernels.
 *
 * Also returns sort_permutation and sorted_reverse_indices so the backward
 * pass can skip its internal radix sort.
 *
 * NOTE: This function is fully asynchronous with no GPU-CPU synchronization.
 *
 * @param keys Input keys tensor (int64 or uint64)
 * @param segment_range Per-table boundary offsets (int64, size = num_tables+1).
 *                      segment_range[t] is the start index of table t's keys.
 * @param num_tables Total number of tables
 * @param input_frequencies Controls frequency counting behavior:
 *                          - Undefined tensor: disable frequency counting
 *                          - Empty tensor (numel==0): count each key as 1
 *                          - Tensor with numel==num_keys: use provided weights
 *
 * @return Tuple of 7 tensors:
 *         - num_uniques: size-1 tensor (view of table_offsets[num_tables])
 *         - unique_keys: compacted unique keys (size=num_keys, first
 *           num_uniques valid)
 *         - reverse_indices: input idx -> global unique idx (size=num_keys)
 *         - table_offsets: cumulative unique counts (size=num_tables+1)
 *         - freq_counters: per-unique-key frequency (empty if disabled)
 *         - sort_permutation: sorted position -> original position
 *           (size=num_keys). For backward reuse.
 *         - sorted_reverse_indices: unique index per sorted position
 *           (size=num_keys). For backward reuse.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor>
segmented_unique_cuda(at::Tensor keys, at::Tensor segment_range,
                      int64_t num_tables,
                      at::Tensor input_frequencies = at::Tensor());

/**
 * @brief OLD hash-based segmented unique (kept for A/B debugging).
 *
 * Takes per-element table_ids (not segment_range). Returns 5 tensors.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
segmented_unique_hashtable_cuda(at::Tensor keys, at::Tensor table_ids,
                                int64_t num_tables,
                                at::Tensor input_frequencies = at::Tensor());

/**
 * @brief Expand table IDs from offsets.
 *
 * Generates a table_id for each element based on offsets structure.
 * This is a helper function to prepare input for segmented_unique_cuda.
 *
 * @param offsets Jagged tensor offsets (int64)
 *                Size = num_features * local_batch_size + 1
 *                Indexed by (feature_id * local_batch_size + batch_id)
 *
 * @param table_offsets_in_feature Feature offsets per table (int64), or None
 *                Size = num_tables + 1
 *                Maps features to tables (adjacent features may share a table)
 *                table_offsets_in_feature[t] is the first feature index for
 * table t If None: each feature is treated as a separate table
 *
 * @param num_tables Number of tables (ignored if table_offsets_in_feature is
 * None)
 * @param local_batch_size Batch size per feature
 * @param num_elements Total number of elements (keys)
 *
 * @return table_ids tensor (int64) with same length as num_elements
 */
at::Tensor expand_table_ids_cuda(
    at::Tensor offsets, c10::optional<at::Tensor> table_offsets_in_feature,
    int64_t num_tables, int64_t local_batch_size, int64_t num_elements);

/**
 * @brief Compute new lengths and offsets by evenly distributing unique keys.
 *
 * This is a GPU kernel that evenly distributes unique keys across (feature,
 * batch) buckets. For each table, unique keys are distributed so each bucket
 * gets (unique_count / num_buckets) keys, with the first (unique_count %
 * num_buckets) buckets getting one extra.
 *
 * @param unique_offsets Cumulative unique counts per table (int64, device)
 * @param table_offsets_in_feature Feature offsets per table (int64, device)
 * @param num_tables Number of tables
 * @param local_batch_size Batch size per feature
 * @param new_lengths_size Total size of output (num_features *
 * local_batch_size)
 *
 * @return Tuple of (new_lengths, new_offsets)
 */
std::tuple<at::Tensor, at::Tensor> compute_dedup_lengths_cuda(
    at::Tensor unique_offsets, at::Tensor table_offsets_in_feature,
    int64_t num_tables, int64_t local_batch_size, int64_t new_lengths_size);

} // namespace dyn_emb

// Python binding
void bind_unique_op(pybind11::module &m);
