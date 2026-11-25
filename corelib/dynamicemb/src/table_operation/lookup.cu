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

#include "kernels.cuh"
#include "../cache_batch.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <limits>

namespace dyn_emb {

template <bool BackingRows = false>
static void table_lookup_single_score(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor keys, at::Tensor table_ids,
    std::optional<at::Tensor> score_input, ScorePolicyType policy_type,
    at::Tensor score_output, at::Tensor founds, at::Tensor indices,
    int64_t num_scores, int64_t const *global_caps_ptr,
    int64_t num_total, int64_t const *num_valid_ptr,
    int64_t const *slot_to_row = nullptr, bool const *need_row = nullptr) {

  auto key_type = get_data_type(keys);

  ScoreType *score_input_ptr = nullptr;
  at::Tensor score_input_tensor;
  if (score_input.has_value() && score_input.value().defined()) {
    at::Tensor in = score_input.value();
    if (in.scalar_type() == torch::kUInt64) {
      score_input_ptr = get_pointer<ScoreType>(score_input);
    } else {
      score_input_tensor = in.view(torch::kUInt64);
      score_input_ptr = score_input_tensor.data_ptr<ScoreType>();
    }
  }

  auto score_output_ptr = BackingRows ? nullptr : score_output.data_ptr<int64_t>();
  auto indices_ptr = indices.data_ptr<IndexType>();
  auto founds_ptr = BackingRows ? nullptr : founds.data_ptr<bool>();
  auto table_ids_ptr = table_ids.data_ptr<int64_t>();
  auto table_bucket_offsets_ptr = table_bucket_offsets.data_ptr<int64_t>();

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  constexpr int BLOCK_SIZE = 256;

  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    auto keys_ptr = get_pointer<KeyType>(keys);

    int64_t total_size =
        sizeof(KeyType) + sizeof(DigestType) + num_scores * sizeof(ScoreType);
    int64_t bucket_bytes = bucket_capacity * total_size;
    int64_t num_buckets =
        table_storage.numel() * table_storage.element_size() / bucket_bytes;

    using Bucket = LinearBucket<KeyType>;
    using Table = LinearBucketTable<Bucket>;

    auto table = Table(reinterpret_cast<uint8_t *>(table_storage.data_ptr()),
                       num_buckets, bucket_capacity, num_scores,
                       global_caps_ptr);

    // A lookup takes a thread per entry of its arrays, with a device count
    // too, rather than the capped grid of the other counted ops: each entry is
    // a probe of the hash table that waits on memory, and a capped grid gives
    // each thread several probes in turn and leaves much of the GPU idle in
    // its last wave (10.5M keys: 3.27 ms against 2.65 ms with a thread each).
    // The threads past the count return at once.
    DISPATCH_SCORE_POLICY(policy_type, PolicyTypeV, [&] {
      table_lookup_kernel<Table, 1, PolicyTypeV, false, false, BackingRows>
          <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE,
             BLOCK_SIZE, 0, stream>>>(
              table, table_bucket_offsets_ptr, num_total, keys_ptr,
              table_ids_ptr, founds_ptr, indices_ptr, score_input_ptr,
              score_output_ptr, Table(), nullptr, num_valid_ptr, {}, slot_to_row, need_row);
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor lookup_backing_rows(const BackingLookupMap &map, at::Tensor keys,
                              at::Tensor tids, const CacheScoreArg &score,
                              const CacheOptionalTensor &num_valid,
                              const CacheOptionalTensor &needed) {
  const auto &[table, offsets, capacity, rows, words, global_caps] = map;
  const auto &[scores, policy] = score;
  auto result = at::empty_like(tids);
  if (keys.numel()) {
    table_lookup_single_score<true>(
        table, offsets, capacity, keys, tids, scores, policy, {}, {}, result, words,
        global_capacities_ptr(global_caps, offsets), keys.numel(),
        counted_num_valid_ptr(num_valid, keys.device()),
        rows ? rows->data_ptr<int64_t>() : nullptr,
        needed ? needed->data_ptr<bool>() : nullptr);
  }
  return result;
}

template <bool CachePrefetch = false>
static void table_lookup_with_overflow_single_score(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor keys, at::Tensor table_ids,
    std::optional<at::Tensor> score_input, ScorePolicyType policy_type,
    at::Tensor score_output, at::Tensor founds, at::Tensor indices,
    at::Tensor ovf_storage, int64_t ovf_bucket_capacity,
    at::Tensor ovf_output_offsets, int64_t num_scores,
    int64_t const *global_caps_ptr, int64_t num_total,
    int64_t const *num_valid_ptr, CacheLookupRows cache = {}) {

  auto key_type = get_data_type(keys);

  ScoreType *score_input_ptr = nullptr;
  at::Tensor score_input_tensor;
  if (score_input.has_value() && score_input.value().defined()) {
    at::Tensor in = score_input.value();
    if (in.scalar_type() == torch::kUInt64) {
      score_input_ptr = get_pointer<ScoreType>(score_input);
    } else {
      score_input_tensor = in.view(torch::kUInt64);
      score_input_ptr = score_input_tensor.data_ptr<ScoreType>();
    }
  }

  auto score_output_ptr =
      CachePrefetch ? nullptr : score_output.data_ptr<int64_t>();
  auto indices_ptr = indices.data_ptr<IndexType>();
  auto founds_ptr = founds.data_ptr<bool>();
  auto table_ids_ptr = CachePrefetch ? nullptr : table_ids.data_ptr<int64_t>();
  auto table_bucket_offsets_ptr = table_bucket_offsets.data_ptr<int64_t>();
  auto ovf_output_offsets_ptr = ovf_output_offsets.data_ptr<int64_t>();

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  constexpr int BLOCK_SIZE = 256;

  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    auto keys_ptr = get_pointer<KeyType>(keys);

    int64_t total_size =
        sizeof(KeyType) + sizeof(DigestType) + num_scores * sizeof(ScoreType);
    int64_t bucket_bytes = bucket_capacity * total_size;
    int64_t num_buckets =
        table_storage.numel() * table_storage.element_size() / bucket_bytes;

    using Bucket = LinearBucket<KeyType>;
    using Table = LinearBucketTable<Bucket>;

    auto table = Table(reinterpret_cast<uint8_t *>(table_storage.data_ptr()),
                       num_buckets, bucket_capacity, num_scores,
                       global_caps_ptr);

    int64_t ovf_bucket_bytes = ovf_bucket_capacity * total_size;
    int64_t ovf_num_buckets =
        ovf_storage.numel() * ovf_storage.element_size() / ovf_bucket_bytes;

    auto ovf_table = Table(reinterpret_cast<uint8_t *>(ovf_storage.data_ptr()),
                           ovf_num_buckets, ovf_bucket_capacity, num_scores);

    // Segmented cache prefetch scans a full bucket with one warp per key.
    constexpr int probe_group = CachePrefetch ? 32 : 1;
    constexpr int keys_per_block = BLOCK_SIZE / probe_group;
    DISPATCH_SCORE_POLICY(policy_type, PolicyTypeV, [&] {
      table_lookup_kernel<Table, probe_group, PolicyTypeV, true, CachePrefetch>
          <<<(num_total + keys_per_block - 1) / keys_per_block,
             BLOCK_SIZE, 0, stream>>>(
              table, table_bucket_offsets_ptr, num_total, keys_ptr,
              table_ids_ptr, founds_ptr, indices_ptr, score_input_ptr,
              score_output_ptr, ovf_table, ovf_output_offsets_ptr,
              num_valid_ptr, cache);
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
table_lookup(at::Tensor table_storage, at::Tensor table_bucket_offsets,
             int64_t bucket_capacity, at::Tensor keys, at::Tensor table_ids,
             std::optional<at::Tensor> score_input, ScorePolicyType policy_type,
             std::optional<at::Tensor> ovf_storage, int64_t ovf_bucket_capacity,
             std::optional<at::Tensor> ovf_output_offsets, int64_t num_scores,
             std::optional<at::Tensor> table_global_capacities,
             const std::optional<at::Tensor> &num_valid,
             std::optional<int64_t> host_num_valid) {
  const auto *num_valid_ptr = counted_num_valid_ptr(num_valid, keys.device());

  int64_t num_total = keys.size(0);
  const int64_t host_n = counted_host_limit(host_num_valid, num_total);

  at::Tensor score_output =
      torch::empty({num_total}, keys.options().dtype(torch::kInt64));
  at::Tensor founds =
      torch::empty({num_total}, keys.options().dtype(torch::kBool));
  at::Tensor indices =
      torch::empty({num_total}, keys.options().dtype(torch::kInt64));

  if (host_n == 0)
    return std::make_tuple(score_output, founds, indices);

  bool use_overflow = ovf_storage.has_value() && ovf_storage.value().defined();
  auto global_caps_ptr =
      global_capacities_ptr(table_global_capacities, table_bucket_offsets);

  if (use_overflow) {
    table_lookup_with_overflow_single_score(
        table_storage, table_bucket_offsets, bucket_capacity, keys, table_ids,
        score_input, policy_type, score_output, founds, indices,
        ovf_storage.value(), ovf_bucket_capacity, ovf_output_offsets.value(),
        num_scores, global_caps_ptr, host_n, num_valid_ptr);
  } else {
    table_lookup_single_score(table_storage, table_bucket_offsets,
                              bucket_capacity, keys, table_ids, score_input,
                              policy_type, score_output, founds, indices,
                              num_scores, global_caps_ptr, host_n, num_valid_ptr);
  }

  return std::make_tuple(score_output, founds, indices);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
lookup_in_cache(at::Tensor keys, at::Tensor key_table_offsets,
                at::Tensor table_storage, at::Tensor table_bucket_offsets,
                int64_t bucket_capacity, at::Tensor counter,
                at::Tensor ovf_storage, int64_t ovf_bucket_capacity,
                at::Tensor ovf_output_offsets,
                std::optional<at::Tensor> score_input,
                ScorePolicyType policy_type, int64_t num_scores,
                std::optional<at::Tensor> table_global_capacities,
                const std::optional<at::Tensor> &num_valid,
                std::optional<int64_t> host_num_valid) {
  TORCH_CHECK(keys.is_cuda(), "lookup_in_cache expects CUDA keys");
  const c10::cuda::CUDAGuard guard(keys.device());
  auto check_vector = [&](const at::Tensor &t, const char *name,
                          at::ScalarType dtype) {
    TORCH_CHECK(t.device() == keys.device() && t.dim() == 1 &&
                    t.is_contiguous() && t.scalar_type() == dtype,
                name, " must be a contiguous ", dtype,
                " vector on the device of the keys");
  };
  auto check_word_vector = [&](const at::Tensor &t, const char *name) {
    const auto dtype = t.scalar_type();
    TORCH_CHECK(dtype == at::kLong || dtype == at::kUInt64,
                name, " must contain signed or unsigned 64-bit integers");
    check_vector(t, name, dtype);
  };

  check_word_vector(keys, "keys");
  check_vector(key_table_offsets, "key_table_offsets", at::kLong);
  check_vector(table_bucket_offsets, "table_bucket_offsets", at::kLong);
  check_vector(ovf_output_offsets, "ovf_output_offsets", at::kLong);
  check_vector(table_storage, "table_storage", at::kByte);
  check_vector(ovf_storage, "ovf_storage", at::kByte);
  check_vector(counter, "counter", at::kInt);
  const int64_t tables = key_table_offsets.numel() - 1;
  TORCH_CHECK(tables > 0 && tables < std::numeric_limits<int>::max(),
              "key_table_offsets must describe at least one table");
  TORCH_CHECK(table_bucket_offsets.numel() == tables + 1 &&
                  ovf_output_offsets.numel() == tables,
              "cache offsets must describe the same tables as key_table_offsets");

  TORCH_CHECK(num_scores == 1 || num_scores == 2,
              "cache slots must have one or two score words");
  TORCH_CHECK(policy_type != ScorePolicyType::LruLfu || num_scores == 2,
              "LruLfu requires two score words per cache slot");
  const int64_t slot_bytes =
      keys.element_size() + sizeof(DigestType) + num_scores * sizeof(ScoreType);
  auto count_slots = [&](const at::Tensor &storage, int64_t capacity) {
    const int64_t slots = storage.numel() / slot_bytes;
    TORCH_CHECK(capacity > 0 && storage.numel() % slot_bytes == 0 &&
                    slots % capacity == 0,
                "cache storage must contain whole buckets");
    return slots;
  };
  const int64_t main_slots = count_slots(table_storage, bucket_capacity);
  const int64_t ovf_slots = count_slots(ovf_storage, ovf_bucket_capacity);
  TORCH_CHECK(ovf_slots / ovf_bucket_capacity == tables,
              "cache must have one overflow bucket per table");
  TORCH_CHECK(counter.numel() == main_slots + ovf_slots,
              "cache must have one reference counter per main and overflow slot");

  if (score_input) {
    check_word_vector(*score_input, "score_input");
    TORCH_CHECK(score_input->numel() == keys.numel(),
                "expected one score per input key");
  } else {
    TORCH_CHECK(policy_type != ScorePolicyType::Assign &&
                    policy_type != ScorePolicyType::Accumulate &&
                    policy_type != ScorePolicyType::LruLfu,
                "this cache score policy requires score_input");
  }
  if (table_global_capacities) {
    check_vector(*table_global_capacities, "table_global_capacities", at::kLong);
  }
  const auto *global_caps =
      global_capacities_ptr(table_global_capacities, table_bucket_offsets);
  const auto *device_n = counted_num_valid_ptr(num_valid, keys.device());
  const int64_t host_n = counted_host_limit(host_num_valid, keys.numel());
  auto miss = at::empty({keys.numel()}, keys.options().dtype(at::kBool));
  auto slots = at::empty({keys.numel()}, keys.options().dtype(at::kLong));
  auto table_ids = at::empty_like(slots);
  if (host_n != 0) {
    CacheLookupRows cache{key_table_offsets.data_ptr<int64_t>(),
                          static_cast<int>(tables), table_ids.data_ptr<int64_t>(),
                          counter.data_ptr<int32_t>()};
    table_lookup_with_overflow_single_score<true>(
        table_storage, table_bucket_offsets, bucket_capacity, keys, {},
        score_input, policy_type, {}, miss, slots, ovf_storage,
        ovf_bucket_capacity, ovf_output_offsets, num_scores, global_caps,
        host_n, device_n, cache);
  }
  return {miss, slots, table_ids};
}

} // namespace dyn_emb
