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
#include "table.cuh"

namespace dyn_emb {

// With `num_valid` only the first *num_valid entries of keys / table_ids are
// erased, and only those of `indices` (when given) are written.
void table_erase(at::Tensor table_storage, at::Tensor table_bucket_offsets,
                 int64_t bucket_capacity, at::Tensor bucket_sizes,
                 at::Tensor keys, at::Tensor table_ids,
                 std::optional<at::Tensor> indices, int64_t num_scores,
                 std::optional<at::Tensor> table_global_capacities,
                 std::optional<at::Tensor> num_valid) {

  int64_t num_total = keys.size(0);
  if (num_total == 0)
    return;

  auto key_type = get_data_type(keys);
  auto bucket_sizes_ = get_pointer<int>(bucket_sizes);
  auto indices_ = get_pointer<IndexType>(indices);
  auto table_ids_ptr = table_ids.data_ptr<int64_t>();
  auto table_bucket_offsets_ptr = table_bucket_offsets.data_ptr<int64_t>();
  auto num_valid_ptr = counted_num_valid_ptr(num_valid, keys.device());
  auto global_caps_ptr =
      global_capacities_ptr(table_global_capacities, table_bucket_offsets);

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  constexpr int BLOCK_SIZE = 256;

  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    auto keys_ = get_pointer<KeyType>(keys);

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

    table_erase_kernel<Table, 1>
        <<<table_grid_size(num_total, BLOCK_SIZE, num_valid_ptr != nullptr),
           BLOCK_SIZE, 0, stream>>>(table, table_bucket_offsets_ptr,
                                    bucket_sizes_, num_total, keys_,
                                    table_ids_ptr, indices_, num_valid_ptr);
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

// The table gate serializes reservation and rollback. At this point a miss owns
// a unique slot but its value still belongs to the previous occupant. Release
// every accepted slot first, then put the recorded victims back in place.
// Negative slots in either input are refused keys and never had a cache row.
using CacheBucket = LinearBucket<uint64_t>;
using CacheRegion = LinearBucketTable<CacheBucket>;

__global__ void rollback_cache_slots_kernel(
    CacheRegion main, CacheRegion overflow, const int64_t *bucket_offsets,
    int *main_sizes, int *overflow_sizes, int64_t n, const int64_t *slots,
    const int64_t *table_ids, const uint64_t *keys, const int64_t *scores) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    const int64_t slot = slots[i];
    const int64_t tid = table_ids[i];
    if (slot < 0 || tid < 0 || uint64_t(tid) >= overflow.num_buckets_) continue;
    const int64_t begin = bucket_offsets[tid];
    const int64_t main_capacity =
        (bucket_offsets[tid + 1] - begin) * main.bucket_capacity();
    const bool in_overflow = slot >= main_capacity;
    if (in_overflow && slot - main_capacity >= overflow.bucket_capacity()) continue;
    const int64_t bucket_id = in_overflow ? tid : begin + slot / main.bucket_capacity();
    const int64_t local = in_overflow ? slot - main_capacity : slot % main.bucket_capacity();
    auto bucket = in_overflow ? overflow[bucket_id] : main[bucket_id];
    auto *size = (in_overflow ? overflow_sizes : main_sizes) + bucket_id;

    const uint64_t key = keys ? keys[i] : CacheBucket::reclaimed_key();
    *bucket.scores(local) = scores ? static_cast<ScoreType>(scores[i]) : 0;
    *bucket.digests(local) = CacheBucket::key_to_digest(key);
    // Publish score and digest before the key (same release as table erase).
    int64_t iter = local;
    bucket.unlock(iter, key);
    atomicAdd(size, keys ? 1 : -1);
  }
}

void table_rollback_cache_placement(
    at::Tensor storage, at::Tensor bucket_offsets, int64_t bucket_capacity,
    at::Tensor bucket_sizes, at::Tensor overflow_storage,
    int64_t overflow_bucket_capacity, at::Tensor overflow_sizes,
    at::Tensor slots, at::Tensor table_ids, at::Tensor evicted_keys,
    at::Tensor evicted_slots, at::Tensor evicted_scores, at::Tensor evicted_table_ids) {
  auto vector = [&](const at::Tensor &tensor, at::ScalarType dtype, int64_t n) {
    TORCH_CHECK(tensor.device() == storage.device() && tensor.scalar_type() == dtype &&
                    tensor.dim() == 1 && tensor.is_contiguous() && tensor.numel() == n,
                "cache rollback needs contiguous vectors on the cache device");
  };
  TORCH_CHECK(storage.is_cuda() && bucket_capacity > 0 && overflow_bucket_capacity > 0,
              "cache rollback needs CUDA storage and positive bucket capacities");
  const int64_t bytes_per_slot = sizeof(uint64_t) + sizeof(DigestType) + sizeof(ScoreType);
  vector(storage, at::kByte, bucket_sizes.numel() * bucket_capacity * bytes_per_slot);
  vector(overflow_storage, at::kByte,
         overflow_sizes.numel() * overflow_bucket_capacity * bytes_per_slot);
  vector(bucket_sizes, at::kInt, bucket_sizes.numel());
  vector(overflow_sizes, at::kInt, overflow_sizes.numel());
  vector(bucket_offsets, at::kLong, overflow_sizes.numel() + 1);
  vector(slots, at::kLong, slots.numel());
  vector(table_ids, at::kLong, slots.numel());
  TORCH_CHECK(evicted_keys.scalar_type() == at::kLong || evicted_keys.scalar_type() == at::kUInt64,
              "cache victim keys must be int64 or uint64");
  vector(evicted_keys, evicted_keys.scalar_type(), evicted_slots.numel());
  vector(evicted_slots, at::kLong, evicted_slots.numel());
  vector(evicted_scores, at::kLong, evicted_slots.numel());
  vector(evicted_table_ids, at::kLong, evicted_slots.numel());

  c10::cuda::CUDAGuard guard(storage.device());
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  CacheRegion main(storage.data_ptr<uint8_t>(), bucket_sizes.numel(), bucket_capacity);
  CacheRegion overflow(overflow_storage.data_ptr<uint8_t>(), overflow_sizes.numel(),
                       overflow_bucket_capacity);
  const auto launch = [&](const at::Tensor &locations, const at::Tensor &ids,
                          const uint64_t *keys, const int64_t *scores) {
    constexpr int block = 256;
    const int64_t n = locations.numel();
    if (n == 0) return;
    rollback_cache_slots_kernel<<<counted_grid((n + block - 1) / block, block),
                                  block, 0, stream>>>(
        main, overflow, bucket_offsets.data_ptr<int64_t>(), bucket_sizes.data_ptr<int>(),
        overflow_sizes.data_ptr<int>(), n, locations.data_ptr<int64_t>(),
        ids.data_ptr<int64_t>(), keys, scores);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  };
  launch(slots, table_ids, nullptr, nullptr);
  launch(evicted_slots, evicted_table_ids, static_cast<const uint64_t *>(evicted_keys.data_ptr()),
         evicted_scores.data_ptr<int64_t>());
}

} // namespace dyn_emb