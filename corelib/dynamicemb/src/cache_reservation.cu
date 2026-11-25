// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "cache_batch.cuh"
#include "table_operation/kernels.cuh"

namespace dyn_emb {
namespace {

// Production cache buckets have 1,024 slots. The ordinary insert path handles
// other widths and sparse batches without allocating a table-sized order.
constexpr int kThreads = 256;
constexpr int kSlotsPerThread = 4;
constexpr int kBucketSize = kThreads * kSlotsPerThread;

struct RankedCacheLru {
  unsigned long long *tickets;
  int *slots;
  ScoreType *scores;

  template <int GroupSize, int BufferDim, typename Bucket>
  __device__ bool operator()(Bucket &bucket, typename Bucket::Iterator &iter,
                             typename Bucket::KeyType &key, ScoreType &score,
                             ScoreType *scratch, const int32_t *pins,
                             int64_t offset, uint64_t ts) const {
    const int64_t bid = offset / kBucketSize;
    // Each ticket is issued once while the entire batch's placements stay
    // locked. The common insertion code rechecks the pin and score after CAS.
    const auto rank = ::atomicAdd(tickets + bid, 1ULL);
    if (rank < kBucketSize && scores[offset + rank] != UINT64_MAX) {
      iter = slots[offset + rank];
      score = scores[offset + rank];
      key = reinterpret_cast<typename Bucket::AtomicKey *>(bucket.keys(iter))
                ->load(cuda::std::memory_order_relaxed);
      if ((Bucket::is_valid(key) || key == Bucket::reclaimed_key()) &&
          *bucket.reduction_score(iter) == score)
        return true;
      // A concurrent mutation changed the snapshot. Re-evaluate eligibility.
    }
    // An exhausted ranking can have newly eligible victims (for example, a
    // concurrent backward released a pin). Recheck before trying overflow.
    score = UINT64_MAX;
    return DefaultEvictor{}.template operator()<GroupSize, BufferDim>(
        bucket, iter, key, score, scratch, pins, offset, ts);
  }
};

template <typename Table>
__global__ void rank_cache_victims(Table table, const int32_t *pins,
                                   int64_t bound, const int64_t *count,
                                   RankedCacheLru ranked, const int64_t *backing_count, int64_t *status) {
  const int64_t bid = blockIdx.x;
  if (bid == 0 && threadIdx.x < 4)
    status[threadIdx.x] = threadIdx.x == 3 ? *backing_count : 0;
  const int64_t n = counted_limit(count, bound);
  if (!n) return;
  if (threadIdx.x == 0) ranked.tickets[bid] = 0;
  using Bucket = typename Table::BucketType;
  using Sort = cub::BlockRadixSort<ScoreType, kThreads, kSlotsPerThread, int>;
  __shared__ typename Sort::TempStorage temp;
  ScoreType scores[kSlotsPerThread];
  int slots[kSlotsPerThread];
  auto bucket = table[bid];
  const int64_t base = bid * kBucketSize;
#pragma unroll
  for (int j = 0; j < kSlotsPerThread; ++j) {
    int slot = threadIdx.x * kSlotsPerThread + j;
    auto key = reinterpret_cast<typename Bucket::AtomicKey *>(bucket.keys(slot))
                   ->load(cuda::std::memory_order_relaxed);
    const bool pinned = pins[base + slot] > 0;
    const bool available =
        !pinned && (Bucket::is_valid(key) || key == Bucket::reclaimed_key());
    slots[j] = slot;
    scores[j] = available ? *bucket.reduction_score(slot) : UINT64_MAX;
  }
  // Equal timestamps preserve the scalar reducer's physical slot tie-break.
  Sort(temp).SortBlockedToStriped(scores, slots);
#pragma unroll
  for (int j = 0; j < kSlotsPerThread; ++j) {
    const int64_t at = base + j * kThreads + threadIdx.x;
    ranked.slots[at] = slots[j];
    ranked.scores[at] = scores[j];
  }
}

} // namespace

std::optional<EvictionResult> try_ranked_cache_reservation(
    const CacheLookupMap &map, at::Tensor sizes, at::Tensor overflow_sizes,
    at::Tensor keys, at::Tensor tids, at::Tensor count, at::Tensor backing_count,
    at::Tensor status) {
  const auto &[storage, offsets, capacity, pins, overflow, overflow_capacity,
               overflow_offsets, words, global_caps] = map;
  const int64_t n = keys.numel(), buckets = sizes.numel();
  // Ranking work and memory are proportional to the cache, so use it only
  // when the input can touch a substantial fraction of the slots.
  if (capacity != kBucketSize || words != 1 || n < 4 * kBucketSize ||
      buckets <= 0 || buckets > (2 * n) / kBucketSize)
    return std::nullopt;
  const auto index_options = tids.options();
  auto tickets = at::empty({buckets}, index_options);
  auto slots = at::empty({buckets * capacity}, sizes.options());
  auto ranked_scores = at::empty({buckets * capacity}, index_options);
  RankedCacheLru ranked{
      reinterpret_cast<unsigned long long *>(tickets.data_ptr()),
      slots.data_ptr<int>(),
      reinterpret_cast<ScoreType *>(ranked_scores.data_ptr())};
  auto indices = at::empty({n}, index_options);
  auto evicted_keys = at::empty_like(keys), evicted_slots = at::empty_like(indices);
  auto evicted_scores = at::empty_like(indices), evicted_tids = at::empty_like(tids);
  auto locks = at::empty_like(indices);
  auto evicted = status.narrow(0, 1, 1);
  const auto *live = count.data_ptr<int64_t>();
  const int64_t tables = offsets.numel() - 1;
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  DISPATCH_KEY_TYPE(get_data_type(keys), Key, [&] {
    using Table = LinearBucketTable<LinearBucket<Key>>;
    constexpr int lanes_per_probe = 32;
    using Traits = InsertKernelTraits<kThreads, lanes_per_probe, 1, 1, 8,
                                      ScorePolicyType::GlobalTimer, false, true, true>;
    Table table(storage.data_ptr<uint8_t>(), buckets, capacity, words,
                global_caps ? global_caps->data_ptr<int64_t>() : nullptr);
    Table ovf(overflow.data_ptr<uint8_t>(), tables, overflow_capacity, words);
    auto *pins_ptr = pins.data_ptr<int32_t>();
    auto **key_slots = reinterpret_cast<Key **>(locks.data_ptr());
    auto *status_ptr = status.data_ptr<int64_t>();
    rank_cache_victims<<<buckets, kThreads, 0, stream>>>(
        table, pins_ptr, n, live, ranked, backing_count.data_ptr<int64_t>(), status_ptr);
    constexpr int keys_per_block = kThreads / lanes_per_probe;
    const auto grid = counted_grid((n + keys_per_block - 1) / keys_per_block, kThreads);
    table_insert_and_evict_kernel<Table, Traits, RankedCacheLru>
        <<<grid, kThreads, 0, stream>>>(
            table, offsets.data_ptr<int64_t>(), sizes.data_ptr<int>(), n, keys.data_ptr<Key>(),
            tids.data_ptr<int64_t>(), nullptr, indices.data_ptr<IndexType>(), nullptr, nullptr,
            key_slots, evicted.data_ptr<CounterType>(), evicted_keys.data_ptr<Key>(),
            evicted_scores.data_ptr<int64_t>(), evicted_slots.data_ptr<IndexType>(),
            evicted_tids.data_ptr<int64_t>(), pins_ptr, ovf, overflow_sizes.data_ptr<int>(),
            pins_ptr + buckets * capacity, overflow_offsets.data_ptr<int64_t>(), live, status_ptr,
            RefusedOutput::PackedStatus, ranked);
    table_unlock_kernel<Table><<<table_grid_size(n, kThreads), kThreads, 0, stream>>>(
        table, n, keys.data_ptr<Key>(), key_slots, live);
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  return EvictionResult{indices, evicted, evicted_keys, evicted_slots,
                        evicted_scores, evicted_tids};
}
} // namespace dyn_emb
