// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "cache_batch.cuh"
#include "cache_row_copy.cuh"
#include "cache_deterministic.h"
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

namespace dyn_emb {
namespace {

using at::Tensor;
using Opt = CacheOptionalTensor;
using MutationLayout =
    std::tuple<Tensor, Tensor, int64_t, Tensor, Opt, Opt, int64_t>;
using CacheCandidates = std::tuple<Tensor, Tensor, Opt, Tensor, Tensor>;
using BackingCandidates = std::tuple<Tensor, Tensor, Tensor, Tensor>;
// Keys, table IDs, optional padded values, live count and optional value rows.
using SpillBatch = std::tuple<Tensor, Tensor, Opt, Tensor, Opt>;
using ExchangeResult =
    std::tuple<int64_t, int64_t, int64_t, std::vector<SpillBatch>>;

struct CachePlacementError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Both tiers use the same hash insertion primitives. Only cache slots have
// an overflow region and pins. Backing slots can own separate value rows.
struct MutableMap {
  Tensor storage, offsets, sizes;
  int64_t bucket_capacity, num_scores, score_fn;
  Opt global_caps, pins, overflow, overflow_sizes, overflow_offsets;
  int64_t overflow_capacity = 0;

  Tensor lookup(Tensor keys, Tensor tids, Opt count = {}) const {
    return std::get<2>(table_lookup(
        storage, offsets, bucket_capacity, keys, tids, {}, ScorePolicyType::Const,
        overflow, overflow_capacity, overflow_offsets, num_scores, global_caps, count));
  }

  EvictionResult insert(Tensor keys, Tensor tids, Tensor count,
                        CacheScoreArg score_arg, bool spill, bool retain,
                        bool deterministic, Opt status = {},
                        std::optional<int64_t> host_n = {}) const {
    const auto &[scores, policy] = score_arg;
    Opt overflow_pins;
    if (overflow) {
      const auto ovf_slots = overflow_capacity * (offsets.numel() - 1);
      overflow_pins = pins->narrow(0, pins->numel() - ovf_slots, ovf_slots);
    }
    auto insert_batch = [&](Tensor k, Tensor t, Opt s, Opt n, Opt report,
                            std::optional<int64_t> host_limit) -> EvictionResult {
      if (overflow || spill) {
        return table_insert_and_evict(
            storage, offsets, bucket_capacity, sizes, k, t, s, policy, pins, {}, {},
            overflow, overflow_capacity, overflow_sizes, overflow_pins, overflow_offsets,
            num_scores, score_fn, {}, global_caps, n, report, host_limit);
      }
      if (retain) {
        auto [slots, evicted, ev_keys, ev_tids] = table_insert_collect_evicted(
            storage, offsets, bucket_capacity, sizes, k, t, s, policy, pins, {}, {},
            num_scores, score_fn, global_caps, n, host_limit);
        return {slots, evicted, ev_keys, Tensor(), Tensor(), ev_tids};
      }
      auto slots = table_insert(storage, offsets, bucket_capacity, sizes, k, t, s,
                                policy, pins, {}, {}, num_scores, score_fn,
                                global_caps, n, host_limit);
      return {slots, Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
    };
    if (!deterministic)
      return insert_batch(keys, tids, scores, count, status, host_n);
    auto adjust_wave_pins = [&](Tensor slots, Tensor table_ids, int32_t delta, Opt n) {
      if (!overflow) return;
      const auto tables = offsets.numel() - 1;
      table_update_counter_with_layout(*pins, slots, delta, offsets, bucket_capacity,
                                      pins->numel() - tables * overflow_capacity, tables,
                                      table_ids, overflow_offsets, overflow_capacity, n);
    };
    auto result = deterministic_cache_insert(
        keys, tids, scores, count, overflow.has_value() || spill || retain,
        [&](Tensor k, Tensor t, Opt s) {
          auto wave = insert_batch(k, t, s, {}, {}, {});
          // The normal placement keeps the whole batch locked. Pin each
          // completed wave so later waves cannot evict this batch's own keys.
          adjust_wave_pins(std::get<0>(wave), t, 1, {});
          return wave;
        },
        [&](Tensor k, Tensor t) { return lookup(k, t); },
        [&](Tensor k, Tensor t) {
          auto b = bucketize_keys(k, t, offsets, sizes.numel(), bucket_capacity, global_caps);
          return std::make_tuple(b[0], b[1], b[2]);
        }, status);
    adjust_wave_pins(std::get<0>(result), tids, -1, count);
    return result;
  }
};

struct ValuePointers {
  const int64_t *ptrs, *emb_dims, *value_dims;
  int64_t max_emb;
};

ValuePointers value_pointers(const CacheValueLayout &layout) {
  const auto &[ptrs, edims, vdims, max_emb, max_value, dtype] = layout;
  return {ptrs.data_ptr<int64_t>(), edims ? edims->data_ptr<int64_t>() : nullptr,
          vdims ? vdims->data_ptr<int64_t>() : nullptr, max_emb};
}

struct SlotLayout {
  const int64_t *bucket_offsets;
  int64_t bucket_capacity;
  int64_t *rows, *row_counters;
  const int64_t *row_limits;
  int32_t *pins;
  const int64_t *overflow_offsets;
  int64_t overflow_capacity, total_main_slots;
};

__device__ int64_t resolve_row(SlotLayout map, int64_t tid, int64_t slot, bool allocate) {
  if (!map.rows || slot < 0)
    return slot;
  CUDA_KERNEL_ASSERT(slot < (map.bucket_offsets[tid + 1] - map.bucket_offsets[tid]) *
                                map.bucket_capacity);
  const int64_t flat = map.bucket_offsets[tid] * map.bucket_capacity + slot;
  int64_t row = map.rows[flat];
  if (row < 0) {
    CUDA_KERNEL_ASSERT(allocate);
    if (allocate) {
      row = static_cast<int64_t>(::atomicAdd(
          reinterpret_cast<unsigned long long *>(map.row_counters + tid), 1ULL));
      CUDA_KERNEL_ASSERT(row < map.row_limits[tid]);
      map.rows[flat] = row;
    }
  }
  return row;
}

constexpr int kThreads = 128;

__device__ __forceinline__ void publish_cache_slot(
    SlotLayout map, int64_t tid, int64_t slot, int64_t output_row,
    int64_t *unique_slots, int64_t output_size) {
  const int64_t overflow = map.overflow_offsets[tid];
  const int64_t flat = slot < overflow ? map.bucket_offsets[tid] * map.bucket_capacity + slot
                                      : map.total_main_slots + tid * map.overflow_capacity + slot - overflow;
  CUDA_KERNEL_ASSERT(output_row >= 0 && output_row < output_size);
  ::atomicAdd(map.pins + flat, 1);
  unique_slots[output_row] = slot;
}

// Assign each tile's destination pointers with contiguous metadata loads, then
// let all four warps issue independent reads from the staged value matrix.
// Keeping the row list in shared memory avoids an extra launch or global list.
template <int RowVectors>
__global__ void install_rows_tiled(ValuePointers table, SlotLayout map, const int64_t *slots,
                                   const int64_t *tids, const uint4 *staged,
                                   const int64_t *outputs, int64_t *unique_slots,
                                   int64_t output_size, int64_t count) {
  constexpr int vectors_per_tile = 512;
  constexpr int tile_rows = vectors_per_tile / RowVectors;
  constexpr int iterations = vectors_per_tile / kThreads;
  __shared__ uintptr_t dst[tile_rows];
  const int t = threadIdx.x;
  const int64_t begin = int64_t(blockIdx.x) * tile_rows;
  int64_t tid = 0, slot = -1;
  if (t < tile_rows) {
    const auto i = begin + t;
    if (i < count) {
      tid = tids[i];
      slot = slots[i];
    }
    dst[t] = slot < 0 ? 0 : uintptr_t(table.ptrs[tid]) + slot * RowVectors * sizeof(uint4);
  }
  // The staged input is contiguous. Fetch it while the threads assigned
  // to metadata resolve each destination, then reuse it after the barrier.
  uint4 payload[iterations];
#pragma unroll
  for (int j = 0; j < iterations; ++j) {
    const int item = j * kThreads + t;
    if (begin + item / RowVectors < count) payload[j] = staged[begin * RowVectors + item];
  }
  __syncthreads();
  if (t < tile_rows && slot >= 0 && unique_slots)
    publish_cache_slot(map, tid, slot, outputs[begin + t], unique_slots, output_size);
#pragma unroll
  for (int j = 0; j < iterations; ++j) {
    const uintptr_t to = dst[(j * kThreads + t) / RowVectors];
    if (!to) continue;
    const auto column = (j * kThreads + t) % RowVectors;
    if (to % alignof(uint4) == 0) {
      reinterpret_cast<uint4 *>(to)[column] = payload[j];
    } else {
      // The native binding also accepts raw pointers into sliced tensors.
      memcpy(reinterpret_cast<char *>(to) + column * sizeof(uint4), &payload[j], sizeof(uint4));
    }
  }
}


// A lane group copies the table's actual row width. Its leader assigns the
// backing slot's value row or publishes and pins a new cache slot.
template <typename Word, bool Cache>
__global__ void install_rows(ValuePointers table, SlotLayout map, const int64_t *slots,
                             const int64_t *tids, const Word *staged, int64_t stride,
                             const int64_t *input_rows, int64_t staged_rows,
                             const int64_t *out_rows, int64_t *unique_slots,
                             int64_t output_size, int64_t bound, const int64_t *count,
                             int lane_shift) {
  const int lanes = 1 << lane_shift;
  const int lane = threadIdx.x & (lanes - 1);
  const int64_t n = counted_limit(count, bound);
  const int groups_per_block = kThreads >> lane_shift;
  for (int64_t i = int64_t(blockIdx.x) * groups_per_block + (threadIdx.x >> lane_shift);
       i < n; i += int64_t(gridDim.x) * groups_per_block) {
    int64_t slot = slots[i];
    if (slot < 0) continue;
    const int64_t tid = tids[i], src = input_rows ? input_rows[i] : i;
    CUDA_KERNEL_ASSERT(src >= 0 && src < staged_rows);
    // Cache slots directly address their value rows. Backing insertions may
    // allocate a row, in which case the group needs its leader's result.
    int64_t row = slot;
    if constexpr (!Cache) {
      if (lane == 0)
        row = resolve_row(map, tid, slot, true);
      const unsigned mask = (~0u >> (32 - lanes)) << ((threadIdx.x & 31) - lane);
      row = __shfl_sync(mask, static_cast<long long>(row), 0, lanes);
    }
    const int64_t width = table.value_dims ? table.value_dims[tid] : stride;
    const int64_t emb = table.emb_dims ? table.emb_dims[tid] : table.max_emb;
    auto *dst = reinterpret_cast<Word *>(table.ptrs[tid]) + row * width;
    copy_cache_row(staged + src * stride, dst, width, emb, table.max_emb, emb,
                   lane, lanes);
    if constexpr (Cache) {
      if (lane == 0 && unique_slots)
        publish_cache_slot(map, tid, slot, out_rows[i], unique_slots, output_size);
    }
  }
}

// A backing victim (or rejected incoming key) leaves the two native tiers.
template <typename Word>
__global__ void copy_victims_to_spill(ValuePointers from, SlotLayout map,
                                      const int64_t *slots, const int64_t *tids,
                                      const int64_t *input_rows, const Word *staged,
                                      Word *spill, int64_t stride, int64_t bound,
                                      const int64_t *count, int lane_shift) {
  const int lanes = 1 << lane_shift, lane = threadIdx.x & (lanes - 1);
  const int groups = kThreads >> lane_shift;
  const int64_t n = counted_limit(count, bound);
  for (int64_t i = int64_t(blockIdx.x) * groups + (threadIdx.x >> lane_shift);
       i < n; i += int64_t(gridDim.x) * groups) {
    const int64_t tid = tids[i], slot = slots[i];
    const int64_t width = from.value_dims ? from.value_dims[tid] : stride;
    const int64_t emb = from.emb_dims ? from.emb_dims[tid] : from.max_emb;
    const Word *src = slot >= 0
        ? reinterpret_cast<const Word *>(from.ptrs[tid]) + resolve_row(map, tid, slot, false) * width
        : staged + input_rows[-slot - 1] * stride;
    copy_cache_row(src, spill + i * stride, width, emb, slot < 0 ? from.max_emb : emb,
                   from.max_emb, lane, lanes);
  }
}

// The backing probe already resolved both ends of each dirty cache row.
template <typename Word>
__global__ void copy_victims(ValuePointers from, const int64_t *sources,
                             const int64_t *targets, const int64_t *layout_tags,
                             Word *spill, int64_t stride, int64_t n, int lane_shift) {
  const int lanes = 1 << lane_shift, lane = threadIdx.x & (lanes - 1);
  const int groups = kThreads >> lane_shift;
  for (int64_t i = int64_t(blockIdx.x) * groups + (threadIdx.x >> lane_shift);
       i < n; i += int64_t(gridDim.x) * groups) {
    const auto *src = reinterpret_cast<const Word *>(sources[i]);
    auto *dst = reinterpret_cast<Word *>(targets[i]);
    if (!dst && !spill) continue;
    const int64_t tag = layout_tags ? layout_tags[i] : 0;
    auto *out = dst ? dst : spill + i * stride;
    if (!tag) {
      copy_cache_region(src, out, stride, lane, lanes);
    } else {
      const int64_t width = from.value_dims[tag - 1], emb = from.emb_dims[tag - 1];
      copy_cache_row(src, out, width, emb, emb, dst ? emb : from.max_emb, lane, lanes);
    }
  }
}

__global__ void copy_victims_dense_256(const int64_t *sources, const int64_t *targets,
                                      int64_t n) {
  const int lane = threadIdx.x & 15;
  constexpr int groups = kThreads / 16;
  const int64_t first = int64_t(blockIdx.x) * groups * 4 + threadIdx.x / 16;
  CacheTransfer256 transfers[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int64_t i = first + j * groups;
    transfers[j].src = i < n ? reinterpret_cast<const uint4 *>(sources[i]) : nullptr;
    transfers[j].dst = i < n ? reinterpret_cast<uint4 *>(targets[i]) : nullptr;
  }
  copy_cache_transfers_256(transfers, lane);
}

template <typename F>
void launch_rows(const Tensor &values, int64_t bound, F &&kernel) {
  if (bound == 0) return;
  const int shift = cache_copy_lane_shift(values.size(1) * values.element_size());
  const int groups = kThreads >> shift;
  const int grid = counted_grid((bound + groups - 1) / groups, kThreads);
  switch (values.element_size()) {
    case 4: kernel(uint32_t{}, grid, shift); break;
    case 2: kernel(uint16_t{}, grid, shift); break;
    case 1: kernel(uint8_t{}, grid, shift); break;
    default: TORCH_CHECK(false, "unsupported cache value word size");
  }
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

ExchangeResult exchange_cache_storage(
    const CacheLookupMap &cache_map, const BackingLookupMap &backing_map,
    const CacheValueLayout &cache_values, const CacheValueLayout &backing_values,
    const MutationLayout &mutation_layout, const CacheScoreArg &cache_score,
    const CacheScoreArg &backing_score, const CacheCandidates &candidates,
    const BackingCandidates &new_backing, Tensor values, Tensor slot_indices,
    bool spill, bool retain, bool deterministic) {
  const auto &[cache_storage, cache_offsets, cache_cap, pins, overflow, ovf_cap,
               ovf_offsets, cache_words, cache_global] = cache_map;
  const auto &[back_storage, back_offsets, back_cap, rows, back_words, back_global] = backing_map;
  const auto &[cache_sizes, ovf_sizes, cache_fn, back_sizes, row_counts,
               row_limits, back_fn] = mutation_layout;
  const auto &[c_keys, c_tids, c_rows, output_rows, c_count] = candidates;
  const auto &[b_keys, b_tids, b_rows, b_count] = new_backing;
  TORCH_CHECK(values.is_cuda() && values.dim() == 2 && values.is_contiguous(),
              "exchange requires a contiguous CUDA matrix of staged values");
  const c10::cuda::CUDAGuard guard(values.device());
  TORCH_CHECK(values.scalar_type() == std::get<5>(cache_values) &&
                  values.scalar_type() == std::get<5>(backing_values) &&
                  values.size(1) == std::get<4>(cache_values) &&
                  values.size(1) == std::get<4>(backing_values) &&
                  std::get<3>(cache_values) == std::get<3>(backing_values),
              "cache and backing must share the staged row layout");
  const auto n = c_keys.numel();
  TORCH_CHECK(c_tids.numel() == n && output_rows.numel() == n &&
                  (!c_rows || c_rows->numel() == n) && b_tids.numel() == b_keys.numel() &&
                  b_rows.numel() == b_keys.numel() && (!rows || (row_counts && row_limits)),
              "inconsistent cache exchange row maps");
  TORCH_CHECK(!deterministic || (cache_words == 1 && back_words == 1 && (spill || !retain)),
              "deterministic exchange supports single-score tables without retained eviction keys");
  counted_num_valid_ptr(c_count, values.device());
  const auto *new_count = counted_num_valid_ptr(b_count, values.device());
  const auto *cache_tids = c_tids.data_ptr<int64_t>();
  const auto *back_tids = b_tids.data_ptr<int64_t>();
  const auto *cache_inputs = c_rows ? c_rows->data_ptr<int64_t>() : nullptr;
  const auto *back_inputs = b_rows.data_ptr<int64_t>();
  auto *unique_slots = slot_indices.data_ptr<int64_t>();
  const auto *outputs = output_rows.data_ptr<int64_t>();
  const int64_t num_tables = cache_offsets.numel() - 1;
  TORCH_CHECK(num_tables > 0 && back_offsets.numel() == num_tables + 1 &&
                  ovf_offsets.numel() == num_tables, "cache exchange table counts differ");
  // The two batches come from stable selection, and the raw metadata are
  // passed independently of a table owner. Check their host-visible shapes
  // and devices before the first operation that changes cache membership.
  const auto check = [&](const Tensor &t, at::ScalarType type, int64_t size) {
    TORCH_CHECK(t.device() == values.device() && t.dim() == 1 && t.is_contiguous() &&
                    t.scalar_type() == type && t.numel() == size,
                "incorrect cache exchange tensor device, dtype or shape");
  };
  const auto same_keys = c_keys.scalar_type();
  TORCH_CHECK(same_keys == at::kLong || same_keys == at::kUInt64,
              "cache exchange requires 64-bit keys");
  check(c_keys, same_keys, n);
  check(b_keys, same_keys, b_keys.numel());
  const auto check_scores = [&](const CacheScoreArg &score, int64_t capacity,
                                int64_t words) {
    const auto &[s, policy] = score;
    TORCH_CHECK((words == 1 || words == 2) &&
                    (policy != ScorePolicyType::LruLfu || words == 2),
                "invalid cache exchange score layout");
    if (s) {
      TORCH_CHECK(s->scalar_type() == at::kUInt64 || s->scalar_type() == at::kLong,
                  "scores must contain 64-bit words");
      check(*s, s->scalar_type(), capacity);
    } else {
      TORCH_CHECK(policy != ScorePolicyType::Assign && policy != ScorePolicyType::Accumulate &&
                      policy != ScorePolicyType::LruLfu,
                  "insertion policy requires input scores");
    }
  };
  check_scores(cache_score, n, cache_words);
  check_scores(backing_score, b_keys.numel(), back_words);
  check(c_tids, at::kLong, n);
  check(output_rows, at::kLong, n);
  if (c_rows) check(*c_rows, at::kLong, n);
  check(b_tids, at::kLong, b_keys.numel());
  check(b_rows, at::kLong, b_keys.numel());
  check(slot_indices, at::kLong, slot_indices.numel());
  check(cache_offsets, at::kLong, num_tables + 1);
  check(back_offsets, at::kLong, num_tables + 1);
  check(ovf_offsets, at::kLong, num_tables);
  check(cache_sizes, at::kInt, cache_sizes.numel());
  check(back_sizes, at::kInt, back_sizes.numel());
  check(ovf_sizes, at::kInt, num_tables);
  TORCH_CHECK(cache_cap > 0 && back_cap > 0 && ovf_cap > 0, "invalid bucket widths");
  const auto slot_bytes = [&](int64_t words) {
    return int64_t(c_keys.element_size() + sizeof(DigestType) + words * sizeof(ScoreType));
  };
  check(cache_storage, at::kByte, cache_sizes.numel() * cache_cap * slot_bytes(cache_words));
  check(overflow, at::kByte, num_tables * ovf_cap * slot_bytes(cache_words));
  check(back_storage, at::kByte, back_sizes.numel() * back_cap * slot_bytes(back_words));
  if (cache_global) check(*cache_global, at::kLong, num_tables);
  if (back_global) check(*back_global, at::kLong, num_tables);
  check(pins, at::kInt, cache_sizes.numel() * cache_cap + num_tables * ovf_cap);
  if (rows) {
    check(*rows, at::kLong, back_sizes.numel() * back_cap);
    check(*row_counts, at::kLong, num_tables);
    check(*row_limits, at::kLong, num_tables);
  }
  for (auto layout : {&cache_values, &backing_values}) {
    check(std::get<0>(*layout), at::kLong, num_tables);
    const auto &emb = std::get<1>(*layout), &widths = std::get<2>(*layout);
    TORCH_CHECK(emb.has_value() == widths.has_value(), "expected both per-table width tensors");
    if (emb) {
      check(*emb, at::kLong, num_tables);
      check(*widths, at::kLong, num_tables);
    }
  }
  MutableMap cache{cache_storage, cache_offsets, cache_sizes, cache_cap,
                   cache_words, cache_fn, cache_global, pins, overflow, ovf_sizes,
                   ovf_offsets, ovf_cap};
  MutableMap back{back_storage, back_offsets, back_sizes, back_cap,
                  back_words, back_fn, back_global, {}, {}, {}, {}};
  const ValuePointers cv = value_pointers(cache_values), bv = value_pointers(backing_values);
  const SlotLayout cm{cache_offsets.data_ptr<int64_t>(), cache_cap, nullptr, nullptr, nullptr,
                      pins.data_ptr<int32_t>(), ovf_offsets.data_ptr<int64_t>(), ovf_cap,
                      pins.numel() - num_tables * ovf_cap};
  const SlotLayout bm{back_offsets.data_ptr<int64_t>(), back_cap,
                      rows ? rows->data_ptr<int64_t>() : nullptr,
                      row_counts ? row_counts->data_ptr<int64_t>() : nullptr,
                      row_limits ? row_limits->data_ptr<int64_t>() : nullptr,
                      nullptr, nullptr, 0, 0};

  std::vector<SpillBatch> spills;
  spills.reserve(2);
  auto status = at::empty({4}, c_tids.options());
  auto host = at::empty({4}, c_tids.options().device(at::kCPU).pinned_memory(true));
  auto stream = at::cuda::getCurrentCUDAStream(values.device().index());
  // Everything a backing hit supplied was staged before admission. Reserving
  // cache slots first makes its placement report the only host decision here.
  try {
    auto ranked = !deterministic && std::get<1>(cache_score) == ScorePolicyType::GlobalTimer
                      ? try_ranked_cache_reservation(cache_map, cache_sizes, ovf_sizes,
                                                     c_keys, c_tids, c_count, b_count, status)
                      : std::nullopt;
    // Ranking publishes the admission count in the same D2H transfer as cache
    // placement. Backing insert, unlock and installation then see the exact
    // host bound, including zero when every cache miss already has a backing row.
    auto report = ranked ? status : status.narrow(0, 0, 3);
    auto [slots, ev_count, ev_keys, ev_slots, ignored, ev_tids] =
        ranked ? std::move(*ranked)
               : cache.insert(c_keys, c_tids, c_count, cache_score, false, false, deterministic, report);
    host.narrow(0, 0, report.numel()).copy_(report, true);
    stream.synchronize();
    const auto *counts = host.data_ptr<int64_t>();
    const int64_t attempted = counts[0], evicted = counts[1], refused = counts[2];
    TORCH_CHECK(attempted >= 0 && attempted <= n && refused >= 0 &&
                    refused <= evicted && evicted <= attempted, "invalid cache placement report");
    if (attempted == 0) return {0, 0, 0, {}};

    const auto backing_limit = std::min(attempted, b_keys.numel());
    const int64_t backing_bound =
        ranked ? std::clamp<int64_t>(counts[3], 0, backing_limit) : backing_limit;
    auto [back_slots, back_count, back_keys, victim_slots, back_scores, victim_tids] =
        back.insert(b_keys, b_tids, b_count, backing_score, spill, retain,
                    deterministic, {}, backing_bound);
    if (spill) {
      auto displaced = at::empty({b_keys.numel(), values.size(1)}, values.options());
      launch_rows(values, backing_bound, [&](auto word, int grid, int shift) {
        using W = decltype(word);
        copy_victims_to_spill<W><<<grid, kThreads, 0, stream>>>(
            bv, bm, victim_slots.data_ptr<int64_t>(), victim_tids.data_ptr<int64_t>(),
            back_inputs, static_cast<const W *>(values.data_ptr()),
            static_cast<W *>(displaced.data_ptr()), values.stride(0), backing_bound,
            back_count.data_ptr<int64_t>(), shift);
      });
      spills.emplace_back(back_keys, victim_tids, displaced, back_count, Opt{});
    } else if (retain) {
      spills.emplace_back(back_keys, victim_tids, Opt{}, back_count, Opt{});
    }

    // All displaced backing rows have been copied out. Assign a row to each
    // admitted key that needs one and install its pre-staged value.
    launch_rows(values, backing_bound, [&](auto word, int grid, int shift) {
      using W = decltype(word);
      install_rows<W, false><<<grid, kThreads, 0, stream>>>(
          bv, bm, back_slots.data_ptr<int64_t>(), back_tids,
          static_cast<const W *>(values.data_ptr()), values.stride(0), back_inputs,
          values.size(0), nullptr, nullptr, 0, backing_bound, new_count, shift);
    });

    const int64_t victims = evicted - refused;
    if (victims > 0) {
      ev_keys = ev_keys.narrow(0, 0, evicted);
      ev_slots = ev_slots.narrow(0, 0, evicted);
      ev_tids = ev_tids.narrow(0, 0, evicted);
      if (refused) {
        // Negative entries report rejected inputs. They have no dirty row.
        auto [size, positions, fields] = flagged_compact(
            ev_slots.ge(0), {ev_keys, ev_slots, ev_tids}, ev_count, false, evicted);
        ev_keys = fields[0]->narrow(0, 0, victims);
        ev_slots = fields[1]->narrow(0, 0, victims);
        ev_tids = fields[2]->narrow(0, 0, victims);
      }
      // Resolve each victim's physical destination after backing insertions
      // have had a chance to evict it.
      auto addresses = lookup_backing_values(
          backing_map, backing_values, ev_keys, ev_tids, {std::nullopt, ScorePolicyType::Const},
          {}, {}, std::get<0>(cache_values), ev_slots);
      Opt gone_values;
      if (spill) gone_values = at::empty({victims, values.size(1)}, values.options());
      if (!spill && !addresses.layout_tags && values.size(1) * values.element_size() == 256) {
        constexpr int rows_per_block = kThreads / 16 * 4;
        copy_victims_dense_256<<<(victims + rows_per_block - 1) / rows_per_block,
                                 kThreads, 0, stream>>>(
            addresses.sources->data_ptr<int64_t>(), addresses.backing.data_ptr<int64_t>(), victims);
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        launch_rows(values, victims, [&](auto word, int grid, int shift) {
          using W = decltype(word);
          copy_victims<W><<<grid, kThreads, 0, stream>>>(
              cv, addresses.sources->data_ptr<int64_t>(), addresses.backing.data_ptr<int64_t>(),
              addresses.layout_tags ? addresses.layout_tags->data_ptr<int64_t>() : nullptr,
              gone_values ? static_cast<W *>(gone_values->data_ptr()) : nullptr,
              values.stride(0), victims, shift);
        });
      }
      if (spill || retain) {
        auto [gone_count, gone_rows, fields] = flagged_compact(
            addresses.backing.eq(0), {ev_keys, ev_tids}, {}, false, {});
        spills.emplace_back(*fields[0], *fields[1], gone_values, gone_count, gone_rows);
      }
    }
    // Dirty cache slots have reached backing or spill buffers. On partial
    // refusal the accepted rows still receive their values, but no new pin
    // belongs to this failed batch and no slot is published for backward.
    const auto row_bytes = values.stride(0) * values.element_size();
    // A null width vector guarantees contiguous uniformly sized rows. Keep
    // the gathered sources and ragged table layouts on the general copier.
    if (!cv.value_dims && !cache_inputs && attempted >= 4096 && attempted <= values.size(0) &&
        reinterpret_cast<uintptr_t>(values.data_ptr()) % alignof(uint4) == 0 &&
        (row_bytes == 256 || row_bytes == 1024 || row_bytes == 2048)) {
      const auto launch_tile = [&](auto row_vectors) {
        constexpr int width = decltype(row_vectors)::value;
        constexpr int rows_per_tile = 512 / width;
        const auto grid = (attempted + rows_per_tile - 1) / rows_per_tile;
        install_rows_tiled<width><<<grid, kThreads, 0, stream>>>(
            cv, cm, slots.data_ptr<int64_t>(), cache_tids,
            static_cast<const uint4 *>(values.data_ptr()), outputs,
            refused ? nullptr : unique_slots, slot_indices.numel(), attempted);
      };
      if (row_bytes == 256)
        launch_tile(std::integral_constant<int, 16>{});
      else if (row_bytes == 1024)
        launch_tile(std::integral_constant<int, 64>{});
      else
        launch_tile(std::integral_constant<int, 128>{});
      DEMB_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      launch_rows(values, attempted, [&](auto word, int grid, int shift) {
        using W = decltype(word);
        install_rows<W, true><<<grid, kThreads, 0, stream>>>(
            cv, cm, slots.data_ptr<int64_t>(), cache_tids,
            static_cast<const W *>(values.data_ptr()), values.stride(0), cache_inputs,
            values.size(0), outputs, refused ? nullptr : unique_slots,
            slot_indices.numel(), attempted, nullptr, shift);
      });
    }
    return {attempted, evicted, refused, std::move(spills)};
  } catch (const std::exception &e) {
    throw CachePlacementError(e.what());
  }
}
} // namespace
} // namespace dyn_emb

void bind_cache_exchange(pybind11::module &m) {
  using namespace dyn_emb;
  namespace py = pybind11;
  py::register_exception<CachePlacementError>(m, "CachePlacementError", PyExc_RuntimeError);
  m.def("exchange_cache_storage", &exchange_cache_storage,
        "Install admitted rows and return cache placement counts and optional storage spills.",
        py::arg("cache_map"), py::arg("backing_map"), py::arg("cache_values"),
        py::arg("backing_values"), py::arg("mutation_layout"), py::arg("cache_score"),
        py::arg("backing_score"), py::arg("cache_candidates"), py::arg("new_backing"),
        py::arg("values"), py::arg("slot_indices"), py::arg("spill"), py::arg("retain"),
        py::arg("deterministic") = false);
}
