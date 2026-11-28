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

#include "cache_batch.cuh"
#include "cache_row_copy.cuh"
#include "segment_offsets.cuh"
#include "table_operation/kernels.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <cub/device/device_radix_sort.cuh>

namespace dyn_emb {
namespace {

constexpr int kEvalLanes = 8;
using Table = LinearBucketTable<LinearBucket<uint64_t>>;
using EvalMap = std::tuple<BackingLookupMap, CacheValueLayout>;
using OverflowMap = std::tuple<at::Tensor, int64_t, at::Tensor>;

// These views refer to the existing bucket and value buffers. The cache stores
// its values at local slot IDs. A backing table can map a global slot to a row.
struct EvalTier {
  Table table;
  const int64_t *buckets, *rows, *ptrs, *emb_dims, *value_dims;
  int64_t emb_dim, value_dim;

  __device__ int64_t find(uint64_t key, int64_t hash, int64_t tid,
                          Table overflow = {}, const int64_t *ovf_offsets = nullptr) const {
    int64_t bucket = 0, begin, cap;
    table.home_bucket(buckets, tid, hash, &bucket, &begin, &cap);
    if (!cap) return -1;
    auto slot = cache_warp_find<false, false, kEvalLanes>(table[bucket], key, hash);
    if (slot >= 0) {
      const int64_t width = table.bucket_capacity();
      return rows ? rows[bucket * width + slot] : (bucket - begin) * width + slot;
    }
    if (ovf_offsets) {
      slot = cache_warp_find<true, false, kEvalLanes>(overflow[tid], key, hash);
      if (slot >= 0) return ovf_offsets[tid] + slot;
    }
    return -1;
  }

  __device__ void copy(int64_t tid, int64_t row, int64_t word_bytes,
                       uint8_t *dst, int lane) const {
    const int64_t stride = value_dims ? value_dims[tid] : value_dim;
    const int64_t bytes = (emb_dims ? emb_dims[tid] : emb_dim) * word_bytes;
    const auto *src = reinterpret_cast<const uint8_t *>(ptrs[tid]) + row * stride * word_bytes;
    copy_cache_region(src, dst, bytes, lane, kEvalLanes);
  }

  __device__ const uint8_t *address(int64_t tid, int64_t row, int64_t word_bytes) const {
    const int64_t stride = value_dims ? value_dims[tid] : value_dim;
    return reinterpret_cast<const uint8_t *>(ptrs[tid]) + row * stride * word_bytes;
  }

  __device__ int64_t width(int64_t tid, int64_t word_bytes) const {
    return (emb_dims ? emb_dims[tid] : emb_dim) * word_bytes;
  }
};

__global__ void lookup_cached_embedding_rows_kernel(
    EvalTier primary, Table overflow, const int64_t *ovf_offsets, EvalTier backing,
    const uint64_t *keys, const int64_t *tids, const int64_t *num_valid,
    int64_t capacity, int64_t word_bytes, uint8_t *output, bool *missing, bool *hits) {
  const int lane = threadIdx.x % kEvalLanes;
  const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kEvalLanes;
  if (i >= counted_limit(num_valid, capacity)) return;
  const auto key = keys[i];
  const auto tid = tids[i];
  const auto hash = Table::hash(key);
  const bool valid = Table::BucketType::is_valid(key);
  int64_t row = valid ? primary.find(key, hash, tid, overflow, ovf_offsets) : -1;
  const bool hit = row >= 0;
  if (hits && !lane) hits[i] = hit;
  if (!hit && valid) row = backing.find(key, hash, tid);
  if (!lane) missing[i] = row < 0;
  if (row < 0) return;
  auto *dst = output + i * primary.emb_dim * word_bytes;
  // Keep each tier's layout in registers; taking the address of a selected
  // kernel-argument struct can spill it to thread-local memory.
  if (hit) primary.copy(tid, row, word_bytes, dst, lane);
  else backing.copy(tid, row, word_bytes, dst, lane);
}

// Encode the four 4-slot words of a digest vector. For each word, bits 0..3
// identify candidate keys and bits 4..7 identify possible empty slots. This
// matches Bucket::probe's order even when the key has the empty-slot digest.
__device__ __forceinline__ uint32_t pack_probe_candidates(
    uint4 digests, uint32_t target, uint32_t empty) {
  using Comparator = Table::BucketType::VectorComparator;
  const uint32_t words[] = {digests.x, digests.y, digests.z, digests.w};
  uint32_t packed = 0;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const uint32_t matches = Comparator::compare(words[w], target);
    const uint32_t vacancies = Comparator::compare(words[w], empty);
    const uint32_t flags = matches | (vacancies << 4);
    // The four byte weights 1, 2, 4, 8 gather the low and high flags into
    // separate nibbles in the high byte of the product.
    packed |= ((flags * 0x01020408u) >> 24) << (w * 8);
  }
  return packed;
}

__device__ __forceinline__ int read_only_probe_128(
    Table::BucketType main, uint64_t key, uint64_t hash,
    const int64_t *bucket_rows) {
  using Bucket = Table::BucketType;
  const uint32_t target = Bucket::digest_to_vector(Bucket::hashcode_to_digest(hash));
  const uint32_t empty = Bucket::digest_to_vector(Bucket::empty_digest());
  const auto *digest = reinterpret_cast<const uint4 *>(main.digests(0));
  const int start = (hash & 127) / 16;
#pragma unroll 1
  for (int group = 0; group < 8; group += 2) {
    const int index = (start + group) & 7;
    const uint4 first = digest[index], second = digest[(index + 1) & 7];
    uint64_t candidates = (uint64_t{pack_probe_candidates(second, target, empty)} << 32) |
                          pack_probe_candidates(first, target, empty);
    // One iteration per possible match or empty slot across 32 digests.
    while (candidates) {
      const int bit = __ffsll(candidates) - 1;
      const int slot = (index * 16 + (bit >> 3) * 4 + (bit & 3)) & 127;
      const bool check_empty = bit & 4;
      auto *candidate = reinterpret_cast<Bucket::AtomicKey *>(main.keys(slot));
      if (bucket_rows && !check_empty)
        asm volatile("prefetch.global.L2 [%0];" : : "l"(bucket_rows + slot));
      const auto stored_key = candidate->load(cuda::std::memory_order_relaxed);
      if (!check_empty && stored_key == key) return slot;
      if (check_empty && stored_key == Bucket::empty_key()) return -1;
      candidates &= candidates - 1;
    }
  }
  return -1;
}

// Most warps lie inside one table. For up to 32 tables, each lane reads
// one table end; the ballot skips all empty segments at the start of a warp.
// Only requests crossing the next boundary need a per-request upper bound.
__device__ __forceinline__ int eval_range_table(const int64_t *ranges, int tables,
                                                int64_t i, int64_t base, int lane, int64_t stride) {
  if (tables == 1) return 0;
  if (tables <= 32) {
    const auto end = lane < tables ? ranges[(lane + 1) * stride] : INT64_MAX;
    const int tid = __popc(__ballot_sync(0xffffffffu, end <= base));
    const auto table_end = __shfl_sync(0xffffffffu, static_cast<long long>(end), tid);
    if (i < table_end) return tid;
  }
  return binary_search_upper_bound(ranges, tables + 1, i, stride);
}

// On a single HBM tier the hash probe is short. Use a different request per
// lane, then shuffle the 32 addresses into groups for coalesced output writes.
template <bool Segmented>
__global__ void lookup_hbm_embedding_rows_kernel(
    EvalTier tier, const uint64_t *keys, const int64_t *tids,
    const int64_t *count, int64_t capacity, int64_t word_bytes,
    uint8_t *output, bool *missing, const uint8_t *defaults, int64_t default_stride, int tables, int64_t range_stride) {
  const int lane = threadIdx.x & 31;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t base = i - lane;
  auto live = counted_limit(count, capacity);
  if constexpr (Segmented) live = min(live, tids[tables * range_stride]);
  if (base >= live) return;

  int64_t table = 0;
  if constexpr (Segmented) table = eval_range_table(tids, tables, i, base, lane, range_stride);
  const uint8_t *src = nullptr;
  int64_t bytes = 0;
  if (i < live) {
    const auto key = keys[i];
    const auto tid = Segmented ? table : tids[i];
    if (Table::BucketType::is_valid(key)) {
      const int64_t hash = Table::hash(key);
      int64_t bucket = 0, begin, cap;
      tier.table.home_bucket(tier.buckets, tid, hash, &bucket, &begin, &cap);
      if (cap) {
        auto main = tier.table[bucket];
        const auto width = tier.table.bucket_capacity();
        int64_t slot = hash % width, steps = 0;
        bool found;
        if (width == 128) {
          slot = read_only_probe_128(main, key, hash, tier.rows ? tier.rows + bucket * 128 : nullptr);
          found = slot >= 0;
        } else {
          found = main.probe<1>(key, slot, steps) == Table::BucketType::ProbeResult::Existed;
        }
        if (found) {
          const auto row = tier.rows ? tier.rows[bucket * width + slot] : (bucket - begin) * width + slot;
          if (row >= 0) {
            src = tier.address(tid, row, word_bytes);
            bytes = tier.width(tid, word_bytes);
          }
        }
      }
    }
    if (missing) missing[i] = !src;
    if (!src && defaults) {
      src = defaults + tid * default_stride;
      bytes = tier.emb_dim * word_bytes;
    }
  }
  for (int j = lane / kEvalLanes; j < 32; j += 32 / kEvalLanes) {
    const auto addr = __shfl_sync(0xffffffff, reinterpret_cast<unsigned long long>(src), j);
    const auto width = __shfl_sync(0xffffffff, static_cast<long long>(bytes), j);
    if (addr) {
      auto *dst = output + (base + j) * tier.emb_dim * word_bytes;
      copy_cache_region(reinterpret_cast<const uint8_t *>(addr), dst, width,
                        lane % kEvalLanes, kEvalLanes);
    }
  }
}

// Look at one flag per lane and cooperate only on rows missing from both tiers.
// The common all-hit case reads a contiguous byte mask and performs no stores.
__global__ void fill_missing_eval_rows(
    const bool *missing, const int64_t *tids, const int64_t *count, int64_t n,
    const uint8_t *defaults, int64_t default_stride, uint8_t *out, int64_t row_bytes) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & 31;
  const int64_t live = counted_limit(count, n);
  unsigned todo = __ballot_sync(0xffffffff, i < live && missing[i]);
  while (todo) {
    const int64_t row = i - lane + __ffs(todo) - 1;
    const auto *src = defaults + (default_stride ? tids[row] * default_stride : 0);
    copy_cache_region(src, out + row * row_bytes, row_bytes, lane, 32);
    todo &= todo - 1;
  }
}

void check_vector(const at::Tensor &t, at::Device device, at::ScalarType dtype,
                  const char *name) {
  TORCH_CHECK(t.device() == device && t.scalar_type() == dtype && t.dim() == 1 &&
              t.is_contiguous(), "invalid ", name, " tensor for lookup_embedding_rows");
}

Table eval_table(const at::Tensor &storage, int64_t capacity, int64_t words,
                 const int64_t *global, at::Device device) {
  check_vector(storage, device, at::kByte, "hash storage");
  TORCH_CHECK(capacity > 0 && (words == 1 || words == 2), "invalid bucket layout");
  const auto slot_bytes = sizeof(uint64_t) + sizeof(DigestType) + words * sizeof(ScoreType);
  const auto bucket_bytes = capacity * slot_bytes;
  TORCH_CHECK(storage.numel() % bucket_bytes == 0, "incomplete bucket");
  return {storage.data_ptr<uint8_t>(), storage.numel() / bucket_bytes, capacity, words, global};
}

EvalTier eval_tier(const EvalMap &spec, at::Device device, int64_t tables) {
  const auto &[map, layout] = spec;
  const auto &[storage, buckets, capacity, rows, words, global] = map;
  const auto &[ptrs, edims, widths, edim, width, dtype] = layout;
  check_vector(buckets, device, at::kLong, "bucket offsets");
  check_vector(ptrs, device, at::kLong, "value pointers");
  TORCH_CHECK(tables > 0 && buckets.numel() == tables + 1 && ptrs.numel() == tables &&
              edim > 0 && width >= edim && capacity % Table::BucketType::VectorDim == 0,
              "invalid row or bucket dimensions");
  TORCH_CHECK(edims.has_value() == widths.has_value(), "expected both row dimensions");
  auto per_table = [&](const CacheOptionalTensor &tensor) -> const int64_t * {
    if (!tensor) return nullptr;
    check_vector(*tensor, device, at::kLong, "per-table metadata");
    TORCH_CHECK(tensor->numel() == tables, "invalid per-table metadata length");
    return tensor->data_ptr<int64_t>();
  };
  const auto table = eval_table(storage, capacity, words, per_table(global), device);
  if (rows) {
    check_vector(*rows, device, at::kLong, "slot-to-row map");
    TORCH_CHECK(rows->numel() == table.num_buckets_ * capacity, "invalid slot-to-row map length");
  }
  return {table, buckets.data_ptr<int64_t>(), rows ? rows->data_ptr<int64_t>() : nullptr,
          ptrs.data_ptr<int64_t>(), per_table(edims), per_table(widths), edim, width};
}

std::tuple<at::Tensor, CacheOptionalTensor, CacheOptionalTensor> lookup_embedding_rows(
    at::Tensor keys, const CacheOptionalTensor &tids, const EvalMap &primary_map,
    const std::optional<OverflowMap> &overflow_map, const std::optional<EvalMap> &backing_map,
    const CacheOptionalTensor &num_valid, std::optional<int64_t> host_num_valid, bool record_hits,
    const CacheOptionalTensor &defaults, const CacheOptionalTensor &table_ranges) {
  TORCH_CHECK(keys.is_cuda() && (keys.scalar_type() == at::kLong || keys.scalar_type() == at::kUInt64),
              "expected 64-bit CUDA keys");
  const auto device = keys.device();
  const c10::cuda::CUDAGuard guard(device);
  check_vector(keys, device, keys.scalar_type(), "keys");
  const int64_t tables = std::get<1>(std::get<0>(primary_map)).numel() - 1;
  TORCH_CHECK(bool(tids) != bool(table_ranges), "supply either table ids or table ranges");
  const auto &table_rows = tids ? *tids : *table_ranges;
  if (tids) {
    check_vector(table_rows, device, at::kLong, "table ids");
  } else {
    TORCH_CHECK(table_rows.device() == device && table_rows.scalar_type() == at::kLong &&
                table_rows.dim() == 1, "expected a CUDA int64 vector of table ranges");
  }
  TORCH_CHECK(table_rows.numel() == (tids ? keys.numel() : tables + 1),
              "invalid table id or range length");
  TORCH_CHECK(!table_ranges || !backing_map, "table ranges require a single storage tier");
  const auto primary = eval_tier(primary_map, device, tables);
  const auto dtype = std::get<5>(std::get<1>(primary_map));
  EvalTier backing{};
  if (backing_map) {
    backing = eval_tier(*backing_map, device, tables);
    TORCH_CHECK(primary.emb_dim == backing.emb_dim && dtype == std::get<5>(std::get<1>(*backing_map)),
                "cache and backing embeddings must have the same shape and dtype");
  }
  Table overflow;
  const int64_t *ovf_offsets = nullptr;
  if (overflow_map) {
    const auto &[storage, cap, offsets] = *overflow_map;
    overflow = eval_table(storage, cap, primary.table.num_scores_, nullptr, device);
    check_vector(offsets, device, at::kLong, "overflow offsets");
    TORCH_CHECK(overflow.num_buckets_ == tables && offsets.numel() == tables,
                "expected one overflow bucket per table");
    ovf_offsets = offsets.data_ptr<int64_t>();
  }
  const int64_t n = counted_host_limit(host_num_valid, keys.numel());
  const auto *count = counted_num_valid_ptr(num_valid, device);
  auto output = at::empty({keys.numel(), primary.emb_dim}, keys.options().dtype(dtype));
  CacheOptionalTensor missing, hits;
  const uint8_t *default_ptr = nullptr;
  int64_t default_stride = 0;
  if (defaults) {
    TORCH_CHECK(defaults->device() == device && defaults->scalar_type() == dtype &&
                defaults->is_contiguous() && defaults->dim() == 2 &&
                (defaults->size(0) == 1 || defaults->size(0) == tables) &&
                defaults->size(1) == primary.emb_dim, "invalid eval default rows");
    default_ptr = static_cast<const uint8_t *>(defaults->data_ptr());
    default_stride = defaults->size(0) == 1 ? 0 : primary.emb_dim * output.element_size();
  }
  if (backing_map || !defaults) {
    missing = at::empty_like(keys, keys.options().dtype(at::kBool));
  }
  auto *misses = missing ? missing->data_ptr<bool>() : nullptr;
  TORCH_CHECK(!record_hits || backing_map, "hit metrics require a cache");
  if (record_hits) hits = at::empty_like(keys, keys.options().dtype(at::kBool));
  if (n) {
    constexpr int threads = 256;
    const int blocks = (n + threads / kEvalLanes - 1) / (threads / kEvalLanes);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto *input = reinterpret_cast<const uint64_t *>(keys.data_ptr());
    const auto *table_ids = table_rows.data_ptr<int64_t>();
    auto *out = static_cast<uint8_t *>(output.data_ptr());
    if (backing_map) {
      lookup_cached_embedding_rows_kernel<<<blocks, threads, 0, stream>>>(
          primary, overflow, ovf_offsets, backing, input, table_ids, count, n,
          output.element_size(), out, misses, hits ? hits->data_ptr<bool>() : nullptr);
      if (defaults) {
        DEMB_CUDA_KERNEL_LAUNCH_CHECK();
        fill_missing_eval_rows<<<(n + threads - 1) / threads, threads, 0, stream>>>(
            misses, table_ids, count, n, default_ptr, default_stride, out,
            primary.emb_dim * output.element_size());
      }
    } else {
      TORCH_CHECK(!overflow_map, "direct HBM lookup expects a single map");
      TORCH_CHECK(primary.table.bucket_capacity_ % Table::BucketType::BufferDim == 0,
                  "direct HBM lookup requires a bucket with complete 16-byte digest vectors");
      const auto launch = [&](auto segmented) {
        constexpr bool from_ranges = decltype(segmented)::value;
        lookup_hbm_embedding_rows_kernel<from_ranges><<<(n + threads - 1) / threads, threads, 0, stream>>>(
            primary, input, table_ids, count, n, output.element_size(), out, misses,
            default_ptr, default_stride, tables, table_rows.stride(0));
      };
      if (table_ranges) launch(std::true_type{});
      else launch(std::false_type{});
    }
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output, defaults ? std::nullopt : missing, hits};
}
// Resolve each distinct (table, key) to the resident value or its encoded
// table default. The pinned/host value is only dereferenced after sorting.
__global__ void resolve_mapped_eval_sources(
    EvalTier primary, const uint64_t *keys, const int64_t *ranges, int tables, int n, int word_bytes, const uint8_t *defaults,
    int64_t default_stride, int64_t *sources) {
  const int i = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / kEvalLanes;
  if (i >= n || i >= ranges[tables]) return;
  const uint64_t key = keys[i];
  const int64_t tid = binary_search_upper_bound(ranges, tables + 1, i);
  const int64_t hash = Table::hash(key);
  const uint8_t *src = defaults + default_stride * tid;
  if (Table::BucketType::is_valid(key)) {
    auto row = primary.find(key, hash, tid);
    if (row >= 0) src = primary.address(tid, row, word_bytes);
  }
  if (!(threadIdx.x % kEvalLanes)) sources[i] = reinterpret_cast<int64_t>(src);
}

__global__ void mapped_eval_order(const int64_t *sources, const int64_t *inverse,
                                  const int64_t *unique_count, const int64_t *count,
                                  int n, int64_t *by_address, int *ordinals) {
  const int i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int64_t uid = i < counted_limit(count, n) ? inverse[i] : -1;
  by_address[i] = uid >= 0 && uid < counted_limit(unique_count, n) ? sources[uid] : 0;
  ordinals[i] = i;
}

// Each lane holds one destination in a run and one stripe of the source row.
// Broadcast destinations while retaining the source stripe in its register.
template <typename T>
__device__ void fan_out_row(const T *src, T *out, int64_t width,
                            int dest, int copies, int lane) {
  for (int64_t base = 0; base < width; base += 32) {
    const auto element = base + lane;
    T value{};
    if (element < width) value = src[element];
    for (int j = 0; j < copies; ++j) {
      const int ordinal = __shfl_sync(0xffffffff, dest, j);
      if (element < width) out[ordinal * width + element] = value;
    }
  }
}

template <bool Continuation>
__global__ void fan_out_mapped_eval_rows(
    const int64_t *addresses, const int *ordinals, int n, int64_t width,
    uint8_t *out) {
  const int lane = threadIdx.x & 31;
  const int64_t warp = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const int64_t i = Continuation ? warp * 32 : warp;
  if (i >= n) return;
  const auto address = addresses[i];
  if (!address) return;
  const bool same = i > 0 && addresses[i - 1] == address;
  if (same != Continuation) return;
  const uint8_t *src = reinterpret_cast<const uint8_t *>(address);
  if constexpr (Continuation) {
    // A hot row can span many tiles. Only the first tile fetches it across
    // PCIe. Independent continuation warps use its published final GPU row.
    int64_t left = 0, right = i - 1;
    while (left < right) {
      const int64_t mid = left + (right - left) / 2;
      if (addresses[mid] < address) left = mid + 1;
      else right = mid;
    }
    src = out + int64_t(ordinals[left]) * width;
  }
  const int64_t q = i + lane;
  const bool match = q < n && (q >> 5) == (i >> 5) && addresses[q] == address;
  const int copies = __popc(__ballot_sync(0xffffffff, match));
  const int dest = match ? ordinals[q] : 0;
  if (!(reinterpret_cast<uintptr_t>(src) % 16) && !(width % 16)) {
    fan_out_row(reinterpret_cast<const uint4 *>(src), reinterpret_cast<uint4 *>(out),
                width / 16, dest, copies, lane);
  } else {
    fan_out_row(src, out, width, dest, copies, lane);
  }
}

at::Tensor grouped_mapped_eval(at::Tensor keys, at::Tensor ranges, at::Tensor reverse,
    at::Tensor count, const EvalMap &primary_map, at::Tensor defaults) {
  const auto device = keys.device();
  TORCH_CHECK(keys.is_cuda() && (keys.scalar_type() == at::kLong || keys.scalar_type() == at::kUInt64),
              "expected 64-bit CUDA keys");
  const c10::cuda::CUDAGuard guard(device);
  check_vector(keys, device, keys.scalar_type(), "keys");
  check_vector(ranges, device, at::kLong, "unique table ranges");
  check_vector(reverse, device, at::kLong, "inverse");
  TORCH_CHECK(reverse.numel() == keys.numel() &&
              keys.numel() < std::numeric_limits<int>::max(), "expected equal input capacities");
  const int64_t tables = std::get<1>(std::get<0>(primary_map)).numel() - 1;
  const auto primary = eval_tier(primary_map, device, tables);
  const auto dtype = std::get<5>(std::get<1>(primary_map));
  TORCH_CHECK(!primary.emb_dims, "grouped eval expects uniform embedding width");
  TORCH_CHECK(ranges.numel() == tables + 1, "expected one unique range per table");
  TORCH_CHECK(defaults.device() == device && defaults.scalar_type() == dtype &&
              defaults.is_contiguous() && defaults.dim() == 2 &&
              (defaults.size(0) == 1 || defaults.size(0) == tables) &&
              defaults.size(1) == primary.emb_dim, "invalid constant default rows");
  const auto *range_ptr = ranges.data_ptr<int64_t>();
  const auto *unique_ptr = range_ptr + tables;
  const auto *count_ptr = counted_num_valid_ptr(count, device);
  const int n = keys.numel();
  auto out = at::empty({n, primary.emb_dim}, keys.options().dtype(dtype));
  if (!n) return out;
  auto sources = at::empty_like(reverse);
  auto addresses = at::empty_like(reverse);
  auto ordinals = at::empty({n}, keys.options().dtype(at::kInt));
  auto sorted = at::empty_like(addresses);
  auto positions = at::empty_like(ordinals);
  constexpr int threads = 256;
  const int64_t blocks = (int64_t(n) + threads - 1) / threads;
  const int64_t probe_blocks = (int64_t(n) * kEvalLanes + threads - 1) / threads;
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int64_t row_bytes = primary.emb_dim * out.element_size();
  resolve_mapped_eval_sources<<<probe_blocks, threads, 0, stream>>>(
      primary, static_cast<const uint64_t *>(keys.data_ptr()), range_ptr, tables,
      n, out.element_size(), static_cast<const uint8_t *>(defaults.data_ptr()),
      defaults.size(0) == 1 ? 0 : row_bytes, sources.data_ptr<int64_t>());
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  mapped_eval_order<<<blocks, threads, 0, stream>>>(sources.data_ptr<int64_t>(),
      reverse.data_ptr<int64_t>(), unique_ptr, count_ptr, n, addresses.data_ptr<int64_t>(),
      ordinals.data_ptr<int>());
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  size_t bytes = 0;
  auto sort = [&](void *storage) {
    return cub::DeviceRadixSort::SortPairs(storage, bytes, addresses.data_ptr<int64_t>(),
        sorted.data_ptr<int64_t>(), ordinals.data_ptr<int>(), positions.data_ptr<int>(),
        n, 0, 64, stream);
  };
  C10_CUDA_CHECK(sort(nullptr));
  auto tmp = at::empty({int64_t(bytes)}, keys.options().dtype(at::kByte));
  C10_CUDA_CHECK(sort(tmp.data_ptr()));
  const int64_t first_blocks = (int64_t(n) * 32 + threads - 1) / threads;
  fan_out_mapped_eval_rows<false><<<first_blocks, threads, 0, stream>>>(
      sorted.data_ptr<int64_t>(), positions.data_ptr<int>(), n, row_bytes,
      static_cast<uint8_t *>(out.data_ptr()));
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  fan_out_mapped_eval_rows<true><<<blocks, threads, 0, stream>>>(
      sorted.data_ptr<int64_t>(), positions.data_ptr<int>(), n, row_bytes,
      static_cast<uint8_t *>(out.data_ptr()));
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
} // namespace
} // namespace dyn_emb

void bind_cache_eval(py::module &m) {
  m.def("grouped_mapped_eval", &dyn_emb::grouped_mapped_eval, py::arg("unique_keys"),
        py::arg("unique_table_ranges"), py::arg("inverse"), py::arg("num_valid"),
        py::arg("primary_map"), py::arg("defaults"));
  m.def("lookup_embedding_rows", &dyn_emb::lookup_embedding_rows, py::arg("keys"),
        py::arg("table_ids"), py::arg("primary_map"), py::arg("overflow"),
        py::arg("backing_map"), py::arg("num_valid") = py::none(),
        py::arg("host_num_valid") = py::none(), py::arg("record_cache_hits") = false,
        py::arg("defaults") = py::none(), py::arg("table_ranges") = py::none());
}
