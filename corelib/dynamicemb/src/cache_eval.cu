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
#include "table_operation/kernels.cuh"
#include <c10/cuda/CUDAGuard.h>

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

// Look ahead by one 16-slot digest vector on a full-width bucket. Loading
// both aligned vectors first lets the device service independent reads while
// each lane validates its candidates. Visit slots in the original order.
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
    const uint4 pair[2] = {digest[index], digest[(index + 1) & 7]};
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const uint4 vec = pair[p];
      const uint32_t chunks[4] = {vec.x, vec.y, vec.z, vec.w};
#pragma unroll
      for (int word = 0; word < 4; ++word) {
        int key_mask = Bucket::VectorComparator::compare(chunks[word], target);
        int empty_mask = Bucket::VectorComparator::compare(chunks[word], empty);
        const int base = (((index + p) & 7) << 4) + 4 * word;
        while (key_mask) {
          int pos = base + Bucket::VectorComparator::equal_index(key_mask);
          auto *candidate = reinterpret_cast<Bucket::AtomicKey *>(main.keys(pos));
          // Fetch the row map entry while checking the atomic key. Its
          // contents are consumed only after the key matches.
          if (bucket_rows) asm volatile("prefetch.global.L2 [%0];" : : "l"(bucket_rows + pos));
          if (candidate->load(cuda::std::memory_order_relaxed) == key) return pos;
        }
        while (empty_mask) {
          int pos = base + Bucket::VectorComparator::equal_index(empty_mask);
          auto *candidate = reinterpret_cast<Bucket::AtomicKey *>(main.keys(pos));
          if (candidate->load(cuda::std::memory_order_relaxed) == Bucket::empty_key()) return -1;
        }
      }
    }
  }
  return -1;
}

// On a single HBM tier the hash probe is short. Use a different request per
// lane, then shuffle the 32 addresses into groups for coalesced output writes.
__global__ void lookup_hbm_embedding_rows_kernel(
    EvalTier tier, const uint64_t *keys, const int64_t *tids,
    const int64_t *count, int64_t capacity, int64_t word_bytes,
    uint8_t *output, bool *missing) {
  const int lane = threadIdx.x & 31;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t base = i - lane;
  const int64_t live = counted_limit(count, capacity);
  if (base >= live) return;

  const uint8_t *src = nullptr;
  int64_t bytes = 0;
  if (i < live) {
    const auto key = keys[i];
    const auto tid = tids[i];
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
    missing[i] = !src;
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

std::tuple<at::Tensor, at::Tensor, CacheOptionalTensor> lookup_embedding_rows(
    at::Tensor keys, at::Tensor tids, const EvalMap &primary_map,
    const std::optional<OverflowMap> &overflow_map, const std::optional<EvalMap> &backing_map,
    const CacheOptionalTensor &num_valid, std::optional<int64_t> host_num_valid, bool record_hits) {
  TORCH_CHECK(keys.is_cuda() && (keys.scalar_type() == at::kLong || keys.scalar_type() == at::kUInt64),
              "expected 64-bit CUDA keys");
  const auto device = keys.device();
  const c10::cuda::CUDAGuard guard(device);
  check_vector(keys, device, keys.scalar_type(), "keys");
  check_vector(tids, device, at::kLong, "table ids");
  TORCH_CHECK(keys.numel() == tids.numel(), "expected one table id per key");
  const int64_t tables = std::get<1>(std::get<0>(primary_map)).numel() - 1;
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
  auto missing = at::empty_like(keys, keys.options().dtype(at::kBool));
  CacheOptionalTensor hits;
  TORCH_CHECK(!record_hits || backing_map, "hit metrics require a cache");
  if (record_hits) hits = at::empty_like(missing);
  if (n) {
    constexpr int threads = 256;
    const int blocks = (n + threads / kEvalLanes - 1) / (threads / kEvalLanes);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto *input = reinterpret_cast<const uint64_t *>(keys.data_ptr());
    const auto *table_ids = tids.data_ptr<int64_t>();
    auto *out = static_cast<uint8_t *>(output.data_ptr());
    if (backing_map) {
      lookup_cached_embedding_rows_kernel<<<blocks, threads, 0, stream>>>(
          primary, overflow, ovf_offsets, backing, input, table_ids, count, n,
          output.element_size(), out, missing.data_ptr<bool>(), hits ? hits->data_ptr<bool>() : nullptr);
    } else {
      TORCH_CHECK(!overflow_map, "direct HBM lookup expects a single map");
      TORCH_CHECK(primary.table.bucket_capacity_ % Table::BucketType::BufferDim == 0,
                  "direct HBM lookup requires a bucket with complete 16-byte digest vectors");
      lookup_hbm_embedding_rows_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
          primary, input, table_ids, count, n, output.element_size(), out, missing.data_ptr<bool>());
    }
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output, missing, hits};
}
} // namespace
} // namespace dyn_emb

void bind_cache_eval(py::module &m) {
  m.def("lookup_embedding_rows", &dyn_emb::lookup_embedding_rows, py::arg("keys"),
        py::arg("table_ids"), py::arg("primary_map"), py::arg("overflow"),
        py::arg("backing_map"), py::arg("num_valid") = py::none(),
        py::arg("host_num_valid") = py::none(), py::arg("record_cache_hits") = false);
}
