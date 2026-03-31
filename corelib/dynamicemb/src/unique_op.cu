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

#include "check.h"
#include "lookup_forward.h"
#include "torch_utils.h"
#include "unique_op.h"
#include "utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>
#include <cuda/std/tuple>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cassert>
#include <limits>

namespace py = pybind11;

namespace dyn_emb {

constexpr int BLOCK_SIZE = 128;

// ============================================================================
// OLD Hash-based Segmented Unique (kept for A/B debugging)
// ============================================================================

template <typename Key, uint32_t m_seed = 0> struct MurmurHash3_32 {
  __forceinline__ __host__ __device__ static uint32_t rotl32(uint32_t x,
                                                             int8_t r) {
    return (x << r) | (x >> (32 - r));
  }
  __forceinline__ __host__ __device__ static uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }
  __forceinline__ __host__ __device__ static uint32_t hash(const Key &key) {
    constexpr int len = sizeof(Key);
    const uint8_t *const data = reinterpret_cast<const uint8_t *>(&key);
    constexpr int nblocks = len / 4;
    uint32_t h1 = m_seed;
    constexpr uint32_t c1 = 0xcc9e2d51;
    constexpr uint32_t c2 = 0x1b873593;
    const uint32_t *const blocks =
        reinterpret_cast<const uint32_t *>(data + nblocks * 4);
    for (int i = -nblocks; i; i++) {
      uint32_t k1 = blocks[i];
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
      h1 = rotl32(h1, 13);
      h1 = h1 * 5 + 0xe6546b64;
    }
    const uint8_t *tail = data + nblocks * 4;
    uint32_t k1 = 0;
    switch (len & 3) {
    case 3: k1 ^= tail[2] << 16; [[fallthrough]];
    case 2: k1 ^= tail[1] << 8;  [[fallthrough]];
    case 1:
      k1 ^= tail[0];
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
    }
    h1 ^= len;
    return fmix32(h1);
  }
  __forceinline__ __host__ __device__ static uint32_t
  hash_combine(uint32_t h1, uint32_t h2) {
    h1 ^= h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
    return h1;
  }
};

__forceinline__ __device__ uint64_t atomicCAS(uint64_t *address,
                                              uint64_t compare, uint64_t val) {
  return static_cast<uint64_t>(
      ::atomicCAS(reinterpret_cast<unsigned long long *>(address),
                  static_cast<unsigned long long>(compare),
                  static_cast<unsigned long long>(val)));
}
__forceinline__ __device__ int64_t atomicCAS(int64_t *address, int64_t compare,
                                             int64_t val) {
  return static_cast<int64_t>(
      ::atomicCAS(reinterpret_cast<unsigned long long *>(address),
                  static_cast<unsigned long long>(compare),
                  static_cast<unsigned long long>(val)));
}

__device__ __forceinline__ int64_t pack_table_val(int64_t table_id,
                                                  int32_t local_idx) {
  return (static_cast<int64_t>(static_cast<int32_t>(table_id)) << 32) |
         static_cast<uint32_t>(local_idx);
}
__device__ __forceinline__ int64_t unpack_table_id(int64_t packed) {
  return static_cast<int64_t>(static_cast<int32_t>(packed >> 32));
}
__device__ __forceinline__ int32_t unpack_local_idx(int64_t packed) {
  return static_cast<int32_t>(packed & 0xFFFFFFFF);
}

template <typename KeyType,
          KeyType empty_key = std::numeric_limits<KeyType>::max(),
          int64_t empty_val = std::numeric_limits<int64_t>::max()>
__global__ void segmented_init_kernel(KeyType *hash_keys, int64_t *hash_vals,
                                      int64_t *table_counters, size_t capacity,
                                      int64_t num_tables) {
  const size_t stride = blockDim.x * gridDim.x;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < capacity;
       idx += stride) {
    hash_keys[idx] = empty_key;
    hash_vals[idx] = empty_val;
  }
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < static_cast<size_t>(num_tables); idx += stride) {
    table_counters[idx] = 0;
  }
}

template <typename KeyType, typename Hasher,
          KeyType empty_key = std::numeric_limits<KeyType>::max(),
          int64_t empty_val = std::numeric_limits<int64_t>::max()>
__global__ void
segmented_unique_kernel(const KeyType *d_keys, const int64_t *d_table_ids,
                        KeyType *d_unique_keys, int64_t *d_output_indices,
                        size_t num_keys, KeyType *hash_keys, int64_t *hash_vals,
                        size_t capacity, int64_t *table_counters,
                        size_t max_keys_per_table, int64_t *frequency_counters,
                        const int64_t *input_frequencies) {
  const size_t stride = blockDim.x * gridDim.x;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_keys;
       idx += stride) {
    const KeyType key = d_keys[idx];
    const int64_t table_id = d_table_ids[idx];
    const int64_t input_freq = input_frequencies ? input_frequencies[idx] : 1;
    uint32_t key_hash = Hasher::hash(key);
    uint32_t tid_hash = Hasher::hash(static_cast<uint32_t>(table_id));
    uint32_t combined_hash = Hasher::hash_combine(key_hash, tid_hash);
    size_t hash_index = combined_hash % capacity;
    bool done = false;
    for (size_t probe = 0; probe < capacity && !done; ++probe) {
      const KeyType existing_key = hash_keys[hash_index];
      if (existing_key == empty_key) {
        const KeyType old_key =
            atomicCAS(&hash_keys[hash_index], empty_key, key);
        if (old_key == empty_key) {
          int32_t local_unique_idx =
              static_cast<int32_t>(atomicAdd(&table_counters[table_id], 1));
          size_t output_pos =
              static_cast<size_t>(table_id) * max_keys_per_table +
              local_unique_idx;
          d_unique_keys[output_pos] = key;
          *reinterpret_cast<volatile int64_t *>(&hash_vals[hash_index]) =
              pack_table_val(table_id, local_unique_idx);
          d_output_indices[idx] = local_unique_idx;
          if (frequency_counters)
            atomicAdd(&frequency_counters[output_pos], input_freq);
          done = true;
        } else if (old_key == key) {
          int64_t packed_val;
          do {
            packed_val =
                *reinterpret_cast<volatile int64_t *>(&hash_vals[hash_index]);
            __nanosleep(1);
          } while (packed_val == empty_val);
          if (unpack_table_id(packed_val) == table_id) {
            int32_t local_idx = unpack_local_idx(packed_val);
            d_output_indices[idx] = local_idx;
            if (frequency_counters) {
              size_t output_pos =
                  static_cast<size_t>(table_id) * max_keys_per_table +
                  local_idx;
              atomicAdd(&frequency_counters[output_pos], input_freq);
            }
            done = true;
          }
        }
      } else if (existing_key == key) {
        int64_t packed_val;
        do {
          packed_val =
              *reinterpret_cast<volatile int64_t *>(&hash_vals[hash_index]);
          __nanosleep(1);
        } while (packed_val == empty_val);
        if (unpack_table_id(packed_val) == table_id) {
          int32_t local_idx = unpack_local_idx(packed_val);
          d_output_indices[idx] = local_idx;
          if (frequency_counters) {
            size_t output_pos =
                static_cast<size_t>(table_id) * max_keys_per_table + local_idx;
            atomicAdd(&frequency_counters[output_pos], input_freq);
          }
          done = true;
        }
      }
      hash_index = (hash_index + 1) % capacity;
    }
    assert(done && "segmented_unique_kernel: hash table full");
  }
}

__device__ __forceinline__ int binary_search_upper_bound(const int64_t *arr,
                                                         int n, int64_t val) {
  int lo = 0, hi = n;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (arr[mid] <= val)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo - 1;
}

template <typename KeyType>
__global__ void compact_keys_and_freq_kernel(
    const KeyType *partitioned_keys, const int64_t *partitioned_freq,
    size_t max_keys_per_table, const int64_t *table_offsets, int64_t num_tables,
    KeyType *output_keys, int64_t *output_freq, const int64_t *d_total_unique) {
  const int64_t total_unique = *d_total_unique;
  const int64_t stride = blockDim.x * gridDim.x;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_unique;
       idx += stride) {
    int table_id =
        binary_search_upper_bound(table_offsets, num_tables + 1, idx);
    int64_t local_idx = idx - table_offsets[table_id];
    size_t src_pos =
        static_cast<size_t>(table_id) * max_keys_per_table + local_idx;
    output_keys[idx] = partitioned_keys[src_pos];
    if (partitioned_freq != nullptr)
      output_freq[idx] = partitioned_freq[src_pos];
  }
}

__global__ void adjust_output_indices_kernel(const int64_t *d_table_ids,
                                             const int64_t *table_offsets,
                                             int64_t *d_output_indices,
                                             size_t num_keys) {
  const size_t stride = blockDim.x * gridDim.x;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_keys;
       idx += stride) {
    int64_t table_id = d_table_ids[idx];
    d_output_indices[idx] += table_offsets[table_id];
  }
}

// Old hash-based segmented_unique: takes per-element table_ids, returns 5 tensors.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
segmented_unique_hashtable_cuda(at::Tensor keys, at::Tensor table_ids,
                                int64_t num_tables,
                                at::Tensor input_frequencies) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  constexpr int OLD_BLOCK = 64;

  const int64_t num_keys = keys.numel();
  const auto device = keys.device();
  const auto key_dtype = keys.scalar_type();
  const int device_sm_count = DeviceProp::getDeviceProp(device.index()).num_sms;

  TORCH_CHECK(keys.numel() == table_ids.numel(),
              "keys and table_ids must have the same length");
  TORCH_CHECK(table_ids.scalar_type() == at::kLong, "table_ids must be int64");
  TORCH_CHECK(num_tables > 0, "num_tables must be positive");

  const bool enable_freq_counting = input_frequencies.defined();
  const bool has_input_freq =
      enable_freq_counting && input_frequencies.numel() > 0;
  if (has_input_freq) {
    TORCH_CHECK(input_frequencies.numel() == num_keys,
                "input_frequencies must have same length as keys");
  }

  if (num_keys == 0) {
    at::Tensor table_offsets = at::zeros(
        {num_tables + 1}, at::TensorOptions().dtype(at::kLong).device(device));
    at::Tensor num_uniques = table_offsets.slice(0, num_tables, num_tables + 1);
    return std::make_tuple(
        num_uniques, at::empty({0}, keys.options()),
        at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device)),
        table_offsets,
        at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device)));
  }

  constexpr int BLOCKS_PER_SM = 4;
  const int grid_size = device_sm_count * BLOCKS_PER_SM;
  const int64_t max_keys_per_table = num_keys;

  at::Tensor partitioned_unique_keys =
      at::empty({num_tables * max_keys_per_table}, keys.options());
  at::Tensor output_indices = at::empty(
      {num_keys}, at::TensorOptions().dtype(at::kLong).device(device));
  at::Tensor table_counters = at::zeros(
      {num_tables}, at::TensorOptions().dtype(at::kLong).device(device));

  at::Tensor partitioned_freq_counters;
  if (enable_freq_counting) {
    partitioned_freq_counters =
        at::zeros({num_tables * max_keys_per_table},
                  at::TensorOptions().dtype(at::kLong).device(device));
  }

  const int64_t capacity = num_keys * 2;
  at::Tensor hash_keys = at::empty({capacity}, keys.options());
  at::Tensor hash_vals = at::empty(
      {capacity}, at::TensorOptions().dtype(at::kLong).device(device));

  dispatch_key_type(key_dtype, [&]<typename KeyType>() {
    segmented_init_kernel<KeyType><<<grid_size, OLD_BLOCK, 0, stream>>>(
        get_pointer<KeyType>(hash_keys), get_pointer<int64_t>(hash_vals),
        get_pointer<int64_t>(table_counters), capacity, num_tables);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();

    segmented_unique_kernel<KeyType, MurmurHash3_32<KeyType>>
        <<<grid_size, OLD_BLOCK, 0, stream>>>(
            get_pointer<const KeyType>(keys),
            get_pointer<const int64_t>(table_ids),
            get_pointer<KeyType>(partitioned_unique_keys),
            get_pointer<int64_t>(output_indices), num_keys,
            get_pointer<KeyType>(hash_keys), get_pointer<int64_t>(hash_vals),
            capacity, get_pointer<int64_t>(table_counters), max_keys_per_table,
            enable_freq_counting
                ? get_pointer<int64_t>(partitioned_freq_counters)
                : nullptr,
            has_input_freq ? get_pointer<const int64_t>(input_frequencies)
                           : nullptr);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  });

  at::Tensor table_offsets = at::zeros(
      {num_tables + 1}, at::TensorOptions().dtype(at::kLong).device(device));
  {
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes, get_pointer<int64_t>(table_counters),
        get_pointer<int64_t>(table_offsets) + 1, num_tables, stream);
    at::Tensor temp_storage =
        at::empty({static_cast<int64_t>(temp_storage_bytes)},
                  at::TensorOptions().dtype(at::kByte).device(device));
    cub::DeviceScan::InclusiveSum(temp_storage.data_ptr(), temp_storage_bytes,
                                  get_pointer<int64_t>(table_counters),
                                  get_pointer<int64_t>(table_offsets) + 1,
                                  num_tables, stream);
  }

  at::Tensor unique_keys = at::empty({num_keys}, keys.options());
  at::Tensor output_freq_counters;
  if (enable_freq_counting) {
    output_freq_counters = at::empty(
        {num_keys}, at::TensorOptions().dtype(at::kLong).device(device));
  } else {
    output_freq_counters =
        at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device));
  }

  dispatch_key_type(key_dtype, [&]<typename KeyType>() {
    compact_keys_and_freq_kernel<<<grid_size, OLD_BLOCK, 0, stream>>>(
        get_pointer<const KeyType>(partitioned_unique_keys),
        enable_freq_counting
            ? get_pointer<const int64_t>(partitioned_freq_counters)
            : nullptr,
        max_keys_per_table, get_pointer<const int64_t>(table_offsets),
        num_tables, get_pointer<KeyType>(unique_keys),
        enable_freq_counting ? get_pointer<int64_t>(output_freq_counters)
                             : nullptr,
        get_pointer<const int64_t>(table_offsets) + num_tables);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  });

  adjust_output_indices_kernel<<<grid_size, OLD_BLOCK, 0, stream>>>(
      get_pointer<const int64_t>(table_ids),
      get_pointer<const int64_t>(table_offsets),
      get_pointer<int64_t>(output_indices), num_keys);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();

  at::Tensor num_uniques = table_offsets.slice(0, num_tables, num_tables + 1);
  return std::make_tuple(num_uniques, unique_keys, output_indices,
                         table_offsets, output_freq_counters);
}

// Atomic operation overloads for 64-bit types
__forceinline__ __device__ long atomicAdd(long *address, long val) {
  return static_cast<long>(
      ::atomicAdd(reinterpret_cast<unsigned long long *>(address),
                  static_cast<unsigned long long>(val)));
}

__forceinline__ __device__ long long atomicAdd(long long *address,
                                               long long val) {
  return static_cast<long long>(
      ::atomicAdd(reinterpret_cast<unsigned long long *>(address),
                  static_cast<unsigned long long>(val)));
}

__forceinline__ __device__ unsigned long atomicAdd(unsigned long *address,
                                                   unsigned long val) {
  return static_cast<unsigned long>(
      ::atomicAdd(reinterpret_cast<unsigned long long *>(address),
                  static_cast<unsigned long long>(val)));
}

// Type dispatch helper
template <typename Func>
void dispatch_key_type(at::ScalarType key_type, Func &&func) {
  if (key_type == at::kLong) {
    func.template operator()<int64_t>();
  } else if (key_type == at::kUInt64) {
    func.template operator()<uint64_t>();
  } else {
    throw std::invalid_argument(
        "Unsupported key dtype: must be int64 or uint64");
  }
}

// ============================================================================
// Sort-based Segmented Unique Implementation
// ============================================================================
//
// We pack each element's (key, table_id) into a CompositeKey struct and use
// CUB's DeviceRadixSort with a custom decomposer to sort by (table_id, key)
// in a single call.  The decomposer tells CUB to treat the struct as
// [key 64-bit LSB | table_id MSB], so the sort orders first by table_id
// then by key, preserving all 64 key bits without any truncation.

struct alignas(16) CompositeKey {
  int64_t key;
  int64_t table_id;
};

// CUB decomposer: returns references to the struct's fields as a tuple.
// First tuple element = least-significant bits, last = most-significant bits.
struct CompositeKeyDecomposer {
  __host__ __device__ ::cuda::std::tuple<int64_t &, int64_t &>
  operator()(CompositeKey &ck) const {
    return {ck.key, ck.table_id};
  }
};

// Kernel: build CompositeKey array and iota permutation in one pass.
__global__ void build_composite_keys_kernel(
    const int64_t *__restrict__ keys, const int64_t *__restrict__ segment_range,
    int num_segments, int64_t num_keys, CompositeKey *__restrict__ composite_out,
    int64_t *__restrict__ perm_out) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_keys;
       i += stride) {
    int lo = 0, hi = num_segments;
    while (lo < hi) {
      int mid = (lo + hi + 1) >> 1;
      if (segment_range[mid] <= i)
        lo = mid;
      else
        hi = mid - 1;
    }
    composite_out[i] = {keys[i], static_cast<int64_t>(lo)};
    perm_out[i] = i;
  }
}

// Functor for CUB TransformInputIterator: marks first-occurrence positions
// by comparing adjacent sorted CompositeKey entries.
struct FirstOccurrenceFlagOp {
  const CompositeKey *sorted;

  __device__ __forceinline__ int64_t operator()(int64_t j) const {
    if (j == 0) return 1;
    return (sorted[j].table_id != sorted[j - 1].table_id ||
            sorted[j].key != sorted[j - 1].key)
               ? 1
               : 0;
  }
};

// Fused kernel: scatter reverse_indices, compact unique_keys, compute freq,
// and write table_offsets.  Reads keys and table_ids directly from sorted
// CompositeKey array — no indirection through sort_perm for key access.
__global__ void __launch_bounds__(128, 16) fused_dedup_scatter_kernel(
    const CompositeKey *__restrict__ sorted,
    const int64_t *__restrict__ inclusive_sum,
    const int64_t *__restrict__ sort_perm,
    const int64_t *__restrict__ input_freq_ptr, int64_t num_keys,
    int num_segments, int64_t *__restrict__ reverse_indices,
    int64_t *__restrict__ unique_keys,
    int64_t *__restrict__ sorted_reverse_indices,
    int64_t *__restrict__ frequency, int64_t *__restrict__ table_offsets,
    bool has_input_freq, bool enable_freq) {

  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (int64_t j = blockIdx.x * blockDim.x + threadIdx.x; j < num_keys;
       j += stride) {
    int64_t u = inclusive_sum[j] - 1;

    bool is_first =
        (j == 0) || (sorted[j].table_id != sorted[j - 1].table_id) ||
        (sorted[j].key != sorted[j - 1].key);

    reverse_indices[sort_perm[j]] = u;
    sorted_reverse_indices[j] = u;

    if (is_first) {
      unique_keys[u] = sorted[j].key;
    }

    if (enable_freq && frequency != nullptr) {
      int64_t fv = has_input_freq ? input_freq_ptr[sort_perm[j]] : 1;
      atomicAdd(&frequency[u], fv);
    }
  }

  // Table offsets: binary-search sorted composite for table boundaries.
  if (blockIdx.x == 0) {
    for (int t = threadIdx.x; t <= num_segments; t += blockDim.x) {
      if (t == 0) {
        table_offsets[0] = 0;
      } else if (t == num_segments) {
        table_offsets[t] = inclusive_sum[num_keys - 1];
      } else {
        int64_t lo = 0, hi = num_keys;
        while (lo < hi) {
          int64_t mid = (lo + hi) >> 1;
          if (sorted[mid].table_id < static_cast<int64_t>(t))
            lo = mid + 1;
          else
            hi = mid;
        }
        table_offsets[t] = (lo > 0) ? inclusive_sum[lo - 1] : 0;
      }
    }
  }
}

// ============================================================================
// segmented_unique_cuda: Sort-based segmented deduplication
// ============================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor>
segmented_unique_cuda(at::Tensor keys, at::Tensor segment_range,
                      int64_t num_tables, at::Tensor input_frequencies) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const int64_t num_keys = keys.numel();
  const auto device = keys.device();
  const int device_sm_count = DeviceProp::getDeviceProp(device.index()).num_sms;

  TORCH_CHECK(keys.scalar_type() == at::kLong, "keys must be int64");
  TORCH_CHECK(segment_range.numel() == num_tables + 1,
              "segment_range must have num_tables+1 elements");
  TORCH_CHECK(segment_range.scalar_type() == at::kLong,
              "segment_range must be int64");
  TORCH_CHECK(num_tables > 0, "num_tables must be positive");
  TORCH_CHECK(num_keys < std::numeric_limits<int32_t>::max(),
              "num_keys must be less than std::numeric_limits<int32_t>::max()");
  TORCH_CHECK(
      num_tables < std::numeric_limits<int32_t>::max(),
      "num_tables must be less than std::numeric_limits<int32_t>::max()");

  const bool enable_freq_counting = input_frequencies.defined();
  const bool has_input_freq =
      enable_freq_counting && input_frequencies.numel() > 0;

  if (has_input_freq) {
    TORCH_CHECK(input_frequencies.numel() == num_keys,
                "input_frequencies must have same length as keys");
  }

  // Handle empty input
  if (num_keys == 0) {
    at::Tensor table_offsets = at::zeros(
        {num_tables + 1}, at::TensorOptions().dtype(at::kLong).device(device));
    at::Tensor num_uniques = table_offsets.slice(0, num_tables, num_tables + 1);
    auto empty_keys = at::empty({0}, keys.options());
    auto empty_long =
        at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device));
    return std::make_tuple(num_uniques, empty_keys, empty_long, table_offsets,
                           empty_long, empty_long, empty_long);
  }

  constexpr int MAX_BLOCKS_PER_SM = 16;
  const int max_grid = device_sm_count * MAX_BLOCKS_PER_SM;

  auto data_driven_grid = [&](int64_t n) {
    return static_cast<int>(
        std::min(static_cast<int64_t>((n + BLOCK_SIZE - 1) / BLOCK_SIZE),
                 static_cast<int64_t>(max_grid)));
  };

  auto opts_long =
      at::TensorOptions().dtype(at::kLong).device(device);

  // ---- Build CompositeKey array + iota permutation ----
  int64_t composite_bytes = num_keys * static_cast<int64_t>(sizeof(CompositeKey));
  at::Tensor composite_in =
      at::empty({composite_bytes},
                at::TensorOptions().dtype(at::kByte).device(device));
  at::Tensor perm_in = at::empty({num_keys}, opts_long);

  build_composite_keys_kernel
      <<<data_driven_grid(num_keys), BLOCK_SIZE, 0, stream>>>(
          keys.data_ptr<int64_t>(), segment_range.data_ptr<int64_t>(),
          static_cast<int>(num_tables), num_keys,
          reinterpret_cast<CompositeKey *>(composite_in.data_ptr()),
          perm_in.data_ptr<int64_t>());
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();

  // ---- Single-pass sort by (table_id, key) using CUB decomposer ----
  int table_bits = 0;
  {
    int64_t t = num_tables;
    while (t > 0) { ++table_bits; t >>= 1; }
  }
  int end_bit = 64 + table_bits;

  at::Tensor composite_out =
      at::empty({composite_bytes},
                at::TensorOptions().dtype(at::kByte).device(device));
  at::Tensor sort_permutation = at::empty({num_keys}, opts_long);

  CompositeKeyDecomposer decomposer;
  {
    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, sort_temp_bytes,
        reinterpret_cast<const CompositeKey *>(composite_in.data_ptr()),
        reinterpret_cast<CompositeKey *>(composite_out.data_ptr()),
        perm_in.data_ptr<int64_t>(), sort_permutation.data_ptr<int64_t>(),
        static_cast<int>(num_keys), decomposer, 0, end_bit, stream);

    at::Tensor sort_temp =
        at::empty({static_cast<int64_t>(sort_temp_bytes)},
                  at::TensorOptions().dtype(at::kByte).device(device));

    cub::DeviceRadixSort::SortPairs(
        sort_temp.data_ptr(), sort_temp_bytes,
        reinterpret_cast<const CompositeKey *>(composite_in.data_ptr()),
        reinterpret_cast<CompositeKey *>(composite_out.data_ptr()),
        perm_in.data_ptr<int64_t>(), sort_permutation.data_ptr<int64_t>(),
        static_cast<int>(num_keys), decomposer, 0, end_bit, stream);
  }

  const CompositeKey *sorted_composite =
      reinterpret_cast<const CompositeKey *>(composite_out.data_ptr());

  // ---- InclusiveSum with TransformInputIterator ----
  at::Tensor inclusive_sum = at::empty({num_keys}, opts_long);

  {
    FirstOccurrenceFlagOp flag_op{sorted_composite};

    cub::TransformInputIterator<int64_t, FirstOccurrenceFlagOp,
                                cub::CountingInputIterator<int64_t>>
        flag_iter(cub::CountingInputIterator<int64_t>(0), flag_op);

    size_t scan_temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, scan_temp_bytes, flag_iter,
                                  inclusive_sum.data_ptr<int64_t>(),
                                  static_cast<int>(num_keys), stream);

    at::Tensor scan_temp =
        at::empty({static_cast<int64_t>(scan_temp_bytes)},
                  at::TensorOptions().dtype(at::kByte).device(device));

    cub::DeviceScan::InclusiveSum(scan_temp.data_ptr(), scan_temp_bytes,
                                  flag_iter,
                                  inclusive_sum.data_ptr<int64_t>(),
                                  static_cast<int>(num_keys), stream);
  }

  // ---- Fused dedup scatter kernel ----
  at::Tensor reverse_indices = at::empty({num_keys}, opts_long);
  at::Tensor unique_keys = at::empty({num_keys}, keys.options());
  at::Tensor sorted_reverse_indices = at::empty({num_keys}, opts_long);
  at::Tensor table_offsets = at::empty({num_tables + 1}, opts_long);

  at::Tensor freq_counters;
  if (enable_freq_counting) {
    freq_counters = at::zeros({num_keys}, opts_long);
  } else {
    freq_counters = at::empty({0}, opts_long);
  }

  fused_dedup_scatter_kernel
      <<<data_driven_grid(num_keys), BLOCK_SIZE, 0, stream>>>(
          sorted_composite, inclusive_sum.data_ptr<int64_t>(),
          sort_permutation.data_ptr<int64_t>(),
          has_input_freq ? input_frequencies.data_ptr<int64_t>() : nullptr,
          num_keys, static_cast<int>(num_tables),
          reverse_indices.data_ptr<int64_t>(),
          unique_keys.data_ptr<int64_t>(),
          sorted_reverse_indices.data_ptr<int64_t>(),
          enable_freq_counting ? freq_counters.data_ptr<int64_t>() : nullptr,
          table_offsets.data_ptr<int64_t>(), has_input_freq,
          enable_freq_counting);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();

  at::Tensor num_uniques = table_offsets.slice(0, num_tables, num_tables + 1);

  return std::make_tuple(num_uniques, unique_keys, reverse_indices,
                         table_offsets, freq_counters, sort_permutation,
                         sorted_reverse_indices);
}

// ============================================================================
// Helper kernel to expand table IDs from jagged offsets
// ============================================================================

__device__ __forceinline__ int64_t find_table_for_index(
    const int64_t *table_offsets_in_feature, const int64_t *offsets,
    int num_tables, int local_batch_size, int64_t global_idx) {
  int lo = 0, hi = num_tables;
  while (lo < hi) {
    int mid = (lo + hi + 1) / 2;
    int64_t table_start_feature =
        table_offsets_in_feature ? table_offsets_in_feature[mid] : mid;
    int64_t table_start_offset = table_start_feature * local_batch_size;
    int64_t table_start_idx = offsets[table_start_offset];
    if (table_start_idx <= global_idx) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return static_cast<int32_t>(lo);
}

__global__ void expand_table_ids_kernel(const int64_t *offsets,
                                        const int64_t *table_offsets_in_feature,
                                        int64_t *table_ids, int num_tables,
                                        int local_batch_size,
                                        int64_t num_elements) {
  const int64_t stride = blockDim.x * gridDim.x;

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements;
       idx += stride) {
    table_ids[idx] = find_table_for_index(
        table_offsets_in_feature, offsets, num_tables, local_batch_size, idx);
  }
}

at::Tensor expand_table_ids_cuda(
    at::Tensor offsets, c10::optional<at::Tensor> table_offsets_in_feature,
    int64_t num_tables, int64_t local_batch_size, int64_t num_elements) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const auto device = offsets.device();
  const int device_sm_count = DeviceProp::getDeviceProp(device.index()).num_sms;

  TORCH_CHECK(offsets.is_cuda(), "offsets must be on CUDA device");
  TORCH_CHECK(local_batch_size > 0, "local_batch_size must be positive");

  if (num_elements == 0) {
    return at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device));
  }

  int64_t num_features = (offsets.size(0) - 1) / local_batch_size;

  const int64_t *table_offsets_ptr = nullptr;
  if (table_offsets_in_feature.has_value() &&
      table_offsets_in_feature.value().numel() > 0) {
    const auto &table_offsets = table_offsets_in_feature.value();
    TORCH_CHECK(table_offsets.is_cuda(),
                "table_offsets_in_feature must be on CUDA device");
    table_offsets_ptr = get_pointer<const int64_t>(table_offsets);
  } else {
    num_tables = num_features;
  }

  constexpr int MAX_BLOCKS_PER_SM = 16;
  const int grid_size =
      std::min((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE,
               static_cast<int64_t>(device_sm_count * MAX_BLOCKS_PER_SM));

  at::Tensor table_ids = at::empty(
      {num_elements}, at::TensorOptions().dtype(at::kLong).device(device));

  expand_table_ids_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
      get_pointer<const int64_t>(offsets), table_offsets_ptr,
      get_pointer<int64_t>(table_ids), num_tables, local_batch_size,
      num_elements);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();

  return table_ids;
}

// Compute dedup lengths and offsets using GPU kernel
std::tuple<at::Tensor, at::Tensor> compute_dedup_lengths_cuda(
    at::Tensor unique_offsets, at::Tensor table_offsets_in_feature,
    int64_t num_tables, int64_t local_batch_size, int64_t new_lengths_size) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const auto device = unique_offsets.device();

  TORCH_CHECK(unique_offsets.is_cuda(),
              "unique_offsets must be on CUDA device");
  TORCH_CHECK(table_offsets_in_feature.is_cuda(),
              "table_offsets_in_feature must be on CUDA device");

  if (new_lengths_size == 0) {
    return std::make_tuple(
        at::empty({0}, at::TensorOptions().dtype(at::kInt).device(device)),
        at::zeros({1}, at::TensorOptions().dtype(at::kLong).device(device)));
  }

  at::Tensor new_lengths = at::empty(
      {new_lengths_size}, at::TensorOptions().dtype(at::kInt).device(device));
  at::Tensor new_offsets =
      at::empty({new_lengths_size + 1},
                at::TensorOptions().dtype(at::kLong).device(device));

  get_new_length_and_offsets(
      reinterpret_cast<uint64_t *>(get_pointer<int64_t>(unique_offsets)),
      get_pointer<int64_t>(table_offsets_in_feature), num_tables,
      new_lengths_size, local_batch_size, DataType::Int32, DataType::Int64,
      get_pointer<int64_t>(new_offsets), get_pointer<int32_t>(new_lengths),
      stream);

  return std::make_tuple(new_lengths, new_offsets);
}

} // namespace dyn_emb

// Python bindings
void bind_unique_op(py::module &m) {
  m.def(
      "segmented_unique_cuda",
      [](at::Tensor keys, at::Tensor segment_range, int64_t num_tables,
         const c10::optional<at::Tensor> &input_frequencies) {
        at::Tensor freq_tensor;
        if (input_frequencies.has_value()) {
          freq_tensor = input_frequencies.value();
        }
        return dyn_emb::segmented_unique_cuda(keys, segment_range, num_tables,
                                              freq_tensor);
      },
      R"doc(
Segmented unique: deduplicate keys per table using two-pass radix sort.

Pass 1 sorts by full 64-bit key; pass 2 stable-sorts by table_id.
This preserves all key bits without truncation.  Adjacent duplicates
are identified via prefix scan.  No hash tables or spin-waits.

NOTE: This function is fully asynchronous with no GPU-CPU synchronization.

Args:
    keys: Input keys tensor (int64 or uint64)
    segment_range: Per-table boundary offsets (int64, size=num_tables+1).
                   segment_range[t] is the start index for table t's keys.
    num_tables: Total number of tables
    input_frequencies: Controls frequency counting behavior:
                       - None: Disable frequency counting
                       - Empty tensor (numel==0): Each key counts as 1
                       - Tensor with numel==num_keys: Use provided frequencies

Returns:
    Tuple of (num_uniques, unique_keys, reverse_indices, table_offsets,
              freq_counters, sort_permutation, sorted_reverse_indices)
)doc",
      py::arg("keys"), py::arg("segment_range"), py::arg("num_tables"),
      py::arg("input_frequencies") = py::none());

  m.def(
      "segmented_unique_hashtable_cuda",
      [](at::Tensor keys, at::Tensor table_ids, int64_t num_tables,
         const c10::optional<at::Tensor> &input_frequencies) {
        at::Tensor freq_tensor;
        if (input_frequencies.has_value()) {
          freq_tensor = input_frequencies.value();
        }
        return dyn_emb::segmented_unique_hashtable_cuda(keys, table_ids,
                                                        num_tables,
                                                        freq_tensor);
      },
      R"doc(
OLD hash-based segmented unique (kept for A/B debugging).

Takes per-element table_ids (not segment_range).

Args:
    keys: Input keys tensor (int64 or uint64)
    table_ids: Per-element table ID tensor (int64)
    num_tables: Total number of tables
    input_frequencies: Controls frequency counting (None/empty/tensor)

Returns:
    Tuple of (num_uniques, unique_keys, reverse_indices, table_offsets,
              freq_counters)
)doc",
      py::arg("keys"), py::arg("table_ids"), py::arg("num_tables"),
      py::arg("input_frequencies") = py::none());

  m.def(
      "expand_table_ids_cuda",
      [](at::Tensor offsets, c10::optional<at::Tensor> table_offsets_in_feature,
         int64_t num_tables, int64_t local_batch_size, int64_t num_elements) {
        return dyn_emb::expand_table_ids_cuda(offsets, table_offsets_in_feature,
                                              num_tables, local_batch_size,
                                              num_elements);
      },
      R"doc(
Expand table IDs from offsets.

Generates a table_id for each element based on the offsets structure.

Args:
    offsets: Jagged tensor offsets (int64)
    table_offsets_in_feature: Feature offsets per table (int64), or None
    num_tables: Number of tables
    local_batch_size: Batch size per feature
    num_elements: Total number of elements (keys)

Returns:
    table_ids tensor (int64) with same length as num_elements
)doc",
      py::arg("offsets"), py::arg("table_offsets_in_feature") = py::none(),
      py::arg("num_tables") = 0, py::arg("local_batch_size") = 1,
      py::arg("num_elements") = 0);

  m.def(
      "compute_dedup_lengths_cuda",
      [](at::Tensor unique_offsets, at::Tensor table_offsets_in_feature,
         int64_t num_tables, int64_t local_batch_size,
         int64_t new_lengths_size) {
        return dyn_emb::compute_dedup_lengths_cuda(
            unique_offsets, table_offsets_in_feature, num_tables,
            local_batch_size, new_lengths_size);
      },
      R"doc(
Compute new lengths and offsets by evenly distributing unique keys.

Args:
    unique_offsets: Cumulative unique counts per table (int64, device)
    table_offsets_in_feature: Feature offsets per table (int64, device)
    num_tables: Number of tables
    local_batch_size: Batch size per feature
    new_lengths_size: Total output size (num_features * local_batch_size)

Returns:
    Tuple of (new_lengths, new_offsets)
)doc",
      py::arg("unique_offsets"), py::arg("table_offsets_in_feature"),
      py::arg("num_tables"), py::arg("local_batch_size"),
      py::arg("new_lengths_size"));
}
