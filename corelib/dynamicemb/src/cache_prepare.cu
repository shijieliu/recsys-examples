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
#include "initializer.cuh"

namespace dyn_emb {
namespace {

using InitializerSpec = std::tuple<std::string, std::vector<float>,
                                   CacheOptionalTensor, CacheOptionalTensor,
                                   float, bool, bool>;

// All per-row inputs and outputs share the compact cache-miss order.
struct StagingRows {
  const int64_t *num_misses;
  const int64_t *num_new;
  const int64_t *miss_table_ids;
  const int64_t *addresses;
  const int64_t *layout_tags;
  const int64_t *embedding_dims;
  const int64_t *value_dims;
  int64_t max_embedding_dim;
  int64_t max_value_dim;
  float initial_accumulator;
  bool fp32_words;
  bool ftrl;
  bool *missing;
};

template <typename V, typename Generator>
__device__ __forceinline__ void initialize_cache_row(
    const StagingRows &rows, Generator &generator, int64_t m,
    V *dst, int64_t emb, int lane, int lanes) {
  for (int64_t col = lane; col < rows.max_embedding_dim; col += lanes)
    dst[col] = TypeConvertFunc<V, float>::convert(generator.generate(m));
  auto *state = dst + rows.max_embedding_dim;
  for (int64_t col = lane; col < rows.max_value_dim - rows.max_embedding_dim;
       col += lanes) {
    if (rows.fp32_words) {
      const uint32_t bits = __float_as_uint(rows.initial_accumulator);
      if constexpr (sizeof(V) == 1)
        reinterpret_cast<uint8_t *>(state)[col] = bits >> (8 * (col % 4));
      if constexpr (sizeof(V) == 2)
        reinterpret_cast<uint16_t *>(state)[col] = bits >> (16 * (col % 2));
    } else {
      const float seed = rows.ftrl && (col < emb || col >= 2 * emb)
                             ? 0.0f : rows.initial_accumulator;
      state[col] = TypeConvertFunc<V, float>::convert(seed);
    }
  }
}

// Compact address metadata first. Each CTA reads a consecutive tile, then
// half warps issue the four value loads before the dependent staging writes.
__device__ void stage_full_rows_256(const StagingRows &rows, uint4 *values) {
  constexpr int groups = 128 / 16;
  constexpr int tile_rows = groups * 4;
  __shared__ const uint4 *sources[tile_rows];
  const int t = threadIdx.x;
  const int64_t n = *rows.num_misses;
  const int64_t stride = int64_t(gridDim.x) * tile_rows;
  for (int64_t begin = int64_t(blockIdx.x) * tile_rows; begin < n; begin += stride) {
    bool unsafe = false;
    if (t < tile_rows) {
      auto ptr = begin + t < n ? rows.addresses[begin + t] : 0;
      sources[t] = reinterpret_cast<const uint4 *>(ptr);
      unsafe = !ptr || (ptr & 15);
      if (begin + t < n) rows.missing[begin + t] = false;
    }
    if (!__syncthreads_or(unsafe)) {
      // The block-wide metadata barrier verified every row is present and
      // aligned. Adjacent threads now write adjacent vectors of staging.
      uint4 data[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const auto *src = sources[j * groups + t / 16] + t % 16;
        auto &v = data[j];
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" :
                     "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(src));
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
        values[begin * 16 + t + j * 128] = data[j];
    } else {
      CacheTransfer256 transfers[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int row = j * groups + t / 16;
        transfers[j].src = sources[row];
        transfers[j].dst = sources[row] ? values + (begin + row) * 16 : nullptr;
      }
      copy_cache_transfers_256(transfers, t % 16);
    }
    if (begin + stride < n) __syncthreads();
  }
}

template <typename V, typename Generator, bool Dense256>
__global__ __launch_bounds__(128, Dense256 ? 8 : 1)
void stage_cache_misses_kernel(StagingRows rows, V *values,
                                          typename Generator::Args args,
                                          int lane_shift) {
  if constexpr (Dense256) {
    if (*rows.num_new == 0) {
      stage_full_rows_256(rows, reinterpret_cast<uint4 *>(values));
      return;
    }
  }
  const int lanes = 1 << lane_shift;
  const int lane = threadIdx.x & (lanes - 1);
  const int64_t worker = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t(gridDim.x) * blockDim.x) >> lane_shift;
  Generator generator(args);
  for (int64_t m = worker >> lane_shift; m < *rows.num_misses; m += stride) {
    const auto *src = reinterpret_cast<const V *>(rows.addresses[m]);
    const int64_t tag = rows.layout_tags ? rows.layout_tags[m] : 0;
    auto *dst = values + m * rows.max_value_dim;
    if (lane == 0)
      rows.missing[m] = !src;
    if (src) {
      if (tag == 0) {
        copy_cache_region(src, dst, rows.max_value_dim, lane, lanes);
      } else {
        const int64_t width = rows.value_dims[tag - 1], emb = rows.embedding_dims[tag - 1];
        copy_cache_row(src, dst, width, emb, emb, rows.max_embedding_dim, lane, lanes);
      }
    } else {
      const int64_t emb = tag ? rows.embedding_dims[tag - 1] : rows.max_embedding_dim;
      initialize_cache_row(rows, generator, m, dst, emb, lane, lanes);
    }
  }
  generator.destroy();
}

// Dispatch the existing per-table/scalar generators on the host. Their
// register state, distributions, per-key DEBUG mapping and parameter order
// are shared with the standalone initializer.
struct StageLaunch {
  const StagingRows &rows;
  const at::Tensor &values;
  int grid, lane_shift;
  static constexpr int block = 128;
  cudaStream_t stream;

  template <typename Generator> void run(typename Generator::Args args) const {
    DISPATCH_VALUE_DATATYPE_FUNCTION(get_data_type(values), V, [&] {
      if (!rows.layout_tags && rows.max_value_dim * sizeof(V) == 256) {
        stage_cache_misses_kernel<V, Generator, true><<<grid, block, 0, stream>>>(
            rows, get_pointer<V>(values), args, lane_shift);
      } else {
        stage_cache_misses_kernel<V, Generator, false><<<grid, block, 0, stream>>>(
            rows, get_pointer<V>(values), args, lane_shift);
      }
    });
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  }

  template <template <bool> class G, bool PerTable>
  void parameterized(const InitializerSpec &spec) const {
    using Generator = G<PerTable>;
    typename Generator::Params params{};
    if constexpr (PerTable) {
      params.table_params = std::get<2>(spec)->data_ptr<float>();
      params.table_ids = rows.miss_table_ids;
    } else {
      const auto &scalars = std::get<1>(spec);
      std::copy(scalars.begin(), scalars.end(), params.values);
    }
    if constexpr (std::is_same_v<Generator, ConstEmbeddingGenerator<PerTable>>) {
      run<Generator>({params});
    } else {
      auto *rng = reinterpret_cast<curandState *>(std::get<3>(spec)->data_ptr());
      run<Generator>({rng, params});
    }
  }

  template <template <bool> class G> void parameters(const InitializerSpec &spec) const {
    if (std::get<2>(spec))
      parameterized<G, true>(spec);
    else
      parameterized<G, false>(spec);
  }
};

void check_vector(const at::Tensor &t, at::Device device, at::ScalarType dtype,
                  int64_t length, const char *name) {
  TORCH_CHECK(t.device() == device && t.scalar_type() == dtype && t.dim() == 1 &&
                  t.is_contiguous() && t.numel() == length,
              "invalid ", name, " in cache preparation");
}

void check_initializer(const InitializerSpec &spec, const at::Tensor &keys,
                       int64_t tables, at::ScalarType dtype) {
  const auto &[mode, scalars, table_params, rng, accum, words, ftrl] = spec;
  const bool random = mode == "normal" || mode == "uniform" ||
                      mode == "truncated_normal";
  TORCH_CHECK(random || mode == "constant" || mode == "debug",
              "unrecognized cache initializer ", mode);
  const int64_t nparams = mode == "debug" ? 0 : mode == "constant" ? 1 :
                          mode == "truncated_normal" ? 4 : 2;
  TORCH_CHECK(static_cast<int64_t>(scalars.size()) == nparams,
              "invalid scalar parameters for ", mode);
  if (table_params) {
    TORCH_CHECK(nparams && table_params->device() == keys.device() &&
                    table_params->scalar_type() == at::kFloat &&
                    table_params->is_contiguous() && table_params->dim() == 2 &&
                    table_params->size(0) == tables && table_params->size(1) == nparams,
                "invalid per-table cache initialization parameters");
  }
  if (random) {
    TORCH_CHECK(rng && rng->device() == keys.device() &&
                    rng->scalar_type() == at::kByte && rng->is_contiguous() &&
                    rng->dim() == 1 && rng->numel() % sizeof(curandState) == 0 &&
                    rng->numel() / sizeof(curandState) >= StageLaunch::block,
                "random initializer requires its CUDA generator-state tensor");
  }
  TORCH_CHECK(dtype == at::kFloat || dtype == at::kHalf ||
                  dtype == at::kBFloat16 || dtype == at::kFloat8_e4m3fn,
              "unsupported cache value dtype");
  TORCH_CHECK(!words || (dtype != at::kFloat && !ftrl),
              "packed FP32 accumulator requires narrow words");
}

// Keep the field order of LookupBatch and CompactedBatch at the Python boundary.
using StagedBatch = std::tuple<at::Tensor, at::Tensor, at::Tensor,
                               CacheOptionalTensor, CacheOptionalTensor, at::Tensor>;
using NewRows = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                           at::Tensor, CacheOptionalTensor, CacheOptionalTensor>;
using PrepareResult =
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
               StagedBatch, NewRows>;

PrepareResult prepare_cache_lookup(
    at::Tensor keys, at::Tensor key_offsets, const CacheLookupMap &cache,
    const BackingLookupMap &backing, const CacheValueLayout &layout,
    const CacheScoreArg &cache_score, const CacheScoreArg &backing_score,
    const InitializerSpec &init, CacheOptionalTensor weights,
    CacheOptionalTensor scores) {
  const auto &[main, buckets, bucket_capacity, pins, overflow, overflow_capacity,
               overflow_offsets, score_words, global_caps] = cache;
  const auto &[back_table, back_buckets, back_capacity, back_rows, back_words,
               back_global] = backing;
  const auto &[ptrs, edims, vdims, max_emb, max_value, dtype] = layout;
  const auto &[mode, scalars, table_params, rng, initial_accum, words, ftrl] = init;
  const auto device = keys.device();
  TORCH_CHECK(keys.is_cuda(), "cache preparation requires CUDA keys");
  const c10::cuda::CUDAGuard guard(device);
  const auto tables = key_offsets.numel() - 1;
  TORCH_CHECK(tables > 0 && max_value >= max_emb && max_emb > 0,
              "invalid cache table dimensions");
  check_vector(key_offsets, device, at::kLong, tables + 1, "key table offsets");
  check_vector(back_buckets, device, at::kLong, tables + 1, "backing bucket offsets");
  check_vector(ptrs, device, at::kLong, tables, "backing value pointers");
  TORCH_CHECK(edims.has_value() == vdims.has_value(), "expected both per-table width tensors");
  if (edims) {
    check_vector(*edims, device, at::kLong, tables, "embedding dimensions");
    check_vector(*vdims, device, at::kLong, tables, "value dimensions");
  }
  check_vector(back_table, device, at::kByte, back_table.numel(), "backing table");
  TORCH_CHECK((back_words == 1 || back_words == 2) && back_capacity > 0,
              "invalid backing score layout or bucket capacity");
  const int64_t slot_bytes = keys.element_size() + sizeof(DigestType) +
                             back_words * sizeof(ScoreType);
  const int64_t backing_slots = back_table.numel() / slot_bytes;
  TORCH_CHECK(back_table.numel() % slot_bytes == 0 && backing_slots % back_capacity == 0,
              "backing storage must contain whole buckets");
  if (back_rows)
    check_vector(*back_rows, device, at::kLong, backing_slots, "backing row map");
  if (back_global)
    check_vector(*back_global, device, at::kLong, tables, "backing global capacities");
  const auto &[score_values, policy] = backing_score;
  if (score_values) {
    TORCH_CHECK(score_values->scalar_type() == at::kLong ||
                    score_values->scalar_type() == at::kUInt64, "invalid backing scores");
    check_vector(*score_values, device, score_values->scalar_type(), keys.numel(),
                 "backing scores");
  } else {
    TORCH_CHECK(policy != ScorePolicyType::Assign && policy != ScorePolicyType::Accumulate &&
                    policy != ScorePolicyType::LruLfu, "backing policy requires scores");
  }
  TORCH_CHECK(policy != ScorePolicyType::LruLfu || back_words == 2,
              "LruLfu requires two backing score words");
  check_initializer(init, keys, tables, dtype);
  auto values = at::empty({keys.numel(), max_value}, keys.options().dtype(dtype));
  auto missing = at::empty({keys.numel()}, keys.options().dtype(at::kBool));
  auto num_unique = key_offsets.narrow(0, tables, 1);
  auto [cache_miss, slots, tids] = lookup_in_cache(
      keys, key_offsets, main, buckets, bucket_capacity, pins, overflow,
      overflow_capacity, overflow_offsets, std::get<0>(cache_score),
      std::get<1>(cache_score), score_words, global_caps, {}, {});
  try {
    // Scores live in storage even when the newest value is in cache.
    auto addresses = lookup_backing_values(backing, layout, keys, tids, backing_score,
                                           num_unique, cache_miss);
    auto [count, positions, fields] = flagged_compact(
        cache_miss, {keys, tids, weights, scores, addresses.backing, addresses.layout_tags},
        num_unique, false, {});
    auto &miss_keys = *fields[0];
    auto &miss_ids = *fields[1];
    // Selection produces the new-row count before staging. On an all-backing-
    // hit batch the payload kernel can skip the initializer, including its
    // RNG worker mapping, without reading a count on the CPU.
    auto [num_new, new_rows, new_fields] = compact_missing_values(
        *fields[4], {miss_keys, miss_ids, fields[2], fields[3], positions}, count);
    if (keys.numel()) {
      StagingRows rows{count.data_ptr<int64_t>(), num_new.data_ptr<int64_t>(),
                       miss_ids.data_ptr<int64_t>(),
                       fields[4]->data_ptr<int64_t>(), fields[5] ? fields[5]->data_ptr<int64_t>() : nullptr,
                       edims ? edims->data_ptr<int64_t>() : nullptr,
                       vdims ? vdims->data_ptr<int64_t>() : nullptr, max_emb, max_value, initial_accum,
                       words, ftrl, missing.data_ptr<bool>()};
      constexpr int block = StageLaunch::block;
      // Preserve the random generator's worker-to-element assignment.
      const int lane_shift = rng ? 5 : cache_copy_lane_shift(max_value * values.element_size());
      const int groups = block >> lane_shift;
      int grid = counted_grid((keys.numel() + groups - 1) / groups, block);
      if (rng && mode != "debug" && mode != "constant")
        grid = std::min<int64_t>(grid, rng->numel() / sizeof(curandState) / block);
      StageLaunch launch{rows, values, grid, lane_shift, at::cuda::getCurrentCUDAStream().stream()};
      if (mode == "debug") {
        // MappingEmbeddingGenerator takes the key modulo an unsigned uint64_t.
        launch.run<MappingEmbeddingGenerator<uint64_t>>(
            {reinterpret_cast<const uint64_t *>(miss_keys.data_ptr()), 100000});
      } else if (mode == "constant") {
        launch.parameters<ConstEmbeddingGenerator>(init);
      } else if (mode == "normal") {
        launch.parameters<NormalEmbeddingGenerator>(init);
      } else if (mode == "uniform") {
        launch.parameters<UniformEmbeddingGenerator>(init);
      } else {
        launch.parameters<TruncatedNormalEmbeddingGenerator>(init);
      }
    }
    StagedBatch staged{miss_keys, miss_ids, values, fields[2], fields[3], missing};
    NewRows new_batch{*new_fields[0], *new_fields[1], num_new, new_rows,
                      *new_fields[4], new_fields[2], new_fields[3]};
    return {cache_miss, slots, tids, count, positions, staged, new_batch};
  } catch (...) {
    // No Python caller has the hit slots yet. Undo pins when allocation or
    // validation of a later stage fails before publishing the result.
    const int64_t main_slots = pins.numel() - tables * overflow_capacity;
    table_update_counter_with_layout(pins, slots, -1, buckets, bucket_capacity,
                                     main_slots, tables, tids, overflow_offsets,
                                     overflow_capacity, num_unique);
    throw;
  }
}

} // namespace
} // namespace dyn_emb

void bind_cache_prepare(py::module &m) {
  m.def("prepare_cache_lookup", &dyn_emb::prepare_cache_lookup,
        "Probe cache and backing, compact cache misses, and stage or initialize their values.",
        py::arg("keys"), py::arg("key_offsets"), py::arg("cache_map"),
        py::arg("backing_map"), py::arg("value_layout"), py::arg("cache_score"),
        py::arg("backing_score"), py::arg("initializer"), py::arg("weights") = py::none(),
        py::arg("scores") = py::none());
}
