// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "cache_deterministic.h"

#include <algorithm>
#include <map>
#include <vector>

namespace dyn_emb {

EvictionResult deterministic_cache_insert(
    at::Tensor keys, at::Tensor tids, std::optional<at::Tensor> score,
    at::Tensor num_valid, bool collect_evictions, InsertWave insert_wave,
    FindSlots lookup_slots, Bucketize bucketize, std::optional<at::Tensor> status) {
  const int64_t capacity = keys.numel();
  const auto n = std::clamp(num_valid.item<int64_t>(), int64_t{0}, capacity);
  auto indices = at::empty({capacity}, tids.options());
  auto ev_keys = at::empty_like(keys);
  auto ev_indices = at::empty_like(indices);
  auto ev_scores = at::empty({capacity}, keys.options().dtype(at::kUInt64));
  auto ev_tids = at::empty_like(tids);
  int64_t num_evicted = 0;
  if (n > 0) {
    auto live_keys = keys.narrow(0, 0, n);
    auto live_tids = tids.narrow(0, 0, n);
    auto [sorted_keys, offsets, input_order] = bucketize(live_keys, live_tids);
    // Bucketize sorts each bucket by key. Wave i visits its i-th key.
    // Store only the occupied positions across all waves.
    auto cpu_offsets = offsets.cpu();
    auto starts = cpu_offsets.accessor<int64_t, 1>();
    auto cpu_order = input_order.cpu();
    auto order = cpu_order.accessor<int64_t, 1>();
    // Backing candidates were all absent at prepare time. A later wave that
    // evicts one of them must spill its staged value: its backing value row
    // has not been installed yet.
    std::map<std::pair<int64_t, int64_t>, int64_t> staged_rows;
    if (collect_evictions && !status) {
      auto cpu_keys = live_keys.view(at::kLong).cpu();
      auto cpu_tids = live_tids.cpu();
      for (int64_t i = 0; i < n; ++i)
        staged_rows.emplace(std::make_pair(cpu_tids.data_ptr<int64_t>()[i],
                                           cpu_keys.data_ptr<int64_t>()[i]), i);
    }
    std::vector<std::vector<int64_t>> waves;
    for (int64_t b = 0; b + 1 < cpu_offsets.numel(); ++b) {
      const auto size = static_cast<size_t>(starts[b + 1] - starts[b]);
      if (waves.size() < size)
        waves.resize(size);
      for (size_t i = 0; i < size; ++i)
        waves[i].push_back(starts[b] + i);
    }
    for (const auto &wave : waves) {
      auto sorted_positions = at::tensor(wave, at::kLong).to(keys.device());
      auto input_rows = input_order.index_select(0, sorted_positions);
      // Index uint64 as words on PyTorch versions without uint64 gathers.
      auto wave_keys = sorted_keys.view(at::kLong)
                           .index_select(0, sorted_positions).view(keys.scalar_type());
      auto wave_tids = tids.index_select(0, input_rows);
      std::optional<at::Tensor> wave_score;
      if (score)
        wave_score = score->view(at::kLong).index_select(0, input_rows)
                         .view(score->scalar_type());
      auto result = insert_wave(wave_keys, wave_tids, wave_score);
      if (!collect_evictions)
        continue;
      const auto m = std::get<1>(result).item<int64_t>();
      if (!m)
        continue;
      auto cpu_slots = std::get<3>(result).narrow(0, 0, m).cpu();
      auto slots = cpu_slots.accessor<int64_t, 1>();
      for (int64_t j = 0; j < m; ++j) {
        if (slots[j] < 0) // Refused input relative to this wave.
          slots[j] = -1 - order[wave.at(-1 - slots[j])];
      }
      if (!staged_rows.empty()) {
        auto victims = std::get<2>(result).narrow(0, 0, m).view(at::kLong).cpu();
        auto tables = std::get<5>(result).narrow(0, 0, m).cpu();
        for (int64_t j = 0; j < m; ++j) {
          auto pos = staged_rows.find({tables.data_ptr<int64_t>()[j],
                                       victims.data_ptr<int64_t>()[j]});
          if (pos != staged_rows.end())
            slots[j] = -1 - pos->second;
        }
      }
      ev_indices.narrow(0, num_evicted, m).copy_(cpu_slots);
      ev_keys.narrow(0, num_evicted, m).copy_(std::get<2>(result).narrow(0, 0, m));
      ev_scores.narrow(0, num_evicted, m).copy_(std::get<4>(result).narrow(0, 0, m));
      ev_tids.narrow(0, num_evicted, m).copy_(std::get<5>(result).narrow(0, 0, m));
      num_evicted += m;
    }
    indices.narrow(0, 0, n).copy_(lookup_slots(live_keys, live_tids));
  }
  auto count = at::full({1}, num_evicted, tids.options());
  if (status) {
    status->select(0, 0).fill_(n);
    status->select(0, 1).fill_(num_evicted);
    status->select(0, 2).copy_(ev_indices.narrow(0, 0, num_evicted).lt(0).sum());
  }
  return {indices, count, ev_keys, ev_indices, ev_scores, ev_tids};
}
} // namespace dyn_emb
