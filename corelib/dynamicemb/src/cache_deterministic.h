// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/ATen.h>
#include <functional>
#include <optional>
#include <tuple>

namespace dyn_emb {
using EvictionResult = std::tuple<at::Tensor, at::Tensor, at::Tensor,
                                  at::Tensor, at::Tensor, at::Tensor>;
using InsertWave = std::function<EvictionResult(at::Tensor, at::Tensor,
                                                std::optional<at::Tensor>)>;
using FindSlots = std::function<at::Tensor(at::Tensor, at::Tensor)>;
using Bucketize = std::function<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
    at::Tensor, at::Tensor)>;

// The explicitly requested deterministic mode serializes keys in each bucket.
// Its host decisions are outside the normal training path.
EvictionResult deterministic_cache_insert(
    at::Tensor keys, at::Tensor tids, std::optional<at::Tensor> score,
    at::Tensor num_valid, bool collect_evictions, InsertWave insert_wave,
    FindSlots lookup_slots, Bucketize bucketize, std::optional<at::Tensor> status);
} // namespace dyn_emb
