# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-wait cache prefetch for the built-in, fixed-capacity #545 benchmark."""

import os
from typing import Literal, Optional

import torch
import triton
import triton.language as tl
from dynamicemb.batched_dynamicemb_function import _copy_cuda_tensor_to_pinned_cpu
from dynamicemb.dynamicemb_config import EvictedItemMode
from dynamicemb.initializer import MultiTableInitializer
from dynamicemb.key_value_table import (
    DynamicEmbCache,
    DynamicEmbStorage,
    _lookup_keys,
    _write_back_evicted,
    load_from_flat,
    score_source,
    store_to_flat,
)
from dynamicemb.types import CopyMode
from dynamicemb_extensions import counted_gather, counted_scatter, flagged_compact


@triton.jit
def _insert_status(n_insert, n_victims, victim_slots, status, BLOCK: tl.constexpr):
    # The counted ABI predates #557. An insert refusal has a negative slot in
    # the live eviction list. Count it along with the sizes in one kernel.
    n = tl.load(n_victims)
    refused = tl.full((BLOCK,), 0, tl.int64)
    for chunk in range(0, tl.cdiv(n, BLOCK)):
        index = chunk * BLOCK + tl.arange(0, BLOCK)
        slot = tl.load(victim_slots + index, mask=index < n, other=0)
        refused += (slot < 0).to(tl.int64)
    tl.store(status, tl.load(n_insert))
    tl.store(status + 1, n)
    tl.store(status + 2, tl.sum(refused, axis=0))


def _load_or_initialize(
    storage, initializer, keys, table_ids, rows, scores, count, dim
):
    """Gather backing rows and use one selection for initialization and backing insertion."""
    state = storage._state
    values = load_from_flat(state, rows, table_ids, CopyMode.VALUE, num_valid=count)
    # _lookup_keys returns a negative value row for a key absent from storage.
    n_new, new_pos, (new_keys, new_tids, new_scores) = flagged_compact(
        rows < 0, [keys, table_ids, scores], count
    )
    initializer(values[:, :dim], keys, table_ids, new_pos, num_valid=n_new)
    if values.size(1) > dim:
        state.optimizer.reset_optimizer_states(
            values[:, dim:],
            indices=new_pos,
            table_ids=table_ids,
            table_emb_dims=state.table_emb_dims,
            num_valid=n_new,
        )
    storage.insert(
        new_keys,
        new_tids,
        counted_gather(values, new_pos, n_new),
        new_scores,
        num_valid=n_new,
    )
    return values


def prefetch_cache(
    cache: DynamicEmbCache,
    storage: DynamicEmbStorage,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    num_unique: torch.Tensor,
    max_emb_dim: int,
    max_val_dim: int,
    initializer: MultiTableInitializer,
    accumulated_frequency: Optional[torch.Tensor],
    admitter,
    external_storage=None,
    key_scores: Optional[torch.Tensor] = None,
    *,
    placement: Literal["early", "late"],
):
    """Reserve miss slots before (early) or after (late) loading their backing rows.

    This experiment runs on one training stream. #545's pin kernel predates the
    atomic counter update required when a backward and a prefetch overlap.
    """
    if placement not in ("early", "late"):
        raise ValueError(f"Unsupported cache reservation placement {placement!r}")
    if (
        type(storage) is not DynamicEmbStorage
        or type(cache) is not DynamicEmbCache
        or storage.can_expand()
        or admitter is not None
        or external_storage is not None
        or EvictedItemMode.RETAIN_KEY in storage._state.evicted_item_mode
        or os.environ.get("DEMB_DETERMINISM_MODE") is not None
    ):
        raise ValueError(
            "The cache PoC requires the benchmark's fixed-capacity built-in layout"
        )
    if unique_keys.numel() == 0:
        return torch.empty_like(unique_keys, dtype=torch.int64), None, None

    # Pin every cache hit before reserving any slot. The backing table owns
    # the scores, so its lookup must process all the unique keys, including hits.
    _, hits, slots = cache.lookup(unique_keys, unique_table_ids, num_valid=num_unique)
    cache.increment_counter(slots, unique_table_ids, num_valid=num_unique)
    score = score_source(storage._state, accumulated_frequency, key_scores)
    _, _, _, backing_rows, _ = _lookup_keys(
        storage._state, unique_keys, unique_table_ids, num_unique, score
    )
    n_miss, miss_pos, (keys, tids, rows, scores) = flagged_compact(
        ~hits, [unique_keys, unique_table_ids, backing_rows, score], num_unique
    )

    if placement == "late":
        values = _load_or_initialize(
            storage, initializer, keys, tids, rows, scores, n_miss, max_emb_dim
        )
    (
        new_slots,
        n_evicted,
        victim_keys,
        victim_slots,
        _,
        victim_tids,
    ) = cache.insert_and_evict(keys, tids, None, num_valid=n_miss)

    # One transfer and one wait. Reading these three integers thereafter is CPU-only.
    status = torch.empty(3, dtype=torch.int64, device=slots.device)
    _insert_status[(1,)](n_miss, n_evicted, victim_slots, status, BLOCK=256)
    installs, evictions, refused = _copy_cuda_tensor_to_pinned_cpu(status).tolist()
    if installs == 0:
        return slots, None, None
    if placement == "early":
        values = _load_or_initialize(
            storage, initializer, keys, tids, rows, scores, n_miss, max_emb_dim
        )
    if refused:
        # A failed batch still inserts its new backing rows. Release just the
        # hits pinned by this call and stop before writing back invalid victims.
        cache.decrement_counter(slots, unique_table_ids, num_valid=num_unique)
        raise RuntimeError(f"The cache could not place {refused} keys of this batch")

    if evictions:
        old_values = load_from_flat(
            cache._state, victim_slots, victim_tids, CopyMode.VALUE, num_valid=n_evicted
        )
    # Capture every evicted row before its cache slot receives a new value.
    store_to_flat(cache._state, new_slots, tids, values, num_valid=n_miss)
    if evictions:
        _write_back_evicted(
            storage, victim_keys, victim_tids, old_values, n_evicted, next_tier=None
        )
    cache.increment_counter(new_slots, tids, num_valid=n_miss)
    counted_scatter(slots, miss_pos, new_slots, n_miss)
    return slots, None, None
