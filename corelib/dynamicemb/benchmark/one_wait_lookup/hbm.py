# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""HBM-only prefetch experiment using DynamicEmb's counted lookup ABI (#545)."""

import os

import torch
import triton
import triton.language as tl
from dynamicemb.batched_dynamicemb_function import _is_hbm_storage
from dynamicemb.dynamicemb_config import EvictedItemMode
from dynamicemb.initializer import _INITIALIZERS
from dynamicemb.key_value_table import (
    DynamicEmbStorage,
    _find_keys,
    get_insert_score_arg,
    score_source,
    store_to_flat,
)
from dynamicemb_extensions import counted_scatter


@triton.jit
def _scatter_slot_rows(
    positions,
    new_slots,
    new_rows,
    slots,
    rows,
    live_count,
    capacity,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    live = (i < capacity) & (i < tl.load(live_count))
    pos = tl.load(positions + i, mask=live, other=-1)
    valid = live & (pos >= 0)
    slot = tl.load(new_slots + i, mask=valid, other=0)
    row = tl.load(new_rows + i, mask=valid, other=0)
    tl.store(slots + pos, slot, mask=valid)
    tl.store(rows + pos, row, mask=valid)


def prefetch_hbm_direct(
    storage,
    unique_keys,
    unique_table_ids,
    num_unique,
    max_emb_dim,
    max_val_dim,
    initializer,
    accumulated_frequency,
    admitter,
    key_scores=None,
    *,
    wait_for_misses=True,
    fuse_slot_scatter=True,
):
    """Serve a fixed-capacity, in-tree HBM lookup without admission or eviction logging.

    A blocking read after hit pinning skips the entire miss path on an all-hit
    batch. Disable it to keep every count on the device. The scatter switch
    isolates the effect of writing both maps in one kernel.
    """
    if (
        type(storage) is not DynamicEmbStorage
        or not _is_hbm_storage(storage)
        or storage.can_expand()
        or admitter is not None
        or key_scores is not None
        or type(initializer) not in _INITIALIZERS.values()
        or EvictedItemMode.RETAIN_KEY in storage._state.evicted_item_mode
        or "DEMB_DETERMINISM_MODE" in os.environ
    ):
        raise ValueError(
            "This experiment requires a fixed-capacity HBM storage and built-in "
            "initializer, with admission, per-key scoring, eviction retention, and "
            "deterministic insertion disabled."
        )

    state = storage._state
    capacity = unique_keys.numel()
    if capacity == 0:
        empty = unique_keys.new_empty((0,), dtype=torch.int64)
        return empty, None, None, empty

    with torch.cuda.nvtx.range("op:storage_find"):
        (
            num_missing,
            missing_keys,
            positions,
            missing_tids,
            missing_scores,
            _,
            _,
            rows,
            slots,
        ) = _find_keys(
            state,
            unique_keys,
            unique_table_ids,
            num_unique,
            score_source(state, accumulated_frequency, key_scores),
            return_slots=True,
        )
    # A later insert must see these pins before choosing an eviction victim.
    storage.increment_counter(slots, unique_table_ids, num_valid=num_unique)
    if wait_for_misses and num_missing.item() == 0:
        return slots, None, None, rows

    values = torch.empty(
        (capacity, max_val_dim), device=unique_keys.device, dtype=state.emb_dtype
    )
    with torch.cuda.nvtx.range("op:initializer"):
        initializer(
            values[:, :max_emb_dim],
            missing_keys,
            missing_tids,
            None,
            num_valid=num_missing,
        )
    if max_val_dim != max_emb_dim:
        state.optimizer.reset_optimizer_states(
            values[:, max_emb_dim:],
            table_ids=missing_tids,
            table_emb_dims=state.table_emb_dims,
            num_valid=num_missing,
        )
    score = get_insert_score_arg(state, capacity, unique_keys.device, missing_scores)
    with torch.cuda.nvtx.range("op:storage_insert"):
        new_rows, new_slots = state.key_index_map.insert(
            missing_keys, missing_tids, score, return_slots=True, num_valid=num_missing
        )
    with torch.cuda.nvtx.range("op:store_to_flat"):
        store_to_flat(state, new_rows, missing_tids, values, num_valid=num_missing)
    storage.increment_counter(new_slots, missing_tids, num_valid=num_missing)

    # These are contiguous vectors from the lookup/insert ops. Each live miss
    # has one unique destination. Pins use slots and values use rows.
    if fuse_slot_scatter:
        with torch.cuda.device(unique_keys.device):
            block_size = 256
            _scatter_slot_rows[(triton.cdiv(capacity, block_size),)](
                positions,
                new_slots,
                new_rows,
                slots,
                rows,
                num_missing,
                capacity,
                BLOCK_SIZE=block_size,
            )
    else:
        counted_scatter(slots, positions, new_slots, num_missing)
        counted_scatter(rows, positions, new_rows, num_missing)
    return slots, None, None, rows
