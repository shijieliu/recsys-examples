# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, ContextManager, Optional, Tuple

import torch
from dynamicemb.async_erase import table_section
from dynamicemb.dynamicemb_config import EvictedItemMode
from dynamicemb.initializer import MultiTableInitializer
from dynamicemb.key_value_table import (
    Cache,
    DynamicEmbCache,
    DynamicEmbStorage,
    DynamicEmbTableState,
    Storage,
    _append_evicted,
    _find_keys,
    _insert_and_evict_keys,
    _lookup_keys,
    _write_back_evicted,
    eval_lookup,
    get_insert_score_arg,
    load_from_flat,
    load_from_flat_by,
    score_source,
    store_to_flat,
)
from dynamicemb.lookup_batch import CompactedBatch, LookupBatch
from dynamicemb.lookup_layout import EmbeddingLayout
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.types import CopyMode, MultiTableAdmitter
from dynamicemb.utils import decode_table_values, gradient_dtype
from dynamicemb_extensions import (
    EvictStrategy,
    expand_table_ids_cuda,
    fill_rows,
    flagged_compact,
    gather_embedding,
    gather_embedding_pooled,
    get_table_range,
    reduce_grads,
    segmented_unique_cuda,
    select_insert_failed_values,
)

# An output dist's allocator of a sequence lookup's rows (output_alloc): for
# (rows, columns, dtype), None (no buffer: the lookup allocates its own), or a
# buffer of those rows and the row of each key there (int32 [rows]), None when
# key i takes row i. The hierarchical dist places the rows in the order its
# GIN puts read and write them (shard/hier_output_dist.py, lookup_buffer).
OutputAlloc = Callable[
    [int, int, torch.dtype], Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]
]


def _copy_cuda_tensor_to_pinned_cpu(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_cuda:
        return tensor.detach().cpu()
    cpu = torch.empty(
        tuple(tensor.shape),
        dtype=tensor.dtype,
        device=torch.device("cpu"),
        pin_memory=True,
    )
    cpu.copy_(tensor.detach(), non_blocking=True)
    torch.cuda.current_stream(tensor.device).synchronize()
    return cpu


def segmented_unique(
    keys: torch.Tensor,
    segment_range: torch.Tensor,
    evict_strategy: Optional[EvictStrategy] = None,
    frequency_counts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Perform segmented unique operation on keys with segment_range.

    This function deduplicates keys within each table segment, using the
    GPU-accelerated segmented_unique_cuda kernel.

    Args:
        keys: Input key tensor (int64 or uint64)
        segment_range: Table boundary offsets where segment_range[i] is the
                       start index for table i (int64)
        evict_strategy: Optional eviction strategy (for LFU mode)
        frequency_counts: Optional input frequency counts per key

    Returns:
        Tuple of (unique_keys, reverse_indices, unique_keys_table_range,
                  output_scores). Unique keys and any scores have input capacity.
        Only their first unique_keys_table_range[-1] entries are live. The
        per-table offsets and the final count are on the input device.
    """
    with torch.cuda.nvtx.range("op:segmented_unique"):
        host_num_keys = keys.size(0)
        host_num_tables = segment_range.size(0) - 1
        device = keys.device

        if host_num_keys == 0:
            empty_keys = torch.empty(0, dtype=keys.dtype, device=device)
            empty_reverse_indices = torch.empty(0, dtype=torch.int64, device=device)
            d_table_range = torch.zeros(
                host_num_tables + 1, dtype=torch.int64, device=device
            )
            return empty_keys, empty_reverse_indices, d_table_range, None

        is_lfu_enabled = (
            evict_strategy == EvictStrategy.KLfu if evict_strategy else False
        )
        need_frequency_output = is_lfu_enabled or frequency_counts is not None

        input_frequencies = None
        if frequency_counts is not None:
            input_frequencies = frequency_counts
        elif need_frequency_output:
            input_frequencies = torch.empty(0, dtype=torch.int64, device=device)

        (
            _num_uniques,
            unique_keys,
            reverse_indices,
            table_offsets,
            freq_counters,
        ) = segmented_unique_cuda(
            keys, segment_range, host_num_tables, input_frequencies
        )

        output_scores = freq_counters if need_frequency_output else None
        return unique_keys, reverse_indices, table_offsets, output_scores


def largest_score_per_unique_key(
    scores: torch.Tensor, reverse_indices: torch.Tensor, host_capacity: int
) -> torch.Tensor:
    """The score of each unique key: the largest of those its occurrences had.

    ``reverse_indices[i]`` is the unique key that input key ``i`` was merged
    into; a key that was not repeated keeps its own score.
    """
    unique_scores = torch.zeros(host_capacity, dtype=torch.int64, device=scores.device)
    return unique_scores.scatter_reduce_(
        0, reverse_indices, scores, reduce="amax", include_self=False
    )


class StorageMode(Enum):
    DEFAULT = auto()
    CACHE = auto()
    HBM_DIRECT = auto()


@dataclass
class PrefetchState:
    unique_keys: torch.Tensor
    reverse_indices: torch.Tensor
    unique_table_ids: torch.Tensor
    lfu_accumulated_frequency: torch.Tensor
    storage_mode: StorageMode
    slot_indices: Optional[torch.Tensor]
    # How many leading entries of the per-key arrays of this state are live, as a
    # one-element int64 device tensor: the arrays keep the length of the input and what
    # lies past ``num_unique`` is undefined (see dynamicemb.types.Storage).
    num_unique: torch.Tensor
    # Sequence output expands duplicates, so it processes num_input rows even
    # when num_unique is smaller. This count reuses the final input-table offset.
    num_input: torch.Tensor
    scores_are_frequencies: bool
    # Value-buffer row per key -- what forward and backward both address values
    # with. It differs from ``slot_indices`` for a table that stores a row per
    # key (a storage): rows are handed out densely while slots come from the
    # key's hash, and the key map of a table that never evicts is deliberately
    # larger than its value buffer, so indexing values by slot would read past the
    # end of it. A cache points this at the slot tensor itself rather than leaving it unset, so a
    # consumer never has to ask which of the two it is holding. Anything indexed
    # by hash slot (the ref counter) uses ``slot_indices``.
    value_row_indices: Optional[torch.Tensor] = None
    # Rejected positions in the unique batch, paired with their device count.
    # The pair is absent when prefetch skips admission. A batch that ran
    # admission can have a device count of zero.
    non_admitted: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    # One int64 per unique key, for tables that take a score per key.
    unique_scores: Optional[torch.Tensor] = None
    # The key_map_generation (rehash count) of the table state the slots were found in: a later
    # prefetch may expand it before this batch's backward runs.
    key_map_generation: int = 0

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Marks the per-batch tensors used on ``stream``: a prefetch pipeline
        makes them on its own stream, and the forward that takes them and its
        backward read them on another, past the point where the caching
        allocator would otherwise hand their blocks to the next prefetch."""
        for tensor in (
            self.unique_keys,
            self.reverse_indices,
            self.unique_table_ids,
            self.lfu_accumulated_frequency,
            self.slot_indices,
            self.value_row_indices,
            self.unique_scores,
            self.num_unique,
            self.num_input,
            *(self.non_admitted or ()),
        ):
            if tensor is not None:
                tensor.record_stream(stream)


def refresh_slots_after_rehash(state: DynamicEmbTableState, batch) -> None:
    """Bring a batch's slots up to date if a rehash moved its keys.

    ``batch`` is anything that holds what a prefetch found: a ``PrefetchState``,
    or the ``ctx`` of ``DynamicEmbeddingFunction`` that took it over (``unique_keys``,
    ``unique_table_ids``, ``slot_indices``, ``key_map_generation`` and ``num_unique``,
    the number of live entries of the arrays). If ``state`` has rehashed since (a
    later batch's prefetch may have, before this one's backward or its no-grad
    forward), the keys are looked up again, read only (their ref counts kept them in
    the table), and ``batch.slot_indices`` and ``batch.key_map_generation`` are
    replaced. A key the prefetch did not admit stays without a slot. Value rows do
    not move: a storage keeps each key's row through a rehash, and a cache does not
    rehash.
    """
    if batch.key_map_generation == state.key_map_generation:
        return
    # Only a table that stores a row per key keeps its value rows through a
    # rehash (``use_rows_``). Without it the row is the slot and the value moves
    # with the key, so ``value_row_indices`` would go stale too: not handled here.
    assert state.key_index_map.use_rows_, (
        "a rehash since the prefetch is only handled for a table that stores a "
        "value row per key (DynamicEmbStorage); a cache does not rehash"
    )
    slots = _lookup_keys(
        state,
        batch.unique_keys,
        batch.unique_table_ids,
        batch.num_unique,
        const_lookup=True,
        return_slots=True,
    )[-1]
    # Only refresh keys this batch actually pinned. A later prefetch may have
    # admitted one of its rejected keys, whose slot here must stay negative.
    num_pinned, pinned_positions, (new_pinned_slots,) = flagged_compact(
        batch.slot_indices >= 0, [slots], batch.num_unique
    )
    # Write the new slots back at their original positions for pin release.
    gather_embedding(
        new_pinned_slots,
        batch.slot_indices,
        out_row=pinned_positions,
        num_valid=num_pinned,
    )
    batch.key_map_generation = state.key_map_generation


def _is_hbm_storage(storage: Storage) -> bool:
    """True if values live in GPU device memory (HBM). Use buffer type, not
    tensor.is_cuda: host memory registered to CUDA address space can report is_cuda True.
    """
    return (
        isinstance(storage, DynamicEmbStorage)
        and storage._state.tables[0].is_device_buffer()
    )


def _apply_admission(
    missing_keys: torch.Tensor,
    missing_indices: torch.Tensor,
    missing_table_ids: torch.Tensor,
    missing_scores: Optional[torch.Tensor],
    values: torch.Tensor,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    max_emb_dim: int,
    freq_for_admission: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
    num_missing: torch.Tensor,
    scores_are_frequencies: bool,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Apply admission filtering for missing keys.

    If an admission strategy is active, runs ``non_admitted_initializer`` over
    the rows it rejects when the strategy asked for one, then compacts
    keys/scores/table_ids/indices to only the admitted subset.

    Returns the device count, keys, scores, table IDs and unique-batch positions
    for the rows to insert. With admission disabled this includes every miss.
    """
    with torch.cuda.nvtx.range("_apply_admission"):
        if admitter is None:
            return (
                num_missing,
                missing_keys,
                missing_scores,
                missing_table_ids,
                missing_indices,
            )
        admit_mask = admitter.admit(
            missing_keys,
            num_missing,
            table_ids=missing_table_ids,
            frequencies=freq_for_admission,
        )
        if (
            scores_are_frequencies
            and freq_for_admission is not None
            and missing_scores is not None
        ):
            missing_scores = freq_for_admission
        non_admitted_initializer = admitter.non_admitted_initializer
        if non_admitted_initializer is not None:
            num_refused, _, (non_admitted_indices,) = flagged_compact(
                admit_mask, [missing_indices], num_missing, invert=True
            )
            with torch.cuda.nvtx.range("op:initializer"):
                non_admitted_initializer(
                    values[:, :max_emb_dim],
                    # The indices select rows of the whole unique batch.
                    unique_keys,
                    num_refused,
                    unique_table_ids,
                    non_admitted_indices,
                )
        with torch.cuda.nvtx.range("op:compact"):
            (
                num_admitted,
                _,
                (
                    keys_to_insert,
                    positions_in_unique,
                    table_ids_to_insert,
                    scores_to_insert,
                ),
            ) = flagged_compact(
                admit_mask,
                [missing_keys, missing_indices, missing_table_ids, missing_scores],
                num_missing,
            )
        return (
            num_admitted,
            keys_to_insert,
            scores_to_insert,
            table_ids_to_insert,
            positions_in_unique,
        )


def _lookup_next_tier(
    next_tier: Storage, rows: LookupBatch, num_valid: torch.Tensor
) -> torch.Tensor:
    """Fill the values of backing misses found in the external tier."""
    assert rows.mask is not None and rows.values is not None
    in_next = torch.zeros_like(rows.mask)
    with torch.cuda.nvtx.range("op:compact"):
        asked = rows.compact(num_valid, copy_weights=False, copy_scores=False)
    with torch.cuda.nvtx.range("op:next_tier_find"):
        result = next_tier.find(
            asked.keys,
            asked.table_ids,
            copy_mode=CopyMode.VALUE,
            num_valid=asked.num_valid,
        )
    found, found_values = result[5], result[7]
    # The external tier answers in request order. Map its hits back to the
    # positions in the original miss buffer.
    num_hits, hit_rows, (miss_rows,) = flagged_compact(
        found, [asked.input_rows], asked.num_valid
    )
    gather_embedding(found_values, rows.values, hit_rows, miss_rows, num_valid=num_hits)
    fill_rows(in_next, True, indices=miss_rows, num_valid=num_hits)
    return in_next


def _insert_into_storage(
    storage: DynamicEmbStorage,
    next_tier: Optional[Storage],
    batch: CompactedBatch,
) -> None:
    """Insert selected rows into storage and pass any spills to the next tier."""
    keys = batch.keys
    table_ids = batch.table_ids
    values = batch.source_values
    assert values is not None
    if next_tier is None:
        storage.insert(
            keys,
            table_ids,
            values,
            batch.scores,
            num_valid=batch.num_valid,
            input_rows=batch.input_rows,
        )
        return
    # The spill operation can also need a refused input's value.
    values = gather_embedding(values, batch.input_rows, num_valid=batch.num_valid)
    # NO_EVICTION is incompatible with external storage. Its storage expands
    # rather than evicting keys, so it does not enter this spill path.
    state = storage._state
    (
        slots,
        num_evicted,
        evicted_keys,
        evicted_table_ids,
        evicted_slots,
        _,
    ) = _insert_and_evict_keys(
        state, keys, table_ids, batch.scores, batch.num_valid
    )
    evicted_values = load_from_flat(
        state,
        evicted_slots,
        evicted_table_ids,
        copy_mode=CopyMode.VALUE,
        num_valid=num_evicted,
    )
    # A refused insertion has a negative slot encoding the incoming row.
    # Spill its input value, since it did not displace an existing value.
    select_insert_failed_values(
        evicted_slots, values, evicted_values, num_valid=num_evicted
    )
    store_to_flat(state, slots, table_ids, values, num_valid=batch.num_valid)
    next_tier.insert(
        evicted_keys,
        evicted_table_ids,
        evicted_values,
        num_valid=num_evicted,
    )


def _prefetch_cache_path(
    cache: DynamicEmbCache,
    storage: DynamicEmbStorage,
    unique_keys: torch.Tensor,
    unique_table_offsets: torch.Tensor,
    num_unique: torch.Tensor,
    max_emb_dim: int,
    max_val_dim: int,
    initializer: MultiTableInitializer,
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
    external_storage: Optional[Storage] = None,
    key_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """Cache prefetch with counter protection and overflow buffer.

    ``storage`` is the tier that owns every key's score; the cache above it
    ranks by timestamp only, and holds the freshest value of a key until it is
    written back. When ``external_storage`` is given (a
    score-less store below ``storage``, e.g. a PS) it holds the keys ``storage`` has
    evicted: keys missing from ``storage`` are fetched from it, and what
    ``storage`` evicts is spilled to it.

    Only admitted keys are inserted into the cache.  Non-admitted keys get
    slot_indices = -1 and their positions (in the unique_keys array) are
    returned so the forward pass can lazily initialize their embeddings.

    Index spaces: ``unique_*`` arrays are indexed by position in the unique batch;
    everything after the cache lookup is over the *misses* only, indexed by miss
    number, and ``miss_pos`` maps a miss number back to its position in the
    unique batch. The misses are compacted into arrays of the capacity of the
    input with a count of their own, ``num_miss``, and every later step takes the
    count of the arrays it is given. Cache insertion reports the placement result
    to the host once, sizing all subsequent work.

    Returns the table IDs, slots, and optional (rejected positions, device count)
    pair. Table IDs and slots are in unique-key order.
    """
    with torch.cuda.nvtx.range("_prefetch_cache_path"):
        device = unique_keys.device
        state = cache._state
        if unique_keys.numel() == 0:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, empty, None

        # 1. Resolve table IDs, probe main and overflow buckets, and pin hits.
        # Placement later fills the -1 slots at the original miss positions.
        with torch.cuda.nvtx.range("op:cache_lookup"):
            miss_in_cache, slot_indices, unique_table_ids = cache.lookup_and_pin(
                unique_keys, unique_table_offsets
            )

        placing = False
        try:
            cache.record_lookup_metrics(
                miss_in_cache, num_unique, mask_marks_misses=True
            )
            # 2. Storage is the only place scores live: a key-only lookup of *every*
            # key -- hits included, and even when nothing missed -- updates its score
            # there without moving a value. The cache only ranks by timestamp.
            # ``score_input`` is what the storage's score is made from, chosen once
            # here and carried with the keys from now on; ``accumulated_frequency``
            # is only for admission below.
            score_input = score_source(
                storage._state, accumulated_frequency, key_scores
            )
            with torch.cuda.nvtx.range("op:storage_find"):
                _, _, _, st_rows, _ = _lookup_keys(
                    storage._state,
                    unique_keys,
                    unique_table_ids,
                    num_unique,
                    score_input,
                )

            # 3. Pack cache misses with the backing rows from the score lookup.
            # output_rows maps each miss back to its unique-key position.
            with torch.cuda.nvtx.range("op:compact"):
                misses = LookupBatch(
                    unique_keys,
                    unique_table_ids,
                    weights=accumulated_frequency if admitter is not None else None,
                    scores=score_input,
                    mask=miss_in_cache,
                ).compact(num_unique, backing_rows=st_rows)
            assert misses.backing_rows is not None
            cache_batch = misses
            into_cache = None
            non_admitted = None

            # 4. Stage values in miss order. Backing insertion can recycle
            # existing rows, so load their values before inserting new keys.
            values = load_from_flat(
                storage._state,
                misses.backing_rows,
                misses.table_ids,
                copy_mode=CopyMode.VALUE,
                num_valid=misses.num_valid,
            )
            rows = LookupBatch(
                misses.keys,
                misses.table_ids,
                values=values,
                weights=misses.weights,
                scores=misses.scores,
                mask=misses.backing_rows < 0,
            )
            in_next = None
            if external_storage is not None:
                in_next = _lookup_next_tier(external_storage, rows, misses.num_valid)
                rows.mask = rows.mask & ~in_next

            # 5. Admission applies only to keys absent from all backing tiers.
            admitted = rows.mask
            if admitter is not None:
                with torch.cuda.nvtx.range("op:compact"):
                    new = rows.compact(
                        misses.num_valid,
                        output_rows=misses.output_rows,
                        copy_scores=False,
                    )
                admit_mask = admitter.admit(
                    new.keys,
                    new.num_valid,
                    table_ids=new.table_ids,
                    frequencies=new.weights,
                )
                held = ~admitted
                gather_embedding(
                    admit_mask, admitted, out_row=new.input_rows, num_valid=new.num_valid
                )
                if (
                    storage._state.evict_strategy == EvictStrategy.KLfu
                    and new.weights is not None
                    and rows.scores is not None
                ):
                    gather_embedding(
                        new.weights,
                        rows.scores,
                        out_row=new.input_rows,
                        num_valid=new.num_valid,
                    )
                with torch.cuda.nvtx.range("op:compact"):
                    num_refused, _, (refused_pos,) = flagged_compact(
                        admit_mask, [new.output_rows], new.num_valid, invert=True
                    )
                    rows.mask = held | admitted
                    cache_batch = rows.compact(
                        misses.num_valid,
                        output_rows=misses.output_rows,
                        copy_weights=False,
                        copy_scores=False,
                    )
                into_cache = cache_batch.input_rows
                non_admitted = (refused_pos, num_refused)

            # 6. Initialize newly admitted rows. A lower-tier hit already has
            # a value but still needs to be inserted into backing storage.
            rows.mask = admitted
            with torch.cuda.nvtx.range("op:compact"):
                if in_next is None:
                    new_backing = rows.compact(misses.num_valid, copy_weights=False)
                    num_to_init, init_rows = new_backing.num_valid, new_backing.input_rows
                else:
                    num_to_init, init_rows, _ = flagged_compact(
                        admitted, [], misses.num_valid
                    )
            with torch.cuda.nvtx.range("op:initializer"):
                initializer(
                    values[:, :max_emb_dim],
                    rows.keys,
                    num_to_init,
                    rows.table_ids,
                    init_rows,
                )
            if max_val_dim != max_emb_dim:
                state.optimizer.reset_optimizer_states(
                    values[:, max_emb_dim:],
                    indices=init_rows,
                    table_ids=rows.table_ids,
                    table_emb_dims=state.table_emb_dims,
                    num_valid=num_to_init,
                )
            if in_next is not None:
                with torch.cuda.nvtx.range("op:compact"):
                    rows.mask = in_next | admitted
                    new_backing = rows.compact(misses.num_valid, copy_weights=False)
            with torch.cuda.nvtx.range("op:storage_insert"):
                _insert_into_storage(storage, external_storage, new_backing)

            # Reserve slots after preparing the backing rows and applying admission.
            with torch.cuda.nvtx.range("op:cache_insert_evict"):
                placing = True
                (
                    cache_slots,
                    num_evicted,
                    evicted_keys,
                    evicted_slots,
                    _,
                    evicted_tids,
                    host_num_refused,
                ) = cache.insert_and_evict(
                    cache_batch.keys,
                    cache_batch.table_ids,
                    report_refused=True,
                    num_valid=cache_batch.num_valid,
                )

            host_num_attempted = cache_slots.numel()
            if host_num_attempted == 0:
                return unique_table_ids, slot_indices, non_admitted

            # The placement report sliced the eviction arrays to their live length.
            # Reuse the GPU count for writeback. On partial refusal the eviction
            # list also has incoming keys with negative slots. Compact those out,
            # since only displaced entries have an old value to write back.
            host_num_evicted = evicted_keys.numel()
            host_num_victims = host_num_evicted - host_num_refused
            num_victims = num_evicted
            if host_num_victims:
                if host_num_refused:
                    (
                        num_victims,
                        _,
                        (evicted_keys, evicted_slots, evicted_tids),
                    ) = flagged_compact(
                        evicted_slots >= 0,
                        [evicted_keys, evicted_slots, evicted_tids],
                        num_evicted,
                    )
                with torch.cuda.nvtx.range("op:load_from_flat"):
                    evicted_values = load_from_flat(
                        state,
                        evicted_slots,
                        evicted_tids,
                        copy_mode=CopyMode.VALUE,
                        num_valid=num_victims,
                    )
                _write_back_evicted(
                    storage,
                    evicted_keys,
                    evicted_tids,
                    evicted_values,
                    num_victims,
                    next_tier=external_storage,
                )
            cache_tids = cache_batch.table_ids[:host_num_attempted]
            with torch.cuda.nvtx.range("op:store_to_flat"):
                store_to_flat(
                    state,
                    cache_slots,
                    cache_tids,
                    values[:host_num_attempted] if into_cache is None else values,
                    num_valid=cache_batch.num_valid,
                    host_num_valid=host_num_attempted,
                    input_rows=None
                    if into_cache is None
                    else into_cache[:host_num_attempted],
                )
        except Exception:
            # Placement can publish a key before its row has been installed or
            # its dirty victim has reached storage. Block future reads and flushes.
            if placing:
                cache._placement_failed = True
            cache.decrement_counter(slot_indices, unique_table_ids, num_valid=num_unique)
            raise

        if host_num_refused:
            # There is no PrefetchState for a failed batch to unpin in backward.
            cache.decrement_counter(
                slot_indices, unique_table_ids, num_valid=num_unique
            )
            raise RuntimeError(
                f"The cache could not place {host_num_refused} keys of this batch: their "
                "buckets and their tables' overflow areas held only keys it may not "
                "evict (pinned by batches in flight, or placed by this batch). "
                "Increase the cache size, or reduce the prefetch pipeline depth or "
                "the batch size."
            )

        # Publish slots at the original unique-key positions so backward can
        # release both sets of pins with num_unique.
        cache.increment_counter(
            cache_slots, cache_tids, num_valid=cache_batch.num_valid
        )
        gather_embedding(
            cache_slots,
            slot_indices,
            out_row=cache_batch.output_rows[:host_num_attempted],
            num_valid=cache_batch.num_valid,
        )
        return unique_table_ids, slot_indices, non_admitted


def _prefetch_hbm_direct_path(
    storage: DynamicEmbStorage,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    num_unique: torch.Tensor,
    max_emb_dim: int,
    max_val_dim: int,
    initializer: MultiTableInitializer,
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
    key_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """HBM-direct prefetch with counter protection.

    Only admitted keys are inserted.  Non-admitted keys get slot_indices = -1
    and their positions are returned for lazy initialization at forward time.

    Returns slots, an optional (rejected positions, device count) pair,
    and the indices of value rows. The two index spaces are kept apart because they only
    coincide by accident: ``slot_indices`` addresses the hash table (and the ref
    counter beside it), ``value_row_indices`` addresses the flat value buffer. A
    storage stores a row per key, so they differ: rows are handed out densely
    while slots come from the key's hash, and the key map of a table that never
    evicts is sized ``1 / load factor`` times the value buffer, so a slot can
    point past the end of it. A table whose row is its slot returns the slot
    tensor itself as the rows, so the caller always gets something it can address
    values with.
    """
    with torch.cuda.nvtx.range("_prefetch_hbm_direct_path"):
        state = storage._state
        device = unique_keys.device
        host_capacity = unique_keys.numel()
        if host_capacity == 0:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return empty, None, empty

        with torch.cuda.nvtx.range("op:storage_find"):
            (
                num_missing,
                missing_keys,
                missing_indices,
                missing_table_ids,
                missing_scores,
                _,
                _,
                rows,
                slot_indices,
                freq_for_admission,
            ) = _find_keys(
                state,
                unique_keys,
                unique_table_ids,
                num_unique,
                score_source(state, accumulated_frequency, key_scores),
                return_slots=True,
                miss_inputs=(accumulated_frequency if admitter is not None else None,),
            )

        # The ref counter is keyed by slot, the values by row: a storage stores a row
        # per key, so these are different tensors and both are kept up to date. A key
        # that was not found has slot -1, which the counter kernel skips.
        storage.increment_counter(slot_indices, unique_table_ids, num_valid=num_unique)

        non_admitted = None
        if admitter is not None:
            admit_mask = admitter.admit(
                missing_keys,
                num_missing,
                table_ids=missing_table_ids,
                frequencies=freq_for_admission,
            )
            if (
                state.evict_strategy == EvictStrategy.KLfu
                and freq_for_admission is not None
                and missing_scores is not None
            ):
                missing_scores = freq_for_admission
            with torch.cuda.nvtx.range("op:compact"):
                (
                    num_admitted,
                    _,
                    (
                        admitted_keys,
                        admitted_tids,
                        admitted_scores,
                        admitted_unique_positions,
                    ),
                ) = flagged_compact(
                    admit_mask,
                    [missing_keys, missing_table_ids, missing_scores, missing_indices],
                    num_missing,
                )
                num_refused, _, (non_admitted_positions,) = flagged_compact(
                    admit_mask, [missing_indices], num_missing, invert=True
                )
            non_admitted = (non_admitted_positions, num_refused)
        else:
            num_admitted = num_missing
            admitted_keys = missing_keys
            admitted_tids = missing_table_ids
            admitted_scores = missing_scores
            admitted_unique_positions = missing_indices

        # Initialize and insert the admitted keys
        init_values = torch.empty(
            host_capacity, max_val_dim, dtype=state.emb_dtype, device=device
        )
        with torch.cuda.nvtx.range("op:initializer"):
            initializer(
                init_values[:, :max_emb_dim],
                admitted_keys,
                num_admitted,
                admitted_tids,
                None,
            )
        if max_val_dim != max_emb_dim:
            state.optimizer.reset_optimizer_states(
                init_values[:, max_emb_dim:],
                table_ids=admitted_tids,
                table_emb_dims=state.table_emb_dims,
                num_valid=num_admitted,
            )
        score_arg = get_insert_score_arg(state, host_capacity, device, admitted_scores)
        with torch.cuda.nvtx.range("op:storage_insert"):
            # HBM-direct is a last tier: this insert may evict, and the victims are
            # dropped (retained when configured, which reads their count: opt-in).
            # This is the forward-path analogue of the _insert_key_values collection.
            if EvictedItemMode.RETAIN_KEY in state.evicted_item_mode:
                (
                    new_indices,
                    num_evicted,
                    evicted_keys,
                    evicted_table_ids,
                    new_slots,
                ) = state.key_index_map.insert(
                    admitted_keys,
                    admitted_tids,
                    score_arg,
                    collect_evicted=True,
                    return_slots=True,
                    num_valid=num_admitted,
                )
                _append_evicted(state, evicted_keys, evicted_table_ids, num_evicted)
            else:
                new_indices, new_slots = state.key_index_map.insert(
                    admitted_keys,
                    admitted_tids,
                    score_arg,
                    return_slots=True,
                    num_valid=num_admitted,
                )
        # The insert returns rows, and the slots as well. Keep both: the counter is
        # indexed by slot, the value buffer by row.
        new_rows = new_indices
        with torch.cuda.nvtx.range("op:store_to_flat"):
            store_to_flat(
                state, new_rows, admitted_tids, init_values, num_valid=num_admitted
            )
        storage.increment_counter(new_slots, admitted_tids, num_valid=num_admitted)

        gather_embedding(
            new_slots, slot_indices, out_row=admitted_unique_positions, num_valid=num_admitted
        )
        if rows is not slot_indices:
            gather_embedding(
                new_rows, rows, out_row=admitted_unique_positions, num_valid=num_admitted
            )
        return slot_indices, non_admitted, rows


def dynamicemb_prefetch(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    cache: Optional[Cache],
    storage: Storage,
    feature_offsets: torch.Tensor,
    initializer: MultiTableInitializer,
    forward_stream: Optional[torch.cuda.Stream] = None,
    evict_strategy=None,
    frequency_counters: Optional[torch.Tensor] = None,
    admitter: Optional[MultiTableAdmitter] = None,
    external_storage: Optional[Storage] = None,
    scores: Optional[torch.Tensor] = None,
    table_gate: Optional[Callable[[], ContextManager]] = None,
) -> PrefetchState:
    """Unified prefetch for all storage types (cache, HBM-direct, generic).

    Returns a PrefetchState containing unique embeddings + optimizer states
    and the metadata needed by forward/backward.

    The host reads the number of unique keys only for a table that can still grow,
    which needs it to decide whether to expand.

    ``scores`` are one int64 per key, for tables that take a score per key. They
    follow the keys through the dedup on a channel of their own, beside the
    frequency counters: where a key repeats in the batch, the largest of its
    scores is the one the table gets.

    ``table_gate`` is what an erase running beside the training takes too (see
    ``async_erase.TableAccessGate``). Prefetch holds it while it reads or changes the
    tables -- from the expansion through the find and the counter increment -- so
    that a key it found is pinned before an erase can look at it.
    """
    with torch.cuda.nvtx.range("dynamicemb_prefetch"):
        assert feature_offsets.numel() > 1
        max_emb_dim = storage.max_embedding_dim()
        max_val_dim = storage.max_value_dim()
        caching = cache is not None

        evict_strat = EvictStrategy(evict_strategy.value) if evict_strategy else None
        # todo: double check
        frequency_counts_int64 = None
        if frequency_counters is not None:
            frequency_counts_int64 = frequency_counters.long()

        if hasattr(storage, "collect_table_sizes"):
            storage.collect_table_sizes(non_blocking=True)

        # The number of unique keys stays on the device. The host reads it (a read that
        # waits for the device) only to decide whether a table has to grow.
        may_grow = (
            hasattr(storage, "expand_if_need")
            and getattr(storage, "can_expand", lambda: True)()
        )

        indices_table_range = get_table_range(offsets, feature_offsets)
        (
            unique_keys,
            reverse_indices,
            unique_indices_table_range,
            lfu_accumulated_frequency,
        ) = segmented_unique(
            indices,
            indices_table_range,
            evict_strat,
            frequency_counts_int64,
        )
        if may_grow:
            # The one read of the step: it waits for the device.
            host_table_offsets = _copy_cuda_tensor_to_pinned_cpu(
                unique_indices_table_range
            )
            host_unique_per_table = host_table_offsets[1:] - host_table_offsets[:-1]
        unique_scores = (
            None
            if scores is None
            else largest_score_per_unique_key(
                scores, reverse_indices, unique_keys.numel()
            )
        )

        # Deduplication keeps input-capacity buffers. The final offset is the
        # live length and stays on the GPU through every unique-row consumer.
        num_unique = unique_indices_table_range[-1:]
        if not caching:
            unique_table_ids = expand_table_ids_cuda(
                unique_indices_table_range, unique_keys.numel()
            )

        slot_indices = None
        non_admitted = None
        value_row_indices = None
        key_map_generation = 0
        if caching:
            storage_mode = StorageMode.CACHE
        elif _is_hbm_storage(storage):
            storage_mode = StorageMode.HBM_DIRECT
        else:
            storage_mode = StorageMode.DEFAULT

        with table_section(table_gate):
            if may_grow:
                storage.expand_if_need(host_unique_per_table)

            if storage_mode == StorageMode.CACHE:
                unique_table_ids, slot_indices, non_admitted = _prefetch_cache_path(
                    cache,
                    storage,
                    unique_keys,
                    unique_indices_table_range,
                    num_unique,
                    max_emb_dim,
                    max_val_dim,
                    initializer,
                    lfu_accumulated_frequency,
                    admitter,
                    external_storage,
                    key_scores=unique_scores,
                )
                # The cache addresses its own value buffer by cache slot: a cache is
                # never NO_EVICTION (it enables overflow, which NO_EVICTION rejects
                # at construction), so the two index spaces coincide here.
                value_row_indices = slot_indices
            elif storage_mode == StorageMode.HBM_DIRECT:
                (
                    slot_indices,
                    non_admitted,
                    value_row_indices,
                ) = _prefetch_hbm_direct_path(
                    storage,
                    unique_keys,
                    unique_table_ids,
                    num_unique,
                    max_emb_dim,
                    max_val_dim,
                    initializer,
                    lfu_accumulated_frequency,
                    admitter,
                    key_scores=unique_scores,
                )
                key_map_generation = storage._state.key_map_generation

        return PrefetchState(
            unique_keys=unique_keys,
            reverse_indices=reverse_indices,
            unique_table_ids=unique_table_ids,
            lfu_accumulated_frequency=lfu_accumulated_frequency,
            slot_indices=slot_indices,
            storage_mode=storage_mode,
            scores_are_frequencies=evict_strat == EvictStrategy.KLfu,
            non_admitted=non_admitted,
            unique_scores=unique_scores,
            value_row_indices=value_row_indices,
            key_map_generation=key_map_generation,
            num_unique=num_unique,
            num_input=indices_table_range[-1:].long(),
        )


def dynamicemb_eval_forward(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    cache: Optional[Cache],
    storage: Storage,
    feature_offsets: torch.Tensor,
    output_dtype: torch.dtype,
    initializer: MultiTableInitializer,
    layout: EmbeddingLayout,
    evict_strategy=None,
    frequency_counters: Optional[torch.Tensor] = None,
    pooling_weights: Optional[torch.Tensor] = None,
    output_alloc: Optional[OutputAlloc] = None,
) -> torch.Tensor:
    """Eval-only forward for all storage configurations (no autograd).

    A sequence lookup's rows go where ``output_alloc`` says, if it has a
    buffer for them (a call on the hierarchical dist whose lookup a pipeline
    runs in eval mode), converted to ``output_dtype`` on the way.
    """
    with torch.cuda.nvtx.range("dynamicemb_eval_forward"):
        assert feature_offsets.numel() > 1

        is_pooling = layout.is_pooling

        evict_strat = EvictStrategy(evict_strategy.value) if evict_strategy else None

        indices_table_range = get_table_range(offsets, feature_offsets)

        if not is_pooling:
            table_ids = expand_table_ids_cuda(
                indices_table_range,
                indices.numel(),
            )
            output_embs = eval_lookup(
                storage,
                indices,
                table_ids,
                initializer,
                cache=cache,
                num_valid=indices_table_range[-1:].long(),
            )
            # Decode first: the allocator's copies below only convert dtypes, so
            # an fp8 table's stored codes (256x the weights) would be copied out.
            # Always: for any other table this is a ``.to`` that returns the
            # tensor itself when the dtype already matches.
            output_embs = decode_table_values(output_embs, output_dtype)
            alloc = (
                output_alloc(*layout.sequence_shape(), output_dtype)
                if output_alloc is not None
                else None
            )
            if alloc is not None:
                out, out_rows = alloc
                if out_rows is None:
                    return out.copy_(output_embs)
                return out.index_copy_(0, out_rows.long(), output_embs.to(out.dtype))
            return output_embs

        frequency_counts_int64 = (
            frequency_counters.long() if frequency_counters is not None else None
        )

        (
            unique_indices,
            reverse_indices,
            unique_indices_table_range,
            lfu_accumulated_frequency,
        ) = segmented_unique(
            indices,
            indices_table_range,
            evict_strat,
            frequency_counts_int64,
        )

        num_unique = unique_indices_table_range[-1:]
        unique_table_ids = expand_table_ids_cuda(
            unique_indices_table_range, unique_indices.numel()
        )

        unique_embs = eval_lookup(
            storage,
            unique_indices,
            unique_table_ids,
            initializer,
            cache=cache,
            num_valid=num_unique,
        )

        output_embs = torch.empty(
            *layout.pooled_shape(),
            dtype=output_dtype,
            device=indices.device,
        )
        gather_embedding_pooled(
            unique_embs,
            output_embs,
            reverse_indices,
            offsets,
            layout.pooling_mode,
            layout.total_D,
            layout.batch_size,
            layout.D_offsets,
            layout.max_D,
            pooling_weights,
            layout.feature_dims_vec4,
        )
        return output_embs


def _generic_forward_path(
    storage: Storage,
    optimizer: BaseDynamicEmbeddingOptimizer,
    unique_keys: torch.Tensor,
    unique_table_ids: torch.Tensor,
    num_unique: torch.Tensor,
    max_emb_dim: int,
    max_val_dim: int,
    emb_dtype: torch.dtype,
    initializer: MultiTableInitializer,
    accumulated_frequency: Optional[torch.Tensor],
    admitter: Optional[MultiTableAdmitter],
    scores_are_frequencies: bool,
    key_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Find, initialize and insert rows for storage without pinned slots.

    When admission can reject keys, backward needs the mask of rows that hit or
    were admitted. Otherwise every live row is written back in its original order.
    """
    with torch.cuda.nvtx.range("_generic_forward_path"):
        if unique_keys.numel() == 0:
            return unique_keys.new_empty((0, max_val_dim), dtype=emb_dtype), None
        with torch.cuda.nvtx.range("op:storage_find"):
            (
                num_missing,
                missing_keys,
                missing_indices,
                missing_table_ids,
                missing_scores,
                founds,
                _,
                unique_values,
            ) = storage.find(
                unique_keys,
                unique_table_ids,
                copy_mode=CopyMode.VALUE,
                lfu_accumulated_frequency=accumulated_frequency,
                key_scores=key_scores,
                num_valid=num_unique,
            )

        freq_for_admission = (
            gather_embedding(
                accumulated_frequency, missing_indices, num_valid=num_missing
            )
            if admitter is not None and accumulated_frequency is not None
            else None
        )
        (
            num_to_insert,
            keys_to_insert,
            scores_to_insert,
            table_ids_to_insert,
            positions_in_unique,
        ) = _apply_admission(
            missing_keys,
            missing_indices,
            missing_table_ids,
            missing_scores,
            unique_values,
            unique_keys,
            unique_table_ids,
            max_emb_dim,
            freq_for_admission,
            admitter,
            num_missing,
            scores_are_frequencies,
        )
        # The table initializes every miss unless admission already initialized
        # the rejected rows. In that case only the rows being inserted remain.
        num_to_init, indices_to_init = num_missing, missing_indices
        if admitter is not None and admitter.non_admitted_initializer is not None:
            num_to_init, indices_to_init = num_to_insert, positions_in_unique
        with torch.cuda.nvtx.range("op:initializer"):
            initializer(
                unique_values[:, :max_emb_dim],
                unique_keys,
                num_to_init,
                unique_table_ids,
                indices_to_init,
            )
        if max_val_dim != max_emb_dim:
            optimizer.reset_optimizer_states(
                unique_values[:, max_emb_dim:],
                indices=missing_indices,
                table_ids=unique_table_ids,
                table_emb_dims=storage.embedding_dims(on_device=True),
                num_valid=num_missing,
            )
        with torch.cuda.nvtx.range("op:storage_insert"):
            storage.insert(
                keys_to_insert,
                table_ids_to_insert,
                unique_values,
                scores_to_insert,
                num_valid=num_to_insert,
                input_rows=positions_in_unique,
            )
        if admitter is None:
            return unique_values, None
        # Backward writes storage hits and newly admitted rows. Preserve the
        # original hit mask in case the storage implementation retains it.
        writeback_mask = founds.clone()
        fill_rows(
            writeback_mask, True, indices=positions_in_unique, num_valid=num_to_insert
        )
        return unique_values, writeback_mask


def release_slot_pins(
    cache: Optional[Cache],
    storage: Storage,
    storage_mode: StorageMode,
    slot_indices: torch.Tensor,
    unique_table_ids: torch.Tensor,
    *,
    num_valid: torch.Tensor,
) -> None:
    """Undo the ref-counter pins that prefetch took on ``slot_indices``.

    The counter sits beside the hash table of whichever of ``cache`` / ``storage``
    served the batch, and is indexed by hash slot (not by value row). Called by
    backward, or by a forward that has no backward (no autograd graph).
    """
    counter_owner = cache if storage_mode == StorageMode.CACHE else storage
    counter_owner.decrement_counter(slot_indices, unique_table_ids, num_valid=num_valid)


class DynamicEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        prefetch_state: PrefetchState,
        offsets: torch.Tensor,
        cache: Optional[Cache],
        storage: Storage,
        output_dtype: torch.dtype,
        initializer: MultiTableInitializer,
        optimizer: BaseDynamicEmbeddingOptimizer,
        layout: EmbeddingLayout,
        admitter=None,
        pooling_weights: Optional[torch.Tensor] = None,
        # Not read: a parameter that requires grad, so that autograd records the
        # op and runs its backward.
        empty_tensor: Optional[torch.Tensor] = None,
        # Allocates the sequence output; an output dist may provide it (the
        # hierarchical dist puts it in a window its node peers read in place,
        # each key's row where its row map says).
        output_alloc: Optional[OutputAlloc] = None,
    ):
        with torch.cuda.nvtx.range("DynamicEmbeddingFunction.forward"):
            max_emb_dim = storage.max_embedding_dim()
            max_val_dim = storage.max_value_dim()
            emb_dtype = storage.embedding_dtype()

            # pooling_weights is validated by the sole caller,
            # BatchedDynamicEmbeddingTablesV2.forward (pooling mode, dtype,
            # numel, and exclusivity with frequency_counters), and again by
            # TORCH_CHECK in gather_embedding_pooled / reduce_grads.
            is_pooling = layout.is_pooling

            use_counter = prefetch_state.slot_indices is not None
            # A sequence lookup whose rows all come from the table, at its own
            # dtype, loads each key's row straight into the output: no unique
            # rows to write and read back in between.
            direct = False
            num_unique = prefetch_state.num_unique
            writeback_mask = None
            if use_counter:
                state = (
                    cache._state
                    if prefetch_state.storage_mode == StorageMode.CACHE
                    else storage._state
                )
                direct = (
                    not is_pooling
                    and prefetch_state.non_admitted is None
                    and output_dtype == state.emb_dtype
                )
                if not direct:
                    with torch.cuda.nvtx.range("op:load_from_flat"):
                        unique_embs = load_from_flat(
                            state,
                            # Value buffer: address by row (the slot, for a
                            # cache).
                            prefetch_state.value_row_indices,
                            prefetch_state.unique_table_ids,
                            copy_mode=CopyMode.EMBEDDING,
                            num_valid=num_unique,
                        )
                if prefetch_state.non_admitted is not None:
                    rejected_positions, num_rejected = prefetch_state.non_admitted
                    row_initializer = (
                        admitter.non_admitted_initializer or initializer
                        if admitter is not None
                        else initializer
                    )
                    with torch.cuda.nvtx.range("op:initializer"):
                        # Whichever initializer owns rejected rows was settled
                        # at construction, so this is one call either way.
                        row_initializer(
                            unique_embs[:, :max_emb_dim],
                            prefetch_state.unique_keys,
                            num_rejected,
                            prefetch_state.unique_table_ids,
                            rejected_positions,
                        )
                unique_values = None
            else:
                assert prefetch_state.storage_mode == StorageMode.DEFAULT
                unique_values, writeback_mask = _generic_forward_path(
                    storage,
                    optimizer,
                    prefetch_state.unique_keys,
                    prefetch_state.unique_table_ids,
                    num_unique,
                    max_emb_dim,
                    max_val_dim,
                    emb_dtype,
                    initializer,
                    prefetch_state.lfu_accumulated_frequency,
                    admitter,
                    scores_are_frequencies=prefetch_state.scores_are_frequencies,
                    key_scores=prefetch_state.unique_scores,
                )
                # find() was asked for CopyMode.VALUE, so a row here is
                # embedding ++ optimizer state; keep only the embedding.
                unique_embs = unique_values[:, :max_emb_dim]

            device = prefetch_state.unique_keys.device
            out_rows = None  # each key's output row, None: key i's is row i
            if is_pooling:
                output_embs = torch.empty(
                    *layout.pooled_shape(),
                    dtype=output_dtype,
                    device=device,
                )
                with torch.cuda.nvtx.range("op:gather_embedding"):
                    gather_embedding_pooled(
                        unique_embs,
                        output_embs,
                        prefetch_state.reverse_indices,
                        offsets,
                        layout.pooling_mode,
                        layout.total_D,
                        layout.batch_size,
                        layout.D_offsets,
                        layout.max_D,
                        pooling_weights,
                        layout.feature_dims_vec4,
                    )
            else:
                alloc = (
                    output_alloc(*layout.sequence_shape(), output_dtype)
                    if output_alloc is not None
                    else None
                )
                # None: no allocator, or it has no buffer for this batch.
                if alloc is None:
                    output_embs = torch.empty(
                        *layout.sequence_shape(),
                        dtype=output_dtype,
                        device=device,
                    )
                else:
                    output_embs, out_rows = alloc
                if direct:
                    with torch.cuda.nvtx.range("op:load_from_flat"):
                        # The reverse map has one entry per original input,
                        # including duplicates. Its live count is num_input.
                        load_from_flat_by(
                            state,
                            prefetch_state.value_row_indices,
                            prefetch_state.unique_table_ids,
                            prefetch_state.reverse_indices,
                            output_embs,
                            out_rows,
                            num_valid=prefetch_state.num_input,
                        )
                else:
                    with torch.cuda.nvtx.range("op:gather_embedding"):
                        gather_embedding(
                            unique_embs,
                            output_embs,
                            prefetch_state.reverse_indices,
                            out_rows,
                            num_valid=prefetch_state.num_input,
                        )

            ctx.unique_keys = prefetch_state.unique_keys
            ctx.reverse_indices = prefetch_state.reverse_indices
            # The backward reads key i's gradient from the row its output took.
            ctx.out_rows = out_rows
            ctx.unique_table_ids = prefetch_state.unique_table_ids
            ctx.cache = cache
            ctx.storage = storage
            ctx.slot_indices = prefetch_state.slot_indices
            # Same split as the forward: the fused optimizer writes into the
            # value buffer (rows), the ref counter is indexed by hash slot.
            ctx.update_value_rows = prefetch_state.value_row_indices
            ctx.key_map_generation = prefetch_state.key_map_generation
            ctx.storage_mode = prefetch_state.storage_mode
            ctx.optimizer = optimizer
            ctx.layout = layout
            ctx.offsets = offsets
            ctx.max_emb_dim = max_emb_dim
            ctx.max_val_dim = max_val_dim
            ctx.emb_dtype = emb_dtype
            ctx.pooling_weights = pooling_weights
            ctx.use_counter = use_counter
            ctx.unique_values = unique_values
            ctx.writeback_mask = writeback_mask
            ctx.num_unique = num_unique

            return output_embs

    @staticmethod
    def backward(ctx, grads):
        with torch.cuda.nvtx.range("DynamicEmbeddingFunction.backward"):
            cache = ctx.cache
            storage = ctx.storage
            optimizer = ctx.optimizer
            grads = grads.contiguous()

            if optimizer.need_gradient_clipping():
                optimizer.clip_gradient(grads)

            layout = ctx.layout
            num_unique = ctx.num_unique
            if layout.is_pooling:
                with torch.cuda.nvtx.range("op:reduce_grads"):
                    unique_grads = reduce_grads(
                        ctx.reverse_indices,
                        grads,
                        ctx.unique_keys.numel(),
                        layout.batch_size,
                        ctx.max_emb_dim,
                        ctx.offsets,
                        layout.D_offsets,
                        layout.pooling_mode,
                        layout.total_D,
                        ctx.pooling_weights,
                        layout.feature_dims_vec4,
                    )
            else:
                with torch.cuda.nvtx.range("op:reduce_grads"):
                    unique_grads = reduce_grads(
                        ctx.reverse_indices,
                        grads,
                        ctx.unique_keys.numel(),
                        layout.batch_size,
                        ctx.max_emb_dim,
                        grad_rows=ctx.out_rows,
                    )

            optimizer.step()

            unique_table_ids = ctx.unique_table_ids

            with torch.cuda.nvtx.range("DynamicEmbeddingFunction.update"):
                if ctx.use_counter:
                    state = (
                        cache._state
                        if ctx.storage_mode == StorageMode.CACHE
                        else storage._state
                    )
                    # A later batch's prefetch may have rehashed the table
                    # since this one's prefetch (prefetch pipelines, several
                    # forwards before their backwards).
                    refresh_slots_after_rehash(state, ctx)
                    with torch.cuda.nvtx.range("op:optimizer_update_fused"):
                        optimizer.fused_update_for_flat_table(
                            unique_grads.to(
                                gradient_dtype(ctx.emb_dtype, unique_grads.dtype)
                            ),
                            # Writes into the value buffer -> rows.
                            ctx.update_value_rows,
                            state.table_ptrs_dev,
                            unique_table_ids,
                            state.table_value_dims,
                            state.table_emb_dims,
                            state.max_emb_dim,
                            state.all_dims_vec4,
                            state.emb_dtype,
                            num_valid=num_unique,
                        )

                    release_slot_pins(
                        cache,
                        storage,
                        ctx.storage_mode,
                        ctx.slot_indices,
                        ctx.unique_table_ids,
                        num_valid=num_unique,
                    )

                if not ctx.use_counter and ctx.unique_values is not None:
                    # Per-table embedding dims (on the CUDA device) let the
                    # padded-buffer kernel place each row's optimizer state at the
                    # max_emb_dim offset using the row's own width. Provided
                    # uniformly across storage types via embedding_dims().
                    table_emb_dims = storage.embedding_dims(on_device=True)
                    # all_dims_vec4 must reflect per-table dims, not just
                    # max_emb_dim: a vec4 store on a row whose edim % 4 != 0 would
                    # run past its m region into v. storage.all_dims_vec4() is the
                    # per-table-correct flag used by load/store too.
                    all_dims_vec4 = storage.all_dims_vec4()
                    with torch.cuda.nvtx.range("op:optimizer_update_padded"):
                        optimizer.update_for_padded_buffer(
                            unique_grads.to(
                                gradient_dtype(ctx.emb_dtype, unique_grads.dtype)
                            ),
                            ctx.unique_values,
                            unique_table_ids,
                            table_emb_dims,
                            ctx.max_emb_dim,
                            ctx.max_val_dim,
                            all_dims_vec4,
                            num_valid=num_unique,
                        )
                    keys, tids = ctx.unique_keys, unique_table_ids
                    count, input_rows = num_unique, None
                    if ctx.writeback_mask is not None:
                        count, input_rows, (keys, tids) = flagged_compact(
                            ctx.writeback_mask, [keys, tids], num_unique
                        )
                    # preserve_existing=True → CONST score policy on insert:
                    # refresh embedding/optimizer buffers only; do not overwrite
                    # table scores (LFU / timestamp / etc.). For HybridStorage,
                    # the HBM tier uses this on backward; existing scores stay.
                    with torch.cuda.nvtx.range("op:storage_insert"):
                        storage.insert(
                            keys,
                            tids,
                            ctx.unique_values,
                            preserve_existing=True,
                            num_valid=count,
                            input_rows=input_rows,
                        )

            return (None,) * 12
