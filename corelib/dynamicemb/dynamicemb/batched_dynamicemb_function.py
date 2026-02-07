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
from typing import Callable, List, Optional, Tuple, Union

import torch
from dynamicemb.dynamicemb_config import (
    DynamicEmbInitializerArgs,
    DynamicEmbPoolingMode,
    dyn_emb_to_torch,
)
from dynamicemb.initializer import BaseDynamicEmbInitializer
from dynamicemb.key_value_table import (
    Cache,
    KeyValueTableCachingFunction,
    KeyValueTableFunction,
    Storage,
)
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer, BaseDynamicEmbeddingOptimizerV2
from dynamicemb.scored_hashtable import (
    GroupedLinearBucketTable,
    GroupedScoredHashTable,
    ScoreArg,
    ScorePolicy,
    ScoreSpec,
    ScoredHashTable,
)
from dynamicemb.types import AdmissionStrategy, Counter
from dynamicemb_extensions import (
    DynamicEmbTable,
    EvictStrategy,
    find_and_initialize,
    find_or_insert,
    gather_embedding,
    get_table_range,
    load_from_combined_table,
    lookup_backward,
    lookup_forward,
    reduce_grads,
    segmented_unique,
    select,
    select_index,
    store_to_combined_table,
)

# =============================================================================
# Data classes for hash table operation results
# =============================================================================


@dataclass
class HashTableLookupResult:
    """Result of a hash table lookup operation."""

    founds: torch.Tensor  # Boolean mask indicating which keys were found
    indices: torch.Tensor  # Indices in the embedding table for found keys
    h_num_missing: int  # Number of missing keys (host value)
    missing_keys: torch.Tensor  # Keys that were not found
    missing_indices: torch.Tensor  # Original positions of missing keys in input
    missing_scores: Optional[torch.Tensor]  # Scores for missing keys (if applicable)


@dataclass
class HashTableInsertResult:
    """Result of a hash table insert operation."""

    indices: torch.Tensor  # Indices where keys were inserted
    num_evicted: int  # Number of evicted keys (host value)
    evicted_keys: Optional[torch.Tensor]  # Keys that were evicted
    evicted_indices: Optional[torch.Tensor]  # Indices of evicted keys
    evicted_scores: Optional[torch.Tensor]  # Scores of evicted keys


@dataclass
class EmbeddingTable:
    """
    Unified physical storage for embeddings.
    
    A single EmbeddingTable can store embeddings for multiple logical tables.
    Each logical table has a corresponding hash table that maps keys to local indices,
    and table_offsets maps local indices to global indices in this unified storage.
    
    When caching is enabled:
    - dev_table: Cache storage (HBM only, fast access)
    - uvm_table: Main storage (UVM/DRAM, slower but larger capacity)
    
    When caching is disabled:
    - dev_table: Main storage if using HBM
    - uvm_table: Main storage if using UVM/DRAM
    """

    dev_table: Optional[torch.Tensor]  # HBM storage (cache when caching enabled, or main storage)
    uvm_table: Optional[torch.Tensor]  # UVM storage (main storage when caching enabled)
    emb_dtype: torch.dtype  # Embedding data type
    emb_dim: int  # Embedding dimension
    val_dim: int  # Total value dimension (emb_dim + optimizer state dim)
    init_optimizer_state: float  # Initial optimizer state value


# =============================================================================
# Hash table operations (scored_hashtable related)
# =============================================================================


def hash_table_lookup(
    key_index_map: ScoredHashTable,
    keys: torch.Tensor,
    score_policy: ScoreSpec,
    scores: Optional[torch.Tensor],
    score_update: bool,
) -> HashTableLookupResult:
    """
    Perform a lookup operation on the hash table.

    Args:
        key_index_map: The scored hash table (key -> index mapping)
        keys: Keys to look up
        score_policy: Score policy specification
        scores: Score values for the keys (optional, depends on evict strategy)
        score_update: Whether to update scores during lookup

    Returns:
        HashTableLookupResult containing lookup results
    """
    batch = keys.numel()
    device = keys.device

    if batch == 0:
        return HashTableLookupResult(
            founds=torch.empty(0, dtype=torch.bool, device=device),
            indices=torch.empty(0, dtype=key_index_map.index_type, device=device),
            h_num_missing=0,
            missing_keys=torch.empty_like(keys),
            missing_indices=torch.empty(0, dtype=torch.long, device=device),
            missing_scores=None,
        )

    founds = torch.empty(batch, dtype=torch.bool, device=device)
    indices = torch.empty(batch, dtype=key_index_map.index_type, device=device)

    # Build score args for lookup
    score_args_lookup = [
        ScoreArg(
            name=score_policy.name,
            value=scores,
            policy=score_policy.policy if score_update else ScorePolicy.CONST,
            is_return=scores is not None,
        )
    ]

    key_index_map.lookup(keys, score_args_lookup, founds, indices)

    # Select missing keys
    missing = torch.logical_not(founds)
    num_missing_0 = torch.empty(1, dtype=torch.long, device=device)
    num_missing_1 = torch.empty(1, dtype=torch.long, device=device)
    missing_keys = torch.empty_like(keys)
    missing_indices = torch.empty(batch, dtype=torch.long, device=device)

    select(missing, keys, missing_keys, num_missing_0)
    select_index(missing, missing_indices, num_missing_1)

    h_num_missing = num_missing_0.cpu().item()

    # Get missing scores if applicable
    missing_scores = None
    if scores is not None and h_num_missing > 0:
        missing_scores = scores[missing_indices[:h_num_missing]]

    return HashTableLookupResult(
        founds=founds,
        indices=indices,
        h_num_missing=h_num_missing,
        missing_keys=missing_keys[:h_num_missing],
        missing_indices=missing_indices[:h_num_missing],
        missing_scores=missing_scores,
    )


def hash_table_insert(
    key_index_map: ScoredHashTable,
    keys: torch.Tensor,
    score_policy: ScoreSpec,
    scores: Optional[torch.Tensor],
    policy_override: Optional[ScorePolicy] = None,
) -> HashTableInsertResult:
    """
    Insert keys into the hash table.

    Args:
        key_index_map: The scored hash table
        keys: Keys to insert
        score_policy: Score policy specification
        scores: Score values for the keys
        policy_override: Optional override for the score policy

    Returns:
        HashTableInsertResult containing insert results (no eviction info)
    """
    h_num_keys = keys.numel()
    device = keys.device

    if h_num_keys == 0:
        return HashTableInsertResult(
            indices=torch.empty(0, dtype=key_index_map.index_type, device=device),
            num_evicted=0,
            evicted_keys=None,
            evicted_indices=None,
            evicted_scores=None,
        )

    policy = policy_override if policy_override is not None else score_policy.policy

    score_args_insert = [
        ScoreArg(
            name=score_policy.name,
            value=scores,
            policy=policy,
            is_return=False,
        )
    ]

    indices = torch.zeros(h_num_keys, dtype=key_index_map.index_type, device=device)
    key_index_map.insert(keys, score_args_insert, indices)

    return HashTableInsertResult(
        indices=indices,
        num_evicted=0,
        evicted_keys=None,
        evicted_indices=None,
        evicted_scores=None,
    )


def hash_table_insert_and_evict(
    key_index_map: ScoredHashTable,
    keys: torch.Tensor,
    score_policy: ScoreSpec,
    scores: Optional[torch.Tensor],
) -> HashTableInsertResult:
    """
    Insert keys into the hash table with eviction support.

    Args:
        key_index_map: The scored hash table
        keys: Keys to insert
        score_policy: Score policy specification
        scores: Score values for the keys

    Returns:
        HashTableInsertResult containing insert and eviction results
    """
    h_num_keys = keys.numel()
    device = keys.device

    if h_num_keys == 0:
        return HashTableInsertResult(
            indices=torch.empty(0, dtype=key_index_map.index_type, device=device),
            num_evicted=0,
            evicted_keys=None,
            evicted_indices=None,
            evicted_scores=None,
        )

    score_args_insert = [
        ScoreArg(
            name=score_policy.name,
            value=scores,
            policy=score_policy.policy,
            is_return=False,
        )
    ]

    indices = torch.zeros(h_num_keys, dtype=key_index_map.index_type, device=device)

    (
        num_evicted,
        evicted_keys,
        evicted_indices,
        evicted_scores_list,
    ) = key_index_map.insert_and_evict(keys, score_args_insert, indices)

    evicted_scores = evicted_scores_list[0] if evicted_scores_list else None

    return HashTableInsertResult(
        indices=indices,
        num_evicted=num_evicted,
        evicted_keys=evicted_keys,
        evicted_indices=evicted_indices,
        evicted_scores=evicted_scores,
    )


# =============================================================================
# Embedding copy operations (dev_table/uvm_table related)
# =============================================================================


def copy_embeddings_by_indices(
    dev_table: Optional[torch.Tensor],
    uvm_table: Optional[torch.Tensor],
    indices: torch.Tensor,
    output_tensor: torch.Tensor,
) -> None:
    """
    Copy embeddings from dev_table/uvm_table to output_tensor using indices.

    Args:
        dev_table: HBM storage tensor (can be None)
        uvm_table: UVM storage tensor (can be None)
        indices: Indices to copy from
        output_tensor: Output tensor to copy to
    """
    if indices.numel() == 0:
        return
    load_from_combined_table(dev_table, uvm_table, indices, output_tensor)


def store_embeddings_by_indices(
    dev_table: Optional[torch.Tensor],
    uvm_table: Optional[torch.Tensor],
    indices: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """
    Store embeddings to dev_table/uvm_table at specified indices.

    Args:
        dev_table: HBM storage tensor (can be None)
        uvm_table: UVM storage tensor (can be None)
        indices: Indices to store at
        values: Values to store
    """
    if indices.numel() == 0:
        return
    store_to_combined_table(dev_table, uvm_table, indices, values)


# =============================================================================
# Result dataclasses for hash_table_processing
# =============================================================================


@dataclass
class HashTableProcessingResult:
    """Result from hash_table_processing function."""

    # Indices for all input keys (for found keys, points to valid location)
    indices: torch.Tensor

    # Information about found keys
    founds: torch.Tensor  # Boolean mask

    # Information about missing keys
    h_num_missing: int
    missing_keys: torch.Tensor
    missing_indices: torch.Tensor  # Indices in input keys
    missing_scores: Optional[torch.Tensor]

    # Indices where missing keys will be inserted (for training)
    # This is the location in the table where new embeddings should be stored
    insert_indices: Optional[torch.Tensor]

    # Keys/indices that need initialization (after admission filtering)
    indices_to_init: torch.Tensor  # Indices in input keys that need initialization

    # Admission info
    admit_mask: Optional[torch.Tensor]  # Mask for admitted keys among missing keys


@dataclass
class HashTableProcessingResultWithCache:
    """Result from hash_table_processing_with_cache function."""

    # Cache lookup results
    cache_founds: torch.Tensor
    cache_indices: torch.Tensor

    # Cache-missed keys info
    h_num_cache_missing: int
    cache_missing_keys: torch.Tensor
    cache_missing_indices: torch.Tensor  # Indices in input keys
    cache_missing_scores: Optional[torch.Tensor]

    # Storage lookup results (for cache-missed keys)
    storage_founds: torch.Tensor  # Relative to cache-missed keys
    storage_indices: torch.Tensor  # Indices in storage table

    # Storage-missed keys info (subset of cache-missed)
    h_num_storage_missing: int
    storage_missing_keys: torch.Tensor
    storage_missing_indices: torch.Tensor  # Indices in cache-missed keys

    # Indices that need initialization (after admission filtering)
    indices_to_init: torch.Tensor  # Indices in cache-missed keys that need initialization

    # Keys/values to insert into cache
    keys_to_cache: torch.Tensor
    cache_insert_indices: torch.Tensor  # Where to store in cache
    scores_to_cache: Optional[torch.Tensor]

    # Evicted from cache (to be stored in storage)
    num_evicted: int
    evicted_keys: Optional[torch.Tensor]
    evicted_cache_indices: Optional[torch.Tensor]
    evicted_scores: Optional[torch.Tensor]
    evicted_storage_indices: Optional[torch.Tensor]

    # Admission info
    admit_mask: Optional[torch.Tensor]


@dataclass
class HashTableOutput:
    """
    Complete output from hash_table_processing functions.
    
    Includes per-table processing results and deduplication context needed
    for embedding operations and backward pass.
    """
    # Per-table processing results (with global indices)
    results: List[HashTableProcessingResult]
    
    # Deduplication context
    unique_indices: torch.Tensor  # Unique indices after deduplication
    inverse: Optional[torch.Tensor]  # Maps original indices to unique indices (None if no dedup)
    h_unique_indices_table_range: torch.Tensor  # CPU tensor with table boundaries for unique indices
    
    # Original indices context (for backward pass)
    indices_table_range: torch.Tensor  # Table boundaries for original indices


@dataclass
class HashTableOutputWithCache:
    """
    Complete output from hash_table_processing_with_cache functions.
    
    Includes per-table processing results and deduplication context needed
    for embedding operations and backward pass.
    """
    # Per-table processing results (with global indices)
    results: List[HashTableProcessingResultWithCache]
    
    # Deduplication context
    unique_indices: torch.Tensor  # Unique indices after deduplication
    inverse: torch.Tensor  # Maps original indices to unique indices
    h_unique_indices_table_range: torch.Tensor  # CPU tensor with table boundaries for unique indices
    unique_indices_table_range: torch.Tensor  # GPU tensor with table boundaries
    
    # Original indices context (for backward pass)
    indices_table_range: torch.Tensor  # Table boundaries for original indices


# =============================================================================
# Index mapping utilities for unified storage
# =============================================================================


def _local_to_global_indices(
    local_indices: torch.Tensor,
    table_offset: int,
) -> torch.Tensor:
    """
    Convert local indices from a logical table to global indices in unified storage.

    Args:
        local_indices: Local indices from hash table lookup/insert
        table_offset: Offset for this table in the global index space

    Returns:
        Global indices in the unified storage
    """
    return local_indices + table_offset


def _apply_table_offset_to_result(
    result: HashTableProcessingResult,
    table_offset: int,
) -> HashTableProcessingResult:
    """
    Apply table offset to convert local indices to global indices in a result.

    Args:
        result: Hash table processing result with local indices
        table_offset: Offset for this table in the global index space

    Returns:
        Result with global indices
    """
    return HashTableProcessingResult(
        indices=result.indices + table_offset,
        founds=result.founds,
        h_num_missing=result.h_num_missing,
        missing_keys=result.missing_keys,
        missing_indices=result.missing_indices,
        missing_scores=result.missing_scores,
        insert_indices=(
            result.insert_indices + table_offset
            if result.insert_indices is not None
            else None
        ),
        indices_to_init=result.indices_to_init,
        admit_mask=result.admit_mask,
    )


def _apply_table_offset_to_cache_result(
    result: HashTableProcessingResultWithCache,
    cache_table_offset: int,
    storage_table_offset: int,
) -> HashTableProcessingResultWithCache:
    """
    Apply table offsets to convert local indices to global indices in a cache result.

    Args:
        result: Hash table processing result with local indices
        cache_table_offset: Offset for this table in the cache's global index space
        storage_table_offset: Offset for this table in the storage's global index space

    Returns:
        Result with global indices
    """
    return HashTableProcessingResultWithCache(
        cache_founds=result.cache_founds,
        cache_indices=result.cache_indices + cache_table_offset,
        h_num_cache_missing=result.h_num_cache_missing,
        cache_missing_keys=result.cache_missing_keys,
        cache_missing_indices=result.cache_missing_indices,
        cache_missing_scores=result.cache_missing_scores,
        storage_founds=result.storage_founds,
        storage_indices=result.storage_indices + storage_table_offset,
        h_num_storage_missing=result.h_num_storage_missing,
        storage_missing_keys=result.storage_missing_keys,
        storage_missing_indices=result.storage_missing_indices,
        indices_to_init=result.indices_to_init,
        keys_to_cache=result.keys_to_cache,
        cache_insert_indices=(
            result.cache_insert_indices + cache_table_offset
            if result.cache_insert_indices is not None
            else None
        ),
        scores_to_cache=result.scores_to_cache,
        num_evicted=result.num_evicted,
        evicted_keys=result.evicted_keys,
        evicted_cache_indices=(
            result.evicted_cache_indices + cache_table_offset
            if result.evicted_cache_indices is not None
            else None
        ),
        evicted_scores=result.evicted_scores,
        evicted_storage_indices=(
            result.evicted_storage_indices + storage_table_offset
            if result.evicted_storage_indices is not None
            else None
        ),
        admit_mask=result.admit_mask,
    )


# =============================================================================
# Hash table processing functions (no embedding operations)
# =============================================================================


def _process_single_table_no_cache(
    key_index_map: ScoredHashTable,
    unique_keys: torch.Tensor,
    score_policy: ScoreSpec,
    scores: Optional[torch.Tensor],
    score_update: bool,
    training: bool,
    admit_strategy: Optional[AdmissionStrategy],
    admission_counter: Optional[Counter],
    accumulated_frequency: Optional[torch.Tensor],
    expand_fn: Optional[Callable],
) -> HashTableProcessingResult:
    """Process a single table (no cache). Internal helper."""
    h_num_total = unique_keys.numel()
    device = unique_keys.device

    if h_num_total == 0:
        return HashTableProcessingResult(
            indices=torch.empty(0, dtype=key_index_map.index_type, device=device),
            founds=torch.empty(0, dtype=torch.bool, device=device),
            h_num_missing=0,
            missing_keys=torch.empty_like(unique_keys),
            missing_indices=torch.empty(0, dtype=torch.long, device=device),
            missing_scores=None,
            insert_indices=None,
            indices_to_init=torch.empty(0, dtype=torch.long, device=device),
            admit_mask=None,
        )

    # Optional: expand table if needed
    if expand_fn is not None:
        expand_fn()

    # 1. Hash table lookup
    lookup_result = hash_table_lookup(
        key_index_map, unique_keys, score_policy, scores, score_update
    )

    if lookup_result.h_num_missing == 0:
        return HashTableProcessingResult(
            indices=lookup_result.indices,
            founds=lookup_result.founds,
            h_num_missing=0,
            missing_keys=lookup_result.missing_keys,
            missing_indices=lookup_result.missing_indices,
            missing_scores=lookup_result.missing_scores,
            insert_indices=None,
            indices_to_init=torch.empty(0, dtype=torch.long, device=device),
            admit_mask=None,
        )

    # 2. Handle admission if configured
    admit_mask = None
    indices_to_init = lookup_result.missing_indices
    keys_to_insert = lookup_result.missing_keys
    scores_to_insert = lookup_result.missing_scores

    if training and admit_strategy is not None and admission_counter is not None:
        # Get frequency counters for admission
        if accumulated_frequency is not None:
            counters_for_admission = accumulated_frequency[lookup_result.missing_indices]
        else:
            counters_for_admission = torch.ones(
                lookup_result.h_num_missing, dtype=torch.int64, device=device
            )

        admit_mask = _do_admission(
            lookup_result.missing_keys,
            counters_for_admission,
            admit_strategy,
            admission_counter,
        )

        # Update indices to init based on admission
        indices_to_init = lookup_result.missing_indices[admit_mask]
        keys_to_insert = lookup_result.missing_keys[admit_mask]
        if scores_to_insert is not None:
            scores_to_insert = scores_to_insert[admit_mask]

    # 3. Insert missing keys into hash table (training only)
    insert_indices = None
    if training and keys_to_insert.numel() > 0:
        insert_result = hash_table_insert(
            key_index_map,
            keys_to_insert,
            score_policy,
            scores_to_insert,
            policy_override=ScorePolicy.ASSIGN,
        )
        insert_indices = insert_result.indices

    return HashTableProcessingResult(
        indices=lookup_result.indices,
        founds=lookup_result.founds,
        h_num_missing=lookup_result.h_num_missing,
        missing_keys=lookup_result.missing_keys,
        missing_indices=lookup_result.missing_indices,
        missing_scores=lookup_result.missing_scores,
        insert_indices=insert_indices,
        indices_to_init=indices_to_init,
        admit_mask=admit_mask,
    )


def hash_table_processing_no_cache(
    hash_table: GroupedScoredHashTable,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    feature_offsets: torch.Tensor,
    evict_strategy: Optional[EvictStrategy],
    score_update: bool,
    training: bool,
    frequency_counters: Optional[torch.Tensor] = None,
    admit_strategy: Optional[AdmissionStrategy] = None,
    admission_counters: Optional[List[Counter]] = None,
) -> HashTableOutput:
    """
    Process keys through hash tables for multiple logical tables (no caching).
    
    Includes deduplication of indices and hash table operations.
    Does NOT perform any embedding operations.

    Uses unified storage design: multiple logical tables share a single physical
    storage tensor. Each table has its own hash table (key_index_map) that returns
    local indices. These are converted to global indices using table_offsets.

    Args:
        hash_table: GroupedScoredHashTable containing all logical tables' hash tables and offsets
        indices: Raw input indices tensor
        offsets: Offsets tensor for features
        feature_offsets: Feature offset boundaries
        evict_strategy: Eviction strategy
        score_update: Whether to update scores during lookup
        training: Whether in training mode
        frequency_counters: Optional frequency counters for LFU
        admit_strategy: Optional admission strategy
        admission_counters: Optional admission counters per table

    Returns:
        HashTableOutput containing per-table results and deduplication context
    """
    # Convert frequency counters to int64 if provided
    frequency_counts_int64 = None
    if frequency_counters is not None:
        frequency_counts_int64 = frequency_counters.long()

    # Get table range for indices
    indices_table_range = get_table_range(offsets, feature_offsets)

    # Deduplication
    if training:
        (
            unique_indices,
            inverse,
            unique_indices_table_range,
            h_unique_indices_table_range,
            lfu_accumulated_frequency,
        ) = segmented_unique(
            indices,
            indices_table_range,
            evict_strategy,
            frequency_counts_int64,
        )
    else:
        # Inference without dedup
        h_unique_indices_table_range = indices_table_range.cpu()
        unique_indices = indices
        inverse = None
        lfu_accumulated_frequency = None

    # Process each table
    num_tables = hash_table.num_tables
    results = []

    for i in range(num_tables):
        begin = h_unique_indices_table_range[i]
        end = h_unique_indices_table_range[i + 1]
        unique_indices_per_table = unique_indices[begin:end]

        lfu_accumulated_frequency_per_table = (
            lfu_accumulated_frequency[begin:end]
            if lfu_accumulated_frequency is not None
            and lfu_accumulated_frequency.numel() > 0
            else None
        )

        # Create scores based on evict strategy
        scores = _create_scores_for_table(
            hash_table,
            unique_indices_per_table.numel(),
            unique_indices_per_table.device,
            evict_strategy,
            lfu_accumulated_frequency_per_table,
        )

        expand_fn = (
            hash_table.get_expand_fn(i)
            if hash_table.expand_fns
            else None
        )

        # Process through this table's hash table (returns local indices)
        result = _process_single_table_no_cache(
            key_index_map=hash_table.get_table_by_idx(i),
            unique_keys=unique_indices_per_table,
            score_policy=hash_table.score_policy,
            scores=scores,
            score_update=score_update,
            training=training,
            admit_strategy=admit_strategy,
            admission_counter=admission_counters[i] if admission_counters else None,
            accumulated_frequency=lfu_accumulated_frequency_per_table,
            expand_fn=expand_fn,
        )

        # Apply table offset to convert local indices to global indices
        table_offset = hash_table.get_table_offset(i)
        result = _apply_table_offset_to_result(result, table_offset)

        results.append(result)

    return HashTableOutput(
        results=results,
        unique_indices=unique_indices,
        inverse=inverse,
        h_unique_indices_table_range=h_unique_indices_table_range,
        indices_table_range=indices_table_range,
    )


def _process_single_table_with_cache(
    cache_key_index_map: ScoredHashTable,
    storage_key_index_map: ScoredHashTable,
    unique_keys: torch.Tensor,
    score_policy: ScoreSpec,
    scores: Optional[torch.Tensor],
    score_update: bool,
    training: bool,
    admit_strategy: Optional[AdmissionStrategy],
    admission_counter: Optional[Counter],
    accumulated_frequency: Optional[torch.Tensor],
    cache_expand_fn: Optional[Callable],
    storage_expand_fn: Optional[Callable],
) -> HashTableProcessingResultWithCache:
    """Process a single table with cache. Internal helper."""
    h_num_total = unique_keys.numel()
    device = unique_keys.device

    # Empty input case
    if h_num_total == 0:
        return _create_empty_cache_result(cache_key_index_map, unique_keys, device)

    # 1. Cache lookup
    cache_lookup_result = hash_table_lookup(
        cache_key_index_map, unique_keys, score_policy, scores, score_update
    )

    # All found in cache
    if cache_lookup_result.h_num_missing == 0:
        return _create_all_cache_hit_result(cache_lookup_result, unique_keys, device)

    # 2. Storage lookup for cache-missed keys
    keys_for_storage = cache_lookup_result.missing_keys
    scores_for_storage = cache_lookup_result.missing_scores

    storage_lookup_result = hash_table_lookup(
        storage_key_index_map,
        keys_for_storage,
        score_policy,
        scores_for_storage,
        score_update,
    )

    # Build storage founds mask
    storage_founds = torch.ones(
        cache_lookup_result.h_num_missing, dtype=torch.bool, device=device
    )
    storage_founds[storage_lookup_result.missing_indices] = False

    # 3. Handle admission for storage-missed keys
    admit_mask = None
    indices_to_init = storage_lookup_result.missing_indices

    if training and admit_strategy is not None and admission_counter is not None:
        if accumulated_frequency is not None:
            indices_in_unique_keys = cache_lookup_result.missing_indices[
                storage_lookup_result.missing_indices
            ]
            counters_for_admission = accumulated_frequency[indices_in_unique_keys]
        else:
            counters_for_admission = torch.ones(
                storage_lookup_result.h_num_missing, dtype=torch.int64, device=device
            )

        admit_mask = _do_admission(
            storage_lookup_result.missing_keys,
            counters_for_admission,
            admit_strategy,
            admission_counter,
        )
        indices_to_init = storage_lookup_result.missing_indices[admit_mask]

    # 4. Determine which keys to cache
    if training:
        if admit_strategy is not None and admit_mask is not None:
            # Include storage-hit + admitted storage-miss
            mask_to_cache = storage_founds.clone()
            admitted_indices = storage_lookup_result.missing_indices[admit_mask]
            mask_to_cache[admitted_indices] = True
        else:
            # All cache-missed keys go to cache
            mask_to_cache = torch.ones(
                cache_lookup_result.h_num_missing, dtype=torch.bool, device=device
            )
    else:
        # Eval: only storage-hit keys
        mask_to_cache = storage_founds

    keys_to_cache = keys_for_storage[mask_to_cache]
    scores_to_cache = (
        scores_for_storage[mask_to_cache] if scores_for_storage is not None else None
    )

    # 5. Insert into cache with eviction
    cache_insert_indices = None
    num_evicted = 0
    evicted_keys = None
    evicted_cache_indices = None
    evicted_scores = None
    evicted_storage_indices = None

    if keys_to_cache.numel() > 0:
        cache_insert_result = hash_table_insert_and_evict(
            cache_key_index_map, keys_to_cache, score_policy, scores_to_cache
        )
        cache_insert_indices = cache_insert_result.indices
        num_evicted = cache_insert_result.num_evicted
        evicted_keys = cache_insert_result.evicted_keys
        evicted_cache_indices = cache_insert_result.evicted_indices
        evicted_scores = cache_insert_result.evicted_scores

        # Insert evicted keys into storage
        if num_evicted > 0:
            storage_insert_result = hash_table_insert(
                storage_key_index_map,
                evicted_keys,
                score_policy,
                evicted_scores,
                policy_override=ScorePolicy.ASSIGN,
            )
            evicted_storage_indices = storage_insert_result.indices

    return HashTableProcessingResultWithCache(
        cache_founds=cache_lookup_result.founds,
        cache_indices=cache_lookup_result.indices,
        h_num_cache_missing=cache_lookup_result.h_num_missing,
        cache_missing_keys=cache_lookup_result.missing_keys,
        cache_missing_indices=cache_lookup_result.missing_indices,
        cache_missing_scores=cache_lookup_result.missing_scores,
        storage_founds=storage_founds,
        storage_indices=storage_lookup_result.indices,
        h_num_storage_missing=storage_lookup_result.h_num_missing,
        storage_missing_keys=storage_lookup_result.missing_keys,
        storage_missing_indices=storage_lookup_result.missing_indices,
        indices_to_init=indices_to_init,
        keys_to_cache=keys_to_cache,
        cache_insert_indices=cache_insert_indices,
        scores_to_cache=scores_to_cache,
        num_evicted=num_evicted,
        evicted_keys=evicted_keys,
        evicted_cache_indices=evicted_cache_indices,
        evicted_scores=evicted_scores,
        evicted_storage_indices=evicted_storage_indices,
        admit_mask=admit_mask,
    )


def hash_table_processing_with_cache(
    storage_hash_table: GroupedScoredHashTable,
    cache_hash_table: GroupedScoredHashTable,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    feature_offsets: torch.Tensor,
    evict_strategy: Optional[EvictStrategy],
    score_update: bool,
    training: bool,
    frequency_counters: Optional[torch.Tensor] = None,
    admit_strategy: Optional[AdmissionStrategy] = None,
    admission_counters: Optional[List[Counter]] = None,
) -> HashTableOutputWithCache:
    """
    Process keys through cache and storage hash tables for multiple logical tables.
    
    Includes deduplication of indices and hash table operations.
    Does NOT perform any embedding operations.

    Uses unified storage design: multiple logical tables share single physical
    cache and storage tensors. Each table has its own hash tables that return
    local indices. These are converted to global indices using table_offsets.

    Args:
        storage_hash_table: GroupedScoredHashTable for storage (all logical tables' hash tables and offsets)
        cache_hash_table: GroupedScoredHashTable for cache (all logical tables' hash tables and offsets)
        indices: Raw input indices tensor
        offsets: Offsets tensor for features
        feature_offsets: Feature offset boundaries
        evict_strategy: Eviction strategy
        score_update: Whether to update scores during lookup
        training: Whether in training mode
        frequency_counters: Optional frequency counters for LFU
        admit_strategy: Optional admission strategy
        admission_counters: Optional admission counters per table

    Returns:
        HashTableOutputWithCache containing per-table results and deduplication context
    """
    # Convert frequency counters to int64 if provided
    frequency_counts_int64 = None
    if frequency_counters is not None:
        frequency_counts_int64 = frequency_counters.long()

    # Get table range for indices
    indices_table_range = get_table_range(offsets, feature_offsets)

    # Deduplication (always done for caching)
    (
        unique_indices,
        inverse,
        unique_indices_table_range,
        h_unique_indices_table_range,
        lfu_accumulated_frequency,
    ) = segmented_unique(
        indices,
        indices_table_range,
        evict_strategy,
        frequency_counts_int64,
    )

    # Process each table
    num_tables = storage_hash_table.num_tables
    assert cache_hash_table.num_tables == num_tables, \
        f"Cache and storage must have same number of tables, got {cache_hash_table.num_tables} vs {num_tables}"

    results = []

    for i in range(num_tables):
        begin = h_unique_indices_table_range[i]
        end = h_unique_indices_table_range[i + 1]
        unique_indices_per_table = unique_indices[begin:end]

        lfu_accumulated_frequency_per_table = (
            lfu_accumulated_frequency[begin:end]
            if lfu_accumulated_frequency is not None
            and lfu_accumulated_frequency.numel() > 0
            else None
        )

        # Create scores based on evict strategy
        scores = _create_scores_for_table(
            storage_hash_table,
            unique_indices_per_table.numel(),
            unique_indices_per_table.device,
            evict_strategy,
            lfu_accumulated_frequency_per_table,
        )

        cache_expand_fn = (
            cache_hash_table.get_expand_fn(i)
            if cache_hash_table.expand_fns
            else None
        )
        storage_expand_fn = (
            storage_hash_table.get_expand_fn(i)
            if storage_hash_table.expand_fns
            else None
        )

        # Process through this table's hash tables (returns local indices)
        result = _process_single_table_with_cache(
            cache_key_index_map=cache_hash_table.get_table_by_idx(i),
            storage_key_index_map=storage_hash_table.get_table_by_idx(i),
            unique_keys=unique_indices_per_table,
            score_policy=storage_hash_table.score_policy,
            scores=scores,
            score_update=score_update,
            training=training,
            admit_strategy=admit_strategy,
            admission_counter=admission_counters[i] if admission_counters else None,
            accumulated_frequency=lfu_accumulated_frequency_per_table,
            cache_expand_fn=cache_expand_fn,
            storage_expand_fn=storage_expand_fn,
        )

        # Apply table offsets to convert local indices to global indices
        cache_offset = cache_hash_table.get_table_offset(i)
        storage_offset = storage_hash_table.get_table_offset(i)
        result = _apply_table_offset_to_cache_result(result, cache_offset, storage_offset)

        results.append(result)

    return HashTableOutputWithCache(
        results=results,
        unique_indices=unique_indices,
        inverse=inverse,
        h_unique_indices_table_range=h_unique_indices_table_range,
        unique_indices_table_range=unique_indices_table_range,
        indices_table_range=indices_table_range,
    )


def _create_empty_cache_result(
    cache_key_index_map: ScoredHashTable,
    unique_keys: torch.Tensor,
    device: torch.device,
) -> HashTableProcessingResultWithCache:
    """Create result for empty input."""
    return HashTableProcessingResultWithCache(
        cache_founds=torch.empty(0, dtype=torch.bool, device=device),
        cache_indices=torch.empty(0, dtype=cache_key_index_map.index_type, device=device),
        h_num_cache_missing=0,
        cache_missing_keys=torch.empty_like(unique_keys),
        cache_missing_indices=torch.empty(0, dtype=torch.long, device=device),
        cache_missing_scores=None,
        storage_founds=torch.empty(0, dtype=torch.bool, device=device),
        storage_indices=torch.empty(0, dtype=torch.long, device=device),
        h_num_storage_missing=0,
        storage_missing_keys=torch.empty_like(unique_keys),
        storage_missing_indices=torch.empty(0, dtype=torch.long, device=device),
        indices_to_init=torch.empty(0, dtype=torch.long, device=device),
        keys_to_cache=torch.empty_like(unique_keys),
        cache_insert_indices=None,
        scores_to_cache=None,
        num_evicted=0,
        evicted_keys=None,
        evicted_cache_indices=None,
        evicted_scores=None,
        evicted_storage_indices=None,
        admit_mask=None,
    )


def _create_all_cache_hit_result(
    cache_lookup_result: HashTableLookupResult,
    unique_keys: torch.Tensor,
    device: torch.device,
) -> HashTableProcessingResultWithCache:
    """Create result when all keys found in cache."""
    return HashTableProcessingResultWithCache(
        cache_founds=cache_lookup_result.founds,
        cache_indices=cache_lookup_result.indices,
        h_num_cache_missing=0,
        cache_missing_keys=cache_lookup_result.missing_keys,
        cache_missing_indices=cache_lookup_result.missing_indices,
        cache_missing_scores=cache_lookup_result.missing_scores,
        storage_founds=torch.empty(0, dtype=torch.bool, device=device),
        storage_indices=torch.empty(0, dtype=torch.long, device=device),
        h_num_storage_missing=0,
        storage_missing_keys=torch.empty_like(unique_keys),
        storage_missing_indices=torch.empty(0, dtype=torch.long, device=device),
        indices_to_init=torch.empty(0, dtype=torch.long, device=device),
        keys_to_cache=torch.empty_like(unique_keys),
        cache_insert_indices=None,
        scores_to_cache=None,
        num_evicted=0,
        evicted_keys=None,
        evicted_cache_indices=None,
        evicted_scores=None,
        evicted_storage_indices=None,
        admit_mask=None,
    )




# =============================================================================
# Embedding initialization and storage functions
# =============================================================================


def embedding_insert_no_cache(
    embedding_table: EmbeddingTable,
    unique_indices: torch.Tensor,
    h_unique_indices_table_range: torch.Tensor,
    ht_results: List[HashTableProcessingResult],
    initializers: List[Callable],
    admit_strategy: Optional[AdmissionStrategy] = None,
) -> None:
    """
    Insert new embeddings into the embedding table (training only).
    
    This function handles:
    - Non-admitted keys initialization (with default values)
    - Missing embeddings initialization (using initializers)
    - Storing newly inserted embeddings to storage
    
    After this function completes, the storage contains all correct embedding values,
    allowing embedding_lookup_no_cache to copy from storage to output buffer.

    Args:
        embedding_table: Single unified embedding table (physical storage)
        unique_indices: All unique indices across tables
        h_unique_indices_table_range: CPU tensor with table boundaries
        ht_results: List of hash table processing results (with global indices)
        initializers: List of initializer functions (one per table)
        admit_strategy: Optional admission strategy
    """
    num_tables = len(ht_results)
    dev_table = embedding_table.dev_table
    uvm_table = embedding_table.uvm_table
    emb_dim = embedding_table.emb_dim
    val_dim = embedding_table.val_dim
    emb_dtype = embedding_table.emb_dtype
    init_optimizer_state = embedding_table.init_optimizer_state

    for i in range(num_tables):
        begin = h_unique_indices_table_range[i]
        end = h_unique_indices_table_range[i + 1]
        unique_keys = unique_indices[begin:end]
        ht_result = ht_results[i]

        if ht_result.h_num_missing == 0:
            continue

        device = unique_keys.device
        h_num_missing = ht_result.h_num_missing

        # Create temporary buffer for missing embeddings only
        missing_embs = torch.empty(h_num_missing, emb_dim, dtype=emb_dtype, device=device)

        # 1. Handle non-admitted keys initialization (if applicable)
        if admit_strategy is not None and ht_result.admit_mask is not None:
            non_admitted_mask = ~ht_result.admit_mask
            # Non-admitted indices are relative to missing_embs buffer (0 to h_num_missing-1)
            non_admitted_local_indices = torch.arange(h_num_missing, device=device)[non_admitted_mask]
            if non_admitted_local_indices.numel() > 0:
                admit_strategy.initialize_non_admitted_embeddings(
                    missing_embs, non_admitted_local_indices
                )

        # 2. Initialize missing embeddings (admitted or all if no admission)
        if ht_result.indices_to_init.numel() > 0:
            # indices_to_init are relative to the missing_embs buffer
            missing_keys = ht_result.missing_keys
            initializers[i](missing_embs, ht_result.indices_to_init, missing_keys)

        # 3. Store newly inserted embeddings
        if ht_result.insert_indices is not None:
            # Determine which missing embeddings were actually inserted
            if ht_result.admit_mask is not None:
                admitted_local_indices = torch.arange(h_num_missing, device=device)[ht_result.admit_mask]
            else:
                admitted_local_indices = torch.arange(h_num_missing, device=device)

            # Prepare values to store (embedding + optimizer state)
            num_to_store = ht_result.insert_indices.numel()
            values_to_store = torch.empty(num_to_store, val_dim, dtype=emb_dtype, device=device)
            values_to_store[:, :emb_dim] = missing_embs[admitted_local_indices, :]
            if val_dim != emb_dim:
                values_to_store[:, emb_dim:] = init_optimizer_state

            # Store using global insert_indices
            store_embeddings_by_indices(dev_table, uvm_table, ht_result.insert_indices, values_to_store)


def embedding_lookup_no_cache(
    embedding_table: EmbeddingTable,
    output_global_indices: torch.Tensor,
    output_embs: torch.Tensor,
) -> None:
    """
    Lookup embeddings from storage table and copy to output buffer.
    
    This is the optimized copy path that avoids the intermediate unique_embs buffer.
    Should be called after embedding_insert_no_cache (for training) to ensure
    all embeddings are properly initialized in storage.

    Args:
        embedding_table: Single unified embedding table (physical storage)
        output_global_indices: Global indices for each output row (expanded from unique)
        output_embs: Output tensor for embeddings (modified in place)
    """
    dev_table = embedding_table.dev_table
    uvm_table = embedding_table.uvm_table
    emb_dim = embedding_table.emb_dim
    val_dim = embedding_table.val_dim

    if emb_dim == val_dim:
        copy_embeddings_by_indices(dev_table, uvm_table, output_global_indices, output_embs)
    else:
        # Need to load full values and extract embeddings
        emb_dtype = embedding_table.emb_dtype
        temp_values = torch.empty(
            output_global_indices.shape[0], val_dim, dtype=emb_dtype, device=output_global_indices.device
        )
        copy_embeddings_by_indices(dev_table, uvm_table, output_global_indices, temp_values)
        output_embs.copy_(temp_values[:, :emb_dim])


def embedding_copy_with_cache(
    embedding_table: EmbeddingTable,
    unique_indices: torch.Tensor,
    unique_embs: torch.Tensor,
    h_unique_indices_table_range: torch.Tensor,
    ht_results: List[HashTableProcessingResultWithCache],
    initializers: List[Callable],
    training: bool,
    admit_strategy: Optional[AdmissionStrategy] = None,
) -> None:
    """
    Perform embedding copy operations for multiple logical tables (with caching).
    Does NOT perform any hash table operations.

    Uses unified storage design:
    - embedding_table.dev_table: Cache storage (HBM, fast access)
    - embedding_table.uvm_table: Main storage (UVM/DRAM)

    Args:
        embedding_table: Single unified embedding table containing both cache (dev_table) and storage (uvm_table)
        unique_indices: All unique indices across tables
        unique_embs: Output tensor for embeddings (modified in place)
        h_unique_indices_table_range: CPU tensor with table boundaries
        ht_results: List of hash table processing results (one per logical table, with global indices)
        initializers: List of initializer functions (one per table)
        training: Whether in training mode
        admit_strategy: Optional admission strategy
    """
    num_tables = len(ht_results)
    # Cache uses dev_table (HBM)
    cache_dev_table = embedding_table.dev_table
    cache_uvm_table = None  # Cache is HBM only
    # Storage uses uvm_table (UVM/DRAM)
    storage_dev_table = None  # Storage is UVM only when caching
    storage_uvm_table = embedding_table.uvm_table
    emb_dim = embedding_table.emb_dim
    val_dim = embedding_table.val_dim
    emb_dtype = embedding_table.emb_dtype
    init_optimizer_state = embedding_table.init_optimizer_state

    for i in range(num_tables):
        begin = h_unique_indices_table_range[i]
        end = h_unique_indices_table_range[i + 1]
        unique_keys = unique_indices[begin:end]
        unique_embs_per_table = unique_embs[begin:end, :]
        ht_result = ht_results[i]

        h_num_total = unique_keys.numel()
        if h_num_total == 0:
            continue

        device = unique_keys.device

        # 1. Copy found embeddings from cache using global indices
        if ht_result.cache_founds.any():
            if emb_dim == val_dim:
                copy_embeddings_by_indices(
                    cache_dev_table, cache_uvm_table, ht_result.cache_indices, unique_embs_per_table
                )
            else:
                temp_values = torch.empty(h_num_total, val_dim, dtype=emb_dtype, device=device)
                copy_embeddings_by_indices(
                    cache_dev_table, cache_uvm_table, ht_result.cache_indices, temp_values
                )
                unique_embs_per_table.copy_(temp_values[:, :emb_dim])

        if ht_result.h_num_cache_missing == 0:
            continue

        # 2. Load values from storage for cache-missed keys using global indices
        values_for_cache_missed = torch.empty(
            ht_result.h_num_cache_missing, val_dim, dtype=emb_dtype, device=device
        )
        copy_embeddings_by_indices(
            storage_dev_table,
            storage_uvm_table,
            ht_result.storage_indices,
            values_for_cache_missed,
        )

        # 3. Handle non-admitted keys initialization
        if training and admit_strategy is not None and ht_result.admit_mask is not None:
            non_admitted_mask = ~ht_result.admit_mask
            non_admitted_indices = ht_result.storage_missing_indices[non_admitted_mask]
            if non_admitted_indices.numel() > 0:
                admit_strategy.initialize_non_admitted_embeddings(
                    values_for_cache_missed[:, :emb_dim], non_admitted_indices
                )

        # 4. Initialize storage-missed embeddings
        if ht_result.indices_to_init.numel() > 0:
            initializers[i](
                values_for_cache_missed[:, :emb_dim],
                ht_result.indices_to_init,
                ht_result.cache_missing_keys,
            )

        # 5. Initialize optimizer state for storage-missed (training)
        if training and val_dim != emb_dim:
            values_for_cache_missed[ht_result.storage_missing_indices, emb_dim:] = (
                init_optimizer_state
            )

        # 6. Copy embeddings to output
        unique_embs_per_table[ht_result.cache_missing_indices, :] = values_for_cache_missed[:, :emb_dim]

        # 7. Store values in cache (at insert locations) using global indices
        if ht_result.cache_insert_indices is not None and ht_result.keys_to_cache.numel() > 0:
            # Determine which values to store in cache
            if training:
                if ht_result.admit_mask is not None:
                    # Storage-hit + admitted storage-miss
                    mask_to_cache = ht_result.storage_founds.clone()
                    admitted_indices = ht_result.storage_missing_indices[ht_result.admit_mask]
                    mask_to_cache[admitted_indices] = True
                else:
                    mask_to_cache = torch.ones(
                        ht_result.h_num_cache_missing, dtype=torch.bool, device=device
                    )
            else:
                mask_to_cache = ht_result.storage_founds

            values_to_cache = values_for_cache_missed[mask_to_cache, :].contiguous()
            store_embeddings_by_indices(
                cache_dev_table, cache_uvm_table, ht_result.cache_insert_indices, values_to_cache
            )

        # 8. Store evicted values to storage using global indices
        if ht_result.num_evicted > 0 and ht_result.evicted_storage_indices is not None:
            evicted_values = torch.empty(
                ht_result.num_evicted, val_dim, dtype=emb_dtype, device=device
            )
            copy_embeddings_by_indices(
                cache_dev_table, cache_uvm_table, ht_result.evicted_cache_indices, evicted_values
            )
            store_embeddings_by_indices(
                storage_dev_table, storage_uvm_table, ht_result.evicted_storage_indices, evicted_values
            )


# =============================================================================
# Combined high-level functions (for convenience)
# =============================================================================


def _do_admission(
    keys: torch.Tensor,
    freqs: torch.Tensor,
    admit_strategy: AdmissionStrategy,
    admission_counter: Counter,
) -> torch.Tensor:
    """
    Perform admission control.

    Returns:
        Boolean mask indicating which keys are admitted
    """
    freq_for_missing_keys = admission_counter.add(keys, freqs, inplace=True)
    admit_mask = admit_strategy.admit(keys, freq_for_missing_keys)
    admitted_keys = keys[admit_mask]
    admission_counter.erase(admitted_keys)
    return admit_mask


# =============================================================================
# High-level update functions (backward pass)
# =============================================================================


def dynamicemb_update_no_cache(
    hash_table: GroupedScoredHashTable,
    embedding_table: EmbeddingTable,
    unique_indices: torch.Tensor,
    unique_grads: torch.Tensor,
    h_unique_indices_table_range: torch.Tensor,
    optimizer: BaseDynamicEmbeddingOptimizerV2,
) -> None:
    """
    Update embeddings with gradients for multiple logical tables (no cache version).

    Uses unified storage design: each logical table has its own hash table,
    and indices are converted to global indices using table_offsets.

    Args:
        hash_table: GroupedScoredHashTable with hash tables and offsets for all logical tables
        embedding_table: Single unified embedding table (physical storage for all tables)
        unique_indices: All unique indices across tables
        unique_grads: All unique gradients across tables
        h_unique_indices_table_range: CPU tensor with table boundaries
        optimizer: Optimizer for updating embeddings
    """
    num_tables = hash_table.num_tables
    dev_table = embedding_table.dev_table
    uvm_table = embedding_table.uvm_table
    emb_dtype = embedding_table.emb_dtype

    score_args_lookup = [
        ScoreArg(
            name=hash_table.score_policy.name,
            value=None,
            policy=ScorePolicy.CONST,
            is_return=False,
        )
    ]

    for i in range(num_tables):
        begin = h_unique_indices_table_range[i]
        end = h_unique_indices_table_range[i + 1]
        unique_keys = unique_indices[begin:end]
        unique_grads_per_table = unique_grads[begin:end, :]

        batch = unique_keys.numel()
        if batch == 0:
            continue

        device = unique_keys.device
        key_index_map = hash_table.get_table_by_idx(i)
        table_offset = hash_table.get_table_offset(i)

        # Lookup to get local indices
        founds = torch.empty(batch, dtype=torch.bool, device=device)
        local_indices = torch.empty(batch, dtype=key_index_map.index_type, device=device)
        key_index_map.lookup(unique_keys, score_args_lookup, founds, local_indices)

        # Convert to global indices
        global_indices = local_indices + table_offset

        # Update embeddings using optimizer with global indices
        optimizer.fused_update_with_index(
            unique_grads_per_table.to(emb_dtype),
            global_indices,
            dev_table,
            uvm_table,
        )


def dynamicemb_update_with_cache(
    storage_hash_table: GroupedScoredHashTable,
    cache_hash_table: GroupedScoredHashTable,
    embedding_table: EmbeddingTable,
    unique_indices: torch.Tensor,
    unique_grads: torch.Tensor,
    h_unique_indices_table_range: torch.Tensor,
    optimizer: BaseDynamicEmbeddingOptimizerV2,
) -> None:
    """
    Update embeddings with gradients for multiple logical tables (with cache version).

    Uses unified storage design:
    - embedding_table.dev_table: Cache storage (HBM)
    - embedding_table.uvm_table: Main storage (UVM/DRAM)

    Args:
        storage_hash_table: GroupedScoredHashTable with storage hash tables and offsets
        cache_hash_table: GroupedScoredHashTable with cache hash tables and offsets
        embedding_table: Single unified embedding table (dev_table=cache, uvm_table=storage)
        unique_indices: All unique indices across tables
        unique_grads: All unique gradients across tables
        h_unique_indices_table_range: CPU tensor with table boundaries
        optimizer: Optimizer for updating embeddings
    """
    num_tables = storage_hash_table.num_tables
    # Cache uses dev_table (HBM)
    cache_dev_table = embedding_table.dev_table
    cache_uvm_table = None  # Cache is HBM only
    # Storage uses uvm_table (UVM/DRAM)
    storage_dev_table = None  # Storage is UVM only when caching
    storage_uvm_table = embedding_table.uvm_table
    emb_dtype = embedding_table.emb_dtype

    score_args_lookup = [
        ScoreArg(
            name=storage_hash_table.score_policy.name,
            value=None,
            policy=ScorePolicy.CONST,
            is_return=False,
        )
    ]

    for i in range(num_tables):
        begin = h_unique_indices_table_range[i]
        end = h_unique_indices_table_range[i + 1]
        unique_keys = unique_indices[begin:end]
        unique_grads_per_table = unique_grads[begin:end, :]

        batch = unique_keys.numel()
        if batch == 0:
            continue

        device = unique_keys.device
        cache_key_index_map = cache_hash_table.get_table_by_idx(i)
        storage_key_index_map = storage_hash_table.get_table_by_idx(i)
        cache_offset = cache_hash_table.get_table_offset(i)
        storage_offset = storage_hash_table.get_table_offset(i)

        # 1. Update cache first - lookup to get local indices
        cache_founds = torch.empty(batch, dtype=torch.bool, device=device)
        cache_local_indices = torch.empty(batch, dtype=cache_key_index_map.index_type, device=device)
        cache_key_index_map.lookup(unique_keys, score_args_lookup, cache_founds, cache_local_indices)

        # Convert to global indices and update
        cache_global_indices = cache_local_indices + cache_offset
        optimizer.fused_update_with_index(
            unique_grads_per_table.to(emb_dtype),
            cache_global_indices,
            cache_dev_table,
            cache_uvm_table,
        )

        # 2. Find cache-missed keys
        cache_missing = torch.logical_not(cache_founds)
        num_missing = torch.empty(1, dtype=torch.long, device=device)
        missing_keys = torch.empty_like(unique_keys)
        missing_indices = torch.empty(batch, dtype=torch.long, device=device)

        select(cache_missing, unique_keys, missing_keys, num_missing)
        select_index(cache_missing, missing_indices, torch.empty(1, dtype=torch.long, device=device))

        h_num_missing = num_missing.cpu().item()

        if h_num_missing == 0:
            continue

        missing_keys = missing_keys[:h_num_missing]
        missing_grads = unique_grads_per_table[missing_indices[:h_num_missing], :].contiguous()

        # 3. Update storage for cache-missed keys
        storage_founds = torch.empty(h_num_missing, dtype=torch.bool, device=device)
        storage_local_indices = torch.empty(
            h_num_missing, dtype=storage_key_index_map.index_type, device=device
        )
        storage_key_index_map.lookup(missing_keys, score_args_lookup, storage_founds, storage_local_indices)

        # Convert to global indices and update
        storage_global_indices = storage_local_indices + storage_offset
        optimizer.fused_update_with_index(
            missing_grads.to(emb_dtype),
            storage_global_indices,
            storage_dev_table,
            storage_uvm_table,
        )


# TODO: BatchedDynamicEmbeddingFunction is more concrete.
class DynamicEmbeddingBagFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        indices: torch.Tensor,
        offsets: torch.Tensor,  # [feature_num * batch_size]
        feature_offsets: torch.Tensor,
        use_index_dedup: bool,
        table_offsets_in_feature: List[int],
        tables: List[DynamicEmbTable],
        scores: List[int],
        total_D: int,
        dims: List[int],
        feature_table_map: List[int],
        embedding_dtype: torch.dtype,
        output_dtype: torch.dtype,
        pooling_mode: DynamicEmbPoolingMode,
        device_num_sms: int,
        device: torch.device,
        optimizer: BaseDynamicEmbeddingOptimizer,
        training: bool,
        eval_initializers: List[DynamicEmbInitializerArgs],
        *args,
    ):
        # TODO: remove unnecessary params.
        # TODO:need check dimension is right
        table_num = len(tables)
        assert table_num == len(table_offsets_in_feature) - 1
        feature_num = len(feature_table_map)

        # split indices, offsets by table.
        indices_list: List[torch.Tensor] = []
        biased_offsets_list: List[torch.Tensor] = []

        feature_num = table_offsets_in_feature[-1]
        feature_batch_size = offsets.shape[0] - 1
        batch_size = feature_batch_size // feature_num
        assert feature_batch_size % feature_num == 0
        # The offsets is on device in torchrec, however, the unique_op and lookup_op are done table by table.
        # So we need to know one index belong to which table, to let op know the boundary.
        # Therefore, copy offsets to cpu is necessary, otherwise, many things will be coupled together.
        # For example, UniqueOp have to accept (indices, offsets, table_offsets_in_feature, table_id) as inputs,
        #   and we have to copy table_offsets_in_feature from cpu to gpu.

        # TODO: if the batch size is large, we can develop a kernel to get: indices boundary.
        indices_table_range = get_table_range(offsets, feature_offsets)
        h_indices_table_range = torch.empty(
            indices_table_range.numel(),
            out=torch.ops.fbgemm.new_unified_tensor(
                # pyre-fixme[6]: Expected `Optional[Type[torch._dtype]]`
                #  for 3rd param but got `Type[Type[torch._dtype]]`.
                torch.zeros(1, device=device, dtype=indices_table_range.dtype),
                [indices_table_range.numel()],
                #  is_host_mapped (bool = False): If True, allocate every UVM tensor
                # using `malloc` + `cudaHostRegister`. Otherwise use
                # `cudaMallocManaged`
                is_host_mapped=True,
            ),
        )
        h_indices_table_range.copy_(indices_table_range)

        for i in range(table_num):
            feature_id_begin, feature_id_end = (
                table_offsets_in_feature[i],
                table_offsets_in_feature[i + 1],
            )
            offset_begin, offset_end = (
                feature_id_begin * batch_size,
                feature_id_end * batch_size,
            )
            # include offset_end to know the boundary of the last feature.
            biased_offsets_list.append(offsets[offset_begin : offset_end + 1])

            indices_begin, indices_end = (
                h_indices_table_range[i],
                h_indices_table_range[i + 1],
            )
            indices_list.append(indices[indices_begin:indices_end])

        unique_indices_list = []
        inverse_indices_list = []
        unique_count_list = []
        for i in range(table_num):
            unique_indices, inverse_indices = torch.unique(
                indices_list[i], sorted=False, return_inverse=True
            )
            unique_indices_list.append(unique_indices)
            inverse_indices_list.append(
                inverse_indices.to(biased_offsets_list[i].dtype)
            )
            unique_count_list.append(inverse_indices.shape[0])

        unique_embedding_list = []
        for i in range(table_num):
            unique_indices = unique_indices_list[i]
            num_unique_indices = unique_indices.shape[0]
            tmp_value_type_torch = dyn_emb_to_torch(tables[i].value_type())
            tmp_unique_embs = torch.empty(
                num_unique_indices, dims[i], dtype=tmp_value_type_torch, device=device
            )

            if training:
                find_or_insert(
                    tables[i],
                    num_unique_indices,
                    unique_indices,
                    tmp_unique_embs,
                    scores[i],
                )
            else:
                find_and_initialize(
                    tables[i],
                    num_unique_indices,
                    unique_indices,
                    tmp_unique_embs,
                    eval_initializers[i].as_ctype(),
                )

            unique_embedding_list.append(tmp_unique_embs)

        if pooling_mode == DynamicEmbPoolingMode.NONE:
            combiner = -1
            # total_embs_num = indices.shape[0]
            total_embs_num = indices.numel()
            # All tables have the same dim.
            embs = torch.empty(
                total_embs_num, dims[0], dtype=output_dtype, device=device
            )
        else:
            if pooling_mode == DynamicEmbPoolingMode.SUM:
                combiner = 0
            elif pooling_mode == DynamicEmbPoolingMode.MEAN:
                combiner = 1
            else:
                raise ValueError("Not support pooling mode.")
            total_embs_num = offsets.shape[0] - 1
            embs = torch.empty(batch_size, total_D, dtype=output_dtype, device=device)

        # TODO:To combine all the table's combiner kernel together, we first need to merge the indices. This may require developing a customized kernel to achieve this.
        accum_D = 0
        for i in range(table_num):
            num_embeddings = biased_offsets_list[i].shape[0] - 1
            lookup_forward(
                unique_embedding_list[i],
                embs,
                biased_offsets_list[i],
                inverse_indices_list[i],
                combiner,
                total_D,
                accum_D,
                dims[i],
                num_embeddings,
                batch_size,
                device_num_sms,
            )
            accum_D += dims[i] * (num_embeddings // batch_size)
            assert num_embeddings % batch_size == 0

        if training:
            backward_tensors = [indices, offsets]
            ctx.save_for_backward(*backward_tensors)
            ctx.tables = tables
            ctx.unique_indices_list = unique_indices_list
            ctx.inverse_indices_list = inverse_indices_list
            ctx.biased_offsets_list = biased_offsets_list
            ctx.dims = dims
            ctx.batch_size = batch_size
            ctx.feature_num = feature_num
            ctx.feature_table_map = feature_table_map
            ctx.device = device
            ctx.optimizer = optimizer
            ctx.scores = scores
            ctx.combiner = combiner

        return embs

    @staticmethod
    def backward(ctx, grad):
        # if we want to do the value check, we shouldn't to update the embeddings ].
        tables = ctx.tables
        unique_indices_list = ctx.unique_indices_list
        inverse_indices_list = ctx.inverse_indices_list
        biased_offsets_list = ctx.biased_offsets_list
        dims = ctx.dims
        batch_size = ctx.batch_size
        ctx.feature_num
        feature_table_map_list = ctx.feature_table_map
        indices, offsets = ctx.saved_tensors
        device = ctx.device
        optimizer = ctx.optimizer
        table_num = len(tables)
        combiner = ctx.combiner

        offsets_list_per_table = []
        for i in range(table_num):
            offsets_list_per_table.append(
                biased_offsets_list[i] - biased_offsets_list[i][0]
            )

        feature_num_per_table = [0] * table_num
        for i in range(len(feature_table_map_list)):
            feature_num_per_table[feature_table_map_list[i]] += 1

        dim_offset_per_table = [0]
        for i in range(table_num):
            dim_offset_per_table.append(
                feature_num_per_table[i] * dims[i] + dim_offset_per_table[i]
            )

        dyn_emb_to_torch(tables[0].value_type())
        dyn_emb_to_torch(tables[0].key_type())

        unique_count_list = []
        for i in range(table_num):
            unique_count_list.append(unique_indices_list[i].shape[0])

        unique_backward_grads_per_table = []
        for i in range(table_num):
            unique_backward_grads_per_table.append(
                torch.zeros(
                    unique_count_list[i] * dims[i], dtype=grad.dtype, device=device
                )
            )

        # dims_tensor = torch.tensor(dims_list,dtype=torch.int32,device=device)
        for i in range(table_num):
            grad_for_table = grad[
                :, dim_offset_per_table[i] : dim_offset_per_table[i + 1]
            ]

            splits = torch.split(grad_for_table, dims[i], dim=-1)
            result = torch.cat(splits, dim=0)
            grad_for_table = result.reshape(-1, dims[i]).contiguous()
            lookup_backward(
                grad_for_table,
                unique_backward_grads_per_table[i],
                unique_indices_list[i],
                inverse_indices_list[i],
                offsets_list_per_table[i],
                dims[i],
                table_num,
                batch_size,
                feature_num_per_table[i],
                inverse_indices_list[i].numel(),
                combiner,
            )

        unique_grads_per_table = []
        for i, unique_grad in enumerate(unique_backward_grads_per_table):
            unique_grads_per_table.append(unique_grad.reshape(-1, dims[i]))

        optimizer.update(tables, unique_indices_list, unique_grads_per_table)

        return (None,) * 20


def dynamicemb_prefetch(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    caches: List[Optional[Cache]],
    storages: List[Storage],
    feature_offsets: torch.Tensor,
    initializers: List[BaseDynamicEmbInitializer],
    training: bool = True,
    forward_stream: Optional[torch.cuda.Stream] = None,
):
    table_num = len(storages)
    assert table_num != 0
    caching = caches[0] is not None

    indices_table_range = get_table_range(offsets, feature_offsets)
    if training or caching:
        (
            unique_indices,
            inverse,
            unique_indices_table_range,
            h_unique_indices_table_range,
            _,
        ) = segmented_unique(indices, indices_table_range)
        # TODO: only return device unique_indices_table_range
        # h_unique_indices_table_range = unique_indices_table_range.cpu()
    else:
        h_unique_indices_table_range = indices_table_range.cpu()
        unique_indices = indices

    for i in range(table_num):
        begin = h_unique_indices_table_range[i]
        end = h_unique_indices_table_range[i + 1]
        unique_indices_per_table = unique_indices[begin:end]

        KeyValueTableCachingFunction.prefetch(
            caches[i],
            storages[i],
            unique_indices_per_table,
            initializers[i],
            training,
            forward_stream,
        )


class DynamicEmbeddingFunctionV2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        caches: List[Optional[Cache]],
        storages: List[Storage],
        feature_offsets: torch.Tensor,
        output_dtype: torch.dtype,
        initializers: List[BaseDynamicEmbInitializer],
        optimizer: BaseDynamicEmbeddingOptimizer,
        enable_prefetch: bool = False,
        input_dist_dedup: bool = False,
        training: bool = True,
        admit_strategy=None,
        evict_strategy=None,
        frequency_counters: Optional[torch.Tensor] = None,
        admission_counter: Optional[list[Counter]] = None,
        *args,
    ):
        table_num = len(storages)
        assert table_num != 0
        emb_dtype = storages[0].embedding_dtype()
        emb_dim = storages[0].embedding_dim()
        caching = caches[0] is not None
        # admit_strategy = storages[0].options.admit_strategy

        # evict_strategy = storages[0].options.score_strategy

        frequency_counts_int64 = None
        if frequency_counters is not None:
            frequency_counts_int64 = frequency_counters.long()

        lfu_accumulated_frequency = None
        indices_table_range = get_table_range(offsets, feature_offsets)
        if training or caching:
            (
                unique_indices,
                inverse,
                unique_indices_table_range,
                h_unique_indices_table_range,
                lfu_accumulated_frequency,
            ) = segmented_unique(
                indices,
                indices_table_range,
                EvictStrategy(evict_strategy.value) if evict_strategy else None,
                frequency_counts_int64,
            )
            # TODO: only return device unique_indices_table_range
            # h_unique_indices_table_range = unique_indices_table_range.cpu()
        else:
            h_unique_indices_table_range = indices_table_range.cpu()
            unique_indices = indices

        unique_embs = torch.empty(
            unique_indices.shape[0], emb_dim, dtype=emb_dtype, device=indices.device
        )

        for i in range(table_num):
            begin = h_unique_indices_table_range[i]
            end = h_unique_indices_table_range[i + 1]
            unique_indices_per_table = unique_indices[begin:end]
            unique_embs_per_table = unique_embs[begin:end, :]
            # Slice lfu_accumulated_frequency to match the table
            lfu_accumulated_frequency_per_table = (
                lfu_accumulated_frequency[begin:end]
                if lfu_accumulated_frequency is not None
                and lfu_accumulated_frequency.numel() > 0
                else None
            )

            if caching:
                KeyValueTableCachingFunction.lookup(
                    caches[i],
                    storages[i],
                    unique_indices_per_table,
                    unique_embs_per_table,
                    initializers[i],
                    enable_prefetch,
                    training,
                    EvictStrategy(evict_strategy.value) if evict_strategy else None,
                    lfu_accumulated_frequency_per_table,
                    admit_strategy,
                    admission_counter[i] if admission_counter else None,
                )
            else:
                KeyValueTableFunction.lookup(
                    storages[i],
                    unique_indices_per_table,
                    unique_embs_per_table,
                    initializers[i],
                    training,
                    EvictStrategy(evict_strategy.value) if evict_strategy else None,
                    lfu_accumulated_frequency_per_table,
                    admit_strategy,
                    admission_counter[i] if admission_counter else None,
                )

        if training or caching:
            output_embs = torch.empty(
                indices.shape[0], emb_dim, dtype=output_dtype, device=indices.device
            )
            gather_embedding(unique_embs, output_embs, inverse)
        else:
            output_embs = unique_embs

        if training:
            # save context
            backward_tensors = [
                indices,
            ]
            ctx.save_for_backward(*backward_tensors)
            ctx.input_dist_dedup = input_dist_dedup
            if input_dist_dedup:
                ctx.unique_indices = unique_indices
                ctx.unique_embs = unique_embs
                ctx.inverse = inverse
            ctx.indices_table_range = indices_table_range
            ctx.h_indices_table_range = indices_table_range.cpu()
            ctx.h_unique_indices_table_range = h_unique_indices_table_range
            ctx.unique_indices_table_range = unique_indices_table_range
            ctx.caches = caches
            ctx.storages = storages
            ctx.optimizer = optimizer
            ctx.enable_prefetch = enable_prefetch

        return output_embs

    @staticmethod
    def backward(ctx, grads):
        # parse context
        (indices,) = ctx.saved_tensors
        indices_table_range = ctx.indices_table_range
        h_indices_table_range = ctx.h_indices_table_range
        h_unique_indices_table_range = ctx.h_unique_indices_table_range
        ctx.unique_indices_table_range
        caches = ctx.caches
        storages = ctx.storages
        optimizer = ctx.optimizer
        caching = caches[0] is not None

        # clip the gradient before reduction
        if optimizer.need_gradient_clipping():
            optimizer.clip_gradient(grads)

        input_dist_dedup = ctx.input_dist_dedup
        if input_dist_dedup:
            unique_indices = ctx.unique_indices
            unique_embs = ctx.unique_embs
            ctx.inverse
        unique_indices, unique_embs = reduce_grads(
            indices, grads, indices_table_range, h_indices_table_range
        )
        optimizer.step()
        table_num = len(storages)
        for i in range(table_num):
            begin = h_unique_indices_table_range[i]
            end = h_unique_indices_table_range[i + 1]
            unique_indices_per_table = unique_indices[begin:end]
            unique_embs_per_table = unique_embs[begin:end, :]

            if caching:
                KeyValueTableCachingFunction.update(
                    caches[i],
                    storages[i],
                    unique_indices_per_table,
                    unique_embs_per_table,
                    optimizer,
                )
            else:
                KeyValueTableFunction.update(
                    storages[i],
                    unique_indices_per_table,
                    unique_embs_per_table,
                    optimizer,
                )

        return (None,) * 17


# =============================================================================
# DynamicEmbeddingFunctionV3 - Uses decoupled hash table and embedding ops
# =============================================================================


class DynamicEmbeddingFunctionV3(torch.autograd.Function):
    """
    Dynamic embedding function that separates hash table operations from
    embedding copy operations.

    This version works directly with the lower-level components:
    - key_index_map (ScoredHashTable): Maps keys to indices
    - dev_table/uvm_table: Store embeddings and optimizer states

    Unlike V2 which uses Cache/Storage abstractions, this version provides
    clearer separation between:
    1. Hash table operations (lookup, insert, evict)
    2. Embedding copy operations (load/store by indices)
    """

    @staticmethod
    def forward(
        ctx,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        # Hash table (GroupedScoredHashTable for multiple logical tables)
        hash_table: GroupedScoredHashTable,
        # Embedding table (unified physical storage)
        # When caching: dev_table=cache (HBM), uvm_table=storage (UVM/DRAM)
        # When no caching: dev_table or uvm_table is main storage
        embedding_table: EmbeddingTable,
        # Optional cache hash table (None if no caching)
        cache_hash_table: Optional[GroupedScoredHashTable],
        feature_offsets: torch.Tensor,
        output_dtype: torch.dtype,
        initializers: List[BaseDynamicEmbInitializer],
        optimizer: BaseDynamicEmbeddingOptimizerV2,
        score_update: bool = True,
        training: bool = True,
        admit_strategy: Optional[AdmissionStrategy] = None,
        evict_strategy: Optional[EvictStrategy] = None,
        frequency_counters: Optional[torch.Tensor] = None,
        admission_counters: Optional[List[Counter]] = None,
        *args,
    ):
        """
        Forward pass for dynamic embedding lookup.

        Args:
            indices: Input indices tensor
            offsets: Offsets tensor for features
            hash_table: Storage hash table (GroupedScoredHashTable for multiple logical tables)
            embedding_table: Unified embedding table. When caching: dev_table=cache (HBM), 
                           uvm_table=storage (UVM/DRAM). When no caching: dev_table or uvm_table is main storage.
            cache_hash_table: Optional cache hash table (GroupedScoredHashTable)
            feature_offsets: Feature offset boundaries
            output_dtype: Output tensor dtype
            initializers: Initializers for missing embeddings
            optimizer: Optimizer for updating embeddings
            score_update: Whether to update scores during lookup
            training: Whether in training mode
            admit_strategy: Optional admission strategy
            evict_strategy: Optional eviction strategy
            frequency_counters: Optional frequency counters for LFU
            admission_counters: Optional admission counters per table
        """
        emb_dtype = embedding_table.emb_dtype
        emb_dim = embedding_table.emb_dim
        caching = cache_hash_table is not None

        # Step 1: Hash table processing (includes deduplication)
        if caching:
            ht_output = hash_table_processing_with_cache(
                storage_hash_table=hash_table,
                cache_hash_table=cache_hash_table,
                indices=indices,
                offsets=offsets,
                feature_offsets=feature_offsets,
                evict_strategy=evict_strategy,
                score_update=score_update,
                training=training,
                frequency_counters=frequency_counters,
                admit_strategy=admit_strategy,
                admission_counters=admission_counters,
            )
        else:
            ht_output = hash_table_processing_no_cache(
                hash_table=hash_table,
                indices=indices,
                offsets=offsets,
                feature_offsets=feature_offsets,
                evict_strategy=evict_strategy,
                score_update=score_update,
                training=training,
                frequency_counters=frequency_counters,
                admit_strategy=admit_strategy,
                admission_counters=admission_counters,
            )

        # Step 2: Embedding operations
        output_embs = torch.empty(
            indices.shape[0], emb_dim, dtype=output_dtype, device=indices.device
        )

        if caching:
            # With cache: use intermediate buffer (cache case is more complex)
            # embedding_table.dev_table is cache, embedding_table.uvm_table is storage
            unique_embs = torch.empty(
                ht_output.unique_indices.shape[0], emb_dim, dtype=emb_dtype, device=indices.device
            )
            embedding_copy_with_cache(
                embedding_table=embedding_table,
                unique_indices=ht_output.unique_indices,
                unique_embs=unique_embs,
                h_unique_indices_table_range=ht_output.h_unique_indices_table_range,
                ht_results=ht_output.results,
                initializers=initializers,
                training=training,
                admit_strategy=admit_strategy,
            )
            gather_embedding(unique_embs, output_embs, ht_output.inverse)
        else:
            # Optimized no-cache path: skip intermediate unique_embs buffer
            
            # Step 2a: Insert new embeddings into table (training only)
            if training:
                embedding_insert_no_cache(
                    embedding_table=embedding_table,
                    unique_indices=ht_output.unique_indices,
                    h_unique_indices_table_range=ht_output.h_unique_indices_table_range,
                    ht_results=ht_output.results,
                    initializers=initializers,
                    admit_strategy=admit_strategy,
                )

            # Step 2b: Lookup embeddings from table to output
            all_unique_global_indices = torch.cat(
                [ht_result.indices for ht_result in ht_output.results]
            )

            if training:
                output_global_indices = all_unique_global_indices[ht_output.inverse]
            else:
                # Inference without dedup: indices == unique_indices
                output_global_indices = all_unique_global_indices

            embedding_lookup_no_cache(
                embedding_table=embedding_table,
                output_global_indices=output_global_indices,
                output_embs=output_embs,
            )

        # Save context for backward
        if training:
            backward_tensors = [indices]
            ctx.save_for_backward(*backward_tensors)
            ctx.indices_table_range = ht_output.indices_table_range
            ctx.h_indices_table_range = ht_output.indices_table_range.cpu()
            ctx.h_unique_indices_table_range = ht_output.h_unique_indices_table_range
            ctx.hash_table = hash_table
            ctx.embedding_table = embedding_table
            ctx.cache_hash_table = cache_hash_table
            ctx.optimizer = optimizer
            ctx.caching = caching

        return output_embs

    @staticmethod
    def backward(ctx, grads):
        """Backward pass for dynamic embedding."""
        (indices,) = ctx.saved_tensors
        indices_table_range = ctx.indices_table_range
        h_indices_table_range = ctx.h_indices_table_range
        h_unique_indices_table_range = ctx.h_unique_indices_table_range
        hash_table = ctx.hash_table
        embedding_table = ctx.embedding_table
        cache_hash_table = ctx.cache_hash_table
        optimizer = ctx.optimizer
        caching = ctx.caching

        # Clip gradients if needed
        if optimizer.need_gradient_clipping():
            optimizer.clip_gradient(grads)

        # Reduce gradients
        unique_indices, unique_grads = reduce_grads(
            indices, grads, indices_table_range, h_indices_table_range
        )
        optimizer.step()

        # Update embeddings for all tables
        if caching:
            dynamicemb_update_with_cache(
                storage_hash_table=hash_table,
                cache_hash_table=cache_hash_table,
                embedding_table=embedding_table,
                unique_indices=unique_indices,
                unique_grads=unique_grads,
                h_unique_indices_table_range=h_unique_indices_table_range,
                optimizer=optimizer,
            )
        else:
            dynamicemb_update_no_cache(
                hash_table=hash_table,
                embedding_table=embedding_table,
                unique_indices=unique_indices,
                unique_grads=unique_grads,
                h_unique_indices_table_range=h_unique_indices_table_range,
                optimizer=optimizer,
            )

        return (None,) * 16


def _create_scores_for_table(
    hash_table: GroupedScoredHashTable,
    num_keys: int,
    device: torch.device,
    evict_strategy: Optional[EvictStrategy],
    accumulated_frequency: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """
    Create scores tensor based on eviction strategy.

    Args:
        hash_table: GroupedScoredHashTable
        num_keys: Number of keys
        device: Device for tensor
        evict_strategy: Eviction strategy
        accumulated_frequency: Pre-computed accumulated frequency for LFU

    Returns:
        Scores tensor or None
    """
    if accumulated_frequency is not None and evict_strategy == EvictStrategy.KLfu:
        return accumulated_frequency

    if evict_strategy == EvictStrategy.KLfu:
        return torch.ones(num_keys, device=device, dtype=torch.long)
    elif evict_strategy == EvictStrategy.KCustomized:
        # For customized strategy, the score should be set externally
        # Return None here, the caller should provide the score
        return None

    return None
