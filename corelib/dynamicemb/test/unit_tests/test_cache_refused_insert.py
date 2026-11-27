# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A key the cache cannot place stops the prefetch and leaves the storage intact.

The insert reports a negative slot for each incoming key it cannot place.
Its refusal counter travels to the host with the eviction and live-key counts.
A failed prefetch releases its hit pins and writes back only genuine victims.
Refused rows were never loaded and would overwrite the storage if written.

The cache here is one bucket (1024 slots, its buckets always hold 1024) plus an
overflow area of 3 buckets: one prefetch of more new keys than those 4096 slots
makes the insert refuse the rest; nothing checks the batch before the insert
does. Unwritten memory is filled with NaN so that a row stored without being
written shows.
"""

import pytest
import torch
import dynamicemb.batched_dynamicemb_function as lookup
from dynamicemb.device_count import full_count
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.key_value_table import get_insert_score_arg, load_from_flat, store_to_flat
from dynamicemb.scored_hashtable import ScoreArg, ScorePolicy
from dynamicemb.types import CopyMode

DIM = 8
STORAGE_CAPACITY = 1 << 16
CACHE_ROWS = 1024  # one bucket of the cache
CACHE_SLOTS = 4 * CACHE_ROWS  # the bucket and the overflow area


@pytest.fixture(autouse=True)
def nan_for_unwritten_memory():
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    yield
    torch.utils.deterministic.fill_uninitialized_memory = fill
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def make_model(cache_rows=CACHE_ROWS, dim=DIM):
    assert torch.cuda.is_available()
    options = [
        DynamicEmbTableOptions(
            dim=dim,
            max_capacity=STORAGE_CAPACITY,
            init_capacity=STORAGE_CAPACITY,
            bucket_capacity=128,
            device_id=torch.cuda.current_device(),
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            # SGD keeps no optimizer state: a row is DIM floats.
            local_hbm_for_values=cache_rows * dim * 4,
            caching=True,
            # A row holds its key: key % 100000 in every column.
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.DEBUG
            ),
        )
    ]
    model = BatchedDynamicEmbeddingTablesV2(
        table_options=options,
        table_names=["t0"],
        feature_table_map=[0],
        output_dtype=torch.float32,
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=EmbOptimType.SGD,
        learning_rate=0.25,
    )
    cache = model._cache
    assert cache is not None
    km = cache._state.key_index_map
    # One bucket and an overflow area of three: the state's capacity counts both.
    assert km.capacity_ == cache_rows
    assert km.overflow_bucket_capacity_ == 3 * CACHE_ROWS
    assert cache._state.capacity == cache_rows + km.overflow_bucket_capacity_
    return model


def prefetch(model, keys):
    """The module's prefetch of one feature holding ``keys``; the state it queued."""
    offsets = torch.tensor([0, keys.numel()], dtype=torch.int64, device=keys.device)
    model.prefetch(keys, offsets)
    return model._prefetch_states[-1]


@pytest.mark.parametrize("preload_backing", [False, True])
def test_refused_keys_stop_the_prefetch_and_keep_the_storage(preload_backing):
    dim = 128 if preload_backing else DIM
    model = make_model(dim=dim)
    # The failing prefetch hits key 1 and evicts a dirty row for key 10001.
    warm_keys = torch.tensor([1, 10001], dtype=torch.int64, device="cuda")
    offsets = torch.tensor([0, 2], dtype=torch.int64, device="cuda")
    with torch.no_grad():
        model(warm_keys, offsets)
    victim = warm_keys[1:]
    victim_tid = torch.zeros_like(victim)
    _, found, slot = model._cache.lookup(victim, victim_tid, num_valid=full_count(victim))
    assert found.item()
    dirty_value = torch.full((1, dim), -7.0, device="cuda")
    store_to_flat(
        model._cache._state, slot, victim_tid, dirty_value, num_valid=full_count(slot)
    )
    keys = torch.arange(1, 6001, dtype=torch.int64, device="cuda")
    if preload_backing:
        # The zero-new report permits deferred reads until the same report
        # reveals refusals. The accepted rows must still get their values.
        stored = keys[1:].float()[:, None].expand(-1, dim).contiguous()
        model._storage.insert(keys[1:], torch.zeros_like(keys[1:]), stored)
    with pytest.raises(
        RuntimeError, match=rf"could not place {keys.numel() - CACHE_SLOTS} keys"
    ):
        prefetch(model, keys)
    torch.cuda.synchronize()

    # No pin is left behind.
    counter = model._cache._state.key_index_map._ref_counter
    assert torch.count_nonzero(counter).item() == 0

    # Every key went into the storage (it has room for all of them) with its
    # initial value. The negative slots of the refused keys must be excluded
    # from the writeback; their rows would contain NaN.
    table_ids = torch.zeros_like(keys)
    found = model._storage.find(
        keys, table_ids, copy_mode=CopyMode.EMBEDDING, num_valid=full_count(keys)
    )
    founds, values = found[5], found[7]
    assert bool(founds.all())
    expected = (keys % 100000).float()[:, None].expand(-1, dim)
    torch.testing.assert_close(values, expected, rtol=0, atol=0)
    if preload_backing:
        _, cached, slots = model._cache.lookup(keys, table_ids)
        reads = load_from_flat(
            model._cache._state, slots[cached], table_ids[cached], copy_mode=CopyMode.EMBEDDING
        )
        torch.testing.assert_close(reads, expected[cached], rtol=0, atol=0)
    found = model._storage.find(
        victim, victim_tid, copy_mode=CopyMode.EMBEDDING, num_valid=full_count(victim)
    )
    assert found[5].item()
    torch.testing.assert_close(found[7], dirty_value, rtol=0, atol=0)


def test_a_batch_that_fits_is_not_refused():
    model = make_model()
    keys = torch.arange(1, CACHE_ROWS + 1, dtype=torch.int64, device="cuda")
    state = prefetch(model, keys)
    torch.cuda.synchronize()
    assert bool((state.slot_indices >= 0).all())
    counter = model._cache._state.key_index_map._ref_counter
    assert counter.sum().item() == keys.numel()


def test_dense_reservation_evicts_oldest_unpinned_rows(monkeypatch):
    monkeypatch.delenv("DEMB_DETERMINISM_MODE", raising=False)
    model = make_model()
    # One full main bucket. Hold 25 rows for an outstanding prefetch and set
    # explicit LRU times so the pinned rows are also the oldest ones.
    old = torch.arange(CACHE_ROWS, device="cuda")
    tids = torch.zeros_like(old)
    with torch.no_grad():
        model(old, old.new_tensor([0, old.numel()]))
    prefetch(model, old[:25])
    table = model._cache._state.key_index_map
    name = table.score_names_[0]
    _, present, _ = table.lookup(old, tids, ScoreArg(name, old, ScorePolicy.ASSIGN))
    assert present.all()

    # More than 4K input rows choose ranked reservation; 500 unique misses
    # must displace exactly the first 500 unpinned rows.
    incoming = torch.arange(CACHE_ROWS, CACHE_ROWS + 500, device="cuda")
    exchange = lookup.exchange_cache_storage

    def publish_before_exchange(*args, **kwargs):
        # A callback or another stream can make a prepared miss resident.
        # Reservation must find it and use its existing slot.
        table.insert(incoming[:1], tids[:1], ScoreArg(name, None, ScorePolicy.GLOBAL_TIMER))
        return exchange(*args, **kwargs)

    monkeypatch.setattr(lookup, "exchange_cache_storage", publish_before_exchange)
    prefetch(model, incoming.repeat(10))
    readonly = ScoreArg(name, None, ScorePolicy.CONST)
    _, present, _ = table.lookup(old, tids, readonly)
    torch.testing.assert_close(present, (old < 25) | (old >= 525))
    _, present, slots = table.lookup(incoming, torch.zeros_like(incoming), readonly)
    assert present.all() and slots.unique().numel() == incoming.numel()
    assert int(table._ref_counter.sum()) == 525


@pytest.mark.parametrize(
    ("num_misses", "scores"),
    [(8, "ascending"), (96, "ascending"), (31, "equal"),
     (31, "shared_bits"), (31, "wide"), (31, "cross_word")],
)
def test_sparse_cache_reservation_keeps_oldest_unpinned_order(monkeypatch, num_misses, scores):
    monkeypatch.delenv("DEMB_DETERMINISM_MODE", raising=False)
    model = make_model(cache_rows=16 * CACHE_ROWS)
    table = model._cache._state.key_index_map
    # Concentrate all new keys in one bucket of an otherwise much larger
    # cache. Eight misses skip ranking, 31 use the short list, and 96
    # exhaust it and exercise the ordinary exact LRU search.
    keys = torch.arange(1, 48 * CACHE_ROWS, device="cuda")
    grouped, offsets, _ = table.bucketize_keys(keys, torch.zeros_like(keys))
    assert offsets[1].item() >= CACHE_ROWS + num_misses
    old = grouped[:CACHE_ROWS].contiguous()
    incoming = grouped[CACHE_ROWS:CACHE_ROWS + num_misses].contiguous()
    tids = torch.zeros_like(old)
    with torch.no_grad():
        model(old, old.new_tensor([0, old.numel()]))
    prefetch(model, old[:25])
    score = torch.arange(CACHE_ROWS, device=old.device)
    if scores == "equal":
        score.zero_()
    elif scores == "shared_bits":
        score = (1 << 40) | ((score % 19) << 10) | 5
    elif scores == "wide":
        score = ((score % 3) << 36) | (score % 19)
    elif scores == "cross_word":
        score = (1 << 33) - 2048 + (score * 71 % 4096)
    name = table.score_names_[0]
    _, present, old_slots = table.lookup(old, tids, ScoreArg(name, score, ScorePolicy.ASSIGN))
    assert present.all()
    # Equal scores preserve the scalar reducer's physical slot tie-break.
    order = old_slots[25:].argsort() + 25
    victims = order[score[order].argsort(stable=True)[:num_misses]]
    expected = torch.ones_like(present)
    expected[victims] = False

    # Capacity >=4096 selects the cooperative reservation, while the exact
    # device count is only num_misses and the spare suffix is uninitialized.
    prefetch(model, incoming.repeat((4096 + num_misses - 1) // num_misses))
    readonly = ScoreArg(name, None, ScorePolicy.CONST)
    _, present, _ = table.lookup(old, tids, readonly)
    torch.testing.assert_close(present, expected)
    _, present, slots = table.lookup(incoming, tids[:num_misses], readonly)
    assert present.all() and slots.unique().numel() == num_misses
    assert int(table._ref_counter.sum()) == 25 + num_misses


def test_failed_preparation_preserves_dirty_victim_and_hit_pins(monkeypatch):
    model = make_model()
    warm = torch.tensor([1, 10001], device="cuda")
    offsets = warm.new_tensor([0, 2])
    model(warm, offsets).sum().backward()  # Both cache rows are now dirty.

    one_row = warm.new_tensor([0, 1])
    in_flight = model(warm[:1], one_row)  # An earlier batch owns a pin on key 1.
    counter = model._cache._state.key_index_map._ref_counter
    pins = counter.clone()
    assert pins.sum().item() == 1

    # Key 1 is a cache hit. Filling the remaining slots displaces key 10001.
    keys = torch.arange(1, CACHE_SLOTS + 1, device="cuda")
    offsets = keys.new_tensor([0, keys.numel()])
    prepare = lookup.prepare_cache_lookup
    fail_once = True

    def fail_during_compaction(*args, **kwargs):
        nonlocal fail_once
        if fail_once:
            fail_once = False
            # The first probe pins the warm row. An invalid weight column then
            # fails while the native op is packing the misses.
            return prepare(
                *args[:-2], torch.zeros_like(args[0], dtype=torch.float32), args[-1], **kwargs
            )
        return prepare(*args, **kwargs)

    monkeypatch.setattr(lookup, "prepare_cache_lookup", fail_during_compaction)
    with pytest.raises(RuntimeError, match="8-byte"):
        model.prefetch(keys, offsets)
    torch.testing.assert_close(counter, pins)
    in_flight.backward(torch.zeros_like(in_flight))
    assert torch.count_nonzero(counter).item() == 0

    # The retry must get fresh rows for the cold keys and the trained values for
    # both warm keys, including the one displaced during the failed prefetch.
    with torch.no_grad():
        recovered = model(keys, offsets)
        victim = model(warm[1:], one_row)
    expected = keys.float()
    expected[0] -= 0.25
    torch.testing.assert_close(recovered, expected[:, None].expand(-1, DIM))
    torch.testing.assert_close(victim, torch.full_like(victim, 10001 - 0.25))
    assert torch.count_nonzero(counter).item() == 0


@pytest.mark.parametrize(
    "host_limit,device_limit,live",
    [(None, None, 6000), (None, 5800, 5800), (5900, None, 5900), (5800, 5900, 5800)],
)
def test_the_insert_counts_the_keys_it_refuses(host_limit, device_limit, live):
    model = make_model()
    state = model._cache._state
    km = state.key_index_map
    keys = torch.arange(1, 6001, dtype=torch.int64, device="cuda")
    num_valid = None if device_limit is None else keys.new_tensor([device_limit])
    table_ids = torch.zeros_like(keys)
    score = get_insert_score_arg(state, keys.numel(), keys.device, None)
    (
        slots,
        num_evicted,
        _,
        evicted_indices,
        _,
        _,
        num_refused,
    ) = km.insert_and_evict_with_counter_and_overflow(
        keys,
        table_ids,
        score,
        report_refused=True,
        num_valid=num_valid,
        host_num_valid=host_limit,
    )
    # The empty cache evicts nothing: every "evicted" key is one it refused, the
    # keys past its slots, each with a negative index.
    assert slots.numel() == live
    assert num_refused == live - CACHE_SLOTS
    assert num_evicted.is_cuda and int(num_evicted) == num_refused
    assert int((evicted_indices < 0).sum()) == num_refused

    # Without the refusal report, all six outputs stay on the device.
    more = torch.arange(10001, 10011, dtype=torch.int64, device="cuda")
    score = get_insert_score_arg(state, more.numel(), more.device, None)
    result = km.insert_and_evict_with_counter_and_overflow(
        more,
        torch.zeros_like(more),
        score,
    )
    assert len(result) == 6


def test_storage_host_bound_guards_table_ids_and_value_row_maps():
    storage = make_model()._storage
    keys = torch.tensor([31, 41, 47], device="cuda")
    table_ids = torch.zeros_like(keys)
    invalid_table_ids = torch.tensor([0, 0, -1], device="cuda")
    rows = torch.tensor([2, 0, -999], device="cuda")
    values = torch.arange(3 * DIM, dtype=torch.float32, device="cuda").view(3, DIM)
    storage.insert(
        keys,
        invalid_table_ids,
        values,
        input_rows=rows,
        host_num_valid=2,
        num_valid=keys.new_tensor([3]),
    )

    # A count-free find looks up the entire input and still returns a device count.
    missing, missing_keys, _, _, _, found, _, result = storage.find(
        keys, table_ids, CopyMode.VALUE
    )
    assert missing.is_cuda and missing.item() == 1
    assert missing_keys.shape == keys.shape
    assert missing_keys[0].item() == 47
    assert found.tolist() == [True, True, False]
    torch.testing.assert_close(result[:2], values[[2, 0]])

    missing, *_ = storage.find(
        keys, invalid_table_ids, CopyMode.VALUE, host_num_valid=2
    )
    assert missing.is_cuda and missing.item() == 0


def test_exchange_deterministic_waves_protect_earlier_placements(monkeypatch):
    model = make_model()
    # Keep every slot but one main-table slot pinned by another batch. A
    # deterministic insert processes these two cache misses in separate waves.
    residents = torch.arange(1, CACHE_SLOTS + 1, device="cuda")
    with torch.no_grad():
        model(residents, residents.new_tensor([0, residents.numel()]))
    _, hits, slots = model._cache.lookup(residents, torch.zeros_like(residents))
    assert hits.all()
    held = residents[slots != 0]
    in_flight = model(held, held.new_tensor([0, held.numel()]))
    counter = model._cache.key_index_map._ref_counter
    pins = counter.clone()
    assert pins.sum().item() == CACHE_SLOTS - 1 and pins[0].item() == 0

    monkeypatch.setenv("DEMB_DETERMINISM_MODE", "1")
    incoming = residents.new_tensor([10000, 10001])
    with pytest.raises(RuntimeError, match="could not place 1 keys"):
        prefetch(model, incoming)
    # The first wave holds the last available slot until the second wave
    # reports refusal. Neither wave leaves a pin on the failed batch's behalf.
    _, found, _ = model._cache.lookup(incoming, torch.zeros_like(incoming))
    assert found.tolist() == [True, False]
    torch.testing.assert_close(counter, pins)
    in_flight.backward(torch.zeros_like(in_flight))
    assert counter.count_nonzero().item() == 0
