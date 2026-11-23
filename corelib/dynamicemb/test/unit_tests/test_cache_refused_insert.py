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
from dynamicemb.key_value_table import get_insert_score_arg, store_to_flat
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


def make_model():
    assert torch.cuda.is_available()
    options = [
        DynamicEmbTableOptions(
            dim=DIM,
            max_capacity=STORAGE_CAPACITY,
            init_capacity=STORAGE_CAPACITY,
            bucket_capacity=128,
            device_id=torch.cuda.current_device(),
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            # SGD keeps no optimizer state: a row is DIM floats.
            local_hbm_for_values=CACHE_ROWS * DIM * 4,
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
    assert km.capacity_ == CACHE_ROWS
    assert km.overflow_bucket_capacity_ == 3 * CACHE_ROWS
    assert cache._state.capacity == CACHE_SLOTS
    return model


def prefetch(model, keys):
    """The module's prefetch of one feature holding ``keys``; the state it queued."""
    offsets = torch.tensor([0, keys.numel()], dtype=torch.int64, device=keys.device)
    model.prefetch(keys, offsets)
    return model._prefetch_states[-1]


def test_refused_keys_stop_the_prefetch_and_keep_the_storage():
    model = make_model()
    # The failing prefetch hits key 1 and evicts a dirty row for key 10001.
    warm_keys = torch.tensor([1, 10001], dtype=torch.int64, device="cuda")
    offsets = torch.tensor([0, 2], dtype=torch.int64, device="cuda")
    with torch.no_grad():
        model(warm_keys, offsets)
    victim = warm_keys[1:]
    victim_tid = torch.zeros_like(victim)
    _, found, slot = model._cache.lookup(victim, victim_tid, num_valid=full_count(victim))
    assert found.item()
    dirty_value = torch.full((1, DIM), -7.0, device="cuda")
    store_to_flat(
        model._cache._state, slot, victim_tid, dirty_value, num_valid=full_count(slot)
    )
    keys = torch.arange(1, 6001, dtype=torch.int64, device="cuda")
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
    expected = (keys % 100000).float()[:, None].expand(-1, DIM)
    torch.testing.assert_close(values, expected, rtol=0, atol=0)
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


def test_failed_backing_load_preserves_dirty_victim_and_hit_pins(monkeypatch):
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
    load_rows = lookup.load_from_flat
    fail_once = True

    def fail_backing_load(state, *args, **kwargs):
        nonlocal fail_once
        if state is model._storage._state and fail_once:
            fail_once = False
            raise torch.cuda.OutOfMemoryError("injected backing load failure")
        return load_rows(state, *args, **kwargs)

    monkeypatch.setattr(lookup, "load_from_flat", fail_backing_load)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="injected backing load"):
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


@pytest.mark.parametrize("padding", [0, 200])
def test_the_insert_counts_the_keys_it_refuses(padding):
    model = make_model()
    state = model._cache._state
    km = state.key_index_map
    keys = torch.arange(1, 6001, dtype=torch.int64, device="cuda")
    live = keys.numel() - padding
    num_valid = keys.new_full((1,), live)
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
        keys, table_ids, score, report_refused=True, num_valid=num_valid
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
        more, torch.zeros_like(more), score,
        num_valid=full_count(more),
    )
    assert len(result) == 6
