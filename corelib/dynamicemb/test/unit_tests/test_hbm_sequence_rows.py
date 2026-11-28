# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.key_value_table import _eval_lookup_storage
from dynamicemb.scored_hashtable import ScoreArg, ScorePolicy, get_scored_table
from dynamicemb_extensions import load_from_flat_table_emb, lookup_embedding_rows


def test_eval_probes_crowded_bucket_in_table_order():
    table = get_scored_table([128], bucket_capacity=128, use_rows=True)
    # A nearly full bucket exercises long probe chains and wraparound. Include
    # misses and reserved keys in the lookup alongside the inserted keys.
    keys = torch.arange(1, 122, dtype=torch.int64, device="cuda")
    score = ScoreArg("timestamp", None, ScorePolicy.CONST)
    table.insert(keys, torch.zeros_like(keys), score)
    queries = torch.arange(-3, 151, device="cuda")
    tids = torch.zeros_like(queries)
    _, found, rows = table.lookup(queries, tids, score)
    assert found.sum().item() == keys.numel()

    payload = torch.arange(128 * 4, dtype=torch.float32, device="cuda").view(128, 4)
    pointers = torch.tensor([payload.data_ptr()], device="cuda")
    layout = (pointers, None, None, 4, 4, payload.dtype)
    mapping = (table.table_storage_, table.table_bucket_offsets_, 128, table.rows_,
               table.num_scores_, table.table_global_capacities_)
    values, missing, _ = lookup_embedding_rows(queries, tids, (mapping, layout), None, None)
    assert torch.equal(missing, ~found)
    assert torch.equal(values[found], payload[rows[found]])

    # DynamicEmb rounds bucket sizes up to 16. A raw caller supplying only
    # four digest bytes cannot satisfy a read of one 16-byte digest vector.
    storage, offsets, _, row_map, words, capacities = mapping
    small = (storage[:storage.numel() // 32], offsets, 4, row_map[:4], words, capacities)
    with pytest.raises(RuntimeError, match="16-byte digest"):
        lookup_embedding_rows(queries, tids, (small, layout), None, None)


@pytest.mark.parametrize("early_empty", [False, True])
@pytest.mark.parametrize("same_as_empty_digest", [False, True])
def test_eval_digest_candidates_preserve_slot_order(early_empty, same_as_empty_digest):
    from dynamicemb.key_ownership import murmur3_fmix64

    def digest(key):
        return (int(murmur3_fmix64(key)) >> 32) & 255

    target = next(
        k for k in range(30000)
        if int(murmur3_fmix64(k)) & 112 == 112
        and (digest(k) == digest(-1)) == same_as_empty_digest
        and digest(k) != digest(-2)
    )
    collision = next(k for k in range(target + 1, target + 30000) if digest(k) == digest(target))
    # Starting at 112, a false digest match forces the probe to continue.
    # After wraparound, the key in slot 3 takes precedence over the empty
    # slot in the same four-slot word. An earlier empty word stops the probe.
    slots = [-2] * 128
    slots[112], slots[0], slots[3] = collision, -1, target
    if early_empty:
        slots[116] = -1
    table = get_scored_table([128], bucket_capacity=128, use_rows=True)
    storage = table.table_storage_
    storage[:1024].view(torch.int64).copy_(torch.tensor(slots, device="cuda"))
    storage[1024:1152].copy_(torch.tensor([digest(k) for k in slots], dtype=torch.uint8, device="cuda"))
    rows = torch.arange(128, device="cuda").roll(5)
    mapping = (storage, table.table_bucket_offsets_, 128, rows, table.num_scores_, None)
    values = torch.arange(128, dtype=torch.float32, device="cuda").reshape(128, 1)
    layout = (torch.tensor([values.data_ptr()], device="cuda"), None, None, 1, 1, values.dtype)
    keys = torch.tensor([target], device="cuda")
    output, missing, _ = lookup_embedding_rows(keys, torch.zeros_like(keys), (mapping, layout), None, None)
    assert missing.item() == early_empty
    if not early_empty:
        torch.testing.assert_close(output[0], values[rows[3]])


@pytest.mark.parametrize("table_stride", [128, 384])
@pytest.mark.parametrize("live,host_bound", [(6, None), (4, 5), (0, 5)])
@pytest.mark.parametrize("reorder", [False, True])
def test_uniform_sequence_row_copy(table_stride, live, host_bound, reorder):
    device = torch.device("cuda")
    tables = [
        torch.arange(4 * table_stride, device=device, dtype=torch.float32)
        .view(4, table_stride)
        .add_(i * 10000)
        for i in range(2)
    ]
    ptrs = torch.tensor([t.data_ptr() for t in tables], device=device)
    strides = torch.full((2,), table_stride, device=device)
    dims = torch.full((2,), 128, device=device)
    rows = [2, -1, 3, 1]
    tids = [1, 0, 0, 1]
    reverse = [2, 0, 0, 1, 3, 2]
    destinations = [4, 2, 5, 0, 3, 1] if reorder else list(range(6))
    # Invalid addresses outside the live prefix must never be dereferenced.
    entries = torch.tensor(reverse[:live] + [999] * (6 - live), device=device)
    out_rows = (
        torch.tensor(destinations, device=device, dtype=torch.int32) if reorder else None
    )
    # The aligned extra columns are part of the allocation, outside the logical view.
    out = torch.full((6, 132), -17.0, device=device)
    expected = out.clone()
    for i, entry in enumerate(reverse[:live]):
        if rows[entry] >= 0:
            expected[destinations[i], :128] = tables[tids[entry]][rows[entry], :128]
    load_from_flat_table_emb(
        ptrs,
        torch.tensor(rows, device=device),
        torch.tensor(tids, device=device),
        out[:, :128],
        strides,
        dims,
        128,
        True,
        entries,
        out_rows,
        num_valid=torch.tensor([live], device=device),
        host_num_valid=host_bound,
        uniform_value_dim=table_stride,
    )
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_eval_full_hbm_table_rows_and_per_table_misses():
    device = torch.device("cuda")
    options = [
        DynamicEmbTableOptions(
            dim=128,
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=torch.cuda.current_device(),
            max_capacity=4096,
            bucket_capacity=128,
            local_hbm_for_values=2**23,
            score_strategy=DynamicEmbScoreStrategy.NO_EVICTION,
            caching=False,
            eval_initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.CONSTANT, value=float(fill)
            ),
        )
        for fill in (-1, -2)
    ]
    model = BatchedDynamicEmbeddingTablesV2(
        table_options=options,
        table_names=["a", "b"],
        feature_table_map=[0, 1],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        output_dtype=torch.float32,
        optimizer=EmbOptimType.SGD,
    )
    storage = model._storage
    payload = torch.arange(3 * 128, device=device, dtype=torch.float32).view(3, 128)
    storage.insert(
        torch.tensor([1871, 1871, 913], device=device),
        torch.tensor([0, 1, 0], device=device),
        payload,
    )
    storage.training = False
    # Use a strided input, repeated keys, distinct bindings for the same key,
    # and an invalid table in the unused suffix of the count-bounded batch.
    keys = torch.tensor(
        [913, 0, 1871, 0, 217, 0, 1871, 0, 441, 0, 1871, 0, 33, 0],
        device=device,
        dtype=torch.int32,
    )[::2]
    tids = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1700, 0], device=device)[::2]
    out = _eval_lookup_storage(
        storage, keys, tids, model._eval_initializer, torch.tensor([6], device=device)
    )
    expect = torch.stack(
        [
            payload[2],
            payload[1],
            torch.full_like(payload[0], -1),
            payload[0],
            torch.full_like(payload[0], -2),
            payload[1],
        ]
    )
    torch.testing.assert_close(out[:6], expect, atol=0, rtol=0)


@pytest.mark.parametrize("hybrid,shared_feature", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("optimizer", [EmbOptimType.SGD, EmbOptimType.ADAM])
def test_mapped_eval_reads_each_tier_and_uses_table_defaults(hybrid, optimizer, shared_feature):
    from dynamicemb.key_value_table import DynamicEmbStorage, HybridStorage, _insert_key_values

    dev = torch.device("cuda")
    dim = 32
    value_dim = dim if optimizer == EmbOptimType.SGD else dim * 3
    options = [
        DynamicEmbTableOptions(
            dim=dim, index_type=torch.int64, embedding_dtype=torch.float32,
            device_id=torch.cuda.current_device(), max_capacity=4096, bucket_capacity=128,
            local_hbm_for_values=4096 * value_dim * 2 if hybrid else 0,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP, caching=False,
            eval_initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.CONSTANT, value=float(-100 - t)),
        )
        for t in range(2)
    ]
    model = BatchedDynamicEmbeddingTablesV2(
        table_options=options, table_names=["a", "b"], feature_table_map=[0, 0, 1] if shared_feature else [0, 1],
        pooling_mode=DynamicEmbPoolingMode.NONE, output_dtype=torch.float32, optimizer=optimizer,
    ).eval()
    storage = model._storage
    assert type(storage) is (HybridStorage if hybrid else DynamicEmbStorage)
    host = storage._host if hybrid else storage._state
    assert not host.tables[0].is_device_buffer()
    payload = torch.arange(4 * value_dim, dtype=torch.float32, device=dev).view(4, value_dim)
    keys = torch.tensor([17, 17, 29, 29], device=dev)
    tids = torch.tensor([0, 1, 0, 1], device=dev)
    if hybrid:
        _insert_key_values(storage._hbm, keys[:2], tids[:2], payload[:2], None, False)
        _insert_key_values(host, keys[2:], tids[2:], payload[2:], None, False)
    else:
        storage.insert(keys, tids, payload)
    q = torch.tensor([17, 29, 29, 47, -1, 17, 47, 29, 17, -3, 47, 29], device=dev)
    offsets = [0, 1, 3, 5, 7, 9, 12] if shared_feature else [0, 3, 7, 9, 12]
    out = model(q, torch.tensor(offsets, device=dev))
    reference_rows = []
    for i, key in enumerate(q.tolist()):
        tid = int(i >= 7)
        if key in (17, 29):
            source = tid + (2 if key == 29 else 0)
            reference_rows.append(payload[source, :dim])
        else:
            reference_rows.append(torch.full((dim,), float(-100 - tid), device=dev))
    reference = torch.stack(reference_rows)
    torch.testing.assert_close(out, reference, atol=0, rtol=0)

    # An independent device bound and host bound apply to the original rows
    # in the direct eval API; the invalid suffix table must never be addressed.
    ids = torch.tensor([0] * 7 + [1] * 2 + [9999] * 3, device=dev)
    bounded = _eval_lookup_storage(storage, q, ids, model._eval_initializer,
        torch.tensor([10], device=dev), host_num_valid=9)
    torch.testing.assert_close(bounded[:9], reference[:9], atol=0, rtol=0)

@pytest.mark.parametrize("tables,range_stride", [(1, 3), (7, 1), (7, 5), (41, 3)])
@pytest.mark.parametrize("shared_default,live,bound", [(True, 81, None), (False, 40, 59), (False, 17, 8), (False, 0, 90)])
def test_hbm_eval_compact_ranges(tables, range_stride, shared_default, live, bound):
    device = torch.device("cuda")
    lengths = [0] * tables
    if tables == 1:
        lengths[0] = 81
    else:
        lengths[1], lengths[3], lengths[-2] = 6, 29, 46
    # Poison the intervening bag offsets so a contiguous read cannot appear correct.
    bag_offsets = torch.full((tables * range_stride + 1,), -1, dtype=torch.int64, device=device)
    ranges = bag_offsets[::range_stride]
    ranges.copy_(torch.tensor([0] + lengths, device=device).cumsum(0))
    active = torch.tensor([t for t in range(tables) if lengths[t]], device=device)
    key_map = get_scored_table([128] * tables, bucket_capacity=128, use_rows=True)
    score = ScoreArg("timestamp", None, ScorePolicy.CONST)
    key_map.insert(torch.full_like(active, 19), active, score)
    dims = torch.tensor([13 + t % 9 for t in range(tables)], device=device)
    strides = torch.tensor([32 + (t % 3) * 4 for t in range(tables)], device=device)
    payloads = [torch.full((128, int(strides[t])), 1000. + t, device=device) for t in range(tables)]
    ptrs = torch.tensor([p.data_ptr() for p in payloads], device=device)
    layout = (ptrs, dims, strides, 24, 40, torch.float32)
    maps = (key_map.table_storage_, key_map.table_bucket_offsets_, 128, key_map.rows_,
            key_map.num_scores_, key_map.table_global_capacities_)
    keys = torch.tensor([19, 887, -1, 19] * 23, device=device)
    tids = torch.tensor([t for t,n in enumerate(lengths) for _ in range(n)] + [9999] * 11, device=device)
    count = torch.tensor([live], device=device)
    defaults = (torch.arange(24 * (1 if shared_default else tables), device=device, dtype=torch.float32)
                .view(-1, 24).add_(-500))
    common = [(maps, layout), None, None, count, bound, False]
    n = min(81, live, len(keys) if bound is None else bound)
    actual, missing, _ = lookup_embedding_rows(keys, None, *common, defaults, table_ranges=ranges)
    expected, _, _ = lookup_embedding_rows(keys, tids, *common, defaults)
    assert missing is None
    for i in range(n):
        width = int(dims[tids[i]]) if int(keys[i]) == 19 else 24
        torch.testing.assert_close(actual[i, :width], expected[i, :width], atol=0, rtol=0)
    # With no default tensor, preserve the original-order missing flags. A
    # capacity suffix after the final range can contain arbitrary values.
    _, missing, _ = lookup_embedding_rows(keys, None, *common, table_ranges=ranges)
    torch.testing.assert_close(missing[:n], keys[:n] != 19)


def test_eval_ranges_falls_back_for_custom_initializer():
    from dynamicemb.key_value_table import eval_lookup

    options = [
        DynamicEmbTableOptions(
            dim=128, index_type=torch.int64, embedding_dtype=torch.float32,
            device_id=torch.cuda.current_device(), max_capacity=4096,
            bucket_capacity=128, local_hbm_for_values=2**23,
            score_strategy=DynamicEmbScoreStrategy.NO_EVICTION, caching=False,
        )
        for _ in range(2)
    ]
    model = BatchedDynamicEmbeddingTablesV2(
        table_options=options, table_names=["a", "b"], feature_table_map=[0, 1],
        pooling_mode=DynamicEmbPoolingMode.NONE, output_dtype=torch.float32,
        optimizer=EmbOptimType.SGD,
    ).eval()
    model._storage.training = False
    keys = torch.tensor([11, 11, 21, 27, 9999, 9999], device="cuda")
    ranges = torch.tensor([0, 2, 4], device="cuda")

    def init(buffer, keys, count, tids, positions):
        selected = positions[:int(count)]
        assert selected.tolist() == [0, 1, 2, 3]
        assert tids[selected].tolist() == [0, 0, 1, 1]
        buffer[selected] = (keys[selected] + tids[selected] * 100 + selected).float()[:, None]

    out = eval_lookup(model._storage, keys, None, init, table_ranges=ranges,
                      num_valid=torch.tensor([8], device="cuda"), host_num_valid=5)
    torch.testing.assert_close(out[:4], torch.tensor([11.,12.,123.,130.], device="cuda")[:,None].expand(-1,128))


def test_mapped_eval_converts_host_values_for_mixed_dtype_hybrid():
    from dataclasses import replace
    from dynamicemb.key_value_table import HybridStorage, _insert_key_values
    from dynamicemb.optimizer import OptimizerArgs, SGDDynamicEmbeddingOptimizer

    primary = DynamicEmbTableOptions(
        dim=8, index_type=torch.int64, embedding_dtype=torch.float32,
        device_id=torch.cuda.current_device(), init_capacity=128, max_capacity=128,
        bucket_capacity=128, local_hbm_for_values=4096, caching=False,
    )
    backing = replace(primary, embedding_dtype=torch.float16, local_hbm_for_values=0)
    storage = HybridStorage([primary], [backing], SGDDynamicEmbeddingOptimizer(OptimizerArgs()))
    keys = torch.tensor([7, 11, 13, 11], device="cuda")
    ids = torch.zeros_like(keys)
    rows = torch.arange(16, device="cuda", dtype=torch.float32).view(2, 8) / 3
    _insert_key_values(storage._hbm, keys[:1], ids[:1], rows[:1], None, False)
    _insert_key_values(storage._host, keys[1:2], ids[1:2], rows[1:].half(), None, False)
    storage.training = False

    def fill_missing(output, keys, count, tids, positions):
        assert positions[:count.item()].tolist() == [2]
        output[2].fill_(-7)

    out = _eval_lookup_storage(storage, keys, ids, fill_missing)
    host_row = rows[1].half().float()
    expected = torch.stack((rows[0], host_row, torch.full_like(rows[0], -7), host_row))
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize("width,dtype,offset", [(128, torch.float32, 0), (13, torch.float16, 1)])
def test_grouped_mapped_eval_reuses_hot_row_without_writing_optimizer_words(width, dtype, offset):
    from dynamicemb_extensions import grouped_mapped_eval

    dev = torch.device("cuda")
    keys = torch.tensor([17, 29, 77, 17, 29, -1], device=dev)
    tids = torch.tensor([0, 0, 0, 1, 1, 1], device=dev)
    inputs = torch.tensor(([0] * 131 + [1, 2] * 17 + [3, 4, 5] * 29), device=dev)
    # Interleave the original destinations, including runs spanning multiple
    # 32-row groups after sorting by source. Repeated table defaults are runs.
    generator = torch.Generator().manual_seed(23)
    inputs = inputs[torch.randperm(inputs.numel(), generator=generator).to(dev)]
    n, live = inputs.numel() + 7, inputs.numel()
    inverse = torch.full((n,), 9999, device=dev)
    inverse[:live] = inputs
    unique = torch.full((n,), -1, device=dev)
    unique[:keys.numel()] = keys
    ranges = torch.tensor([0, 3, 6], device=dev)
    table = get_scored_table([128, 128], bucket_capacity=128, use_rows=True)
    score = ScoreArg("timestamp", None, ScorePolicy.CONST)
    selected = [0, 1, 3, 4]
    table.insert(keys[selected], tids[selected], score)
    _, _, rows = table.lookup(keys[selected], tids[selected], score)
    stride = width + offset + 3
    buffers = [torch.zeros((128, stride), dtype=dtype, pin_memory=True) for _ in range(2)]
    for item, row in zip(selected, rows.tolist()):
        buffers[tids[item].item()][row, offset:offset + width].copy_(
            torch.arange(width, dtype=dtype) + (item + 1) * 32
        )
    ptrs = torch.tensor([buf[:, offset:].data_ptr() for buf in buffers], device=dev)
    layout = ptrs, None, None, width, stride, dtype
    mapping = (table.table_storage_, table.table_bucket_offsets_, 128, table.rows_,
               table.num_scores_, table.table_global_capacities_)
    defaults = torch.empty((2, width), dtype=dtype, device=dev)
    defaults[0].fill_(-31)
    defaults[1].fill_(-55)
    result = grouped_mapped_eval(
        unique, ranges, inverse, torch.tensor([live], device=dev), (mapping, layout), defaults
    )
    expected = torch.stack([
        torch.arange(width, device=dev, dtype=dtype) + (i + 1) * 32
        if i in selected else defaults[tids[i]] for i in range(6)
    ])
    torch.testing.assert_close(result[:live], expected[inputs], atol=0, rtol=0)
    empty = grouped_mapped_eval(
        unique, ranges, inverse, torch.tensor([0], device=dev), (mapping, layout), defaults
    )
    assert empty.shape == result.shape


def test_grouped_mapped_eval_public_sequence_respects_device_tail(monkeypatch):
    import dynamicemb.key_value_table as kv

    dev = torch.device("cuda")
    options = [
        DynamicEmbTableOptions(
            dim=256, index_type=torch.int64, embedding_dtype=torch.float16,
            device_id=torch.cuda.current_device(), init_capacity=128, max_capacity=128,
            bucket_capacity=128, local_hbm_for_values=0, caching=False,
            eval_initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.CONSTANT, value=float(-t - 1)
            ),
        )
        for t in range(2)
    ]
    model = BatchedDynamicEmbeddingTablesV2(
        table_options=options, table_names=["a", "b"], feature_table_map=[0, 1],
        pooling_mode=DynamicEmbPoolingMode.NONE, output_dtype=torch.float32, optimizer=EmbOptimType.SGD,
    ).eval()
    q = torch.tensor([7, 9, 7, -1, 7, 9, 11, 9], device=dev)
    tids = torch.tensor([0, 1], device=dev)
    weights = torch.arange(512, device=dev, dtype=torch.float16).view(2, 256)
    model._storage.insert(q[[0, 5]], tids, weights)
    grouped = kv.grouped_mapped_eval
    calls = []

    def capture(*args):
        calls.append(args[0].shape)
        return grouped(*args)

    monkeypatch.setattr(kv, "_MIN_GROUPED_HOST_EVAL_KEYS", 1)
    monkeypatch.setattr(kv, "grouped_mapped_eval", capture)
    offsets = torch.tensor([0, 4, 8], device=dev)
    expected = torch.stack([
        weights[0], torch.full_like(weights[0], -1), weights[0], torch.full_like(weights[0], -1),
        torch.full_like(weights[0], -2), weights[1], torch.full_like(weights[0], -2), weights[1],
    ])
    torch.testing.assert_close(model(q, offsets), expected.float(), atol=0, rtol=0)
    # The populated input capacity runs past the final device-side table
    # boundary. Neither unique nor scatter may dereference that suffix.
    ranges = torch.tensor([0, 4, 7], device=dev)
    limited = kv.eval_lookup(model._storage, q, None, model._eval_initializer, table_ranges=ranges)
    torch.testing.assert_close(limited[:7], expected[:7], atol=0, rtol=0)
    assert len(calls) == 2
