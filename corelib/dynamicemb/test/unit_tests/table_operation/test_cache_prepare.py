# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from dynamicemb.scored_hashtable import (
    ScoreArg,
    ScorePolicy,
    ScoreSpec,
    get_scored_table,
)
from dynamicemb.utils import decode_table_values
from dynamicemb_extensions import (
    CurandStateContext,
    PoolingMode,
    exchange_cache_storage,
    gather_embedding_pooled,
    prepare_cache_lookup,
)


def _ids(values):
    return torch.tensor(values, dtype=torch.int64, device="cuda")


def _maps(cache, backing):
    cache_map = (
        cache.table_storage_,
        cache.table_bucket_offsets_,
        cache.bucket_capacity_,
        cache._ref_counter,
        cache.overflow_table_storage_,
        cache.overflow_bucket_capacity_,
        cache.overflow_output_offsets_,
        cache.num_scores_,
        cache.table_global_capacities_,
    )
    backing_map = (
        backing.table_storage_,
        backing.table_bucket_offsets_,
        backing.bucket_capacity_,
        backing.rows_,
        backing.num_scores_,
        backing.table_global_capacities_,
    )
    return cache_map, backing_map


def _fixture(dtype):
    def table(**layout):
        return get_scored_table(
            capacity=[256, 128],
            bucket_capacity=128,
            key_type=torch.int64,
            score_specs=[ScoreSpec(name="score", policy=ScorePolicy.ACCUMULATE)],
            **layout,
        )

    cache = table(enable_overflow=True)
    backing = table(use_rows=True)
    const = ScoreArg("score", None, ScorePolicy.CONST)
    cache.insert(_ids([11, 12]), _ids([0, 1]), const)
    keys = _ids([11, 12, 13, 10, 11, 12, 11, 12])
    tids = _ids([0, 0, 0, 1, 1, 1])
    held = _ids([0, 1, 3, 5])
    rows = backing.insert(
        keys[held],
        tids[held],
        ScoreArg("score", torch.zeros_like(held), ScorePolicy.ASSIGN),
    )
    # Keep distinct raw values per table, row and coordinate. The second
    # table has wider embeddings; optimizer state starts at max_emb_dim in
    # the staged output, and at its table's own width in the backing buffer.
    widths = [9, 12]
    buffers = [
        (torch.arange(2 * width, device="cuda").reshape(2, width) + t * 16).to(dtype)
        for t, width in enumerate(widths)
    ]
    cache_map, backing_map = _maps(cache, backing)
    layout = (
        _ids([b.data_ptr() for b in buffers]),
        _ids([3, 4]),
        _ids(widths),
        4,
        12,
        dtype,
    )
    scores = torch.arange(1, keys.numel() + 1, device="cuda")
    args = (
        keys,
        _ids([0, 3, 6]),
        cache_map,
        backing_map,
        layout,
        (None, ScorePolicy.CONST),
        (scores, ScorePolicy.ACCUMULATE),
    )
    return args, buffers, rows, backing, (held, tids)


@pytest.mark.parametrize(
    "mode,dtype,words,ftrl",
    [
        ("constant", torch.float16, True, False),
        ("debug", torch.float32, False, True),
        ("constant", torch.float8_e4m3fn, True, False),
        ("uniform", torch.float32, False, False),
        ("normal", torch.bfloat16, False, False),
        ("truncated_normal", torch.float16, False, False),
    ],
)
def test_prepare_cache_rows(mode, dtype, words, ftrl):
    args, buffers, source_rows, backing, (held, tids) = _fixture(dtype)
    keys, _, cache_map, _, _, _, (scores, _) = args
    weights = scores + 10
    scalars = {
        "debug": [],
        "constant": [1.0],
        "uniform": [0.5, 1.5],
        "normal": [1.0, 0.25],
        "truncated_normal": [0.0, 1.0, -1.0, 1.0],
    }[mode]
    params = torch.tensor([[0.5], [1.0]], device="cuda") if mode == "constant" else None
    rng = (
        CurandStateContext().state
        if mode in {"uniform", "normal", "truncated_normal"}
        else None
    )
    init = (mode, scalars, params, rng, 0.75, words, ftrl)
    mask, slots, result_ids, n, pos, staged, new = prepare_cache_lookup(
        *args, init, weights, scores
    )
    mk, mt, values, mw, ms, absent = staged
    new_keys, new_ids, num_new, new_rows, new_pos, new_weights, new_scores = new
    assert mask[:6].tolist() == [False, True, True, True, True, False]
    assert torch.equal(mask[:6], slots[:6] == -1)
    assert int(cache_map[3].sum()) == 2
    assert result_ids[:6].tolist() == tids.tolist()
    assert int(n) == 4 and pos[:4].tolist() == [1, 2, 3, 4]
    assert mk[:4].tolist() == [12, 13, 10, 11] and mt[:4].tolist() == [0, 0, 1, 1]
    assert mw[:4].tolist() == weights[pos[:4]].tolist()
    assert ms[:4].tolist() == scores[pos[:4]].tolist()
    assert absent[:4].tolist() == [False, True, False, True]
    assert int(num_new) == 2 and new_rows[:2].tolist() == [1, 3]
    assert new_keys[:2].tolist() == [13, 11] and new_ids[:2].tolist() == [0, 1]
    assert new_pos[:2].tolist() == [2, 4]
    assert new_weights[:2].tolist() == weights[new_pos[:2]].tolist()
    assert new_scores[:2].tolist() == scores[new_pos[:2]].tolist()
    looked_up, *_ = backing.lookup(
        keys[held], tids[held], ScoreArg("score", None, ScorePolicy.CONST)
    )
    assert looked_up.tolist() == scores[held].tolist()

    for m, t, src, dim in [(0, 0, 1, 3), (2, 1, 2, 4)]:
        raw = buffers[t][int(source_rows[src])].view(torch.uint8)
        row = values[m].view(torch.uint8)
        word = values.element_size()
        assert torch.equal(row[: dim * word], raw[: dim * word])
        assert torch.equal(row[4 * word : (4 + 2 * dim) * word], raw[dim * word :])
    for m, dim in [(1, 3), (3, 4)]:
        emb = decode_table_values(values[m, :4], torch.float32)
        if mode == "constant":
            assert bool((emb == (0.5 if m == 1 else 1.0)).all())
        elif mode == "debug":
            assert bool((emb == mk[m]).all())
        elif mode == "uniform":
            assert bool(((emb >= 0.5) & (emb <= 1.5)).all())
        elif mode == "truncated_normal":
            assert bool((emb.abs() <= 1).all())
        else:
            assert bool(emb.isfinite().all()) and emb.unique().numel() > 1
        state = values[m, 4:]
        if words:
            state = state.view(torch.uint8).view(torch.float32)
        if ftrl:
            assert bool((state[:dim] == 0).all()) and bool(
                (state[dim : 2 * dim] == 0.75).all()
            )
        else:
            assert bool((state == 0.75).all())


def test_prepare_releases_hits_on_failure():
    args, buffers, _, _, _ = _fixture(torch.float32)
    init = ("constant", [0.0], None, None, 0.0, False, False)
    # The invalid compacted field is discovered after lookup pinned cache hits.
    with pytest.raises(RuntimeError, match="8-byte"):
        prepare_cache_lookup(
            *args, init, torch.zeros_like(args[0], dtype=torch.float32)
        )
    assert buffers and not bool(args[2][3].any())


@pytest.mark.parametrize("mode,params", [
    ("normal", [0.0, 0.75]),
    ("uniform", [-0.75, 0.75]),
    ("truncated_normal", [0.0, 0.4, -0.5, 0.5]),
])
def test_dense_copy_retains_rng_sequence(mode, params):
    # Use the same initialized curandStateXORWOW workers for both preparations.
    # Four warps repeatedly process interleaved backing hits and new rows.
    initialized = CurandStateContext().state
    props = torch.cuda.get_device_properties(initialized.device)
    workers = props.multi_processor_count * props.max_threads_per_multi_processor
    state = initialized.reshape(workers, -1)[:128].flatten().clone()
    uniform_state = state.clone()
    n, dim, width = 149, 64, 128
    cache = get_scored_table([512], bucket_capacity=128, enable_overflow=True)
    backing = get_scored_table([512], bucket_capacity=128, use_rows=True)
    keys = torch.arange(1, n + 1, device="cuda")
    held = torch.nonzero(keys % 11 < 8).flatten()
    physical = backing.insert(
        keys[held], torch.zeros_like(held), ScoreArg("timestamp", None, ScorePolicy.CONST)
    )
    buffer = torch.empty((n, width), device="cuda", dtype=torch.float16)
    buffer[physical] = keys[held, None].expand(-1, width).to(buffer.dtype)
    cache_map, backing_map = _maps(cache, backing)

    def prepare(widths, rng):
        layout = (_ids([buffer.data_ptr()]), *widths, dim, width, buffer.dtype)
        const = (None, ScorePolicy.CONST)
        return prepare_cache_lookup(
            keys, _ids([0, n]), cache_map, backing_map, layout, const, const,
            (mode, params, None, rng, 0.625, False, False))
    generic = prepare((_ids([dim]), _ids([width])), state)
    dense = prepare((None, None), uniform_state)
    assert int(generic[3]) == int(dense[3]) == n
    assert int(dense[6][2]) == n - held.numel()
    assert torch.equal(dense[5][2], generic[5][2])
    assert torch.equal(dense[5][5], generic[5][5])
    assert torch.equal(dense[5][2][held], buffer[physical])
    assert torch.equal(state, uniform_state)

    # Once every row exists, dense preparation only copies. The raw-copy
    # branch must leave all random workers at the same position.
    rest = torch.nonzero(keys % 11 >= 8).flatten()
    added = backing.insert(
        keys[rest], torch.zeros_like(rest), ScoreArg("timestamp", None, ScorePolicy.CONST)
    )
    buffer[added] = keys[rest, None].expand(-1, width).to(buffer.dtype)
    before = state.clone()
    generic = prepare((_ids([dim]), _ids([width])), state)
    dense = prepare((None, None), uniform_state)
    assert int(dense[6][2]) == 0 and not bool(dense[5][5].any())
    assert torch.equal(dense[5][2], generic[5][2])
    assert torch.equal(dense[5][2], keys[:, None].expand(-1, width).to(buffer.dtype))
    assert torch.equal(before, state) and torch.equal(before, uniform_state)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float8_e4m3fn])
def test_unaligned_dense_cache_moves_raw_words(dtype):
    def buffer(rows):
        element = torch.empty((), dtype=dtype).element_size()
        raw = torch.full((rows * 256 + 2 * element,), 91, device="cuda", dtype=torch.uint8)
        payload = raw[element:-element].view(rows, 256)
        values = payload.view(dtype)
        assert values.data_ptr() % 16 != 0
        layout = (_ids([values.data_ptr()]), None, None, values.shape[1], values.shape[1], dtype)
        return raw, payload, values, layout

    cache = get_scored_table([128], bucket_capacity=128, enable_overflow=True)
    backing = get_scored_table([1024], bucket_capacity=128, use_rows=True)
    cache_map, backing_map = _maps(cache, backing)
    keys = torch.arange(1, 130, device="cuda")
    tids = torch.zeros_like(keys)
    score = ScoreArg("timestamp", None, ScorePolicy.CONST)
    held = cache.insert(keys[:-1], tids[:-1], score)
    physical = backing.insert(keys, tids, score)
    cache_owner, cache_raw, _, cache_values = buffer(128 + cache.overflow_bucket_capacity_)
    back_owner, back_raw, _, backing_values = buffer(keys.numel())
    col = torch.arange(256, device="cuda")
    newest = (keys[:-1, None] + col).to(torch.uint8)
    cache_raw[held] = newest
    back_raw[physical[-1]] = (3 * col).to(torch.uint8)

    const = (None, ScorePolicy.CONST)
    _, slots, _, count, positions, staged, new = prepare_cache_lookup(
        keys[-1:], _ids([0, 1]), cache_map, backing_map, backing_values, const, const,
        ("constant", [0.0], None, None, 0.0, False, False))
    assert int(count) == 1 and int(new[2]) == 0
    assert torch.equal(staged[2][0].view(torch.uint8), back_raw[physical[-1]])
    stage_owner, stage_raw, stage_values, _ = buffer(1)
    stage_raw.copy_(staged[2].view(torch.uint8))
    mutation = (cache.bucket_sizes, cache.overflow_bucket_sizes, cache.score_fn_key_,
                backing.bucket_sizes, backing.row_counters_, backing.row_limits_, backing.score_fn_key_)
    attempted, evicted, refused, _ = exchange_cache_storage(
        cache_map, backing_map, cache_values, backing_values, mutation,
        (None, ScorePolicy.GLOBAL_TIMER), const, (staged[0], staged[1], None, positions, count),
        (new[0], new[1], new[3], new[2]), stage_values, slots, False, False)
    assert (attempted, evicted, refused) == (1, 1, 0)
    _, found, _ = cache.lookup(keys[:-1], tids[:-1], score)
    gone = torch.nonzero(~found).flatten()
    assert gone.numel() == 1
    assert torch.equal(back_raw[physical[gone]], newest[gone])
    assert torch.equal(cache_raw[slots[0]], stage_raw[0])
    for owner in (cache_owner, back_owner, stage_owner):
        assert int(owner[0]) == int(owner[-1]) == 91


@pytest.mark.parametrize("dim", [32, 64, 256, 512])
@pytest.mark.parametrize("mode, weighted", [
    (PoolingMode.KSum, False), (PoolingMode.KMean, False), (PoolingMode.KSum, True)
])
def test_pinned_pool_rows(dim, mode, weighted):
    # Two physical tables with unequal row strides. Some inputs point to a missing slot.
    # The input includes empty bags, repeated keys, partial thread blocks and long bags.
    device = "cuda"
    strides = [dim + 4, dim + 8]
    tables = [
        torch.arange(7 * stride + offset, dtype=torch.float, device=device)[offset:].view(7, stride)
        for offset, stride in enumerate(strides)
    ]
    counts = [0, 1, 4, 19, 33, 65, 2, 0, 17, 32]
    offsets = torch.tensor([0] + counts, device=device, dtype=torch.long).cumsum(0)
    n = sum(counts)
    tids = [1, 1, 0, 1, 0, 0, 1]
    slots = [3, -1, 3, 0, -1, 5, 1]
    positions = torch.arange(n, device=device) % len(tids)
    weights = (torch.arange(n, device=device, dtype=torch.float32) % 9 - 4) / 4 if weighted else None
    # A sliced output also exercises the scalar fallback for unaligned stores.
    offset = int(dim == 32 and weighted)
    output = torch.full((10 * dim + offset,), float("nan"), device=device)[offset:].view(5, 2 * dim)
    source = (
        torch.tensor([t.data_ptr() for t in tables], device=device), torch.tensor(strides, device=device),
        torch.tensor(slots, device=device), torch.tensor(tids, device=device), torch.float32,
    )
    gather_embedding_pooled(source, output, positions, offsets, mode, 2 * dim, 5,
                            None, dim, weights, True)
    zeros = torch.zeros(dim, device=device)
    values = torch.stack([tables[t][s, :dim] if s >= 0 else zeros for t, s in zip(tids, slots)])[positions]
    if weights is not None:
        values *= weights.unsqueeze(1)
    expected = torch.empty_like(output)
    start = 0
    for i, count in enumerate(counts):
        table, batch = divmod(i, 5)
        summed = values[start:start + count].sum(0)
        if mode == PoolingMode.KMean and count:
            summed /= count
        expected[batch, table * dim:(table + 1) * dim] = summed
        start += count
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-4)
