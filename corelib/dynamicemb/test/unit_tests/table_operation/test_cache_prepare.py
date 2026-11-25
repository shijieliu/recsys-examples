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
from dynamicemb_extensions import CurandStateContext, prepare_cache_lookup


def _ids(values):
    return torch.tensor(values, dtype=torch.int64, device="cuda")


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
