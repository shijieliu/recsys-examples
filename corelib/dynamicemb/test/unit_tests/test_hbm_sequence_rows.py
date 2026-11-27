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
from dynamicemb_extensions import load_from_flat_table_emb


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
