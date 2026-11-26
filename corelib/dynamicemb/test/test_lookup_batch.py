# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from dynamicemb.lookup_batch import LookupBatch


def test_compact_lookup_batch_preserves_payload_and_row_maps():
    keys = torch.tensor([10, 11, 12, 13, 14], device="cuda")
    table_ids = torch.tensor([0, 1, 0, 1, 0], device="cuda")
    values = torch.arange(20, device="cuda").view(5, 4)
    weights = keys + 100
    scores = keys + 1000
    output_rows = torch.tensor([8, 9, 5, 7, 2], device="cuda")
    batch = LookupBatch(
        keys,
        table_ids,
        values=values,
        weights=weights,
        scores=scores,
        mask=torch.tensor([True, False, True, True, True], device="cuda"),
    )
    num_valid = torch.tensor([4], device="cuda")
    selected = batch.compact(num_valid, host_num_valid=3, output_rows=output_rows)

    assert selected.num_valid.item() == 2
    assert selected.source_values is values
    assert selected.input_rows[:2].tolist() == [0, 2]
    assert selected.output_rows[:2].tolist() == [8, 5]
    assert selected.keys[:2].tolist() == [10, 12]
    assert selected.table_ids[:2].tolist() == [0, 0]
    assert selected.weights[:2].tolist() == [110, 112]
    assert selected.scores[:2].tolist() == [1010, 1012]
