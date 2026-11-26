# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import dynamicemb_extensions as ops


@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("inverse_dtype", [torch.int64, torch.uint64])
def test_sorted_pooled_sgd_cache_rows(offset, inverse_dtype):
    # Includes groups on both sides of 32-occurrence chunk boundaries. The two
    # hottest keys span several of the previous reduction's atomic windows.
    lengths = [1, 12, 31, 33, 65, 97, 1, 1445, 8193, 11]
    n, live = sum(lengths), len(lengths)
    device = "cuda"
    def longs(x):
        return torch.tensor(x, dtype=torch.int64, device=device)

    torch.manual_seed(7331)
    inverse = torch.repeat_interleave(torch.arange(live), torch.tensor(lengths))
    inverse = inverse[torch.randperm(n)].to(device=device, dtype=inverse_dtype)
    keys_per_table = 256
    batch = n // 6
    offsets = torch.linspace(0, n, 2 * batch + 1, device=device).long()
    raw = torch.randn(batch * 128 + offset, device=device)
    # Small integers make every possible addition order exactly representable.
    # Both raw casts deliberately retain the optionally unaligned view.
    grad = raw[offset:].round_().reshape(batch, 128)
    capacity = live + 17
    table_ids = torch.arange(capacity, device=device) % 2
    slots = torch.arange(capacity, device=device) // 2
    slots[1::3] += keys_per_table
    rows = slots.clone()
    slots[2] = -1
    rows[3] = -1
    table_offsets, overflow_offsets, count = longs([0, 2, 4]), longs([256, 256]), longs([live])
    pin_args = (table_offsets, 128, overflow_offsets, 256)
    pins = torch.full((1024,), 3, device=device, dtype=torch.int32)
    expected_pins = pins.clone()
    ref, actual = [], []
    for _ in range(2):
        base = torch.randn(512 * 64 + offset, device=device)
        ref.append(base.clone()[offset:].reshape(512,64))
        actual.append(base.clone()[offset:].reshape(512,64))

    def ptrs(tables):
        return longs([t.data_ptr() for t in tables])

    # Use an aligned gradient as the reference reduction kernel requests
    # aligned FP32 vectors. The new cache operation accepts either alignment.
    reduced = ops.reduce_grads(inverse, grad.clone(), capacity, batch, 64, offsets,
                               None, ops.PoolingMode.KSum, 128, None, True)
    ops.sgd_update_and_unpin_cache(reduced, rows, slots, table_ids, ptrs(ref),
                                   *pin_args, expected_pins, count, .125)
    ops.reduce_pooled_cache_sgd(inverse, grad, offsets, batch, rows, slots, table_ids,
                                ptrs(actual), *pin_args, pins, count, .125)
    for a, b in zip(actual, ref):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))
    assert torch.equal(pins, expected_pins)
