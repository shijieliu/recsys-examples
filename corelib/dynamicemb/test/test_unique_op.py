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

import pytest
import torch
from dynamicemb.device_count import full_count
from dynamicemb_extensions import (
    expand_table_ids_cuda,
    fill_rows,
    flagged_compact,
    gather_embedding,
    segmented_unique_cuda,
    select_insert_failed_values,
)


def make_segmented_range(table_assignments: torch.Tensor, num_tables: int, device):
    """Sort keys by table and return (sort_indices, segmented_range)."""
    sort_idx = torch.argsort(table_assignments, stable=True)
    sorted_assignments = table_assignments[sort_idx]
    counts = torch.bincount(sorted_assignments, minlength=num_tables).to(torch.int64)
    segmented_range = torch.zeros(num_tables + 1, dtype=torch.int64, device=device)
    segmented_range[1:] = torch.cumsum(counts, dim=0)
    return sort_idx, segmented_range


@pytest.fixture
def setup_device():
    assert torch.cuda.is_available()
    device_id = 0
    return torch.device(f"cuda:{device_id}")


# ============================================================================
# Segmented Unique Tests
# ============================================================================


def test_segmented_unique_basic(setup_device):
    """Test basic segmented unique operation with large input (1M keys)."""
    device = setup_device
    torch.cuda.get_device_properties(device).multi_processor_count

    num_tables = 10
    num_keys = 1_000_000
    num_unique_per_table = 10000  # Each table has ~10K unique keys

    # Generate keys with controlled uniqueness per table
    keys_raw = torch.randint(
        0, num_unique_per_table, (num_keys,), dtype=torch.int64, device=device
    )

    # Assign tables, sort keys by table to satisfy segmented_range contract
    table_assignments = torch.randint(
        0, num_tables, (num_keys,), dtype=torch.int64, device=device
    )
    sort_idx, segmented_range = make_segmented_range(
        table_assignments, num_tables, device
    )
    keys = keys_raw[sort_idx]

    (
        num_uniques,
        unique_keys,
        output_indices,
        table_offsets,
        freq_counters,
    ) = segmented_unique_cuda(keys, segmented_range, num_tables)
    torch.cuda.synchronize()

    # Check table offsets
    table_offsets_cpu = table_offsets.cpu()
    assert table_offsets_cpu[0].item() == 0, "First offset should be 0"

    # Verify offsets are non-decreasing
    for i in range(num_tables):
        assert (
            table_offsets_cpu[i + 1] >= table_offsets_cpu[i]
        ), "Table offsets should be non-decreasing"

    # Check that output indices correctly reconstruct keys
    unique_keys_cpu = unique_keys.cpu()
    output_indices_cpu = output_indices.cpu()
    keys_cpu = keys.cpu()

    reconstructed = unique_keys_cpu[output_indices_cpu]
    assert torch.equal(reconstructed, keys_cpu), "Reconstruction failed"

    # freq_counters should be empty when not requested
    assert (
        freq_counters.numel() == 0
    ), "freq_counters should be empty when not requested"

    total_unique = num_uniques.item()
    assert (
        total_unique == table_offsets_cpu[-1].item()
    ), "num_uniques should match table_offsets[-1]"
    print(
        f"Segmented unique basic test passed: {total_unique} unique from {num_keys} keys, {num_tables} tables"
    )


def test_segmented_unique_overlapping_keys(setup_device):
    """Test segmented unique with same keys in different tables (1M keys)."""
    device = setup_device
    torch.cuda.get_device_properties(device).multi_processor_count

    num_tables = 8
    num_keys = 1_000_000
    num_unique_keys = 1000  # Small unique key space to maximize overlaps

    # Same keys appear across all tables - should be counted separately per table
    keys_raw = torch.randint(
        0, num_unique_keys, (num_keys,), dtype=torch.int64, device=device
    )

    table_assignments = torch.randint(
        0, num_tables, (num_keys,), dtype=torch.int64, device=device
    )
    sort_idx, segmented_range = make_segmented_range(
        table_assignments, num_tables, device
    )
    keys = keys_raw[sort_idx]

    num_uniques, unique_keys, output_indices, table_offsets, _ = segmented_unique_cuda(
        keys, segmented_range, num_tables
    )
    torch.cuda.synchronize()

    table_offsets_cpu = table_offsets.cpu()

    # Each table should have at most num_unique_keys unique keys
    for i in range(num_tables):
        table_count = table_offsets_cpu[i + 1].item() - table_offsets_cpu[i].item()
        assert (
            table_count <= num_unique_keys
        ), f"Table {i} has more unique keys than possible"

    # Total unique count from table_offsets[num_tables]
    total_unique = table_offsets_cpu[num_tables].item()

    # Verify reconstruction
    unique_keys_cpu = unique_keys.cpu()
    output_indices_cpu = output_indices.cpu()
    keys_cpu = keys.cpu()

    reconstructed = unique_keys_cpu[output_indices_cpu]
    assert torch.equal(reconstructed, keys_cpu), "Reconstruction failed"

    print(
        f"Segmented unique overlapping keys test passed: {total_unique} unique from {num_keys} keys"
    )


def test_segmented_unique_empty_tables(setup_device):
    """Test segmented unique with some empty tables (1M keys)."""
    device = setup_device
    torch.cuda.get_device_properties(device).multi_processor_count

    num_tables = 10
    num_keys = 1_000_000

    # Assign keys only to active tables (tables 2, 5, 7 will be empty)
    active_tables = [0, 1, 3, 4, 6, 8, 9]
    active_tables_tensor = torch.tensor(active_tables, dtype=torch.int64, device=device)
    table_assignments = active_tables_tensor[
        torch.randint(
            0, len(active_tables), (num_keys,), dtype=torch.int64, device=device
        )
    ]

    keys_raw = torch.randint(0, 10000, (num_keys,), dtype=torch.int64, device=device)
    sort_idx, segmented_range = make_segmented_range(
        table_assignments, num_tables, device
    )
    keys = keys_raw[sort_idx]

    num_uniques, unique_keys, output_indices, table_offsets, _ = segmented_unique_cuda(
        keys, segmented_range, num_tables
    )
    torch.cuda.synchronize()

    table_offsets_cpu = table_offsets.cpu()

    # Check empty tables have 0 count
    empty_tables = [2, 5, 7]
    for t in empty_tables:
        count = table_offsets_cpu[t + 1].item() - table_offsets_cpu[t].item()
        assert count == 0, f"Table {t} should be empty, got {count} keys"

    # Check active tables have non-zero counts
    for t in active_tables:
        count = table_offsets_cpu[t + 1].item() - table_offsets_cpu[t].item()
        # Active tables should have some keys (unless extremely unlucky with random)
        # Just verify it doesn't exceed max possible
        assert count <= 10000, f"Table {t} has more unique keys than possible"

    # Verify reconstruction
    unique_keys_cpu = unique_keys.cpu()
    output_indices_cpu = output_indices.cpu()
    keys_cpu = keys.cpu()

    reconstructed = unique_keys_cpu[output_indices_cpu]
    assert torch.equal(reconstructed, keys_cpu), "Reconstruction failed"

    total_unique = num_uniques.item()
    print(
        f"Segmented unique empty tables test passed: {total_unique} unique, {len(empty_tables)} empty tables"
    )


def test_segmented_unique_empty_input(setup_device):
    """Test segmented unique with empty input."""
    device = setup_device
    torch.cuda.get_device_properties(device).multi_processor_count

    keys = torch.tensor([], dtype=torch.int64, device=device)
    num_tables = 3
    segmented_range = torch.zeros(num_tables + 1, dtype=torch.int64, device=device)

    (
        num_uniques,
        unique_keys,
        output_indices,
        table_offsets,
        freq_counters,
    ) = segmented_unique_cuda(keys, segmented_range, num_tables)
    torch.cuda.synchronize()

    assert unique_keys.numel() == 0, "Empty input should return empty unique keys"
    assert output_indices.numel() == 0, "Empty input should return empty indices"
    assert num_uniques.item() == 0, "Empty input should have 0 unique keys"
    assert (
        table_offsets.numel() == num_tables + 1
    ), "Table offsets should have num_tables+1 elements"
    assert torch.all(table_offsets == 0), "All offsets should be 0 for empty input"
    assert freq_counters.numel() == 0, "Empty input should return empty freq_counters"

    print("Segmented unique empty input test passed")


def test_segmented_unique_random(setup_device):
    """Test segmented unique with random data (1M keys)."""
    device = setup_device
    torch.cuda.get_device_properties(device).multi_processor_count

    num_tables = 16
    num_keys = 1_000_000

    keys_raw = torch.randint(0, 100000, (num_keys,), dtype=torch.int64, device=device)
    table_assignments = torch.randint(
        0, num_tables, (num_keys,), dtype=torch.int64, device=device
    )
    sort_idx, segmented_range = make_segmented_range(
        table_assignments, num_tables, device
    )
    keys = keys_raw[sort_idx]

    num_uniques, unique_keys, output_indices, table_offsets, _ = segmented_unique_cuda(
        keys, segmented_range, num_tables
    )
    torch.cuda.synchronize()

    # Verify reconstruction
    unique_keys_cpu = unique_keys.cpu()
    output_indices_cpu = output_indices.cpu()
    keys_cpu = keys.cpu()

    reconstructed = unique_keys_cpu[output_indices_cpu]
    assert torch.equal(reconstructed, keys_cpu), "Reconstruction failed for random test"

    # Verify table offsets are non-decreasing
    table_offsets_cpu = table_offsets.cpu()
    for i in range(num_tables):
        assert (
            table_offsets_cpu[i + 1] >= table_offsets_cpu[i]
        ), "Table offsets should be non-decreasing"

    total_unique = num_uniques.item()
    print(
        f"Segmented unique random test passed: {total_unique} unique from {num_keys} keys, {num_tables} tables"
    )


def test_segmented_unique_stress(setup_device):
    """Stress test with very large input (4M keys, many tables)."""
    device = setup_device
    torch.cuda.get_device_properties(device).multi_processor_count

    num_tables = 32
    num_keys = 4_000_000

    keys_raw = torch.randint(0, 500000, (num_keys,), dtype=torch.int64, device=device)
    table_assignments = torch.randint(
        0, num_tables, (num_keys,), dtype=torch.int64, device=device
    )
    sort_idx, segmented_range = make_segmented_range(
        table_assignments, num_tables, device
    )
    keys = keys_raw[sort_idx]

    # Warmup
    torch.cuda.synchronize()

    import time

    start = time.perf_counter()

    num_uniques, unique_keys, output_indices, table_offsets, _ = segmented_unique_cuda(
        keys, segmented_range, num_tables
    )
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    # Verify reconstruction
    unique_keys_cpu = unique_keys.cpu()
    output_indices_cpu = output_indices.cpu()
    keys_cpu = keys.cpu()

    reconstructed = unique_keys_cpu[output_indices_cpu]
    assert torch.equal(reconstructed, keys_cpu), "Reconstruction failed for stress test"

    total_unique = table_offsets.cpu()[-1].item()
    throughput = num_keys / elapsed / 1e6
    print(
        f"Segmented unique stress test: {total_unique} unique from {num_keys} keys in {elapsed*1000:.2f}ms ({throughput:.2f}M keys/s)"
    )


def test_segmented_unique_with_frequency_counters(setup_device):
    """Test segmented unique with frequency counting enabled."""
    device = setup_device
    torch.cuda.get_device_properties(device).multi_processor_count

    num_tables = 4
    num_keys = 100000

    keys_raw = torch.randint(0, 1000, (num_keys,), dtype=torch.int64, device=device)
    table_assignments = torch.randint(
        0, num_tables, (num_keys,), dtype=torch.int64, device=device
    )
    sort_idx, segmented_range = make_segmented_range(
        table_assignments, num_tables, device
    )
    keys = keys_raw[sort_idx]

    # Enable frequency counting by passing an empty tensor (numel==0)
    # This enables counting with each key occurrence counted as 1
    empty_freq_tensor = torch.empty(0, dtype=torch.int64, device=device)

    (
        num_uniques,
        unique_keys,
        output_indices,
        table_offsets,
        freq_counters,
    ) = segmented_unique_cuda(keys, segmented_range, num_tables, empty_freq_tensor)
    torch.cuda.synchronize()

    # freq_counters should have values
    total_unique = num_uniques.item()
    assert (
        freq_counters.numel() == num_keys
    ), "freq_counters should have num_keys elements"

    # Sum of frequencies should equal num_keys (each input counted once)
    freq_sum = freq_counters[:total_unique].sum().item()
    assert (
        freq_sum == num_keys
    ), f"Sum of frequencies should be {num_keys}, got {freq_sum}"

    # Verify reconstruction still works
    unique_keys_cpu = unique_keys.cpu()
    output_indices_cpu = output_indices.cpu()
    keys_cpu = keys.cpu()

    reconstructed = unique_keys_cpu[output_indices_cpu]
    assert torch.equal(
        reconstructed, keys_cpu
    ), "Reconstruction failed with freq counters"

    print(
        f"Segmented unique with frequency counters test passed: {total_unique} unique, freq_sum={freq_sum}"
    )


def test_segmented_unique_with_custom_frequencies(setup_device):
    """Test segmented unique with custom input frequencies."""
    device = setup_device
    torch.cuda.get_device_properties(device).multi_processor_count

    num_tables = 2
    num_keys = 1000

    keys_raw = torch.randint(0, 100, (num_keys,), dtype=torch.int64, device=device)
    table_assignments = torch.randint(
        0, num_tables, (num_keys,), dtype=torch.int64, device=device
    )
    sort_idx, segmented_range = make_segmented_range(
        table_assignments, num_tables, device
    )
    keys = keys_raw[sort_idx]

    # Custom frequencies: each key occurrence has frequency 2 (sorted same as keys)
    input_frequencies = torch.full((num_keys,), 2, dtype=torch.int64, device=device)

    (
        num_uniques,
        unique_keys,
        output_indices,
        table_offsets,
        freq_counters,
    ) = segmented_unique_cuda(keys, segmented_range, num_tables, input_frequencies)
    torch.cuda.synchronize()

    total_unique = num_uniques.item()

    # Sum of frequencies should equal 2 * num_keys (each input counted as 2)
    freq_sum = freq_counters[:total_unique].sum().item()
    assert (
        freq_sum == 2 * num_keys
    ), f"Sum of frequencies should be {2 * num_keys}, got {freq_sum}"

    # Verify reconstruction still works
    unique_keys_cpu = unique_keys.cpu()
    output_indices_cpu = output_indices.cpu()
    keys_cpu = keys.cpu()

    reconstructed = unique_keys_cpu[output_indices_cpu]
    assert torch.equal(
        reconstructed, keys_cpu
    ), "Reconstruction failed with custom freq"

    print(f"Segmented unique with custom frequencies test passed: freq_sum={freq_sum}")


def test_expand_table_ids(setup_device):
    """Test expand_table_ids_cuda (identity mapping, local_batch_size=1)."""
    device = setup_device

    # 3 tables with 5, 8, 3 keys respectively; total 16 elements
    counts = torch.tensor([5, 8, 3], dtype=torch.int64)
    segmented_range = torch.zeros(4, dtype=torch.int64, device=device)
    segmented_range[1:] = torch.cumsum(counts, dim=0).to(device)
    num_elements = segmented_range[-1].item()

    table_ids = expand_table_ids_cuda(segmented_range, num_elements)
    torch.cuda.synchronize()

    assert table_ids.numel() == num_elements
    assert table_ids.dtype == torch.int64

    table_ids_cpu = table_ids.cpu()
    expected = torch.repeat_interleave(torch.arange(3), counts)
    assert torch.equal(
        table_ids_cpu, expected
    ), f"table_ids mismatch: {table_ids_cpu} vs {expected}"

    # Edge: empty input
    empty_ids = expand_table_ids_cuda(segmented_range, 0)
    assert empty_ids.numel() == 0

    print(f"expand_table_ids test passed: {num_elements} elements, 3 tables")


# ============================================================================
# Flagged Compact Tests
# ============================================================================


def _flagged_compact_reference(flags, inputs):
    """Pure-PyTorch reference for flagged_compact."""
    idx = torch.where(flags)[0]
    h_count = idx.numel()
    outputs = []
    for t in inputs:
        if t is None:
            outputs.append(None)
        else:
            outputs.append(t[idx])
    return h_count, idx, outputs


@pytest.mark.parametrize(
    "N, flag_mode, input_spec",
    [
        pytest.param(1000, "random", ["t", "t"], id="basic"),
        pytest.param(512, "all_true", ["t"], id="all_true"),
        pytest.param(512, "all_false", ["t"], id="all_false"),
        pytest.param(0, "random", ["t"], id="empty_input"),
        pytest.param(256, "random", [], id="no_inputs"),
        pytest.param(500, "random", ["t", None], id="optional_none"),
        pytest.param(300, "random", [None, "t", None, "t"], id="multiple_none"),
        pytest.param(4_000_000, "random", ["t", "t", "t"], id="large"),
        pytest.param(1000, "random", ["t"] * 6, id="max_inputs"),
        pytest.param(100, "all_true", ["t"], id="preserves_dtype"),
    ],
)
def test_flagged_compact(setup_device, N, flag_mode, input_spec):
    device = setup_device

    if N == 0:
        flags = torch.empty(0, dtype=torch.bool, device=device)
    elif flag_mode == "all_true":
        flags = torch.ones(N, dtype=torch.bool, device=device)
    elif flag_mode == "all_false":
        flags = torch.zeros(N, dtype=torch.bool, device=device)
    else:
        flags = torch.randint(0, 2, (N,), dtype=torch.bool, device=device)

    inputs = []
    for spec in input_spec:
        if spec is None:
            inputs.append(None)
        elif N == 0:
            inputs.append(torch.empty(0, dtype=torch.int64, device=device))
        else:
            inputs.append(
                torch.randint(0, 2**60, (N,), dtype=torch.int64, device=device)
            )

    count, indices, outputs = flagged_compact(flags, inputs, full_count(flags))
    assert count.device == flags.device and count.dtype == torch.int64
    assert count.shape == (1,)
    h_count = count.item()
    ref_count, ref_idx, ref_outputs = _flagged_compact_reference(flags, inputs)

    assert h_count == ref_count, f"count mismatch: {h_count} vs {ref_count}"
    assert indices.shape == (N,)
    assert torch.equal(indices[:h_count], ref_idx), "indices mismatch"
    assert len(outputs) == len(ref_outputs)
    for i, (out, ref) in enumerate(zip(outputs, ref_outputs)):
        if ref is None:
            assert out is None, f"output {i} should be None"
        else:
            assert out.shape == (N,)
            assert torch.equal(out[:h_count], ref), f"output {i} mismatch"
            assert out.dtype == torch.int64, f"output {i} dtype mismatch"


@pytest.mark.parametrize("n", [-3, 0, 1, 31, 64, 80])
@pytest.mark.parametrize("invert", [False, True])
def test_flagged_compact_ignores_the_tail(setup_device, n, invert):
    capacity = 64
    live = min(max(n, 0), capacity)
    keys = torch.arange(capacity, device=setup_device)
    flags = keys % 3 == 0
    flags[live:] = not invert  # Every entry in the unused tail would be selected.
    num_valid = torch.tensor([-1, n, -1], device=setup_device)[1:2]

    count, positions, (selected, skip) = flagged_compact(
        flags, [keys.view(torch.uint64), None], num_valid, invert=invert
    )

    ref = (flags[:live] != invert).nonzero().flatten()
    assert count.item() == ref.numel()
    assert torch.equal(positions[: ref.numel()], ref)
    assert skip is None and selected.dtype == torch.uint64
    assert torch.equal(selected[: ref.numel()].view(torch.int64), keys[ref])


@pytest.mark.parametrize("n", [1, 2**21 + 1, 2**22])
def test_flagged_compact_large_prefix(setup_device, n):
    capacity = 2**22
    keys = torch.randint(0, 1 << 40, (capacity,), device=setup_device)
    flags = torch.rand(capacity, device=setup_device) < 0.3
    count, positions, (selected,) = flagged_compact(
        flags, [keys], torch.tensor([n], device=setup_device)
    )
    ref = flags[:n].nonzero().flatten()
    assert count.item() == ref.numel()
    assert torch.equal(positions[: ref.numel()], ref)
    assert torch.equal(selected[: ref.numel()], keys[ref])


def test_flagged_compact_rejects_a_seventh_input(setup_device):
    flags = torch.ones(8, dtype=torch.bool, device=setup_device)
    with pytest.raises(RuntimeError, match="at most 6"):
        flagged_compact(
            flags, [torch.arange(8, device=setup_device)] * 7, full_count(flags)
        )


@pytest.mark.parametrize("capacity", [0, 3])
def test_flagged_compact_requires_device_count(setup_device, capacity):
    flags = torch.ones(capacity, dtype=torch.bool, device=setup_device)
    with pytest.raises(TypeError):
        flagged_compact(flags, [])
    with pytest.raises(TypeError):
        flagged_compact(flags, [], None)
    with pytest.raises(RuntimeError, match="num_valid"):
        flagged_compact(flags, [], torch.tensor([capacity], device="cpu"))


def test_row_selection_validates_count_on_empty_input(setup_device):
    indices = torch.empty(0, dtype=torch.int64, device=setup_device)
    values = torch.empty((0, 4), device=setup_device)
    for invalid in (torch.tensor([0]), indices.new_zeros(1, dtype=torch.int32)):
        with pytest.raises(RuntimeError, match="num_valid"):
            select_insert_failed_values(indices, values, values, num_valid=invalid)


def test_flagged_compact_does_not_read_the_count(setup_device):
    keys = torch.arange(64, device=setup_device)
    flags = keys % 2 == 0
    n = torch.tensor([37], device=setup_device)
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        count, positions, _ = flagged_compact(flags, [keys, None], n)
        chained, _, _ = flagged_compact(flags, [], count, invert=True)
    finally:
        torch.cuda.set_sync_debug_mode(previous)

    assert count.item() == 19 and chained.item() == 9
    assert torch.equal(positions[:19], keys[:37:2])


# ============================================================================
# Tests with a device count
# ============================================================================
# ``num_valid``, a one-element int64 device tensor, says how many leading entries of the
# capacity-CAP arrays are live; the rest is garbage that no op may read or write. The
# reference for n is a torch computation on the exact slice [:n].

CAP = 1000
NS = [0, 1, CAP // 2, CAP - 1, CAP]
BIG = 1 << 22  # more entries than a capped grid has threads: the ops must stride
HUGE = 1 << 40
INT_OF_SIZE = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
# The n sweep on one layout, then every word width (1, 2, 4, 8, 16 bytes) and a row
# wider than a block (1100 columns) at one n. ``inner``: the dims of a row, () for 1-D.
ROW_CASES = [(n, torch.float32, (8,)) for n in NS] + [
    (CAP // 2, torch.bool, ()),
    (CAP // 2, torch.float16, (3,)),
    (CAP // 2, torch.float8_e4m3fn, (32,)),
    (CAP // 2, torch.float32, (3,)),
    (CAP // 2, torch.int64, ()),
    (CAP // 2, torch.uint64, (40,)),
    (CAP // 2, torch.bfloat16, (1100,)),
]


def _nv(n):
    """``num_valid`` as the step passes it: a view into a longer int64 tensor."""
    return torch.tensor([-7, n, 99], device="cuda")[1:2]


def _bits(t):
    """``t`` as a signed integer tensor of its element size: exact to compare, and
    indexable for the dtypes torch has few ops for (uint64)."""
    return t.contiguous().view(INT_OF_SIZE[t.element_size()])


def _same(a, b):
    return torch.equal(_bits(a), _bits(b))


def _rand(shape, dtype):
    if dtype == torch.float8_e4m3fn:
        # FP8 rows can also hold raw optimizer-state bytes.
        return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda").view(
            dtype
        )
    x = torch.randint(0, 100, shape, dtype=torch.int64, device="cuda")
    if dtype == torch.uint64:
        return x.view(torch.uint64)
    return x % 2 == 1 if dtype == torch.bool else x.to(dtype)


def _sentinel(shape, dtype):
    """An output pre-filled with an all-ones bit pattern, which no test value has."""
    out = torch.empty(shape, dtype=dtype, device="cuda")
    out.view(INT_OF_SIZE[out.element_size()]).fill_(-1)
    return out


def _garbage_idx(n, rows, unique=False):
    """(idx, sel): idx[:n] holds row ids in [0, rows) (distinct if ``unique``) or -1,
    idx[n:] is garbage, out-of-range ids of both signs; sel are the positions below n
    that hold a row id."""
    if unique:
        idx = torch.randperm(rows, device="cuda")[:CAP]
    else:
        idx = torch.randint(0, rows, (CAP,), device="cuda")
    idx[torch.rand(CAP, device="cuda") < 0.2] = -1
    sel = (idx[:n] >= 0).nonzero().flatten()
    junk = torch.randint(rows, HUGE, (CAP - n,), device="cuda")
    junk[1::2] *= -1
    idx[n:] = junk
    return idx, sel


@pytest.mark.parametrize("n, dtype, inner", ROW_CASES, ids=str)
def test_gather_embedding_with_device_count(n, dtype, inner):
    rows = 600
    src = _rand((rows, *inner), dtype)
    idx, sel = _garbage_idx(n, rows)
    out = gather_embedding(src, idx, num_valid=_nv(n))
    assert out.dtype == dtype and out.shape == (CAP, *inner)
    assert torch.equal(_bits(out)[sel], _bits(src)[idx[sel]])


@pytest.mark.parametrize("n, dtype, inner", ROW_CASES, ids=str)
def test_gather_embedding_with_output_rows(n, dtype, inner):
    shape = (CAP + 20, *inner)
    src = _rand((CAP, *inner), dtype)
    idx, sel = _garbage_idx(n, shape[0], unique=True)
    ref = _sentinel(shape, dtype)
    _bits(ref)[idx[sel]] = _bits(src)[sel]

    dst = _sentinel(shape, dtype)
    gather_embedding(src, dst, out_row=idx, num_valid=_nv(n))
    assert _same(dst, ref)  # the garbage tail wrote nothing
    dst = _sentinel(shape, dtype)
    gather_embedding(src[:n], dst, out_row=idx[:n], num_valid=_nv(n))
    assert _same(dst, ref)


@pytest.mark.parametrize(
    "n, dtype, value",
    [(n, torch.int64, -1) for n in NS] + [(CAP // 2, torch.int64, 7)]
    # a bool destination takes True for any value but 0
    + [(CAP // 2, torch.bool, v) for v in (True, 2, 0)],
    ids=str,
)
def test_fill_rows_with_device_count(n, dtype, value):
    idx, sel = _garbage_idx(n, CAP)
    ref = torch.zeros(CAP, dtype=dtype, device="cuda")
    ref[idx[sel]] = bool(value) if dtype == torch.bool else value

    dst = torch.zeros_like(ref)
    fill_rows(dst, value, indices=idx, num_valid=_nv(n))
    assert _same(dst, ref)  # the garbage tail wrote nothing
    dst = torch.zeros_like(ref)
    fill_rows(dst, value, indices=idx[:n], num_valid=_nv(n))
    assert _same(dst, ref)


@pytest.mark.parametrize("count", [-3, CAP + 9])
def test_count_is_clamped_to_the_capacity(count):
    # idx is the first CAP entries of a longer array: a loop to the raw count would
    # read live entries past the capacity, and write dst[5] for them.
    idx = torch.full((CAP + 16,), 5, device="cuda")
    idx[:CAP] = -1  # skipped
    dst = torch.zeros(CAP, dtype=torch.int64, device="cuda")
    fill_rows(dst, 1, indices=idx[:CAP], num_valid=_nv(count))
    assert not dst.any()


def test_fill_rows_rejects_3d_destination():
    with pytest.raises(RuntimeError):
        fill_rows(
            torch.zeros(8, 4, 2, device="cuda"),
            1,
            num_valid=_nv(8),
            indices=torch.arange(8, device="cuda"),
        )


def test_rows_of_strided_views():
    # Views of the middle columns of a wider tensor: a row pitch that is no multiple
    # of 16 bytes and a base 4 bytes in, so rows move in 4-byte words; the other
    # columns must stay as they were.
    n, c = CAP // 2, 6
    wide = torch.randn(700, c + 4, device="cuda")
    idx, sel = _garbage_idx(n, 700)
    out = gather_embedding(wide[:, 1 : 1 + c], idx, num_valid=_nv(n))
    assert torch.equal(out[sel], wide[idx[sel], 1 : 1 + c])

    idx, sel = _garbage_idx(n, CAP, unique=True)
    rows = torch.randn(CAP, c, device="cuda")
    dst = torch.full((CAP, c + 4), -1.0, device="cuda")
    ref = dst.clone()
    ref[idx[sel], 1 : 1 + c] = rows[sel]
    gather_embedding(rows, dst[:, 1 : 1 + c], out_row=idx, num_valid=_nv(n))
    assert torch.equal(dst, ref)


@pytest.mark.parametrize(
    "width,src_pad,src_offset,dst_pad,dst_offset",
    [
        (128, 0, 0, 0, 0),
        (256, 0, 0, 0, 0),
        (32, 4, 1, 0, 0),
        (32, 0, 0, 4, 1),
        (32, 1, 0, 0, 0),
        (32, 0, 0, 1, 0),
        (31, 1, 0, 1, 0),
        (260, 0, 0, 0, 0),
    ],
    ids=[
        "vec4", "vec4-wide", "src-offset", "dst-offset",
        "src-pitch", "dst-pitch", "odd-width", "wide-fallback",
    ],
)
def test_gather_embedding_converts_rows(
    width, src_pad, src_offset, dst_pad, dst_offset
):
    src = torch.randn(3, width + src_pad, device="cuda")
    src = src[:, src_offset : src_offset + width]
    dst = torch.full((3, width + dst_pad), -1, dtype=torch.float16, device="cuda")
    dst_view = dst[:, dst_offset : dst_offset + width]
    indices = torch.tensor([2, 0, 1], device="cuda")
    expected = dst.clone()
    expected[:2, dst_offset : dst_offset + width] = src[indices[:2]].half()
    gather_embedding(src, dst_view, indices, num_valid=_nv(2))
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)


@pytest.mark.parametrize("n", [1, BIG // 2 + 1, BIG])
def test_row_ops_over_a_capped_grid(n):
    num_valid = torch.tensor([n], device="cuda")
    keys = torch.randint(0, 1 << 40, (BIG,), device="cuda")
    idx = torch.randperm(BIG, device="cuda")  # n live entries, all distinct, all valid

    gathered = gather_embedding(keys, idx, num_valid=num_valid)
    assert torch.equal(gathered[:n], keys[idx[:n]])

    scattered = torch.zeros_like(keys)
    gather_embedding(keys, scattered, out_row=idx, num_valid=num_valid)
    ref = torch.zeros_like(keys)
    ref[idx[:n]] = keys[:n]
    assert torch.equal(scattered, ref)

    filled = torch.zeros_like(keys)
    fill_rows(filled, 7, indices=idx, num_valid=num_valid)
    ref = torch.zeros_like(keys)
    ref[idx[:n]] = 7
    assert torch.equal(filled, ref)


def test_row_ops_do_not_read_the_count():
    flags = torch.rand(CAP, device="cuda") < 0.5
    keys = torch.arange(CAP, device="cuda")
    rows = torch.zeros(CAP, 4, device="cuda")
    src = torch.ones(CAP, 4, device="cuda")
    count = _nv(CAP // 2)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")  # a .item() / .cpu() of the count raises
    try:
        gather_embedding(rows, keys, num_valid=count)
        gather_embedding(src, rows, out_row=keys, num_valid=count)
        fill_rows(flags, True, indices=keys, num_valid=count)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.parametrize("reduce", ["sum", "max", "min"])
@pytest.mark.parametrize("num_keys,key_range", [(1000, 100), (200000, 5000)])
def test_segmented_unique_reduce(setup_device, reduce, num_keys, key_range):
    """A repeated key's value is reduced by `reduce` (sum / max / min)."""
    device = setup_device
    num_tables = 3

    keys_raw = torch.randint(
        0, key_range, (num_keys,), dtype=torch.int64, device=device
    )
    table_assignments = torch.randint(
        0, num_tables, (num_keys,), dtype=torch.int64, device=device
    )
    sort_idx, segmented_range = make_segmented_range(
        table_assignments, num_tables, device
    )
    keys = keys_raw[sort_idx]
    # Large non-negative values: a narrower accumulator would lose bits.
    values = torch.randint(0, 2**40, (num_keys,), dtype=torch.int64, device=device)

    num_uniques, unique_keys, output_indices, _, freq = segmented_unique_cuda(
        keys, segmented_range, num_tables, values, reduce
    )
    torch.cuda.synchronize()

    total_unique = num_uniques.item()
    assert torch.equal(unique_keys[:total_unique][output_indices], keys)

    # Reference: reduce `values` into the unique slot each input maps to.
    ref_op = {"sum": "sum", "max": "amax", "min": "amin"}[reduce]
    init = {"sum": 0, "max": 0, "min": 2**62}[reduce]
    # Every unique slot has at least one occurrence, so include_self=False
    # leaves no slot at `init`.
    expected = torch.full((total_unique,), init, dtype=torch.int64, device=device)
    expected.scatter_reduce_(
        0, output_indices, values, reduce=ref_op, include_self=(reduce == "sum")
    )
    assert torch.equal(freq[:total_unique], expected)


@pytest.mark.parametrize("reduce", ["max", "min"])
def test_segmented_unique_reduce_without_input_frequencies(setup_device, reduce):
    """No input values: every occurrence counts as 1, so max/min of any key is 1."""
    device = setup_device
    keys = torch.tensor([5, 5, 7, 5, 9, 9], dtype=torch.int64, device=device)
    segmented_range = torch.tensor([0, 6], dtype=torch.int64, device=device)
    empty = torch.empty(0, dtype=torch.int64, device=device)
    num_uniques, _, _, _, freq = segmented_unique_cuda(
        keys, segmented_range, 1, empty, reduce
    )
    torch.cuda.synchronize()
    assert torch.equal(
        freq[: num_uniques.item()],
        torch.ones(num_uniques.item(), dtype=torch.int64, device=device),
    )


def test_segmented_unique_reduce_rejects_unknown(setup_device):
    device = setup_device
    keys = torch.tensor([1, 2], dtype=torch.int64, device=device)
    segmented_range = torch.tensor([0, 2], dtype=torch.int64, device=device)
    with pytest.raises(Exception, match="reduce must be"):
        segmented_unique_cuda(
            keys,
            segmented_range,
            1,
            torch.empty(0, dtype=torch.int64, device=device),
            "avg",
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_segmented_unique_rejects_non_int64_frequencies(setup_device, dtype):
    device = setup_device
    keys = torch.tensor([1, 2, 1], dtype=torch.int64, device=device)
    segmented_range = torch.tensor([0, 3], dtype=torch.int64, device=device)
    freq = torch.ones(3, dtype=dtype, device=device)
    with pytest.raises(Exception, match="must be int64"):
        segmented_unique_cuda(keys, segmented_range, 1, freq)


def test_segmented_unique_rejects_non_contiguous_frequencies(setup_device):
    device = setup_device
    keys = torch.tensor([1, 2, 1], dtype=torch.int64, device=device)
    segmented_range = torch.tensor([0, 3], dtype=torch.int64, device=device)
    freq = torch.ones(6, dtype=torch.int64, device=device)[::2]
    with pytest.raises(Exception, match="must be contiguous"):
        segmented_unique_cuda(keys, segmented_range, 1, freq)
