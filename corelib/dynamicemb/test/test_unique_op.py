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
from dynamicemb_extensions import (
    expand_table_ids_cuda,
    flagged_compact,
    segmented_unique_cuda,
    segmented_unique_hashtable_cuda,
)


def _table_ids_to_segment_range(table_ids, num_tables, device):
    """Convert sorted per-element table_ids to segment_range boundaries."""
    segment_range = torch.zeros(num_tables + 1, dtype=torch.int64, device=device)
    if table_ids.numel() == 0:
        return segment_range
    segment_range[num_tables] = table_ids.numel()
    for t in range(1, num_tables):
        mask = table_ids >= t
        if mask.any():
            segment_range[t] = torch.where(mask)[0][0].item()
        else:
            segment_range[t] = table_ids.numel()
    return segment_range


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
    num_unique_per_table = 10000

    keys = torch.randint(
        0, num_unique_per_table, (num_keys,), dtype=torch.int64, device=device
    )

    table_ids = torch.sort(
        torch.randint(0, num_tables, (num_keys,), dtype=torch.int64, device=device)
    ).values

    segment_range = _table_ids_to_segment_range(table_ids, num_tables, device)

    (
        num_uniques,
        unique_keys,
        output_indices,
        table_offsets,
        freq_counters,
        _sort_perm,
        _sorted_rev_idx,
    ) = segmented_unique_cuda(keys, segment_range, num_tables)
    torch.cuda.synchronize()

    table_offsets_cpu = table_offsets.cpu()
    assert table_offsets_cpu[0].item() == 0, "First offset should be 0"

    for i in range(num_tables):
        assert (
            table_offsets_cpu[i + 1] >= table_offsets_cpu[i]
        ), "Table offsets should be non-decreasing"

    unique_keys_cpu = unique_keys.cpu()
    output_indices_cpu = output_indices.cpu()
    keys_cpu = keys.cpu()

    reconstructed = unique_keys_cpu[output_indices_cpu]
    assert torch.equal(reconstructed, keys_cpu), "Reconstruction failed"

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
    num_unique_keys = 1000

    keys = torch.randint(
        0, num_unique_keys, (num_keys,), dtype=torch.int64, device=device
    )

    table_ids = torch.sort(
        torch.randint(0, num_tables, (num_keys,), dtype=torch.int64, device=device)
    ).values

    segment_range = _table_ids_to_segment_range(table_ids, num_tables, device)

    num_uniques, unique_keys, output_indices, table_offsets, _, _, _ = (
        segmented_unique_cuda(keys, segment_range, num_tables)
    )
    torch.cuda.synchronize()

    table_offsets_cpu = table_offsets.cpu()

    for i in range(num_tables):
        table_count = table_offsets_cpu[i + 1].item() - table_offsets_cpu[i].item()
        assert (
            table_count <= num_unique_keys
        ), f"Table {i} has more unique keys than possible"

    total_unique = table_offsets_cpu[num_tables].item()

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

    active_tables = [0, 1, 3, 4, 6, 8, 9]
    table_ids_list = torch.randint(
        0, len(active_tables), (num_keys,), dtype=torch.int64, device=device
    )
    active_tables_tensor = torch.tensor(active_tables, dtype=torch.int64, device=device)
    table_ids = torch.sort(active_tables_tensor[table_ids_list]).values

    keys = torch.randint(0, 10000, (num_keys,), dtype=torch.int64, device=device)

    segment_range = _table_ids_to_segment_range(table_ids, num_tables, device)

    num_uniques, unique_keys, output_indices, table_offsets, _, _, _ = (
        segmented_unique_cuda(keys, segment_range, num_tables)
    )
    torch.cuda.synchronize()

    table_offsets_cpu = table_offsets.cpu()

    empty_tables = [2, 5, 7]
    for t in empty_tables:
        count = table_offsets_cpu[t + 1].item() - table_offsets_cpu[t].item()
        assert count == 0, f"Table {t} should be empty, got {count} keys"

    for t in active_tables:
        count = table_offsets_cpu[t + 1].item() - table_offsets_cpu[t].item()
        assert count <= 10000, f"Table {t} has more unique keys than possible"

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

    num_tables = 3
    segment_range = torch.zeros(num_tables + 1, dtype=torch.int64, device=device)

    keys = torch.tensor([], dtype=torch.int64, device=device)

    (
        num_uniques,
        unique_keys,
        output_indices,
        table_offsets,
        freq_counters,
        _sort_perm,
        _sorted_rev_idx,
    ) = segmented_unique_cuda(keys, segment_range, num_tables)
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

    keys = torch.randint(0, 100000, (num_keys,), dtype=torch.int64, device=device)

    table_ids = torch.sort(
        torch.randint(0, num_tables, (num_keys,), dtype=torch.int64, device=device)
    ).values

    segment_range = _table_ids_to_segment_range(table_ids, num_tables, device)

    num_uniques, unique_keys, output_indices, table_offsets, _, _, _ = (
        segmented_unique_cuda(keys, segment_range, num_tables)
    )
    torch.cuda.synchronize()

    unique_keys_cpu = unique_keys.cpu()
    output_indices_cpu = output_indices.cpu()
    keys_cpu = keys.cpu()

    reconstructed = unique_keys_cpu[output_indices_cpu]
    assert torch.equal(reconstructed, keys_cpu), "Reconstruction failed for random test"

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

    keys = torch.randint(0, 500000, (num_keys,), dtype=torch.int64, device=device)

    table_ids = torch.sort(
        torch.randint(0, num_tables, (num_keys,), dtype=torch.int64, device=device)
    ).values

    segment_range = _table_ids_to_segment_range(table_ids, num_tables, device)

    torch.cuda.synchronize()

    import time

    start = time.perf_counter()

    num_uniques, unique_keys, output_indices, table_offsets, _, _, _ = (
        segmented_unique_cuda(keys, segment_range, num_tables)
    )
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

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

    keys = torch.randint(0, 1000, (num_keys,), dtype=torch.int64, device=device)
    table_ids = torch.sort(
        torch.randint(0, num_tables, (num_keys,), dtype=torch.int64, device=device)
    ).values

    segment_range = _table_ids_to_segment_range(table_ids, num_tables, device)

    empty_freq_tensor = torch.empty(0, dtype=torch.int64, device=device)

    (
        num_uniques,
        unique_keys,
        output_indices,
        table_offsets,
        freq_counters,
        _sort_perm,
        _sorted_rev_idx,
    ) = segmented_unique_cuda(keys, segment_range, num_tables, empty_freq_tensor)
    torch.cuda.synchronize()

    total_unique = num_uniques.item()
    assert (
        freq_counters.numel() == num_keys
    ), "freq_counters should have num_keys elements"

    freq_sum = freq_counters[:total_unique].sum().item()
    assert (
        freq_sum == num_keys
    ), f"Sum of frequencies should be {num_keys}, got {freq_sum}"

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

    keys = torch.randint(0, 100, (num_keys,), dtype=torch.int64, device=device)
    table_ids = torch.sort(
        torch.randint(0, num_tables, (num_keys,), dtype=torch.int64, device=device)
    ).values

    segment_range = _table_ids_to_segment_range(table_ids, num_tables, device)

    input_frequencies = torch.full((num_keys,), 2, dtype=torch.int64, device=device)

    (
        num_uniques,
        unique_keys,
        output_indices,
        table_offsets,
        freq_counters,
        _sort_perm,
        _sorted_rev_idx,
    ) = segmented_unique_cuda(keys, segment_range, num_tables, input_frequencies)
    torch.cuda.synchronize()

    total_unique = num_uniques.item()

    freq_sum = freq_counters[:total_unique].sum().item()
    assert (
        freq_sum == 2 * num_keys
    ), f"Sum of frequencies should be {2 * num_keys}, got {freq_sum}"

    unique_keys_cpu = unique_keys.cpu()
    output_indices_cpu = output_indices.cpu()
    keys_cpu = keys.cpu()

    reconstructed = unique_keys_cpu[output_indices_cpu]
    assert torch.equal(
        reconstructed, keys_cpu
    ), "Reconstruction failed with custom freq"

    print(f"Segmented unique with custom frequencies test passed: freq_sum={freq_sum}")


@pytest.mark.parametrize(
    "num_tables, num_keys, num_unique_per_table",
    [
        pytest.param(10, 1_000_000, 10000, id="1M_10T_10K"),
        pytest.param(8, 1_000_000, 1000, id="1M_8T_1K_overlap"),
        pytest.param(4, 100_000, 1000, id="100K_4T_1K"),
        pytest.param(16, 1_000_000, 100000, id="1M_16T_100K"),
        pytest.param(2, 1000, 100, id="1K_2T_100"),
        pytest.param(10, 655360, 10000, id="bench_config"),
    ],
)
def test_compare_old_vs_new_unique(
    setup_device, num_tables, num_keys, num_unique_per_table
):
    """A/B comparison: old hash-based vs new sort-based segmented_unique.

    Verifies that both produce the same num_uniques and that unique_keys sets
    match per table.
    """
    device = setup_device

    keys = torch.randint(
        0, num_unique_per_table, (num_keys,), dtype=torch.int64, device=device
    )
    table_ids = torch.sort(
        torch.randint(0, num_tables, (num_keys,), dtype=torch.int64, device=device)
    ).values
    segment_range = _table_ids_to_segment_range(table_ids, num_tables, device)

    # --- NEW sort-based ---
    (
        new_num_uniques,
        new_unique_keys,
        new_reverse,
        new_offsets,
        _,
        _,
        _,
    ) = segmented_unique_cuda(keys, segment_range, num_tables)
    torch.cuda.synchronize()

    # --- OLD hash-based ---
    (
        old_num_uniques,
        old_unique_keys,
        old_reverse,
        old_offsets,
        _,
    ) = segmented_unique_hashtable_cuda(keys, table_ids, num_tables)
    torch.cuda.synchronize()

    new_total = new_num_uniques.item()
    old_total = old_num_uniques.item()

    # Both should reconstruct correctly
    new_recon = new_unique_keys[new_reverse]
    old_recon = old_unique_keys[old_reverse]
    assert torch.equal(new_recon, keys), "NEW reconstruction failed"
    assert torch.equal(old_recon, keys), "OLD reconstruction failed"

    # Per-table unique counts
    new_offsets_cpu = new_offsets.cpu()
    old_offsets_cpu = old_offsets.cpu()
    mismatches = []
    for t in range(num_tables):
        nc = new_offsets_cpu[t + 1].item() - new_offsets_cpu[t].item()
        oc = old_offsets_cpu[t + 1].item() - old_offsets_cpu[t].item()
        if nc != oc:
            mismatches.append((t, oc, nc))

    if mismatches:
        detail = "; ".join(
            f"table {t}: old={oc} new={nc}" for t, oc, nc in mismatches
        )
        print(f"MISMATCH! Total old={old_total} new={new_total}  detail: {detail}")

    # Per-table unique key sets should match
    for t in range(num_tables):
        ns, ne = new_offsets_cpu[t].item(), new_offsets_cpu[t + 1].item()
        os_, oe = old_offsets_cpu[t].item(), old_offsets_cpu[t + 1].item()
        new_set = set(new_unique_keys[ns:ne].cpu().tolist())
        old_set = set(old_unique_keys[os_:oe].cpu().tolist())
        if new_set != old_set:
            extra_new = new_set - old_set
            extra_old = old_set - new_set
            print(
                f"  Table {t} key-set mismatch: "
                f"|new|={len(new_set)} |old|={len(old_set)} "
                f"extra_in_new={len(extra_new)} extra_in_old={len(extra_old)}"
            )

    assert new_total == old_total, (
        f"num_uniques mismatch: old={old_total} vs new={new_total}"
    )
    print(
        f"A/B match: {old_total} uniques from {num_keys} keys, {num_tables} tables"
    )


def test_new_unique_minimality(setup_device):
    """Verify the new segmented_unique produces minimal unique counts.

    Checks that unique_keys within each table has no duplicates,
    which would inflate num_uniques while still passing reconstruction.
    """
    device = setup_device
    num_tables = 10
    num_keys = 1_000_000
    num_unique_per_table = 10000

    keys = torch.randint(
        0, num_unique_per_table, (num_keys,), dtype=torch.int64, device=device
    )
    table_ids = torch.sort(
        torch.randint(0, num_tables, (num_keys,), dtype=torch.int64, device=device)
    ).values
    segment_range = _table_ids_to_segment_range(table_ids, num_tables, device)

    (
        num_uniques,
        unique_keys,
        reverse_indices,
        table_offsets,
        _,
        _,
        _,
    ) = segmented_unique_cuda(keys, segment_range, num_tables)
    torch.cuda.synchronize()

    offsets_cpu = table_offsets.cpu()
    total_dups = 0
    for t in range(num_tables):
        s = offsets_cpu[t].item()
        e = offsets_cpu[t + 1].item()
        table_uniques = unique_keys[s:e].cpu()
        n_distinct = table_uniques.unique().numel()
        n_reported = e - s
        if n_distinct != n_reported:
            total_dups += n_reported - n_distinct
            print(
                f"  Table {t}: reported {n_reported} uniques but only {n_distinct} distinct"
            )

    assert total_dups == 0, (
        f"unique_keys contains {total_dups} duplicate entries across tables"
    )
    total = num_uniques.item()
    print(f"Minimality check passed: {total} truly unique from {num_keys} keys")


def test_expand_table_ids(setup_device):
    """Test expand_table_ids_cuda helper function."""
    device = setup_device
    torch.cuda.get_device_properties(device).multi_processor_count

    num_tables = 2
    local_batch_size = 3
    features_per_table = 2
    num_features = num_tables * features_per_table

    lengths = torch.tensor(
        [
            2,
            1,
            3,
            1,
            2,
            1,
            3,
            2,
            2,
            1,
            1,
            2,
        ],
        dtype=torch.int64,
        device=device,
    )

    offsets = torch.zeros(len(lengths) + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(lengths, dim=0)

    table_offsets_in_feature = torch.tensor([0, 2, 4], dtype=torch.int64, device=device)

    num_elements = offsets[-1].item()

    table_ids = expand_table_ids_cuda(
        offsets,
        table_offsets_in_feature,
        num_tables,
        local_batch_size,
        num_elements,
    )
    torch.cuda.synchronize()

    assert (
        table_ids.numel() == num_elements
    ), f"Expected {num_elements} table_ids, got {table_ids.numel()}"
    assert table_ids.dtype == torch.int64, "table_ids should be int64"

    table_ids_cpu = table_ids.cpu()

    table0_end_offset_idx = 2 * local_batch_size
    table0_end = offsets[table0_end_offset_idx].item()

    assert torch.all(
        table_ids_cpu[:table0_end] == 0
    ), f"First table elements should have table_id=0, got {table_ids_cpu[:table0_end]}"
    assert torch.all(
        table_ids_cpu[table0_end:] == 1
    ), f"Second table elements should have table_id=1, got {table_ids_cpu[table0_end:]}"

    print(f"expand_table_ids test passed: {num_elements} elements, {num_tables} tables")


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

    h_count, indices, outputs = flagged_compact(flags, inputs)
    ref_count, ref_idx, ref_outputs = _flagged_compact_reference(flags, inputs)

    assert h_count == ref_count, f"count mismatch: {h_count} vs {ref_count}"
    assert torch.equal(indices, ref_idx), "indices mismatch"
    assert len(outputs) == len(ref_outputs)
    for i, (out, ref) in enumerate(zip(outputs, ref_outputs)):
        if ref is None:
            assert out is None, f"output {i} should be None"
        else:
            assert torch.equal(out, ref), f"output {i} mismatch"
            assert out.dtype == torch.int64, f"output {i} dtype mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
