#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Usage:
#   NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 tests/test_hier_all2all.py
#   NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=8 tests/test_hier_all2all.py
"""
Tests for hierarchical all2all output distribution.

Covers:
  T1  - TopologyInfo detection
  T2  - check_hier_a2a_requirements
  T3  - HierAll2AllManager init + fallback
  T4  - Forward correctness: hier vs raw all2all (single node)
  T5  - Backward correctness
  T6  - Multiple embedding dims
  T7  - Repeated forward (training loop simulation)
  T8  - HierarchicalSequenceEmbeddingDist end-to-end
"""

import argparse
import os
import sys
import traceback

import importlib
import importlib.util

import torch
import torch.distributed as dist

# Import hier_all2all directly by file path to bypass dynamicemb.__init__
# which depends on the C++ extension.
_parent = os.path.join(os.path.dirname(__file__), "..")


def _load_module_from_file(mod_name, file_path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


hier_mod = _load_module_from_file(
    "dynamicemb.hier_all2all",
    os.path.join(_parent, "dynamicemb", "hier_all2all.py"),
)
TopologyInfo = hier_mod.TopologyInfo
HierAll2AllManager = hier_mod.HierAll2AllManager
check_hier_a2a_requirements = hier_mod.check_hier_a2a_requirements


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_data(rank, world_size, num_features, max_seq_len, D, device,
                  dtype=torch.float32, seed_offset=0):
    """Generate synthetic sequence embedding data.

    Returns tensors mimicking what KJTAllToAll + gather_embedding produce.
    Uses identity recat (sparse_features_recat = arange) for simplicity.
    """
    torch.manual_seed(42 + rank + seed_offset)
    F = num_features
    W = world_size

    # lengths_after_input_dist: [F * W] — per-feature, per-source-rank row counts
    lengths = torch.randint(1, max_seq_len + 1, (F * W,), device=device)

    # input_splits[r] = total rows this rank sends to rank r
    input_splits = []
    for r in range(W):
        s = sum(lengths[r * F + f].item() for f in range(F))
        input_splits.append(s)

    total_send = sum(input_splits)
    output_embs = torch.randn(total_send, D, device=device, dtype=dtype)

    # Identity recat
    sparse_features_recat = torch.arange(F * W, dtype=torch.int32, device=device)

    # Exchange splits -> output_splits
    input_splits_t = torch.tensor(input_splits, dtype=torch.int64, device=device)
    all_splits = [torch.empty(W, dtype=torch.int64, device=device) for _ in range(W)]
    dist.all_gather(all_splits, input_splits_t)
    output_splits = [all_splits[r][rank].item() for r in range(W)]

    total_recv = sum(output_splits)
    unbucketize_permute = torch.arange(total_recv, dtype=torch.int64, device=device)

    return (output_embs, lengths, input_splits, output_splits,
            sparse_features_recat, unbucketize_permute)


def raw_all2all_forward(pg, output_embs, input_splits, output_splits, D, device):
    """Reference: raw all_to_all_single (no recat/unbucketize)."""
    total_recv = sum(output_splits)
    recv = torch.empty(total_recv, D, dtype=output_embs.dtype, device=device)
    in_sizes = [s * D for s in input_splits]
    out_sizes = [s * D for s in output_splits]
    dist.all_to_all_single(recv.view(-1), output_embs.view(-1), out_sizes, in_sizes, group=pg)
    return recv


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_topology_detection(rank, world_size):
    """T1: TopologyInfo.detect() produces correct mappings."""
    pg = dist.group.WORLD
    topo = TopologyInfo.detect(pg)

    assert topo.my_global_rank == rank, f"rank mismatch: {topo.my_global_rank} vs {rank}"
    assert topo.world_size == world_size
    assert topo.local_world_size > 0
    assert topo.num_nodes == world_size // topo.local_world_size
    assert topo.my_local_rank == rank % topo.local_world_size
    assert topo.my_node == rank // topo.local_world_size
    assert len(topo.rank_to_node) == world_size
    assert len(topo.rank_to_local) == world_size
    assert topo.rank_to_node[rank] == topo.my_node
    assert topo.rank_to_local[rank] == topo.my_local_rank

    for n in range(topo.num_nodes):
        for lr, r in enumerate(topo.node_to_ranks[n]):
            assert topo.rank_to_node[r] == n
            assert topo.rank_to_local[r] == lr

    return True


def test_requirements_check(rank, world_size):
    """T2: check_hier_a2a_requirements on H100."""
    pg = dist.group.WORLD
    topo = TopologyInfo.detect(pg)
    ok, reason = check_hier_a2a_requirements(pg, topo)
    assert ok, f"Requirements check failed: {reason}"
    return True


def test_manager_init(rank, world_size):
    """T3: HierAll2AllManager initializes without crash."""
    pg = dist.group.WORLD
    device = torch.device(f"cuda:{rank}")

    manager = HierAll2AllManager(
        pg=pg, num_features=4, max_rows_per_rank=10000,
        D=128, device=device, dtype=torch.float32,
    )
    assert not manager.fallback, f"Manager fell back: {manager._fallback_reason}"
    return True


def test_forward_correctness(rank, world_size):
    """T4: Hier forward matches raw all_to_all_single."""
    pg = dist.group.WORLD
    device = torch.device(f"cuda:{rank}")
    D = 128; F = 2; max_seq = 10

    (output_embs, lengths, input_splits, output_splits,
     sfrecat, unbuck) = generate_data(rank, world_size, F, max_seq, D, device)

    # Reference: raw all2all
    ref_out = raw_all2all_forward(pg, output_embs.clone(), input_splits,
                                   output_splits, D, device)
    dist.barrier()

    # Hier path: uses SequenceEmbeddingsAllToAll internally (reference mode)
    # With identity recat and identity unbucketize, the result should match
    # raw all2all.
    max_rows = max_seq * F * world_size
    manager = HierAll2AllManager(
        pg=pg, num_features=F, max_rows_per_rank=max_rows, D=D, device=device,
    )
    assert not manager.fallback

    hier_out = manager.forward(
        output_embs=output_embs.clone(),
        lengths_after_input_dist=lengths,
        input_splits=input_splits,
        output_splits=output_splits,
        sparse_features_recat=sfrecat,
        unbucketize_permute=unbuck,
        batch_size_per_rank=None,
    )

    assert ref_out.shape == hier_out.shape, (
        f"Shape mismatch: {ref_out.shape} vs {hier_out.shape}"
    )
    assert torch.allclose(ref_out, hier_out, atol=1e-5), (
        f"Forward mismatch: max_diff={(ref_out - hier_out).abs().max().item()}"
    )
    return True


def test_forward_bf16(rank, world_size):
    """T4b: Forward with bfloat16."""
    pg = dist.group.WORLD
    device = torch.device(f"cuda:{rank}")
    D = 64; F = 2; max_seq = 8

    (output_embs, lengths, input_splits, output_splits,
     sfrecat, unbuck) = generate_data(rank, world_size, F, max_seq, D, device,
                                       dtype=torch.bfloat16)

    ref_out = raw_all2all_forward(pg, output_embs.clone(), input_splits,
                                   output_splits, D, device)
    dist.barrier()

    max_rows = max_seq * F * world_size
    manager = HierAll2AllManager(
        pg=pg, num_features=F, max_rows_per_rank=max_rows, D=D, device=device,
        dtype=torch.bfloat16,
    )
    hier_out = manager.forward(
        output_embs=output_embs.clone(), lengths_after_input_dist=lengths,
        input_splits=input_splits, output_splits=output_splits,
        sparse_features_recat=sfrecat, unbucketize_permute=unbuck,
    )

    assert torch.allclose(ref_out, hier_out, atol=1e-2), (
        f"bf16 mismatch: max_diff={(ref_out - hier_out).abs().max().item()}"
    )
    return True


def test_backward_correctness(rank, world_size):
    """T5: Backward all2all with transposed splits."""
    pg = dist.group.WORLD
    device = torch.device(f"cuda:{rank}")
    D = 128; F = 2; max_seq = 8

    (output_embs, lengths, input_splits, output_splits,
     sfrecat, unbuck) = generate_data(rank, world_size, F, max_seq, D, device)

    max_rows = max_seq * F * world_size
    manager = HierAll2AllManager(
        pg=pg, num_features=F, max_rows_per_rank=max_rows, D=D, device=device,
    )

    # Forward to cache splits
    hier_out = manager.forward(
        output_embs=output_embs.clone(), lengths_after_input_dist=lengths,
        input_splits=input_splits, output_splits=output_splits,
        sparse_features_recat=sfrecat, unbucketize_permute=unbuck,
    )

    # Backward
    grad_final = torch.randn_like(hier_out)
    grad_out = manager.backward(
        grad_final=grad_final, sparse_features_recat=sfrecat,
        unbucketize_permute=unbuck,
    )

    # Reference backward: all2all with transposed splits
    ref_grad = raw_all2all_forward(pg, grad_final.clone(), output_splits,
                                    input_splits, D, device)

    assert grad_out.shape == ref_grad.shape, (
        f"Bwd shape: {grad_out.shape} vs {ref_grad.shape}"
    )
    assert torch.allclose(grad_out, ref_grad, atol=1e-5), (
        f"Bwd mismatch: max_diff={(grad_out - ref_grad).abs().max().item()}"
    )
    return True


def test_multiple_dims(rank, world_size):
    """T6: Test with D=64, 128, 256."""
    pg = dist.group.WORLD
    device = torch.device(f"cuda:{rank}")
    F = 2; max_seq = 5

    for D in [64, 128, 256]:
        (output_embs, lengths, input_splits, output_splits,
         sfrecat, unbuck) = generate_data(rank, world_size, F, max_seq, D, device)

        ref_out = raw_all2all_forward(pg, output_embs.clone(), input_splits,
                                       output_splits, D, device)
        dist.barrier()

        max_rows = max_seq * F * world_size
        manager = HierAll2AllManager(
            pg=pg, num_features=F, max_rows_per_rank=max_rows, D=D, device=device,
        )
        hier_out = manager.forward(
            output_embs=output_embs.clone(), lengths_after_input_dist=lengths,
            input_splits=input_splits, output_splits=output_splits,
            sparse_features_recat=sfrecat, unbucketize_permute=unbuck,
        )

        assert torch.allclose(ref_out, hier_out, atol=1e-5), (
            f"D={D}: max_diff={(ref_out - hier_out).abs().max().item()}"
        )
        dist.barrier()

    return True


def test_repeated_forward(rank, world_size):
    """T7: Multiple forward calls (training loop)."""
    pg = dist.group.WORLD
    device = torch.device(f"cuda:{rank}")
    D = 128; F = 2; max_seq = 8
    max_rows = max_seq * F * world_size

    manager = HierAll2AllManager(
        pg=pg, num_features=F, max_rows_per_rank=max_rows, D=D, device=device,
    )

    for it in range(5):
        (output_embs, lengths, input_splits, output_splits,
         sfrecat, unbuck) = generate_data(
            rank, world_size, F, max_seq, D, device, seed_offset=it * 100)

        ref_out = raw_all2all_forward(pg, output_embs.clone(), input_splits,
                                       output_splits, D, device)
        dist.barrier()

        hier_out = manager.forward(
            output_embs=output_embs.clone(), lengths_after_input_dist=lengths,
            input_splits=input_splits, output_splits=output_splits,
            sparse_features_recat=sfrecat, unbucketize_permute=unbuck,
        )
        assert torch.allclose(ref_out, hier_out, atol=1e-5), (
            f"Iter {it}: max_diff={(ref_out - hier_out).abs().max().item()}"
        )
        dist.barrier()

    return True


def test_hier_seq_dist_e2e(rank, world_size):
    """T8: HierarchicalSequenceEmbeddingDist end-to-end via output_dist module."""
    # Load output_dist module
    output_dist_mod = _load_module_from_file(
        "dynamicemb.output_dist",
        os.path.join(_parent, "dynamicemb", "output_dist.py"),
    )
    HierarchicalSequenceEmbeddingDist = output_dist_mod.HierarchicalSequenceEmbeddingDist

    from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext

    pg = dist.group.WORLD
    device = torch.device(f"cuda:{rank}")
    D = 128; F = 2; max_seq = 8

    (output_embs, lengths, input_splits, output_splits,
     sfrecat, unbuck) = generate_data(rank, world_size, F, max_seq, D, device)

    ref_out = raw_all2all_forward(pg, output_embs.clone(), input_splits,
                                   output_splits, D, device)
    dist.barrier()

    max_rows = max_seq * F * world_size
    hier_dist = HierarchicalSequenceEmbeddingDist(
        pg=pg, num_features=F, max_rows_per_rank=max_rows,
        D=D, device=device, dtype=torch.float32,
    )

    ctx = SequenceShardingContext(
        features_before_input_dist=None,
        lengths_after_input_dist=lengths,
        input_splits=input_splits,
        output_splits=output_splits,
        batch_size_per_rank=None,
        sparse_features_recat=sfrecat,
        unbucketize_permute_tensor=unbuck,
    )

    awaitable = hier_dist.forward(output_embs.clone(), sharding_ctx=ctx)
    hier_out = awaitable.wait()

    assert ref_out.shape == hier_out.shape
    assert torch.allclose(ref_out, hier_out, atol=1e-5), (
        f"E2E: max_diff={(ref_out - hier_out).abs().max().item()}"
    )
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    ("T1: topology_detection", test_topology_detection),
    ("T2: requirements_check", test_requirements_check),
    ("T3: manager_init", test_manager_init),
    ("T4: forward_correctness_fp32", test_forward_correctness),
    ("T4b: forward_correctness_bf16", test_forward_bf16),
    ("T5: backward_correctness", test_backward_correctness),
    ("T6: multiple_dims", test_multiple_dims),
    ("T7: repeated_forward", test_repeated_forward),
    ("T8: hier_seq_dist_e2e", test_hier_seq_dist_e2e),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=str, default=None)
    args, _ = parser.parse_known_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Hierarchical All2All Tests — {world_size} GPUs")
        print(f"{'='*60}")

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in ALL_TESTS:
        if args.filter and args.filter not in name:
            continue
        dist.barrier()
        try:
            result = test_fn(rank, world_size)
            if rank == 0:
                print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            if rank == 0:
                print(f"  [FAIL] {name}: {e}")
                traceback.print_exc()
            failed += 1
            errors.append((name, str(e)))
        dist.barrier()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Results: {passed} passed, {failed} failed")
        if errors:
            print("\nFailures:")
            for n, e in errors:
                print(f"  - {n}: {e}")
        print(f"{'='*60}\n")

    dist.destroy_process_group()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
