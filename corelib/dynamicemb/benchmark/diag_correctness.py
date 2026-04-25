#!/usr/bin/env python3
"""
Diagnostic: isolate correctness mismatch between torchrec and hier.

Tests:
  1. Identity recat + identity unbucketize -> should match
  2. Random recat + identity unbucketize -> isolates recat handling
  3. Identity recat + random unbucketize -> isolates unbucketize handling
  4. Random recat + random unbucketize -> full comparison (benchmark scenario)

Usage:
  torchrun --nproc_per_node=8 benchmark/diag_correctness.py
"""

import os
import sys

import torch
import torch.distributed as dist
from torchrec.distributed.dist_data import SequenceEmbeddingsAllToAll

_parent = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _parent)

import importlib, importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

hier_mod = _load_module(
    "dynamicemb.hier_all2all",
    os.path.join(_parent, "dynamicemb", "hier_all2all.py"),
)
HierAll2AllManager = hier_mod.HierAll2AllManager


def generate_data(rank, W, F, B, max_seq_len, D, device, dtype,
                  random_recat=False, random_unbucketize=False):
    torch.manual_seed(42 + rank)
    bspr = [B + r for r in range(W)]

    parts = []
    for r in range(W):
        parts.append(torch.randint(1, max_seq_len + 1, (bspr[r] * F,), device=device))
    lengths = torch.cat(parts)

    input_splits = []
    offset = 0
    for r in range(W):
        n = bspr[r] * F
        input_splits.append(lengths[offset:offset + n].sum().item())
        offset += n

    total_send = sum(input_splits)
    output_embs = torch.randn(total_send, D, device=device, dtype=dtype)

    total_feats = sum(bspr[r] * F for r in range(W))

    if random_recat:
        # Block-diagonal random permutation (like the benchmark)
        feat_offset = 0
        blocks = []
        for r in range(W):
            n = bspr[r] * F
            blocks.append(torch.randperm(n, device=device) + feat_offset)
            feat_offset += n
        sparse_features_recat = torch.cat(blocks).to(torch.int32)
    else:
        sparse_features_recat = torch.arange(total_feats, dtype=torch.int32, device=device)

    input_splits_t = torch.tensor(input_splits, dtype=torch.int64, device=device)
    output_splits_t = torch.empty_like(input_splits_t)
    dist.all_to_all_single(output_splits_t, input_splits_t,
                           output_split_sizes=[1]*W, input_split_sizes=[1]*W)
    output_splits = output_splits_t.tolist()
    total_recv = sum(output_splits)

    if random_unbucketize:
        unbucketize_permute = torch.randperm(total_recv, device=device).to(torch.int64)
    else:
        unbucketize_permute = torch.arange(total_recv, dtype=torch.int64, device=device)

    return (output_embs, lengths, input_splits, output_splits,
            sparse_features_recat, unbucketize_permute, bspr,
            total_send, total_recv)


def run_test(label, rank, W, device, D=128, F=100, B=32, max_seq_len=50,
             random_recat=False, random_unbucketize=False):
    dtype = torch.bfloat16

    (output_embs, lengths, input_splits, output_splits,
     sparse_features_recat, unbucketize_permute, bspr,
     total_send, total_recv) = generate_data(
        rank, W, F, B, max_seq_len, D, device, dtype,
        random_recat=random_recat, random_unbucketize=random_unbucketize,
    )

    features_per_rank = [bspr[r] * F for r in range(W)]

    # Baseline: torchrec
    torchrec_dist = SequenceEmbeddingsAllToAll(
        pg=dist.group.WORLD,
        features_per_rank=features_per_rank,
        device=device,
    )
    awaitable = torchrec_dist(
        local_embs=output_embs,
        lengths=lengths,
        input_splits=input_splits,
        output_splits=output_splits,
        sparse_features_recat=sparse_features_recat,
        unbucketize_permute_tensor=unbucketize_permute,
        batch_size_per_rank=bspr,
    )
    baseline_result = awaitable.wait()

    # Hier (fused path)
    total_feats = sum(features_per_rank)
    max_rows = max_seq_len * total_feats
    manager = HierAll2AllManager(
        pg=dist.group.WORLD,
        num_features=total_feats,
        max_rows_per_rank=max_rows,
        D=D,
        device=device,
        dtype=dtype,
    )
    manager._scatter_map_cache.clear()
    hier_result = manager.forward(
        output_embs=output_embs,
        lengths_after_input_dist=lengths,
        input_splits=input_splits,
        output_splits=output_splits,
        sparse_features_recat=sparse_features_recat,
        unbucketize_permute=unbucketize_permute,
        batch_size_per_rank=bspr,
    )
    torch.cuda.synchronize()

    # Also test reference path (no fused kernel) by setting env
    os.environ["HIER_A2A_FUSED"] = "0"
    manager2 = HierAll2AllManager(
        pg=dist.group.WORLD,
        num_features=total_feats,
        max_rows_per_rank=max_rows,
        D=D,
        device=device,
        dtype=dtype,
    )
    ref_result = manager2.forward(
        output_embs=output_embs,
        lengths_after_input_dist=lengths,
        input_splits=input_splits,
        output_splits=output_splits,
        sparse_features_recat=sparse_features_recat,
        unbucketize_permute=unbucketize_permute,
        batch_size_per_rank=bspr,
    )
    os.environ["HIER_A2A_FUSED"] = "1"
    torch.cuda.synchronize()

    diff = (baseline_result.float() - hier_result.float()).abs().max().item()
    ref_diff = (baseline_result.float() - ref_result.float()).abs().max().item()
    hier_vs_ref = (hier_result.float() - ref_result.float()).abs().max().item()
    status = "OK" if diff <= 0.01 else "MISMATCH"

    b_sorted = baseline_result.float().sort(dim=0).values
    h_sorted = hier_result.float().sort(dim=0).values
    sorted_diff = (b_sorted - h_sorted).abs().max().item()
    sorted_status = "same_set" if sorted_diff <= 0.01 else "diff_set"

    if rank == 0:
        print(f"  {label:50s}:", flush=True)
        print(f"    fused_vs_torchrec={diff:.4f} ({status})  "
              f"sorted={sorted_diff:.4f} ({sorted_status})", flush=True)
        print(f"    ref_vs_torchrec={ref_diff:.4f}  "
              f"fused_vs_ref={hier_vs_ref:.4f}", flush=True)

    del manager, manager2, torchrec_dist
    torch.cuda.empty_cache()


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    W = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Correctness Diagnostic: W={W}")
        print(f"{'='*70}")

    run_test("identity recat + identity unbucketize",
             rank, W, device, random_recat=False, random_unbucketize=False)
    run_test("random recat + identity unbucketize",
             rank, W, device, random_recat=True, random_unbucketize=False)
    run_test("identity recat + random unbucketize",
             rank, W, device, random_recat=False, random_unbucketize=True)
    run_test("random recat + random unbucketize",
             rank, W, device, random_recat=True, random_unbucketize=True)

    if rank == 0:
        print(f"{'='*70}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
