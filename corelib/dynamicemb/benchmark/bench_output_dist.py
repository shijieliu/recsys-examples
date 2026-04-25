#!/usr/bin/env python3
"""
Benchmark: torchrec SequenceEmbeddingsAllToAll vs hier fused output_dist.

Baseline = torchrec SequenceEmbeddingsAllToAll (permute + all2all + unbucketize)
Hier     = HierAll2AllManager.forward (fused outcast + gather)

Correctness: hier result is checked against baseline result.

Usage:
  torchrun --nproc_per_node=8 benchmark/bench_output_dist.py \
    --D=128 --num_features=10 --max_seq_len=50 --warmup=100 --iters=500

  # Sweep mode (CSV output):
  torchrun --nproc_per_node=8 benchmark/bench_output_dist.py --sweep
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
from torchrec.distributed.dist_data import SequenceEmbeddingsAllToAll

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

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


def generate_data(rank, world_size, num_features, batch_size, max_seq_len, D,
                   device, dtype=torch.bfloat16):
    """Generate synthetic data compatible with torchrec SequenceEmbeddingsAllToAll.

    Generates lengths in 1D layout: [B*F*W] entries. Each rank contributes
    B*F contiguous entries. Uses variable_batch_size path (permute_1D) by
    making batch_size_per_rank slightly non-uniform.
    """
    torch.manual_seed(42 + rank)
    F = num_features
    W = world_size
    B = batch_size

    # Use slightly different batch sizes per rank to trigger variable_batch_size
    # path in torchrec (avoids permute_2D shape constraints).
    # Rank 0 gets B, rank 1 gets B+1, etc.
    bspr = [B + r for r in range(W)]

    # lengths: bspr[r]*F entries per rank r
    parts = []
    for r in range(W):
        parts.append(torch.randint(1, max_seq_len + 1, (bspr[r] * F,), device=device))
    lengths = torch.cat(parts)

    # input_splits: total rows sent to each dest rank
    input_splits = []
    offset = 0
    for r in range(W):
        n = bspr[r] * F
        input_splits.append(lengths[offset:offset + n].sum().item())
        offset += n

    total_send = sum(input_splits)
    output_embs = torch.randn(total_send, D, device=device, dtype=dtype)

    # sparse_features_recat: random permutation that preserves rank boundaries.
    total_feats = sum(bspr[r] * F for r in range(W))
    feat_offset = 0
    blocks = []
    for r in range(W):
        n = bspr[r] * F
        blocks.append(torch.randperm(n, device=device) + feat_offset)
        feat_offset += n
    sparse_features_recat = torch.cat(blocks).to(torch.int32)

    # Exchange splits
    input_splits_t = torch.tensor(input_splits, dtype=torch.int64, device=device)
    output_splits_t = torch.empty_like(input_splits_t)
    dist.all_to_all_single(
        output_splits_t, input_splits_t,
        output_split_sizes=[1] * W, input_split_sizes=[1] * W,
    )
    output_splits = output_splits_t.tolist()
    total_recv = sum(output_splits)

    unbucketize_permute = torch.randperm(total_recv, device=device).to(torch.int64)

    return (output_embs, lengths, input_splits, output_splits,
            sparse_features_recat, unbucketize_permute, bspr,
            total_send, total_recv)


def bench(name, fn, warmup, iters, rank):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    if rank == 0:
        print(f"  {name:45s}: {ms*1000:8.1f} us", flush=True)
    return ms


def run_one(D, F, B, max_seq_len, warmup, iters, dtype, rank, W, device):
    """Run one benchmark configuration. Returns (diff, baseline_us, hier_us) or None on error."""
    dtype_torch = DTYPE_MAP[dtype]
    elem_bytes = torch.tensor([], dtype=dtype_torch).element_size()

    (output_embs, lengths, input_splits, output_splits,
     sparse_features_recat, unbucketize_permute, batch_size_per_rank,
     total_send, total_recv) = generate_data(
        rank, W, F, B, max_seq_len, D, device, dtype=dtype_torch,
    )

    # Baseline: torchrec SequenceEmbeddingsAllToAll
    # features_per_rank[r] = bspr[r] * F (number of length entries from rank r)
    features_per_rank = [batch_size_per_rank[r] * F for r in range(W)]
    torchrec_dist = SequenceEmbeddingsAllToAll(
        pg=dist.group.WORLD,
        features_per_rank=features_per_rank,
        device=device,
    )

    def baseline_fwd():
        # Match output_dist.py convention: pass splits WITHOUT swapping.
        # torchrec and hier use the same split semantics.
        awaitable = torchrec_dist(
            local_embs=output_embs,
            lengths=lengths,
            input_splits=input_splits,
            output_splits=output_splits,
            sparse_features_recat=sparse_features_recat,
            unbucketize_permute_tensor=unbucketize_permute,
            batch_size_per_rank=batch_size_per_rank,
        )
        return awaitable.wait()

    # Hier: HierAll2AllManager.forward
    total_feats = sum(features_per_rank)
    max_rows = max_seq_len * total_feats
    manager = HierAll2AllManager(
        pg=dist.group.WORLD,
        num_features=total_feats,
        max_rows_per_rank=max_rows,
        D=D,
        device=device,
        dtype=dtype_torch,
    )

    def hier_fwd():
        return manager.forward(
            output_embs=output_embs,
            lengths_after_input_dist=lengths,
            input_splits=input_splits,
            output_splits=output_splits,
            sparse_features_recat=sparse_features_recat,
            unbucketize_permute=unbucketize_permute,
            batch_size_per_rank=batch_size_per_rank,
        )

    # Correctness
    baseline_result = baseline_fwd()
    hier_result = hier_fwd()
    torch.cuda.synchronize()

    diff = (baseline_result.float() - hier_result.float()).abs().max().item()

    # Benchmark
    torch.cuda.profiler.start()
    baseline_ms = bench("baseline", baseline_fwd, warmup, iters, rank=-1)
    hier_ms = bench("hier", hier_fwd, warmup, iters, rank=-1)
    torch.cuda.profiler.stop()

    baseline_us = baseline_ms * 1000
    hier_us = hier_ms * 1000

    del manager, torchrec_dist
    torch.cuda.empty_cache()

    return diff, baseline_us, hier_us, total_send, total_recv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--num_features", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--sweep", action="store_true",
                        help="Run full sweep: D=[16,32,64,128], dtype=[bf16,fp16,fp32]")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    W = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    F = args.num_features
    B = args.batch_size

    if args.sweep:
        dims = [16, 32, 64, 128]
        dtypes = ["bf16", "fp16", "fp32"]
        warmup = 50
        iters = 200

        if rank == 0:
            print(f"W,D,dtype,B,total_send,total_recv,data_MB,torchrec_us,hier_us,speedup,diff,status",
                  flush=True)

        for D in dims:
            for dtype in dtypes:
                dtype_torch = DTYPE_MAP[dtype]
                elem_bytes = torch.tensor([], dtype=dtype_torch).element_size()
                try:
                    diff, base_us, hier_us, tsend, trecv = run_one(
                        D, F, B, args.max_seq_len, warmup, iters,
                        dtype, rank, W, device,
                    )
                    speedup = base_us / hier_us if hier_us > 0 else float('inf')
                    data_mb = tsend * D * elem_bytes / 1e6
                    status = "OK" if diff <= 0.01 else "MISMATCH"
                    if rank == 0:
                        print(f"{W},{D},{dtype},{B},{tsend},{trecv},{data_mb:.2f},"
                              f"{base_us:.1f},{hier_us:.1f},{speedup:.2f}x,"
                              f"{diff:.4f},{status}", flush=True)
                except Exception as e:
                    if rank == 0:
                        print(f"{W},{D},{dtype},,,,,,,{e},ERROR", flush=True)

        dist.destroy_process_group()
        return

    # Single-config mode
    D = args.D
    dtype = args.dtype
    dtype_torch = DTYPE_MAP[dtype]
    elem_bytes = torch.tensor([], dtype=dtype_torch).element_size()

    (output_embs, lengths, input_splits, output_splits,
     sparse_features_recat, unbucketize_permute, batch_size_per_rank,
     total_send, total_recv) = generate_data(
        rank, W, F, B, args.max_seq_len, D, device, dtype=dtype_torch,
    )

    features_per_rank = [batch_size_per_rank[r] * F for r in range(W)]
    torchrec_dist = SequenceEmbeddingsAllToAll(
        pg=dist.group.WORLD,
        features_per_rank=features_per_rank,
        device=device,
    )

    def baseline_fwd():
        awaitable = torchrec_dist(
            local_embs=output_embs,
            lengths=lengths,
            input_splits=input_splits,
            output_splits=output_splits,
            sparse_features_recat=sparse_features_recat,
            unbucketize_permute_tensor=unbucketize_permute,
            batch_size_per_rank=batch_size_per_rank,
        )
        return awaitable.wait()

    total_feats = sum(features_per_rank)
    max_rows = args.max_seq_len * total_feats
    manager = HierAll2AllManager(
        pg=dist.group.WORLD,
        num_features=total_feats,
        max_rows_per_rank=max_rows,
        D=D,
        device=device,
        dtype=dtype_torch,
    )

    def hier_fwd():
        return manager.forward(
            output_embs=output_embs,
            lengths_after_input_dist=lengths,
            input_splits=input_splits,
            output_splits=output_splits,
            sparse_features_recat=sparse_features_recat,
            unbucketize_permute=unbucketize_permute,
            batch_size_per_rank=batch_size_per_rank,
        )

    baseline_result = baseline_fwd()
    hier_result = hier_fwd()
    torch.cuda.synchronize()

    diff = (baseline_result.float() - hier_result.float()).abs().max().item()

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Output Dist Benchmark: W={W}, B~={B}, D={D}, F={F}, "
              f"seq={args.max_seq_len}, dtype={dtype}")
        print(f"total_send={total_send}, total_recv={total_recv}, "
              f"data={total_send * D * elem_bytes / 1e6:.2f} MB ({dtype})")
        print(f"correctness diff: {diff:.6f}"
              f"{'  *** MISMATCH ***' if diff > 0.01 else '  OK'}")
        print(f"hier fallback: {manager.fallback}")
        print(f"warmup={args.warmup}, iters={args.iters}")
        print(f"{'='*70}")

    torch.cuda.profiler.start()
    baseline_ms = bench(
        "torchrec SequenceEmbeddingsAllToAll", baseline_fwd,
        args.warmup, args.iters, rank)
    hier_ms = bench(
        "hier fused (outcast + gather)", hier_fwd,
        args.warmup, args.iters, rank)
    torch.cuda.profiler.stop()

    if rank == 0:
        speedup = baseline_ms / hier_ms if hier_ms > 0 else float('inf')
        print(f"\n  Speedup: {speedup:.2f}x")
        print(f"{'='*70}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
