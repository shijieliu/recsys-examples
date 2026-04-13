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

import contextlib
import csv
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest
import torch
import torchrec
from benchmark_utils import GPUTimer
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    CacheAlgorithm,
    EmbeddingLocation,
    PoolingMode,
    RecordCacheMetrics,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)

try:
    from fbgemm_gpu.runtime_monitor import StdLogStatsReporterConfig

    _HAS_STATS_REPORTER = True
except ImportError:
    _HAS_STATS_REPORTER = False


# ── Constants ────────────────────────────────────────────────────────────────

REPORT_INTERVAL = 10
WARMUP_ITERS = 5

GPU_PEAK_BW_GB_S = {
    "H100 SXM": 3350,
    "H100 NVL": 3350,
    "H100 PCIe": 2039,
    "H100": 2039,
    "H200": 4800,
    "A100 SXM": 2039,
    "A100 PCIe": 2039,
    "A100": 2039,
    "L40": 864,
    "V100": 900,
}


def get_peak_bandwidth():
    """Return peak HBM bandwidth (GB/s) for the current GPU, or 0 if unknown."""
    if not torch.cuda.is_available():
        return 0
    name = torch.cuda.get_device_name()
    for key in sorted(GPU_PEAK_BW_GB_S, key=len, reverse=True):
        if key in name:
            return GPU_PEAK_BW_GB_S[key]
    return 0


# ── Utility helpers ──────────────────────────────────────────────────────────


def get_emb_precision(s):
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[s]


def get_fbgemm_precision(s):
    return {"fp32": SparseType.FP32, "fp16": SparseType.FP16, "bf16": SparseType.BF16}[
        s
    ]


_DYN_OPT = {
    "sgd": EmbOptimType.EXACT_SGD,
    "exact_sgd": EmbOptimType.EXACT_SGD,
    "adam": EmbOptimType.ADAM,
    "exact_adagrad": EmbOptimType.EXACT_ADAGRAD,
    "exact_row_wise_adagrad": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
}

_FBGEMM_OPT = {
    "sgd": EmbOptimType.EXACT_SGD,
    "exact_sgd": EmbOptimType.EXACT_SGD,
    "adam": EmbOptimType.ADAM,
    "exact_adagrad": EmbOptimType.EXACT_ADAGRAD,
    "exact_row_wise_adagrad": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
}

_DYN_POOL = {
    "none": DynamicEmbPoolingMode.NONE,
    "sum": DynamicEmbPoolingMode.SUM,
    "mean": DynamicEmbPoolingMode.MEAN,
}

_FBGEMM_POOL = {
    "none": PoolingMode.NONE,
    "sum": PoolingMode.SUM,
    "mean": PoolingMode.MEAN,
}

_OPT_STATE_DIM = {
    "sgd": lambda d: 0,
    "adam": lambda d: 2 * d,
    "exact_adagrad": lambda d: d,
    "exact_row_wise_adagrad": lambda d: 1,
}


def table_idx_to_name(i):
    return f"t_{i}"


def feature_idx_to_name(i):
    return f"cate_{i}"


def dtype_size(dt):
    return torch.tensor([], dtype=dt).element_size()


# ── BenchmarkConfig ──────────────────────────────────────────────────────────


@dataclass
class BenchmarkConfig:
    batch_size: int = 65536
    num_embeddings_per_feature: List[int] = field(
        default_factory=lambda: [24 * 1024 * 1024]
    )
    embedding_dim: int = 128
    optimizer_type: str = "adam"
    caching: bool = False
    cache_algorithm: str = "lru"
    gpu_ratio: float = 1.0
    hbm_for_embeddings: List[int] = field(
        default_factory=lambda: [36 * (1024**3)]
    )
    feature_distribution: str = "pow-law"
    alpha: float = 1.05
    pooling_mode: str = "none"
    max_hotness: int = 10
    num_iterations: int = 100
    emb_precision: str = "fp32"
    output_dtype: str = "fp32"
    use_index_dedup: bool = False
    learning_rate: float = 0.1
    eps: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    seed: int = 42

    @property
    def num_tables(self):
        return len(self.num_embeddings_per_feature)

    @property
    def value_dim(self):
        opt_fn = _OPT_STATE_DIM.get(self.optimizer_type, lambda d: 0)
        return self.embedding_dim + opt_fn(self.embedding_dim)

    @property
    def mode(self):
        if self.caching:
            return "caching"
        return "gpu" if self.gpu_ratio >= 1.0 else "no_caching"

    def label(self):
        caps = "_".join(
            f"{e // (1024 * 1024)}M" for e in self.num_embeddings_per_feature
        )
        return (
            f"T{self.num_tables}_B{self.batch_size}_D{self.embedding_dim}_"
            f"{self.optimizer_type}_{self.mode}_"
            f"pool={self.pooling_mode}_cap={caps}"
        )


# ── GPU-accelerated data generation ─────────────────────────────────────────


def generate_sparse_features_gpu(cfg: BenchmarkConfig, device: torch.device):
    """Batch-generate all sparse features on GPU.

    All random number generation happens in bulk GPU calls.  Only the final
    KJT construction loops in Python (unavoidable since KJT is a Python object).
    """
    num_tables = cfg.num_tables
    num_iters = cfg.num_iterations
    bs = cfg.batch_size
    feature_names = [feature_idx_to_name(i) for i in range(num_tables)]
    is_pooling = cfg.pooling_mode != "none"

    if is_pooling:
        all_lengths = torch.randint(
            1,
            cfg.max_hotness + 1,
            (num_iters, bs * num_tables),
            device=device,
            dtype=torch.int64,
        )
    else:
        all_lengths = torch.ones(
            num_iters, bs * num_tables, device=device, dtype=torch.int64
        )

    if cfg.feature_distribution == "random":
        total_vals = int(all_lengths.sum().item())
        all_values = torch.randint(
            0, (2**63) - 1, (total_vals,), device=device, dtype=torch.int64
        )
    elif cfg.feature_distribution in ("pow-law", "zipf"):
        from dataset_generator import PowerLaw, zipf

        per_table_lengths = all_lengths.view(num_iters, num_tables, bs)
        per_table_totals = per_table_lengths.sum(dim=(0, 2))

        per_table_vals = []
        for t in range(num_tables):
            n_samples = int(per_table_totals[t].item())
            cap = cfg.num_embeddings_per_feature[t]
            if cfg.feature_distribution == "pow-law":
                vals = PowerLaw(1, cap, cfg.alpha, n_samples, device)
            else:
                vals = zipf(0, cap, cfg.alpha, n_samples, device)
            per_table_vals.append(vals.to(torch.int64))

        per_table_iter_counts = per_table_lengths.sum(dim=2)
        per_table_offsets = []
        for t in range(num_tables):
            cs = torch.zeros(num_iters + 1, device=device, dtype=torch.long)
            torch.cumsum(per_table_iter_counts[:, t], dim=0, out=cs[1:])
            per_table_offsets.append(cs)

        total_vals = int(all_lengths.sum().item())
        all_values = torch.empty(total_vals, device=device, dtype=torch.int64)
        pos = 0
        for i in range(num_iters):
            for t in range(num_tables):
                s = int(per_table_offsets[t][i].item())
                e = int(per_table_offsets[t][i + 1].item())
                cnt = e - s
                all_values[pos : pos + cnt] = per_table_vals[t][s:e]
                pos += cnt
    else:
        raise ValueError(f"Unsupported distribution: {cfg.feature_distribution}")

    iter_counts = all_lengths.sum(dim=1)
    iter_offsets = torch.zeros(num_iters + 1, device=device, dtype=torch.long)
    torch.cumsum(iter_counts, dim=0, out=iter_offsets[1:])

    res = []
    for i in range(num_iters):
        s = int(iter_offsets[i].item())
        e = int(iter_offsets[i + 1].item())
        res.append(
            torchrec.KeyedJaggedTensor(
                keys=feature_names,
                values=all_values[s:e],
                lengths=all_lengths[i],
            )
        )
    return res


# ── Model creation ───────────────────────────────────────────────────────────


def create_dynamic_embedding_tables(cfg: BenchmarkConfig, device: torch.device):
    table_options = []
    for i in range(cfg.num_tables):
        table_options.append(
            DynamicEmbTableOptions(
                index_type=torch.int64,
                embedding_dtype=get_emb_precision(cfg.emb_precision),
                dim=cfg.embedding_dim,
                max_capacity=cfg.num_embeddings_per_feature[i],
                local_hbm_for_values=cfg.hbm_for_embeddings[i],
                bucket_capacity=128,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.NORMAL,
                ),
                score_strategy=(
                    DynamicEmbScoreStrategy.LFU
                    if cfg.cache_algorithm == "lfu"
                    else DynamicEmbScoreStrategy.TIMESTAMP
                ),
                caching=cfg.caching,
            )
        )

    var = BatchedDynamicEmbeddingTablesV2(
        table_options=table_options,
        table_names=[table_idx_to_name(i) for i in range(cfg.num_tables)],
        use_index_dedup=cfg.use_index_dedup,
        pooling_mode=_DYN_POOL[cfg.pooling_mode],
        output_dtype=get_emb_precision(cfg.output_dtype),
        device=device,
        optimizer=_DYN_OPT[cfg.optimizer_type],
        learning_rate=cfg.learning_rate,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
        beta1=cfg.beta1,
        beta2=cfg.beta2,
    )

    var.fill_tables(load_factor=0.95)

    return var


def create_split_table_batched_embeddings(cfg: BenchmarkConfig, device: torch.device):
    optimizer = _FBGEMM_OPT[cfg.optimizer_type]
    D = cfg.embedding_dim
    Es = cfg.num_embeddings_per_feature
    cache_alg = (
        CacheAlgorithm.LRU if cfg.cache_algorithm == "lru" else CacheAlgorithm.LFU
    )
    pooling = _FBGEMM_POOL[cfg.pooling_mode]

    if cfg.caching:
        kwargs = {}
        if _HAS_STATS_REPORTER:
            kwargs["stats_reporter_config"] = StdLogStatsReporterConfig(
                REPORT_INTERVAL
            )
        emb = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (e, D, EmbeddingLocation.MANAGED_CACHING, ComputeDevice.CUDA)
                for e in Es
            ],
            optimizer=optimizer,
            weights_precision=get_fbgemm_precision(cfg.emb_precision),
            stochastic_rounding=False,
            cache_load_factor=cfg.gpu_ratio,
            cache_algorithm=cache_alg,
            pooling_mode=pooling,
            output_dtype=get_fbgemm_precision(cfg.output_dtype),
            device=device,
            learning_rate=cfg.learning_rate,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            beta1=cfg.beta1,
            beta2=cfg.beta2,
            bounds_check_mode=BoundsCheckMode.NONE,
            record_cache_metrics=RecordCacheMetrics(True, False),
            **kwargs,
        ).cuda()
    else:
        loc = (
            EmbeddingLocation.MANAGED
            if abs(cfg.gpu_ratio - 1.0) > 1e-3
            else EmbeddingLocation.DEVICE
        )
        emb = SplitTableBatchedEmbeddingBagsCodegen(
            [(e, D, loc, ComputeDevice.CUDA) for e in Es],
            optimizer=optimizer,
            weights_precision=get_fbgemm_precision(cfg.emb_precision),
            stochastic_rounding=False,
            pooling_mode=pooling,
            output_dtype=get_fbgemm_precision(cfg.output_dtype),
            device=device,
            learning_rate=cfg.learning_rate,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            beta1=cfg.beta1,
            beta2=cfg.beta2,
            bounds_check_mode=BoundsCheckMode.NONE,
        ).cuda()
    return emb


# ── Benchmark execution ──────────────────────────────────────────────────────


def benchmark_train_eval(
    model, sparse_features, timer, num_iterations,
    use_profiler=False, trace_prefix="",
):
    """Measure train / forward-only / eval latencies (ms per iteration).

    When *use_profiler* is True the train loop is also wrapped with
    ``torch.profiler`` so a single pass yields both latency numbers and
    profiler data.

    The measured loops are bracketed with cudaProfilerStart/Stop so external
    profilers (nsys, ncu) launched with ``--profile-from-start off`` capture
    only the benchmark iterations, not setup.
    """
    model.train()

    n_warm = min(WARMUP_ITERS, num_iterations)
    for i in range(n_warm):
        sf = sparse_features[i]
        output = model(sf.values(), sf.offsets())
        grad = torch.empty_like(output)
        output.backward(grad)
    torch.cuda.synchronize()

    if use_profiler:
        from torch.profiler import ProfilerActivity, profile, schedule

        if num_iterations >= 8:
            wait, warmup, active = 1, 2, num_iterations - 3
        else:
            wait, warmup, active = 0, 1, max(1, num_iterations - 1)
        prof_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            record_shapes=True,
            with_stack=True,
        )
    else:
        prof_ctx = contextlib.nullcontext()

    torch.cuda.cudart().cudaProfilerStart()

    with prof_ctx as prof:
        timer.start()
        for i in range(num_iterations):
            sf = sparse_features[i]
            torch.cuda.nvtx.range_push(f"train_iter_{i}")
            torch.cuda.nvtx.range_push("forward")
            output = model(sf.values(), sf.offsets())
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("backward")
            grad = torch.empty_like(output)
            output.backward(grad)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()
            if use_profiler:
                prof.step()
        timer.stop()
    train_ms = timer.elapsed_time() / num_iterations

    timer.start()
    for i in range(num_iterations):
        sf = sparse_features[i]
        output = model(sf.values(), sf.offsets())
    timer.stop()
    fwd_ms = timer.elapsed_time() / num_iterations

    bwd_ms = train_ms - fwd_ms

    model.eval()
    timer.start()
    for i in range(num_iterations):
        sf = sparse_features[i]
        output = model(sf.values(), sf.offsets())
    timer.stop()
    eval_ms = timer.elapsed_time() / num_iterations

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    if use_profiler:
        trace_file = f"{trace_prefix}trace.json"
        prof.export_chrome_trace(trace_file)
        print(f"  Chrome trace -> {trace_file}")
        print(
            prof.key_averages().table(sort_by="device_time_total", row_limit=40)
        )

    return {
        "train_ms": train_ms,
        "forward_ms": fwd_ms,
        "backward_ms": bwd_ms,
        "eval_ms": eval_ms,
        "profiler": prof if use_profiler else None,
    }


# ── Kernel pattern definitions ────────────────────────────────────────────────


KERNEL_NAME_PATTERNS = {
    "segmented_unique": [
        "segmented_unique", "segmented_init_kernel",
        "compact_keys_and_freq",
    ],
    "expand_table_ids": ["expand_table_ids"],
    "get_table_range": ["get_table_range"],
    "hash_find": ["table_lookup_kernel", "lookup", "find_kernel", "_find"],
    "hash_insert": [
        "table_insert_and_evict", "table_insert_kernel",
        "insert_and_evict", "insert_kernel",
        "select_insert_failed", "update_counter_with_layout",
    ],
    "load_from_flat": [
        "load_from_flat_table_kernel",
        "load_from_flat_table", "load_from_flat",
    ],
    "gather_embedding": [
        "one_to_one_warp",
        "multi_to_one_cta_per_ev", "multi_to_one_warp_per_ev",
        "forwardsequencefusedcopy", "forwardpooledfusedcopy",
        "gather_embedding",
        "add_offset_kernel", "get_new_length_and_offsets",
    ],
    "reduce_grads": [
        "multi_to_one_reduce",
        "reduce_grads",
    ],
    "optimizer_update": [
        "update4_with_index_flat_table",
        "update_with_index_flat_table",
        "vecoptimizer",
        "sgd_update", "adam_update",
        "adagrad_update", "rowwise_adagrad",
        "update_for_flat_table", "update_for_padded_buffer",
    ],
    "store_to_flat": [
        "store_to_flat_table_kernel",
        "store_to_flat_table", "store_to_flat",
    ],
    "flagged_compact": ["flagged_compact", "deviceselect", "devicescan"],
    "key_manipulation": [
        "compose_segmented_key", "decompose_segmented_key",
    ],
    "scoring": ["assign_scores", "no_eviction_assign_scores"],
    "initializer": ["initialize_with_index", "init_curand"],
}


# ── Pre-compute N_unique via segmented_unique ────────────────────────────────


def precompute_unique_counts(sparse_features, num_tables, device):
    """Return list of N_unique per iteration (cheap GPU operation)."""
    from dynamicemb_extensions import (
        expand_table_ids_cuda,
        get_table_range,
        segmented_unique_cuda,
    )

    feature_offsets = torch.arange(num_tables + 1, device=device, dtype=torch.int64)
    counts = []
    for kjt in sparse_features:
        indices = kjt.values()
        offsets = kjt.offsets()
        table_range = get_table_range(offsets, feature_offsets)
        table_ids = expand_table_ids_cuda(
            table_range, None, num_tables, 1, indices.numel(),
        )
        num_uniques, _, _, _, _ = segmented_unique_cuda(
            indices, table_ids, num_tables, None
        )
        counts.append(num_uniques.item())
    return counts


# ── Bandwidth computation ────────────────────────────────────────────────────


def get_kernel_patterns(cfg: BenchmarkConfig, avg_n_unique, avg_n_total):
    """Return kernel-group dict with 'patterns' and 'bytes' per group.

    Only includes ops where we can estimate data movement.
    """
    emb_dim = cfg.embedding_dim
    elem = dtype_size(get_emb_precision(cfg.emb_precision))
    out_elem = dtype_size(get_emb_precision(cfg.output_dtype))
    vdim = cfg.value_dim
    bs = cfg.batch_size
    total_D = emb_dim * cfg.num_tables
    is_pooling = cfg.pooling_mode != "none"
    Nu = avg_n_unique
    Nt = avg_n_total

    byte_counts = {
        "load_from_flat": Nu * emb_dim * elem,
        "store_to_flat": Nu * vdim * elem,
        "gather_embedding": (
            (Nu * emb_dim * elem + bs * total_D * out_elem)
            if is_pooling
            else (Nu + Nt) * emb_dim * out_elem
        ),
        "reduce_grads": (Nt + Nu) * emb_dim * elem,
        "optimizer_update": Nu * (emb_dim + 2 * vdim) * elem,
        "segmented_unique": (2 * Nt + Nu) * 8,
        "hash_find": Nu * 16,
        "hash_insert": Nu * 32,
    }

    return {
        name: {"patterns": KERNEL_NAME_PATTERNS[name], "bytes": byte_counts[name]}
        for name in byte_counts
        if name in KERNEL_NAME_PATTERNS
    }


def compute_bandwidth_report(prof, avg_n_unique, avg_n_total, cfg: BenchmarkConfig):
    """Match profiler kernel events to known ops and compute achieved BW."""
    kernels = get_kernel_patterns(cfg, avg_n_unique, avg_n_total)

    peak_bw = get_peak_bandwidth()
    events = prof.key_averages()
    rows = []
    for name, info in kernels.items():
        matched = [
            e
            for e in events
            if e.self_device_time_total > 0
            and any(p in e.key.lower() for p in info["patterns"])
        ]
        if not matched:
            continue
        avg_us = sum(e.device_time_total / e.count for e in matched if e.count > 0)
        if avg_us <= 0:
            continue
        data_bytes = info["bytes"]
        bw = (data_bytes / 1e9) / (avg_us / 1e6)
        pct = f"{100 * bw / peak_bw:.1f}%" if peak_bw else "N/A"
        rows.append(
            {
                "kernel": name,
                "avg_time_us": avg_us,
                "data_mb": data_bytes / 1e6,
                "bw_gb_s": bw,
                "pct_peak": pct,
            }
        )
    return rows


# ── NVTX-based operation breakdown ────────────────────────────────────────────


def _is_cuda_kernel(e):
    """Return True if *e* represents an actual CUDA kernel (not a CPU/autograd wrapper).

    Wrapper events (ProfilerStep*, autograd nodes, aten:: ops) have non-zero
    self CPU time and their self CUDA time double-counts child kernels.  Real
    CUDA kernels (and memcpy/memset) have self_cpu_time_total == 0.
    """
    if e.self_device_time_total <= 0:
        return False
    if e.self_cpu_time_total > 0:
        return False
    if e.key.startswith("ProfilerStep"):
        return False
    return True


def analyze_nvtx_breakdown(prof, num_active_iters=None):
    """Categorize CUDA kernel time by DynamicEmb operation using KERNEL_NAME_PATTERNS.

    Only actual CUDA kernels are counted (autograd wrappers and profiler
    bookkeeping events are excluded to avoid double-counting).

    Returns (rows, top_unmatched) where rows is a list of dicts with keys:
    op, cuda_time_us, pct_total, count, avg_per_iter_us; and top_unmatched is
    a list of profiler events not matched by any pattern.
    """
    events = prof.key_averages()
    kernel_events = [e for e in events if _is_cuda_kernel(e)]

    assigned_keys = set()
    rows = []
    for op_name, patterns in KERNEL_NAME_PATTERNS.items():
        matched = [
            e for e in kernel_events
            if any(p in e.key.lower() for p in patterns)
        ]
        time_us = sum(e.self_device_time_total for e in matched)
        count = sum(e.count for e in matched)
        for e in matched:
            assigned_keys.add(e.key)
        rows.append({
            "op": op_name,
            "cuda_time_us": time_us,
            "count": count,
        })

    total_device_us = sum(e.self_device_time_total for e in kernel_events)
    categorized_us = sum(r["cuda_time_us"] for r in rows)
    rest_us = max(0.0, total_device_us - categorized_us)

    unmatched = [e for e in kernel_events if e.key not in assigned_keys]
    top_unmatched = sorted(
        unmatched, key=lambda e: e.self_device_time_total, reverse=True
    )

    rows.append({"op": "rest", "cuda_time_us": rest_us, "count": 0})

    for r in rows:
        r["pct_total"] = 100.0 * r["cuda_time_us"] / total_device_us if total_device_us > 0 else 0.0
        if num_active_iters and num_active_iters > 0:
            r["avg_per_iter_us"] = r["cuda_time_us"] / num_active_iters
        else:
            r["avg_per_iter_us"] = r["cuda_time_us"]

    rows.append({
        "op": "TOTAL",
        "cuda_time_us": total_device_us,
        "pct_total": 100.0,
        "count": 0,
        "avg_per_iter_us": total_device_us / num_active_iters if num_active_iters else total_device_us,
    })
    return rows, top_unmatched


def format_nvtx_report(breakdown, top_unmatched=None, max_unmatched=15):
    """Format the NVTX breakdown as a human-readable table."""
    cols = [
        ("Operation", 22),
        ("CUDA Time(us)", 14),
        ("Avg/Iter(us)", 13),
        ("% Total", 8),
        ("Count", 6),
    ]
    header = " | ".join(_fmt(n, w) for n, w in cols)
    sep = "-+-".join("-" * w for _, w in cols)
    lines = [
        "",
        "  DynamicEmb Operation Breakdown",
        f"  {header}",
        f"  {sep}",
    ]
    total_device_us = 0
    for r in breakdown:
        is_total = r["op"] == "TOTAL"
        is_rest = r["op"] == "rest"
        if is_total:
            total_device_us = r["cuda_time_us"]
        count_str = "--" if (is_rest or is_total) else str(r["count"])
        lines.append(
            "  "
            + " | ".join([
                _fmt(r["op"], 22),
                _fmt(f"{r['cuda_time_us']:.1f}", 14),
                _fmt(f"{r['avg_per_iter_us']:.1f}", 13),
                _fmt(f"{r['pct_total']:.1f}%", 8),
                _fmt(count_str, 6),
            ])
        )
        if is_rest:
            lines.append(f"  {sep}")

    if top_unmatched:
        lines.append("")
        lines.append("  Top unmatched kernels (in 'rest'):")
        ucols = [("Kernel", 60), ("CUDA Time(us)", 14), ("% Total", 8), ("Count", 6)]
        lines.append("  " + " | ".join(_fmt(n, w) for n, w in ucols))
        lines.append("  " + "-+-".join("-" * w for _, w in ucols))
        for e in top_unmatched[:max_unmatched]:
            pct = 100.0 * e.self_device_time_total / total_device_us if total_device_us > 0 else 0.0
            name = e.key[:60]
            lines.append(
                "  " + " | ".join([
                    _fmt(name, 60),
                    _fmt(f"{e.self_device_time_total:.1f}", 14),
                    _fmt(f"{pct:.1f}%", 8),
                    _fmt(str(e.count), 6),
                ])
            )

    return "\n".join(lines)


# ── Summary tables ───────────────────────────────────────────────────────────


def _fmt(val, width):
    """Right-align a string to *width*."""
    return f"{val:>{width}}"


def format_summary_table(results):
    if not results:
        return "No results."
    cols = [
        ("label", 50),
        ("T", 3),
        ("batch", 9),
        ("optim", 8),
        ("cch", 3),
        ("pool", 4),
        ("dyn_fwd", 9),
        ("dyn_bwd", 9),
        ("dyn_trn", 9),
        ("dyn_evl", 9),
        ("trc_fwd", 9),
        ("trc_bwd", 9),
        ("trc_trn", 9),
        ("trc_evl", 9),
    ]
    header = " | ".join(_fmt(n, w) for n, w in cols)
    sep = "-+-".join("-" * w for _, w in cols)
    lines = [header, sep]

    for r in results:
        vals = [
            (r.get("label", "")[:50], 50),
            (str(r.get("num_tables", "")), 3),
            (str(r.get("batch_size", "")), 9),
            (r.get("optimizer_type", ""), 8),
            ("Y" if r.get("caching") else "N", 3),
            (r.get("pooling_mode", ""), 4),
            (f"{r.get('dyn_forward_ms', 0):.3f}", 9),
            (f"{r.get('dyn_backward_ms', 0):.3f}", 9),
            (f"{r.get('dyn_train_ms', 0):.3f}", 9),
            (f"{r.get('dyn_eval_ms', 0):.3f}", 9),
            (f"{r.get('trc_forward_ms', 0):.3f}", 9),
            (f"{r.get('trc_backward_ms', 0):.3f}", 9),
            (f"{r.get('trc_train_ms', 0):.3f}", 9),
            (f"{r.get('trc_eval_ms', 0):.3f}", 9),
        ]
        lines.append(" | ".join(_fmt(v, w) for v, w in vals))
    return "\n".join(lines)


def format_bandwidth_table(rows):
    if not rows:
        return "  (no matching kernels found -- inspect full profiler output above)"
    cols = [
        ("kernel", 22),
        ("avg_us", 10),
        ("data_MB", 10),
        ("BW_GB/s", 10),
        ("%peak", 8),
    ]
    header = " | ".join(_fmt(n, w) for n, w in cols)
    sep = "-+-".join("-" * w for _, w in cols)
    lines = [header, sep]
    for r in rows:
        lines.append(
            " | ".join(
                [
                    _fmt(r["kernel"], 22),
                    _fmt(f"{r['avg_time_us']:.1f}", 10),
                    _fmt(f"{r['data_mb']:.2f}", 10),
                    _fmt(f"{r['bw_gb_s']:.1f}", 10),
                    _fmt(r["pct_peak"], 8),
                ]
            )
        )
    return "\n".join(lines)


def write_results(results, json_path=None, csv_path=None):
    if json_path:
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4, default=str)
        print(f"Results -> {json_path}")
    if csv_path and results:
        flat = []
        for r in results:
            row = {k: v for k, v in r.items() if k != "bandwidth"}
            flat.append(row)
        keys = list(flat[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(flat)
        print(f"Results -> {csv_path}")


# ── Single benchmark run ─────────────────────────────────────────────────────


def run_single_benchmark(
    cfg: BenchmarkConfig,
    device: torch.device,
    timer: GPUTimer,
    use_profiler: bool = True,
) -> Dict[str, Any]:
    print(f"\n{'=' * 80}")
    print(f"Config: {cfg.label()}")
    print(f"{'=' * 80}")

    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.empty_cache()

    timer.start()
    dynamic_emb = create_dynamic_embedding_tables(cfg, device)
    timer.stop()
    print(f"  DynamicEmb created in {timer.elapsed_time() / 1000:.3f} s")

    timer.start()
    torchrec_emb = create_split_table_batched_embeddings(cfg, device)
    timer.stop()
    print(f"  TorchRec created in {timer.elapsed_time() / 1000:.3f} s")

    timer.start()
    sparse_features = generate_sparse_features_gpu(cfg, device)
    timer.stop()
    print(f"  Data generated in {timer.elapsed_time() / 1000:.3f} s")

    unique_counts = precompute_unique_counts(sparse_features, cfg.num_tables, device)
    avg_n_unique = sum(unique_counts) / len(unique_counts)
    avg_n_total = sum(sf.values().numel() for sf in sparse_features) / len(
        sparse_features
    )
    print(f"  Avg N_unique={avg_n_unique:.0f}  Avg N_total={avg_n_total:.0f}")

    if cfg.caching:
        dynamic_emb.set_record_cache_metrics(True)
        dynamic_emb.reset_cache_states()
        torchrec_emb.reset_cache_states()

    bw_results: List[Dict] = []
    nvtx_breakdown: List[Dict] = []

    print("\n  >> DynamicEmb benchmark")
    dyn = benchmark_train_eval(
        dynamic_emb, sparse_features, timer, cfg.num_iterations,
        use_profiler=use_profiler,
        trace_prefix=f"dynamicemb_{cfg.label()}_",
    )
    if dyn["profiler"]:
        prof = dyn["profiler"]
        n_iters = cfg.num_iterations
        num_active = (n_iters - 3) if n_iters >= 8 else max(1, n_iters - 1)
        bw_results = compute_bandwidth_report(prof, avg_n_unique, avg_n_total, cfg)
        nvtx_breakdown, top_unmatched = analyze_nvtx_breakdown(prof, num_active_iters=num_active)
        print(format_nvtx_report(nvtx_breakdown, top_unmatched))

    if cfg.caching:
        dynamic_emb.reset_cache_states()
        torchrec_emb.reset_cache_states()

    print("\n  >> TorchRec benchmark")
    trc = benchmark_train_eval(
        torchrec_emb, sparse_features, timer, cfg.num_iterations,
        use_profiler=use_profiler,
        trace_prefix=f"torchrec_{cfg.label()}_",
    )

    if cfg.caching:
        dynamic_emb.set_record_cache_metrics(False)

    result = {
        "label": cfg.label(),
        "num_tables": cfg.num_tables,
        "batch_size": cfg.batch_size,
        "embedding_dim": cfg.embedding_dim,
        "optimizer_type": cfg.optimizer_type,
        "caching": cfg.caching,
        "pooling_mode": cfg.pooling_mode,
        "num_embeddings_per_feature": cfg.num_embeddings_per_feature,
        "feature_distribution": cfg.feature_distribution,
        "avg_n_unique": avg_n_unique,
        "avg_n_total": avg_n_total,
        "dyn_forward_ms": dyn["forward_ms"],
        "dyn_backward_ms": dyn["backward_ms"],
        "dyn_train_ms": dyn["train_ms"],
        "dyn_eval_ms": dyn["eval_ms"],
        "trc_forward_ms": trc["forward_ms"],
        "trc_backward_ms": trc["backward_ms"],
        "trc_train_ms": trc["train_ms"],
        "trc_eval_ms": trc["eval_ms"],
    }
    if bw_results:
        result["bandwidth"] = bw_results
    if nvtx_breakdown:
        result["nvtx_breakdown"] = nvtx_breakdown

    print(
        f"\n  DynamicEmb  train={dyn['train_ms']:.3f}  fwd={dyn['forward_ms']:.3f}"
        f"  bwd={dyn['backward_ms']:.3f}  eval={dyn['eval_ms']:.3f} ms"
    )
    print(
        f"  TorchRec    train={trc['train_ms']:.3f}  fwd={trc['forward_ms']:.3f}"
        f"  bwd={trc['backward_ms']:.3f}  eval={trc['eval_ms']:.3f} ms"
    )
    if bw_results:
        print("\n  Bandwidth (DynamicEmb):")
        print(format_bandwidth_table(bw_results))

    del dynamic_emb, torchrec_emb, sparse_features
    torch.cuda.empty_cache()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic config generation from CLI params
# ═══════════════════════════════════════════════════════════════════════════════

CACHE_GPU_RATIO = 0.1


def _cache_hbm(gpu_ratio, cap_per_table, num_tables, dim, optimizer_type):
    """HBM for caching mode: gpu_ratio fraction of the full table per table."""
    opt_fn = _OPT_STATE_DIM.get(optimizer_type, lambda d: 0)
    value_dim = dim + opt_fn(dim)
    per_table = int(gpu_ratio * cap_per_table * value_dim * 4)
    return [per_table] * num_tables


def generate_configs(params):
    """Build a list of BenchmarkConfig from the CLI-driven params dict."""
    configs = []
    for nt in params["num_tables"]:
        for cap in params["cap_per_table"]:
            caps = [cap] * nt
            for bs in params["batch_sizes"]:
                for opt in params["optimizers"]:
                    for pool in params["pooling_modes"]:
                        for dim in params["dims"]:
                            for mode in params["modes"]:
                                if mode == "gpu":
                                    configs.append(BenchmarkConfig(
                                        batch_size=bs,
                                        num_embeddings_per_feature=caps,
                                        embedding_dim=dim,
                                        hbm_for_embeddings=[sys.maxsize] * nt,
                                        optimizer_type=opt,
                                        caching=False,
                                        gpu_ratio=1.0,
                                        pooling_mode=pool,
                                        num_iterations=params["num_iterations"],
                                    ))
                                elif mode == "caching":
                                    configs.append(BenchmarkConfig(
                                        batch_size=bs,
                                        num_embeddings_per_feature=caps,
                                        embedding_dim=dim,
                                        hbm_for_embeddings=_cache_hbm(
                                            CACHE_GPU_RATIO, cap, nt, dim, opt,
                                        ),
                                        optimizer_type=opt,
                                        caching=True,
                                        cache_algorithm="lru",
                                        gpu_ratio=CACHE_GPU_RATIO,
                                        pooling_mode=pool,
                                        num_iterations=params["num_iterations"],
                                    ))
                                elif mode == "no-caching":
                                    configs.append(BenchmarkConfig(
                                        batch_size=bs,
                                        num_embeddings_per_feature=caps,
                                        embedding_dim=dim,
                                        hbm_for_embeddings=[0] * nt,
                                        optimizer_type=opt,
                                        caching=False,
                                        gpu_ratio=0.1,
                                        pooling_mode=pool,
                                        num_iterations=params["num_iterations"],
                                    ))
    return configs


# ── Test entry point ──────────────────────────────────────────────────────────


def pytest_generate_tests(metafunc):
    """Dynamically parametrize 'cfg' from CLI options."""
    if "cfg" not in metafunc.fixturenames:
        return
    from conftest import build_benchmark_params

    params = build_benchmark_params(metafunc.config)
    configs = generate_configs(params)
    metafunc.parametrize("cfg", configs, ids=[c.label() for c in configs])


def test_benchmark(cfg, device, timer, use_torch_profiler):
    result = run_single_benchmark(cfg, device, timer, use_profiler=use_torch_profiler)
    assert "error" not in result
