#!/bin/bash
# Run the dynamicemb benchmark via pytest.
#
# Benchmark modes (--mode):
#   gpu         -- full table in HBM, gpu_ratio=1.0 (default)
#   caching     -- 10% HBM with LRU caching
#   no-caching  -- 10% HBM without caching (UVM / eviction)
#   all         -- run all three modes
#
# Configuration options (comma-separated values for sweeps):
#   --num-tables     Number of tables (default: 10)
#   --cap-per-table  Capacity per table, e.g. 1M, 24M (default: 1M)
#   --batch-size     Batch size (default: 65536)
#   --optimizer      sgd, adam, exact_adagrad, exact_row_wise_adagrad (default: sgd)
#   --pooling        none, sum, mean (default: none)
#   --dim            Embedding dimension (default: 128)
#   --num-iterations Number of benchmark iterations (default: 100)
#   --mode           gpu, caching, no-caching, all (default: gpu)
#
# Profiling (default: torch profiler with NVTX breakdown):
#   --profile         Enable torch profiler with operation breakdown (default: on)
#   --no-profile      Disable torch profiler (latency-only)
#
# External profiling (nsys / ncu):
#   The benchmark loop uses cudaProfilerStart/Stop so external tools can
#   capture only the measured iterations.  Launch with e.g.:
#     nsys profile --capture-range=cudaProfiler ./benchmark/benchmark_batched_dynamicemb_tables.sh
#     ncu --profile-from-start off ./benchmark/benchmark_batched_dynamicemb_tables.sh
#
# Usage examples:
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh                               # defaults
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh --optimizer sgd,adam           # sweep optimizers
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh --mode gpu,caching            # sweep modes
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh --num-tables 1,5,10           # sweep table counts
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh --no-profile                  # latency only
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh -k "adam"                     # pytest filter
#   ./benchmark/benchmark_batched_dynamicemb_tables.sh --co                          # list configs

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

torchrun --nnodes 1 --nproc_per_node 1 \
    -m pytest ./benchmark/benchmark_batched_dynamicemb_tables.py \
    -v -s "$@"
