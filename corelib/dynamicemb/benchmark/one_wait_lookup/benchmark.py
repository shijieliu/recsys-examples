# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare Python prefetch variants on a shared #545 training workload."""

import argparse
import json
import os
import statistics
from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
from unittest.mock import patch

VARIANTS = {
    "hbm": ("baseline", "fused", "one_wait"),
    "caching": ("baseline", "early", "late"),
}


def lookup_variant(storage, name):
    """Select a prefetch implementation for this single-threaded benchmark."""
    if name not in VARIANTS[storage]:
        raise ValueError(f"Unknown {storage} variant {name!r}")
    if name == "baseline":
        return nullcontext()

    from dynamicemb import batched_dynamicemb_function as lookup

    if storage == "hbm":
        from hbm import prefetch_hbm_direct

        prefetch = partial(
            prefetch_hbm_direct,
            wait_for_misses=name == "one_wait",
            fuse_slot_scatter=True,
        )
        return patch.object(lookup, "_prefetch_hbm_direct_path", prefetch)

    from cache import prefetch_cache

    return patch.object(
        lookup, "_prefetch_cache_path", partial(prefetch_cache, placement=name)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", choices=VARIANTS, default="hbm")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--inputs", type=int, default=2)
    parser.add_argument("--hotness", type=int, default=4)
    parser.add_argument("--collections", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20, help="Steps per timed window")
    parser.add_argument("--windows", type=int, default=4, help="Windows per variant")
    args = parser.parse_args()
    if (
        min(
            args.batch_size,
            args.inputs,
            args.hotness,
            args.collections,
            args.steps,
            args.windows,
        )
        < 1
    ):
        parser.error("All sizes and timing counts must be positive")
    if os.getenv("DEMB_DETERMINISM_MODE") is not None:
        parser.error("Unset DEMB_DETERMINISM_MODE for this counted-insert benchmark")

    import benchmark_many_collections as bench
    import dynamicemb
    import dynamicemb_extensions
    import torch

    config = bench.Config(
        storage=args.storage,
        batch_size=args.batch_size,
        inputs=args.inputs,
        hotness=args.hotness,
        collections=args.collections,
        optimizer="sgd",
    )
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(config.seed)
    work = bench.Workload(config, device, verbose=False)
    if args.storage == "caching":
        # #557 removed this pre-insert host capacity check. Disable it on
        # these benchmark-owned modules so all arms measure the same work.
        for module in work.modules:
            module._prefetch_outstanding_keys = None

    names = VARIANTS[args.storage]
    timings = {name: {"wall_ms": [], "host_ms": []} for name in names}
    for name in names:
        with lookup_variant(args.storage, name):
            for _ in range(2 * args.inputs):
                work.step()

    for window in range(args.windows):
        offset = window % len(names)
        for name in names[offset:] + names[:offset]:
            with lookup_variant(args.storage, name):
                # Start each arm on the same input, even for an odd window size.
                align = -work.count % args.inputs
                sample = bench.measure(work, align, args.steps, windows=1)
            for metric, values in timings[name].items():
                values.append(sample[metric]["median"])

    results = {}
    for name in names:
        with lookup_variant(args.storage, name):
            for _ in range(-work.count % args.inputs):
                work.step()
            profile = bench.profile_steps(work.step, max(4, args.inputs))
        results[name] = {
            "windows": timings[name],
            "median_ms": {
                metric: statistics.median(values)
                for metric, values in timings[name].items()
            },
            "profile_per_step": profile,
        }

    # Reading cache metrics synchronizes. Sample a whole input cycle outside
    # the timing and profiler windows using the stock metric implementation.
    cache = bench.cache_round(work)
    print(
        json.dumps(
            {
                "config": asdict(config),
                "steps_per_window": args.steps,
                "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "dynamicemb": dynamicemb.__file__,
                "extension": dynamicemb_extensions.__file__,
                "cache_round": cache,
                "cache_hit_rate": bench.hit_rate({"cache": cache}),
                "variants": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
