# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Combine one lookup's NCU traffic with the same unprofiled batch's latency."""

import argparse
import csv
import json
import math
from pathlib import Path


BYTES = {
    "hbm_read": "dram__bytes_read.sum",
    "hbm_write": "dram__bytes_write.sum",
    "pcie_rx": "pcie__read_bytes.sum",
    "pcie_tx": "pcie__write_bytes.sum",
}
BYTE_SCALE = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}


def summarize(raw, timings, capture, hbm_gbps, pcie_gbps):
    if not all(math.isfinite(rate) and rate > 0 for rate in (hbm_gbps, pcie_gbps)):
        raise ValueError("bandwidth ceilings must be finite and positive")
    if len(capture) != 1 or capture[0]["profile"] not in (
        "lookup-train",
        "lookup-eval",
    ):
        raise ValueError("select one lookup-train or lookup-eval case per capture")
    config = capture[0]
    phase = config["profile"].split("-", 1)[1]
    step = config["iteration"]
    matching = [case for case in timings if case["label"] == config["label"]]
    if len(matching) != 1:
        raise ValueError("expected one matching case in the unprofiled timings")
    elapsed_ms = matching[0]["lookup_steps_ms"][phase][step]
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise ValueError("invalid unprofiled lookup latency")

    csv_rows = csv.DictReader(raw)
    units = next(csv_rows)
    scale = {name: BYTE_SCALE[units[metric]] for name, metric in BYTES.items()}
    total = dict.fromkeys(BYTES, 0.0)
    launches = 0
    for row in csv_rows:
        if not row.get("ID"):
            continue
        for name, metric in BYTES.items():
            count = float(row[metric].replace(",", "")) * scale[name]
            if not math.isfinite(count) or count < 0:
                raise ValueError(f"invalid {metric} for kernel {row['ID']}")
            total[name] += count
        launches += 1
    if not launches:
        raise ValueError("no kernel measurements in the NCU report")

    # SI GB/s times 1e6 converts ms to bytes. PCIe RX and TX can overlap.
    traffic = {
        "hbm": total["hbm_read"] + total["hbm_write"],
        "pcie_rx": total["pcie_rx"],
        "pcie_tx": total["pcie_tx"],
    }
    rates = {name: size / (1e6 * elapsed_ms) for name, size in traffic.items()}
    ceilings = {"hbm": hbm_gbps, "pcie_rx": pcie_gbps, "pcie_tx": pcie_gbps}
    floor_ms = max(size / (1e6 * ceilings[name]) for name, size in traffic.items())
    return {
        **config,
        "kernel_launches": launches,
        "unprofiled_latency_ms": elapsed_ms,
        "physical_bytes": total,
        "GB_per_s": rates,
        "ceiling_GB_per_s": ceilings,
        "percent_of_ceiling": {
            name: 100 * rates[name] / ceilings[name] for name in rates
        },
        "observed_traffic_floor_ms": floor_ms,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raw_csv", type=Path, help="NCU --page raw --csv for one lookup"
    )
    parser.add_argument("timings", type=Path, help="unprofiled benchmark JSON")
    parser.add_argument(
        "capture", type=Path, help="profiled benchmark JSON naming phase and step"
    )
    parser.add_argument(
        "--hbm-gbps",
        type=float,
        required=True,
        help="aggregate HBM read + write ceiling",
    )
    parser.add_argument(
        "--pcie-gbps",
        type=float,
        required=True,
        help="one-direction physical PCIe ceiling",
    )
    args = parser.parse_args()
    with args.raw_csv.open(newline="") as raw:
        result = summarize(
            raw,
            json.loads(args.timings.read_text()),
            json.loads(args.capture.read_text()),
            args.hbm_gbps,
            args.pcie_gbps,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
