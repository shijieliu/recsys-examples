# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch


def full_count(rows: torch.Tensor) -> torch.Tensor:
    """Create a device count for a buffer whose entire first dimension is live.

    Callers with a count from compaction or deduplication pass that count through.
    The first dimension of a padded buffer describes only its capacity.
    """
    return rows.new_full((1,), rows.size(0), dtype=torch.int64)


def read_host_count(
    capacity: int, num_valid: Optional[torch.Tensor], host_num_valid: Optional[int]
) -> int:
    """Resolve the live prefix for an operation that needs its length on the CPU."""
    limit = (
        capacity if host_num_valid is None else max(0, min(capacity, host_num_valid))
    )
    if num_valid is None or limit == 0:
        return limit
    return max(0, min(limit, int(num_valid.item())))
