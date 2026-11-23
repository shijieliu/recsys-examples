# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


def full_count(rows: torch.Tensor) -> torch.Tensor:
    """Create a device count for a buffer whose entire first dimension is live.

    Callers with a count from compaction or deduplication pass that count through.
    The first dimension of a padded buffer describes only its capacity.
    """
    return rows.new_full((1,), rows.size(0), dtype=torch.int64)
