# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

import torch
from dynamicemb_extensions import flagged_compact


@dataclass
class LookupBatch:
    """Fields aligned by key within a lookup step.

    Weights are per-key inputs, such as the frequency increments used in
    admission. Per-occurrence pooling weights live outside the batch.
    """

    keys: torch.Tensor
    table_ids: torch.Tensor
    values: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None

    def compact(
        self,
        num_valid: Optional[torch.Tensor] = None,
        *,
        host_num_valid: Optional[int] = None,
        invert: bool = False,
        output_rows: Optional[torch.Tensor] = None,
        backing_rows: Optional[torch.Tensor] = None,
        copy_weights: bool = True,
        copy_scores: bool = True,
    ) -> "CompactedBatch":
        """Pack selected metadata. Keep wide values in the source buffer."""
        if self.mask is None:
            raise ValueError("Compaction requires a row mask")
        count, input_rows, (keys, table_ids, weights, scores, outputs, backing) = (
            flagged_compact(
                self.mask,
                [
                    self.keys,
                    self.table_ids,
                    self.weights if copy_weights else None,
                    self.scores if copy_scores else None,
                    output_rows,
                    backing_rows,
                ],
                num_valid,
                invert=invert,
                host_num_valid=host_num_valid,
            )
        )
        return CompactedBatch(
            keys,
            table_ids,
            count,
            input_rows,
            input_rows if outputs is None else outputs,
            weights=weights,
            scores=scores,
            source_values=self.values,
            backing_rows=backing,
        )


@dataclass(frozen=True)
class CompactedBatch:
    """Selected key metadata and its device length, with explicit row mappings.

    input_rows selects rows from source_values. output_rows maps back to the
    destination supplied by the caller, or to the input when no map is supplied.
    Only the leading num_valid entries of the selected arrays are live.
    """

    keys: torch.Tensor
    table_ids: torch.Tensor
    num_valid: torch.Tensor
    input_rows: torch.Tensor
    output_rows: torch.Tensor
    weights: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
    source_values: Optional[torch.Tensor] = None
    backing_rows: Optional[torch.Tensor] = None
