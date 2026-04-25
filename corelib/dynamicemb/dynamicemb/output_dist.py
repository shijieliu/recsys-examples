import logging
from typing import Dict, List, Optional, Union, cast

import torch
from torch import distributed as dist
from torchrec.distributed.dist_data import (
    PooledEmbeddingsReduceScatter,
    SequenceEmbeddingsAllToAll,
    VariableBatchPooledEmbeddingsReduceScatter,
)
from torchrec.distributed.embedding_sharding import (
    BaseEmbeddingDist,
    EmbeddingShardingContext,
)
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
from torchrec.distributed.types import Awaitable, CommOp, QuantizedCommCodecs

from .hier_all2all import HierAll2AllManager

logger = logging.getLogger(__name__)


class RwSequenceEmbeddingDist(
    BaseEmbeddingDist[SequenceShardingContext, torch.Tensor, torch.Tensor]
):
    """
    Redistributes sequence embedding tensor in RW fashion with an AlltoAll operation.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        num_features (int): total number of features.
        device (Optional[torch.device]): device on which buffers will be allocated.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        num_features: int,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__()
        self._dist = SequenceEmbeddingsAllToAll(
            pg,
            [num_features] * pg.size(),
            device,
            codecs=(
                qcomm_codecs_registry.get(
                    CommOp.SEQUENCE_EMBEDDINGS_ALL_TO_ALL.name, None
                )
                if qcomm_codecs_registry
                else None
            ),
        )

    def forward(
        self,
        local_embs: torch.Tensor,
        sharding_ctx: Optional[SequenceShardingContext] = None,
    ) -> Awaitable[torch.Tensor]:
        """
        Performs AlltoAll operation on sequence embeddings tensor.

        Args:
            local_embs (torch.Tensor): tensor of values to distribute.
            sharding_ctx (SequenceShardingContext): shared context from KJTAllToAll
                operation.

        Returns:
            Awaitable[torch.Tensor]: awaitable of sequence embeddings.
        """
        assert sharding_ctx is not None
        return self._dist(
            local_embs,
            lengths=sharding_ctx.lengths_after_input_dist,
            input_splits=sharding_ctx.input_splits,
            output_splits=sharding_ctx.output_splits,
            batch_size_per_rank=sharding_ctx.batch_size_per_rank,
            sparse_features_recat=sharding_ctx.sparse_features_recat,
            unbucketize_permute_tensor=sharding_ctx.unbucketize_permute_tensor,
        )


class HierarchicalAwaitable(Awaitable[torch.Tensor]):
    """Awaitable that wraps an already-computed tensor from HierAll2AllManager."""

    def __init__(self, tensor: torch.Tensor) -> None:
        super().__init__()
        self._tensor = tensor

    def _wait_impl(self) -> torch.Tensor:
        return self._tensor


class HierarchicalSequenceEmbeddingDist(
    BaseEmbeddingDist[SequenceShardingContext, torch.Tensor, torch.Tensor]
):
    """Redistributes sequence embedding tensor using hierarchical NVLink + IB all2all.

    Replaces baseline NCCL SequenceEmbeddingsAllToAll with a custom hierarchical
    all2all targeting Hopper GPUs. Uses outcast NVLink writes for intra-node and
    NCCL GIN puts for inter-node, with fused recat and unbucketize.

    Sequence embeddings only -- pooled embeddings use reduce-scatter (out of scope).

    Falls back to baseline RwSequenceEmbeddingDist if:
    - Non-Hopper GPU
    - No P2P access
    - GIN init failure (multi-node)
    - Quantized communication requested

    Args:
        pg: ProcessGroup for communication.
        num_features: total number of features.
        max_rows_per_rank: worst-case rows any rank handles per output_dist call.
        D: embedding dimension (uniform within this dist instance).
        device: device on which buffers will be allocated.
        dtype: embedding data type.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        num_features: int,
        max_rows_per_rank: int,
        D: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self._pg = pg
        self._num_features = num_features
        self._D = D

        device = device or torch.device(f"cuda:{torch.cuda.current_device()}")

        # Initialize hierarchical all2all manager
        self._manager = HierAll2AllManager(
            pg=pg,
            num_features=num_features,
            max_rows_per_rank=max_rows_per_rank,
            D=D,
            device=device,
            dtype=dtype,
        )

        # Baseline fallback dist (created lazily if needed)
        self._fallback_dist: Optional[SequenceEmbeddingsAllToAll] = None
        if self._manager.fallback:
            self._fallback_dist = SequenceEmbeddingsAllToAll(
                pg,
                [num_features] * pg.size(),
                device,
            )
            logger.info(
                "HierarchicalSequenceEmbeddingDist: using baseline fallback"
            )

    @property
    def is_fallback(self) -> bool:
        return self._manager.fallback

    def forward(
        self,
        local_embs: torch.Tensor,
        sharding_ctx: Optional[SequenceShardingContext] = None,
    ) -> Awaitable[torch.Tensor]:
        """Performs hierarchical all2all on sequence embeddings.

        Args:
            local_embs: tensor of values to distribute.
            sharding_ctx: shared context from KJTAllToAll operation.

        Returns:
            Awaitable[torch.Tensor]: awaitable of sequence embeddings.
        """
        assert sharding_ctx is not None

        # Use baseline fallback if hierarchical is not available
        if self._fallback_dist is not None:
            return self._fallback_dist(
                local_embs,
                lengths=sharding_ctx.lengths_after_input_dist,
                input_splits=sharding_ctx.input_splits,
                output_splits=sharding_ctx.output_splits,
                batch_size_per_rank=sharding_ctx.batch_size_per_rank,
                sparse_features_recat=sharding_ctx.sparse_features_recat,
                unbucketize_permute_tensor=sharding_ctx.unbucketize_permute_tensor,
            )

        # Hierarchical path
        result = self._manager.forward(
            output_embs=local_embs,
            lengths_after_input_dist=sharding_ctx.lengths_after_input_dist,
            input_splits=sharding_ctx.input_splits,
            output_splits=sharding_ctx.output_splits,
            sparse_features_recat=sharding_ctx.sparse_features_recat,
            unbucketize_permute=sharding_ctx.unbucketize_permute_tensor,
            batch_size_per_rank=sharding_ctx.batch_size_per_rank,
        )

        return HierarchicalAwaitable(result)


class RwPooledEmbeddingDist(
    BaseEmbeddingDist[EmbeddingShardingContext, torch.Tensor, torch.Tensor]
):
    """
    Redistributes pooled embedding tensor in RW fashion by performing a reduce-scatter
    operation.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for reduce-scatter communication.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        embedding_dims: List[int],
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__()

        self._dist: Optional[
            Union[
                PooledEmbeddingsReduceScatter,
                VariableBatchPooledEmbeddingsReduceScatter,
            ]
        ] = None
        self._pg = pg
        self._qcomm_codecs_registry = qcomm_codecs_registry
        self._codecs: Optional[QuantizedCommCodecs] = (
            qcomm_codecs_registry.get(
                CommOp.POOLED_EMBEDDINGS_REDUCE_SCATTER.name, None
            )
            if qcomm_codecs_registry
            else None
        )
        self._embedding_dims = embedding_dims

    def forward(
        self,
        local_embs: torch.Tensor,
        sharding_ctx: Optional[EmbeddingShardingContext] = None,
    ) -> Awaitable[torch.Tensor]:
        """
        Performs reduce-scatter pooled operation on pooled embeddings tensor.

        Args:
            local_embs (torch.Tensor): pooled embeddings tensor to distribute.
            sharding_ctx (Optional[EmbeddingShardingContext]): shared context from
                KJTAllToAll operation.

        Returns:
            Awaitable[torch.Tensor]: awaitable of pooled embeddings tensor.
        """
        if self._dist is None:
            self._create_output_dist_module(sharding_ctx)

        if sharding_ctx is None:
            return cast(PooledEmbeddingsReduceScatter, self._dist)(local_embs)
        elif sharding_ctx.variable_batch_per_feature:
            return cast(VariableBatchPooledEmbeddingsReduceScatter, self._dist)(
                local_embs,
                batch_size_per_rank_per_feature=sharding_ctx.batch_size_per_rank_per_feature,
                embedding_dims=self._embedding_dims,
            )
        else:
            return cast(PooledEmbeddingsReduceScatter, self._dist)(
                local_embs,
                input_splits=sharding_ctx.batch_size_per_rank,
            )

    def _create_output_dist_module(
        self, sharding_ctx: Optional[EmbeddingShardingContext] = None
    ) -> None:
        if sharding_ctx is not None and sharding_ctx.variable_batch_per_feature:
            self._dist = VariableBatchPooledEmbeddingsReduceScatter(
                pg=self._pg,
                codecs=self._codecs,
            )
        else:
            self._dist = PooledEmbeddingsReduceScatter(
                pg=self._pg,
                codecs=self._codecs,
            )
