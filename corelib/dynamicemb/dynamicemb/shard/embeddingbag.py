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

from typing import Dict, List, Optional

import torch
from torch import nn
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    EmbeddingShardingContext,
    EmbeddingShardingInfo,
    KJTListSplitsAwaitable,
)
from torchrec.distributed.embedding_types import EmbeddingComputeKernel, ShardingType
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.types import (
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
)
from torchrec.modules.embedding_configs import data_type_to_dtype
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from ..dynamicemb_config import DynamicEmbKernel
from ..scored_kjt import without_scores
from .input_dist import HierStartAwaitable, RwPooledHierInputDist
from .per_key_score import collect_per_key_score, require_scores
from .rw_sharding import RwPooledDynamicEmbeddingSharding
from .twrw_sharding import TwRwPooledDynamicEmbeddingSharding


class ShardedDynamicEmbeddingBagCollection(ShardedEmbeddingBagCollection):
    supported_compute_kernels: List[str] = [
        kernel.value for kernel in EmbeddingComputeKernel
    ] + [DynamicEmbKernel]

    def __init__(self, *args, per_key_score: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # Whether this collection's tables take a score per key; the same on every
        # rank. See :func:`collect_per_key_score`.
        self._per_key_score = per_key_score
        self._direct_hier_input = False
        self.register_buffer("_sum_feature_mask", None, persistent=False)

    def input_dist(self, ctx, features):
        # TorchRec prepares the KJT under no_grad, including permutations
        # that would discard this flag before the requester sees the weights.
        if (
            self._is_weighted
            and torch.is_grad_enabled()
            and isinstance(features, KeyedJaggedTensor)
        ):
            weights = features.weights_or_none()
            if weights is not None and weights.requires_grad:
                raise ValueError(
                    "Pooling weights with requires_grad=True are unsupported"
                )
        if self._per_key_score:
            if self.training:
                require_scores(features)
            else:
                # Evaluation reads no score, so it does not have to be given
                # any; one given is dropped here, on every rank alike, so the
                # ranks agree on whether the distribution sends scores.
                features = without_scores(features)
        # TorchRec's KJT.split reads the per-feature totals on the host even
        # when the only split is the entire KJT. The hierarchical second wait
        # already supplies those totals. After TorchRec initializes its feature
        # order, pass that single split directly to its distributor.
        if (
            self.training
            and self._direct_hier_input
            and self._input_dists[0].hier.takes_hier_dist
            and isinstance(features, KeyedJaggedTensor)
            and not features.variable_stride_per_key()
        ):
            distributor = self._input_dists[0]
            distributor.training = True
            with torch.no_grad():
                if self._has_features_permute:
                    features = features.permute(
                        self._features_order, self._features_order_tensor
                    )
                if self._has_mean_pooling_callback:
                    lengths = features.lengths()
                    weights = features.weights_or_none()
                    if weights is not None:
                        lengths = torch.ops.fbgemm.segment_sum_csr(
                            1, features.offsets().int(), weights
                        )
                    # TorchRec adds epsilon to every divisor, including SUM
                    # features of a mixed collection.
                    divisor = lengths.reshape(-1, features.stride()).T
                    if self._sum_feature_mask is not None:
                        divisor = divisor.masked_fill(self._sum_feature_mask, 1)
                    divisor = divisor + 1e-6
                    width = self._embedding_dims[0]
                    ctx.divisor = (
                        divisor.unsqueeze(-1)
                        .expand(-1, -1, width)
                        .reshape(features.stride(), -1)
                    )
                request = distributor(features)
            ctx.sharding_contexts.append(request.sharding_context)
            return KJTListSplitsAwaitable([request], ctx)
        # TorchRec keeps these lazy distributors in a plain list. A later
        # eval()/train() must select the path for this batch as well.
        for distributor in self._input_dists:
            if isinstance(distributor, RwPooledHierInputDist):
                distributor.training = self.training
        pending = super().input_dist(ctx, features)
        # EBC creates each context after calling that sharding's input dist.
        # Use the context its hierarchical second wait will populate.
        for i, request in enumerate(pending.awaitables):
            if isinstance(request, HierStartAwaitable):
                ctx.sharding_contexts[i] = request.sharding_context
        return pending

    def _create_input_dist(self, *args, **kwargs):
        super()._create_input_dist(*args, **kwargs)
        # Lazy modules start in training mode even when created by an eval call.
        for input_dist in self._input_dists:
            input_dist.training = self.training
        self._direct_hier_input = len(self._input_dists) == 1 and isinstance(
            self._input_dists[0], RwPooledHierInputDist
        )
        if self._direct_hier_input:
            # Leave TorchRec in charge of remapping shared input features
            # across tables in its mixed-pooling divisor.
            features = self._embedding_shardings[0].feature_names()
            if self._has_mean_pooling_callback and len(set(features)) != len(features):
                self._direct_hier_input = False
                return
            if self._has_mean_pooling_callback:
                sums = self._pooling_type_to_rs_features.get("SUM", ())
                if sums:
                    self._sum_feature_mask = torch.tensor(
                        [f in sums for f in features], device=self._device
                    )

    def reset_parameters(self) -> None:
        # TorchRec registers an empty weight for shards a rank does not hold but
        # skips CUSTOMIZED_KERNEL tables; a table-row-wise DynamicEmb table lives
        # on one node, so the other nodes' ranks need a placeholder before
        # reset_parameters and state_dict look for a weight to touch. It has the
        # same (1, 1) meta shape the kernel registers on owner ranks, so a
        # checkpoint planner sees one identical entry per rank and dedups it.
        dtypes = {
            config.name: data_type_to_dtype(config.data_type)
            for config in self._embedding_bag_configs
        }
        for table_name, parameter_sharding in self.module_sharding_plan.items():
            if (
                parameter_sharding.compute_kernel
                != EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value
            ):
                continue
            if not hasattr(self.embedding_bags[table_name], "weight"):
                self.embedding_bags[table_name].register_parameter(
                    "weight",
                    nn.Parameter(
                        torch.empty(
                            (1, 1),
                            device=torch.device("meta"),
                            dtype=dtypes[table_name],
                        )
                    ),
                )
        super().reset_parameters()

    @classmethod
    def create_embedding_bag_sharding(
        cls,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        permute_embeddings: bool = False,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> EmbeddingSharding[
        EmbeddingShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
    ]:
        """
        override this function to provide customized EmbeddingSharding
        """
        sharding_type = sharding_infos[0].param_sharding.sharding_type

        if sharding_type == ShardingType.ROW_WISE.value:
            return RwPooledDynamicEmbeddingSharding(
                sharding_infos=sharding_infos,
                env=env,
                device=device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        elif sharding_type == ShardingType.TABLE_ROW_WISE.value:
            return TwRwPooledDynamicEmbeddingSharding(
                sharding_infos=sharding_infos,
                env=env,
                device=device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        else:
            return super().create_embedding_bag_sharding(
                sharding_infos=sharding_infos,
                env=env,
                device=device,
                permute_embeddings=permute_embeddings,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )


class DynamicEmbeddingBagCollectionSharder(EmbeddingBagCollectionSharder):
    """
    DynamicEmbeddingBagCollectionSharder extends the EmbeddingBagCollectionSharder class from the TorchREC repo.
    The usage is completely consistent with TorchREC's EmbeddingBagCollectionSharder.
    """

    def shard(
        self,
        module: EmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedEmbeddingBagCollection:
        per_key_score = collect_per_key_score(params)
        return ShardedDynamicEmbeddingBagCollection(
            module=module,
            table_name_to_parameter_sharding=params,
            env=env,
            fused_params=self.fused_params,
            device=device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
            module_fqn=module_fqn,
            per_key_score=per_key_score,
        )
