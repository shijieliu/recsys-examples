# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun or srun, one process per GPU, on two or more GPUs."""

import os
import sys
from unittest.mock import patch

import torch
import torch.distributed as dist
import torchrec
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbTableOptions,
    EmbOptimType,
    get_table_value_bytes,
)
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import DynamicEmbeddingBagCollectionSharder, quiesce
from dynamicemb.shard.hier_output_dist import _PooledOutputFunction
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import Topology
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import PoolingType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "benchmark"))
from benchmark_utils import init_distributed  # noqa: E402


def model(output_dist, weighted, device, world_size, local_size):
    tables = [
        torchrec.EmbeddingBagConfig(
            name="shared",
            feature_names=["a", "b"],
            embedding_dim=32,
            num_embeddings=65536,
            pooling=PoolingType.SUM,
        ),
        torchrec.EmbeddingBagConfig(
            name="other",
            feature_names=["c"],
            embedding_dim=32,
            num_embeddings=65536,
            pooling=PoolingType.SUM if weighted else PoolingType.MEAN,
        ),
    ]
    ebc = torchrec.EmbeddingBagCollection(
        tables=tables, is_weighted=weighted, device=torch.device("meta")
    )
    constraints = {
        table.name: DynamicEmbParameterConstraints(
            sharding_types=[ShardingType.ROW_WISE.value],
            enforce_hbm=True,
            use_dynamicemb=True,
            dynamicemb_options=DynamicEmbTableOptions(
                global_hbm_for_values=get_table_value_bytes(
                    table, EmbOptimType.SGD, world_size
                ),
                dist_type="roundrobin",
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.DEBUG
                ),
                output_dist=output_dist,
                output_dist_capacity=65536,
            ),
        )
        for table in tables
    }
    sharder = DynamicEmbeddingBagCollectionSharder(
        fused_params={"optimizer": EmbOptimType.SGD, "learning_rate": 1.0}
    )
    planner = DynamicEmbeddingShardingPlanner(
        topology=Topology(
            world_size=world_size, local_world_size=local_size, compute_device="cuda"
        ),
        constraints=constraints,
    )
    plan = planner.collective_plan(ebc, [sharder], dist.group.WORLD)
    return DistributedModelParallel(
        module=ebc, device=device, sharders=[sharder], plan=plan
    )


def batch(rank, weighted, device, long_bag=False):
    # Input features are permuted. Key 7 repeats within bags, across ranks,
    # and in the two features sharing a table. Each feature has empty bags.
    keys = [7, 7, rank, rank + 17, 7, 7, 3, 3, 7, 3, rank, 7, 7, 7, 4]
    lengths = [0, 3, 1, 1, 2, 0, 3, 0, 1, 2, 0, 2]
    if long_bag:
        # Exercise the cooperative bag sum and the backward's heavy-key partials.
        keys.extend([7] * 1024)
        lengths[-1] += 1024
    weights = None
    if weighted:
        weights = (torch.arange(len(keys), device=device).float() % 5 - 2) * 0.5
    return torchrec.KeyedJaggedTensor.from_lengths_sync(
        keys=["c", "a", "b"],
        values=torch.tensor(keys, device=device),
        lengths=torch.tensor(lengths, dtype=torch.int32, device=device),
        weights=weights,
    )


def check(weighted, rank, world_size, local_size, device):
    reference, hierarchical = (
        model(which, weighted, device, world_size, local_size)
        for which in ("torchrec", "hierarchical")
    )
    with (
        patch.object(
            _PooledOutputFunction, "forward", wraps=_PooledOutputFunction.forward
        ) as fwd,
        patch.object(
            _PooledOutputFunction, "backward", wraps=_PooledOutputFunction.backward
        ) as bwd,
    ):
        for step in range(3):
            expected = reference(batch(rank, weighted, device, long_bag=(step == 1)))
            actual = hierarchical(batch(rank, weighted, device, long_bag=(step == 1)))
            assert actual.keys() == expected.keys()
            torch.testing.assert_close(
                actual.values(), expected.values(), atol=1e-4, rtol=1e-5
            )
            grad = torch.arange(actual.values().numel(), device=device).reshape_as(
                actual.values()
            )
            grad = ((grad + rank + step) % 7 - 3).to(actual.values().dtype)
            expected.values().backward(grad)
            actual.values().backward(grad)
            assert fwd.call_count == bwd.call_count == step + 1
        reference.eval()
        hierarchical.eval()
        with torch.no_grad():
            expected = reference(batch(rank, weighted, device))
            actual = hierarchical(batch(rank, weighted, device))
            torch.testing.assert_close(
                actual.values(), expected.values(), atol=1e-4, rtol=1e-5
            )
            assert fwd.call_count == 3
    quiesce(hierarchical)


if __name__ == "__main__":
    rank, world_size, local_rank, local_size = init_distributed()
    assert world_size >= 2
    for weighted in (False, True):
        check(weighted, rank, world_size, local_size, torch.device("cuda", local_rank))
    dist.destroy_process_group()
    if rank == 0:
        print("Hierarchical mixed SUM/MEAN and weighted SUM passed.")
