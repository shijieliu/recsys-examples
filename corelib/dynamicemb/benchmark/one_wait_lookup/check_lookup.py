# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""GPU correctness checks. Run with pytest on the counted DynamicEmb runtime."""

import pytest
import torch
from benchmark import lookup_variant
from benchmark_many_collections import Config, Workload, cache_round, make_collection
from dynamicemb import DynamicEmbInitializerArgs, DynamicEmbInitializerMode
from dynamicemb.initializer import MultiTableInitializer
from dynamicemb.types import CopyMode


def set_initializer(module, tables, mode, device):
    args = DynamicEmbInitializerArgs(mode=mode, value=0.0)
    module._initializer = MultiTableInitializer.create([args] * tables, device)


@pytest.mark.parametrize(
    ("storage", "variant"),
    [("hbm", "one_wait"), ("hbm", "fused"), ("caching", "early"), ("caching", "late")],
)
def test_training_matches_counted_lookup(storage, variant):
    device = torch.device("cuda", torch.cuda.current_device())
    cfg = Config(
        storage=storage,
        collections=1,
        batch_size=2048,
        dim=8,
        num_embeddings=131072,
        inputs=2,
        lr=0.05,
    )
    outputs = {}
    for label in ("baseline", variant):
        torch.manual_seed(cfg.seed)
        work = Workload(cfg, device, verbose=False)
        module = work.modules[0]
        if storage == "caching":
            module._prefetch_outstanding_keys = None
        set_initializer(module, cfg.tables, DynamicEmbInitializerMode.CONSTANT, device)
        with lookup_variant(storage, label):
            outputs[label] = [work.step()[0].detach().clone() for _ in range(6)]
        if storage == "caching":
            # Verify the two inputs actually exercise eviction and reload.
            assert all(step["evicted"] > 0 for step in cache_round(work))
    assert outputs["baseline"][2].abs().max().item() > 0
    for actual, expected in zip(outputs[variant], outputs["baseline"]):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("placement", ["early", "late"])
def test_refused_insert_preserves_backing_and_releases_hit_pin(placement):
    device = torch.device("cuda", torch.cuda.current_device())
    cfg = Config(
        storage="caching", collections=1, tables=1, dim=8, num_embeddings=65536
    )
    cache_rows = 1024
    module = make_collection(cfg, 0, device, cache_rows / cfg.num_embeddings)
    module._prefetch_outstanding_keys = None
    set_initializer(module, 1, DynamicEmbInitializerMode.DEBUG, device)
    # A single bucket and its overflow area can hold 4096 keys.
    assert module.cache._state.capacity == 4 * cache_rows
    keys = torch.arange(1, 6001, dtype=torch.int64, device=device)
    with lookup_variant("caching", placement):
        # One existing cache hit takes a pin in the failing batch.
        single = module(keys[:1], torch.tensor([0, 1], device=device))
        single.backward(torch.zeros_like(single))
        offsets = torch.tensor([0, keys.numel()], device=device)
        with pytest.raises(RuntimeError, match="1904"):
            module.prefetch(keys, offsets)

    assert (
        torch.count_nonzero(module.cache._state.key_index_map._ref_counter).item() == 0
    )
    found = module.tables.find(
        keys, torch.zeros_like(keys), copy_mode=CopyMode.EMBEDDING
    )
    assert found[5].all().item()
    expected = keys.float().unsqueeze(1).expand(-1, cfg.dim)
    torch.testing.assert_close(found[7], expected, rtol=0, atol=0)
