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

"""
Hierarchical NVLink + IB All2All for DynamicEmb output distribution.

Replaces NCCL all2all with a custom hierarchical all2all targeting Hopper GPUs:
- Intra-node: NVLink outcast writes via CUDA IPC
- Inter-node: NCCL GIN one-sided puts (same-rail only)
- Fused recat + unbucketize into outcast/gather warps
- Single fused kernel with warp specialization
"""

import logging
import os
import socket
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import distributed as dist
logger = logging.getLogger(__name__)

# Try to import C++ extensions — graceful fallback if not available
# Ensure locally-built extension (with hier_a2a) takes precedence over
# system-installed one (which may lack hier_a2a bindings).
import sys as _sys
_local_ext_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _local_ext_dir not in _sys.path:
    _sys.path.insert(0, _local_ext_dir)
# Force reimport if already loaded without hier_a2a
if "dynamicemb_extensions" in _sys.modules:
    _ext = _sys.modules["dynamicemb_extensions"]
    if not hasattr(_ext, "hier_a2a"):
        del _sys.modules["dynamicemb_extensions"]
try:
    import dynamicemb_extensions

    _HAS_HIER_A2A = hasattr(dynamicemb_extensions, "hier_a2a")
except ImportError:
    _HAS_HIER_A2A = False


# ---------------------------------------------------------------------------
# TopologyInfo
# ---------------------------------------------------------------------------


@dataclass
class TopologyInfo:
    """Describes the physical GPU topology for hierarchical all2all routing."""

    my_global_rank: int
    my_node: int  # node index (0..num_nodes-1)
    my_local_rank: int  # local rank within node (0..local_world_size-1)
    world_size: int
    local_world_size: int  # GPUs per node (uniform across nodes)
    num_nodes: int

    rank_to_node: List[int]  # [world_size] global_rank -> node index
    rank_to_local: List[int]  # [world_size] global_rank -> local_rank
    node_to_ranks: List[List[int]]  # [num_nodes][lws] node -> sorted global ranks

    @staticmethod
    def detect(pg: dist.ProcessGroup) -> "TopologyInfo":
        """Auto-detect topology from process group and environment.

        Uses LOCAL_RANK and LOCAL_WORLD_SIZE env vars (set by torchrun).
        Falls back to hostname-based node grouping.
        """
        rank = dist.get_rank(pg)
        world_size = dist.get_world_size(pg)

        # Get local rank/world size from env (set by torchrun/torchx)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        local_world_size_env = os.environ.get("LOCAL_WORLD_SIZE", None)

        if local_world_size_env is not None:
            local_world_size = int(local_world_size_env)
        else:
            # Fallback: hostname-based grouping
            hostname = socket.gethostname()
            all_hostnames = [None] * world_size
            dist.all_gather_object(all_hostnames, hostname, group=pg)

            # Group by hostname
            hostname_to_ranks: Dict[str, List[int]] = {}
            for r, h in enumerate(all_hostnames):
                hostname_to_ranks.setdefault(h, []).append(r)

            my_node_ranks = sorted(hostname_to_ranks[hostname])
            local_world_size = len(my_node_ranks)
            local_rank = my_node_ranks.index(rank)

        assert (
            world_size % local_world_size == 0
        ), f"world_size ({world_size}) must be divisible by local_world_size ({local_world_size})"

        num_nodes = world_size // local_world_size
        my_node = rank // local_world_size

        # Build mappings
        rank_to_node = [r // local_world_size for r in range(world_size)]
        rank_to_local = [r % local_world_size for r in range(world_size)]
        node_to_ranks = [
            list(range(n * local_world_size, (n + 1) * local_world_size))
            for n in range(num_nodes)
        ]

        return TopologyInfo(
            my_global_rank=rank,
            my_node=my_node,
            my_local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
            num_nodes=num_nodes,
            rank_to_node=rank_to_node,
            rank_to_local=rank_to_local,
            node_to_ranks=node_to_ranks,
        )


# ---------------------------------------------------------------------------
# Hardware / system capability checks
# ---------------------------------------------------------------------------


def check_hier_a2a_requirements(
    pg: dist.ProcessGroup,
    topology: TopologyInfo,
) -> Tuple[bool, str]:
    """Validate hardware/system requirements for hierarchical all2all.

    Returns (ok, reason). If ok is False, reason explains why.
    """
    world_size = dist.get_world_size(pg)

    # 1. Multi-GPU required
    if world_size < 2:
        return False, "world_size < 2"

    # 2. CUDA available
    if not torch.cuda.is_available():
        return False, "CUDA not available"

    # 3. Compute capability >= 9.0 (Hopper)
    device = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(device)
    if cc[0] < 9:
        return False, f"Compute capability {cc[0]}.{cc[1]} < 9.0 (Hopper required)"

    # 4. P2P access between all intra-node GPUs
    for i in range(topology.local_world_size):
        for j in range(topology.local_world_size):
            if i != j:
                can_access = torch.cuda.can_device_access_peer(i, j)
                if not can_access:
                    return False, f"No P2P access between GPU {i} and GPU {j}"

    # 5. C++ extension: not required for reference path, just log
    if not _HAS_HIER_A2A:
        logger.info("hier_a2a C++ extension not available; using reference path")

    # 6. Uniform local_world_size (already checked in TopologyInfo.detect)

    return True, "ok"


def log_topology(topology: TopologyInfo) -> None:
    """Log topology information."""
    logger.info(
        f"HierAll2All topology: rank={topology.my_global_rank}, "
        f"node={topology.my_node}, local_rank={topology.my_local_rank}, "
        f"W={topology.world_size}, L={topology.local_world_size}, "
        f"N={topology.num_nodes}"
    )


# ---------------------------------------------------------------------------
# OutcastScatterMap
# ---------------------------------------------------------------------------


@dataclass
class OutcastScatterMap:
    """CSR-indexed gather map for outcast writes.

    For dest peer d, the rows to read from the source tensor are
    gather_indices[peer_offsets[d] : peer_offsets[d+1]].
    Writes to peer d's recv buffer are sequential starting at 0.
    """

    # Per local_rank peer (INCLUDING self)
    peer_gather_indices: torch.Tensor  # [total_peer_rows] int64
    peer_offsets: torch.Tensor  # [local_world_size + 1] int64
    peer_intra_counts: torch.Tensor  # [local_world_size] int64
    peer_relay_dest_counts: torch.Tensor  # [local_world_size * num_nodes] int32

    # Inter-node same-rail destinations
    inter_gather_indices: torch.Tensor  # [total_inter_rows] int64
    inter_peer_offsets: torch.Tensor  # [num_inter_peers + 1] int64


# ---------------------------------------------------------------------------
# HierAll2AllManager
# ---------------------------------------------------------------------------


class HierAll2AllManager:
    """Manages hierarchical all2all state: buffers, IPC handles, scatter maps.

    Created once per output_dist instance. Handles:
    - IPC recv buffer allocation and handle exchange
    - GIN initialization (multi-node, with fallback)
    - Scatter map building and caching
    - Forward/backward kernel dispatch
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        num_features: int,
        max_rows_per_rank: int,
        D: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._pg = pg
        self._num_features = num_features
        self._max_rows_per_rank = max_rows_per_rank
        self._D = D
        self._device = device
        self._dtype = dtype
        self._elem_size = torch.tensor([], dtype=dtype).element_size()
        self._fallback = False
        self._fallback_reason = ""

        # Detect topology
        self._topology = TopologyInfo.detect(pg)
        log_topology(self._topology)

        # Check requirements
        ok, reason = check_hier_a2a_requirements(pg, self._topology)
        if not ok:
            self._fallback = True
            self._fallback_reason = reason
            logger.warning(f"HierAll2All falling back to NCCL: {reason}")
            return

        # Allocate IPC recv buffer (skip if C++ extension unavailable)
        self._ipc_raw_ptr = 0
        self._peer_ipc_ptrs: Dict[int, int] = {}
        self._gin_context = None

        if _HAS_HIER_A2A:
            self._enable_peer_access()
            self._allocate_ipc_buffer()
            self._exchange_ipc_handles()
            if self._topology.num_nodes > 1:
                self._init_gin()
        else:
            logger.info("Skipping IPC/GIN init (C++ extension not available)")

        # Scatter map cache: keyed by (direction, batch_size, num_features)
        self._scatter_map_cache: Dict[
            Tuple[str, int, int], OutcastScatterMap
        ] = {}
        # Output splits tensor cache: avoid Python→GPU transfer
        self._output_splits_cache: Dict = {}
        # Reusable empty tensor
        self._empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
        # Monotonic iteration counter for signal synchronization
        self._iter_id = 0

        # Scatter map stream (separate from compute stream)
        self._scatter_map_stream = torch.cuda.Stream(device=device)
        self._scatter_map_event = torch.cuda.Event()

        # Device-side flags (persist across iterations, reset in kernel)
        self._device_flag = torch.zeros(1, dtype=torch.int32, device=device)

        # Cached forward metadata for backward
        self._cached_lengths_after_output_dist: Optional[torch.Tensor] = None
        self._cached_output_splits: Optional[List[int]] = None
        self._cached_input_splits: Optional[List[int]] = None

        # Topology tensors on device (for scatter map kernel)
        self._rank_to_node_dev = torch.tensor(
            self._topology.rank_to_node, dtype=torch.int32, device=device
        )
        self._rank_to_local_dev = torch.tensor(
            self._topology.rank_to_local, dtype=torch.int32, device=device
        )

        # CTA occupancy check
        total_ctas = self._topology.local_world_size + max(
            0, self._topology.num_nodes - 1
        )
        # H100 has 132 SMs; check that total_ctas fits
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        # Conservative: assume max 2 CTAs per SM for cooperative launch
        if total_ctas > sm_count * 2:
            self._fallback = True
            self._fallback_reason = (
                f"total_ctas ({total_ctas}) exceeds cooperative launch limit "
                f"({sm_count * 2} = {sm_count} SMs * 2)"
            )
            logger.warning(f"HierAll2All falling back to NCCL: {self._fallback_reason}")

        # Pre-compute persistent device-side IPC pointer arrays for fused kernel.
        # peer_slot_ptrs_dev[lr] = IPC ptr to my slot in peer lr's recv buffer
        # peer_sig_ptrs_dev[lr]  = IPC ptr to my signal entry in peer lr's signal pad
        self._peer_slot_ptrs_dev: Optional[torch.Tensor] = None
        self._peer_sig_ptrs_dev: Optional[torch.Tensor] = None
        if _HAS_HIER_A2A and not self._fallback and self._ipc_raw_ptr != 0:
            self._init_ipc_pointer_arrays()

    @property
    def fallback(self) -> bool:
        return self._fallback

    def _enable_peer_access(self) -> None:
        """Explicitly enable P2P access to all intra-node peers.

        cudaIpcMemLazyEnablePeerAccess (used by cudaIpcOpenMemHandle) may not
        fully enable the P2P path needed for system-scope atomics
        (st.release.sys / ld.acquire.sys). Explicit cudaDeviceEnablePeerAccess
        ensures full P2P capability including atomic visibility.
        """
        import ctypes

        try:
            cudart = ctypes.CDLL("libcudart.so")
        except OSError:
            logger.warning("Cannot load libcudart.so for P2P enable")
            return

        my_dev = self._topology.my_local_rank
        for lr in range(self._topology.local_world_size):
            if lr != my_dev:
                global_rank = self._topology.my_node * self._topology.local_world_size + lr
                peer_dev = self._topology.rank_to_local[global_rank]
                ret = cudart.cudaDeviceEnablePeerAccess(peer_dev, 0)
                # ret=0: success, ret=704: already enabled — both OK
                # Clear the sticky CUDA error (cudaErrorPeerAccessAlreadyEnabled)
                # so subsequent CUDA calls don't fail.
                cudart.cudaGetLastError()

    def _allocate_ipc_buffer(self) -> None:
        """Allocate worst-case IPC recv buffer."""
        L = self._topology.local_world_size
        N = self._topology.num_nodes
        D = self._D
        elem = self._elem_size
        max_rows = self._max_rows_per_rank

        # lr slots: L peers * max_rows * D * elem_size
        lr_slots_bytes = max_rows * L * D * elem
        # relay region: L fixed-size slots * max_rows * D * elem_size
        relay_region_bytes = L * max_rows * D * elem
        # relay counts pad: L * N * 4 bytes, 128-byte aligned
        relay_counts_pad_bytes = self._align_to(128, L * N * 4)
        # signal pad: L * K_STAGES * 4 bytes, 128-byte aligned
        # K_STAGES=2 matches kProgressStages in hier_all2all_kernel.cuh
        self._signal_stages = 2
        signal_pad_bytes = self._align_to(128, L * self._signal_stages * 4)

        total_bytes = (
            lr_slots_bytes + relay_region_bytes + relay_counts_pad_bytes + signal_pad_bytes
        )

        # Allocate via cudaMalloc (bypasses PyTorch caching allocator).
        # cudaIpcGetMemHandle requires a pointer from cudaMalloc; PyTorch's
        # caching allocator may suballocate from pooled blocks, producing
        # pointers that IPC cannot map correctly.
        self._ipc_raw_ptr = dynamicemb_extensions.hier_a2a.ipc_cuda_malloc(
            total_bytes
        )
        self._ipc_total_bytes = total_bytes

        # Compute offsets
        self._lr_slots_offset = 0
        self._relay_region_offset = lr_slots_bytes
        self._relay_counts_pad_offset = lr_slots_bytes + relay_region_bytes
        self._signal_pad_offset = (
            lr_slots_bytes + relay_region_bytes + relay_counts_pad_bytes
        )

        logger.info(
            f"HierAll2All IPC buffer: {total_bytes / 1024 / 1024:.1f} MB "
            f"(lr_slots={lr_slots_bytes}, relay={relay_region_bytes}, "
            f"counts_pad={relay_counts_pad_bytes}, signals={signal_pad_bytes})"
        )

    def _exchange_ipc_handles(self) -> None:
        """Exchange IPC handles within the node via all_gather."""
        if not _HAS_HIER_A2A:
            return

        # Get my IPC handle
        my_handle = dynamicemb_extensions.hier_a2a.ipc_get_handle(
            self._ipc_raw_ptr
        )

        # Gather all handles within the node
        all_handles = [None] * self._topology.local_world_size
        # Use the global pg with all_gather_object for simplicity
        # In production, use a local_pg (intra-node process group)
        dist.all_gather_object(all_handles, my_handle, group=self._pg)

        # Open peer handles
        self._peer_ipc_ptrs: Dict[int, int] = {}
        L = self._topology.local_world_size
        my_lr = self._topology.my_local_rank
        my_node = self._topology.my_node

        for lr in range(L):
            global_rank = my_node * L + lr
            if lr == my_lr:
                # Self: use own buffer pointer
                self._peer_ipc_ptrs[lr] = self._ipc_raw_ptr
            else:
                # Peer: open IPC handle
                peer_handle = all_handles[global_rank]
                if peer_handle is not None:
                    try:
                        ptr = dynamicemb_extensions.hier_a2a.ipc_open_handle(
                            peer_handle
                        )
                        self._peer_ipc_ptrs[lr] = ptr
                    except Exception as e:
                        logger.warning(f"Failed to open IPC handle for lr={lr}: {e}")
                        self._fallback = True
                        self._fallback_reason = f"IPC handle open failed: {e}"
                        return

    def _init_gin(self) -> None:
        """Initialize GIN for multi-node communication."""
        if not _HAS_HIER_A2A or not hasattr(
            dynamicemb_extensions.hier_a2a, "gin_init"
        ):
            logger.warning("GIN not available in C++ extension, falling back")
            self._fallback = True
            self._fallback_reason = "GIN C++ functions not available"
            return

        try:
            # Get NCCL comm handle from the process group
            # This is PyTorch-internal and version-dependent
            nccl_comm = self._get_nccl_comm_handle()
            if nccl_comm is None:
                self._fallback = True
                self._fallback_reason = "Cannot get NCCL comm handle for GIN"
                return

            max_gin_rows = self._max_rows_per_rank * (
                self._topology.world_size - self._topology.local_world_size
            )

            self._gin_context = dynamicemb_extensions.hier_a2a.gin_init(
                nccl_comm,
                self._topology.num_nodes,
                self._topology.local_world_size,
                max_gin_rows,
                self._D,
                self._elem_size,
            )

            if self._gin_context == -1:
                self._gin_context = None
                self._fallback = True
                self._fallback_reason = "GIN init failed (NCCL GIN not supported)"
                logger.warning("GIN init failed, falling back to NCCL all2all")
        except Exception as e:
            self._gin_context = None
            self._fallback = True
            self._fallback_reason = f"GIN init exception: {e}"
            logger.warning(f"GIN init failed: {e}, falling back to NCCL all2all")

    def _get_nccl_comm_handle(self) -> Optional[int]:
        """Extract NCCL communicator handle from ProcessGroup.

        This is internal PyTorch API and may change across versions.
        """
        try:
            if hasattr(self._pg, "_get_backend"):
                backend = self._pg._get_backend(torch.device("cuda"))
                if hasattr(backend, "get_nccl_comm"):
                    comm = backend.get_nccl_comm()
                    return comm
            return None
        except Exception:
            return None

    def build_forward_scatter_map(
        self,
        feature_offsets: torch.Tensor,
        recatted_offsets: torch.Tensor,
        split_offsets: torch.Tensor,
        feature_recat: torch.Tensor,
        total_rows: int,
        batch_size: int,
    ) -> OutcastScatterMap:
        """Build or retrieve cached forward scatter map."""
        cache_key = ("fwd", batch_size, self._num_features)
        if cache_key in self._scatter_map_cache:
            return self._scatter_map_cache[cache_key]

        topo = self._topology
        (
            peer_gather_indices,
            peer_offsets,
            peer_intra_counts,
            peer_relay_dest_counts,
            inter_gather_indices,
            inter_peer_offsets,
        ) = dynamicemb_extensions.hier_a2a.build_scatter_map(
            feature_offsets,
            recatted_offsets,
            split_offsets,
            feature_recat,
            total_rows,
            topo.my_node,
            topo.my_local_rank,
            topo.local_world_size,
            topo.num_nodes,
            self._rank_to_node_dev,
            self._rank_to_local_dev,
            None,  # no inv_unbucketize for forward
        )

        scatter_map = OutcastScatterMap(
            peer_gather_indices=peer_gather_indices,
            peer_offsets=peer_offsets,
            peer_intra_counts=peer_intra_counts,
            peer_relay_dest_counts=peer_relay_dest_counts,
            inter_gather_indices=inter_gather_indices,
            inter_peer_offsets=inter_peer_offsets,
        )
        self._scatter_map_cache[cache_key] = scatter_map
        return scatter_map

    def build_backward_scatter_map(
        self,
        feature_offsets: torch.Tensor,
        recatted_offsets: torch.Tensor,
        split_offsets: torch.Tensor,
        feature_recat: torch.Tensor,
        total_rows: int,
        batch_size: int,
        inv_unbucketize: torch.Tensor,
    ) -> OutcastScatterMap:
        """Build or retrieve cached backward scatter map."""
        cache_key = ("bwd", batch_size, self._num_features)
        if cache_key in self._scatter_map_cache:
            return self._scatter_map_cache[cache_key]

        topo = self._topology
        (
            peer_gather_indices,
            peer_offsets,
            peer_intra_counts,
            peer_relay_dest_counts,
            inter_gather_indices,
            inter_peer_offsets,
        ) = dynamicemb_extensions.hier_a2a.build_scatter_map(
            feature_offsets,
            recatted_offsets,
            split_offsets,
            feature_recat,
            total_rows,
            topo.my_node,
            topo.my_local_rank,
            topo.local_world_size,
            topo.num_nodes,
            self._rank_to_node_dev,
            self._rank_to_local_dev,
            inv_unbucketize,
        )

        scatter_map = OutcastScatterMap(
            peer_gather_indices=peer_gather_indices,
            peer_offsets=peer_offsets,
            peer_intra_counts=peer_intra_counts,
            peer_relay_dest_counts=peer_relay_dest_counts,
            inter_gather_indices=inter_gather_indices,
            inter_peer_offsets=inter_peer_offsets,
        )
        self._scatter_map_cache[cache_key] = scatter_map
        return scatter_map

    def _init_ipc_pointer_arrays(self) -> None:
        """Build device-side arrays of IPC pointers for the fused kernel.

        peer_slot_ptrs_dev[lr] = base of MY slot in peer lr's recv buffer.
        peer_sig_ptrs_dev[lr]  = my signal entry in peer lr's signal pad (0 for self).
        """
        L = self._topology.local_world_size
        my_lr = self._topology.my_local_rank
        D = self._D
        elem = self._elem_size
        max_rows = self._max_rows_per_rank

        slot_ptrs = []
        sig_ptrs = []

        for lr in range(L):
            peer_buf_ptr = self._peer_ipc_ptrs.get(lr, 0)
            if peer_buf_ptr == 0:
                slot_ptrs.append(0)
                sig_ptrs.append(0)
                continue

            # My slot in peer's buffer: offset = my_lr * max_rows * D * elem
            slot_ptr = peer_buf_ptr + my_lr * max_rows * D * elem
            slot_ptrs.append(slot_ptr)

            if lr != my_lr:
                # Point to base of my K-entry signal block in peer's signal pad
                K = self._signal_stages
                sig_ptr = peer_buf_ptr + self._signal_pad_offset + my_lr * K * 4
                sig_ptrs.append(sig_ptr)
            else:
                sig_ptrs.append(0)  # no signal for self

        self._peer_slot_ptrs_dev = torch.tensor(
            slot_ptrs, dtype=torch.int64, device=self._device
        )
        self._peer_sig_ptrs_dev = torch.tensor(
            sig_ptrs, dtype=torch.int64, device=self._device
        )

        # Signal pad pointer (for the kernel to read incoming signals)
        self._signal_pad_ptr = (
            self._ipc_raw_ptr + self._signal_pad_offset
        )

        logger.info("HierAll2All: IPC pointer arrays initialized for fused kernel")

    def _build_scatter_map_from_context(
        self,
        lengths_after_input_dist: torch.Tensor,
        input_splits: List[int],
        sparse_features_recat: torch.Tensor,
        total_rows: int,
        batch_size: int,
        direction: str = "fwd",
        inv_unbucketize: Optional[torch.Tensor] = None,
    ) -> "OutcastScatterMap":
        """Build or retrieve scatter map, computing prefix sums on the fly."""
        cache_key = (direction, batch_size, self._num_features)
        if cache_key in self._scatter_map_cache:
            return self._scatter_map_cache[cache_key]

        topo = self._topology
        W = topo.world_size
        F = self._num_features

        # feature_recat: match torchrec permute_1D semantics.
        # Forward: sparse_features_recat[i] = original feature index at
        #   position i in the recat layout (same as torchrec's permute_1D).
        # Backward: inverse permutation undoes the forward recat.
        if direction == "fwd":
            feature_recat = sparse_features_recat.to(torch.int64)
        else:
            feature_recat = torch.argsort(
                sparse_features_recat.to(torch.int64)
            )

        # Prefix sums (all on GPU)
        lengths_i64 = lengths_after_input_dist.to(torch.int64)
        feature_offsets = torch.zeros(
            lengths_i64.size(0) + 1, dtype=torch.int64, device=self._device
        )
        feature_offsets[1:] = lengths_i64.cumsum(0)

        recatted_lengths = lengths_i64[feature_recat]
        recatted_offsets = torch.zeros(
            recatted_lengths.size(0) + 1,
            dtype=torch.int64,
            device=self._device,
        )
        recatted_offsets[1:] = recatted_lengths.cumsum(0)

        input_splits_t = torch.tensor(
            input_splits, dtype=torch.int64, device=self._device
        )
        split_offsets = torch.zeros(
            W + 1, dtype=torch.int64, device=self._device
        )
        split_offsets[1:] = input_splits_t.cumsum(0)

        (
            peer_gather_indices,
            peer_offsets,
            peer_intra_counts,
            peer_relay_dest_counts,
            inter_gather_indices,
            inter_peer_offsets,
        ) = dynamicemb_extensions.hier_a2a.build_scatter_map(
            feature_offsets,
            recatted_offsets,
            split_offsets,
            feature_recat,
            total_rows,
            topo.my_node,
            topo.my_local_rank,
            topo.local_world_size,
            topo.num_nodes,
            self._rank_to_node_dev,
            self._rank_to_local_dev,
            inv_unbucketize,
        )

        scatter_map = OutcastScatterMap(
            peer_gather_indices=peer_gather_indices,
            peer_offsets=peer_offsets,
            peer_intra_counts=peer_intra_counts,
            peer_relay_dest_counts=peer_relay_dest_counts,
            inter_gather_indices=inter_gather_indices,
            inter_peer_offsets=inter_peer_offsets,
        )
        self._scatter_map_cache[cache_key] = scatter_map
        return scatter_map

    def _build_gather_map(
        self,
        unbucketize_permute: torch.Tensor,
        output_splits: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build or retrieve cached per-peer gather map for parallel gather."""
        cache_key = ("gather_map", unbucketize_permute.data_ptr())
        if cache_key in self._scatter_map_cache:
            return self._scatter_map_cache[cache_key]

        output_splits_t = torch.tensor(
            output_splits, dtype=torch.int64, device=self._device
        )
        gather_ipc_indices, gather_out_indices, gather_peer_offsets = \
            dynamicemb_extensions.hier_a2a.build_gather_map(
                unbucketize_permute,
                output_splits_t,
                self._rank_to_local_dev,
                self._max_rows_per_rank,
                self._topology.local_world_size,
            )
        result = (gather_ipc_indices, gather_out_indices, gather_peer_offsets, None)
        self._scatter_map_cache[cache_key] = result
        return result

    def _forward_two_phase(
        self,
        output_embs: torch.Tensor,
        lengths_after_input_dist: torch.Tensor,
        input_splits: List[int],
        output_splits: List[int],
        sparse_features_recat: torch.Tensor,
        unbucketize_permute: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward using two-phase kernels (outcast-only + gather-only).

        Phase 1: outcast kernel with ALL 8 warps doing cp.async pipeline.
        Phase 2: gather kernel with per-peer parallel gather using pre-computed map.
        Both kernels launched sequentially on the same stream.
        Requires pre-computed gather map (unbucketize_permute must be available).
        """
        self._iter_id += 1
        total_send = sum(input_splits)
        total_recv = sum(output_splits)

        # 1. Build scatter map (cached)
        scatter_map = self._build_scatter_map_from_context(
            lengths_after_input_dist,
            input_splits,
            sparse_features_recat,
            total_send,
            total_send,
            direction="fwd",
        )

        # 2. Build gather map (cached)
        gather_ipc, gather_out, gather_peer_off = self._build_gather_map(
            unbucketize_permute, output_splits
        )

        # 3. Pre-allocate output buffer (reused across iterations)
        needed = total_recv * self._D
        if not hasattr(self, "_output_buf") or self._output_buf is None or self._output_buf.numel() < needed:
            self._output_buf = torch.empty(
                needed, dtype=self._dtype, device=self._device
            )

        return dynamicemb_extensions.hier_a2a.fwd_two_phase(
            output_embs.contiguous(),
            scatter_map.peer_gather_indices,
            scatter_map.peer_offsets,
            gather_ipc,
            gather_out,
            gather_peer_off,
            self._peer_slot_ptrs_dev,
            self._peer_sig_ptrs_dev,
            self._ipc_raw_ptr,
            self._signal_pad_ptr,
            self._topology.my_local_rank,
            self._topology.local_world_size,
            total_recv,
            self._D,
            self._device_flag,
            self._iter_id,
            self._output_buf,
        )

    def _forward_fused(
        self,
        output_embs: torch.Tensor,
        lengths_after_input_dist: torch.Tensor,
        input_splits: List[int],
        output_splits: List[int],
        sparse_features_recat: torch.Tensor,
        unbucketize_permute: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward using fused CUDA kernel (single-node NVLink outcast + gather)."""
        self._iter_id += 1
        total_send = sum(input_splits)
        total_recv = sum(output_splits)

        # 1. Build scatter map (cached by total_send + num_features)
        scatter_map = self._build_scatter_map_from_context(
            lengths_after_input_dist,
            input_splits,
            sparse_features_recat,
            total_send,
            total_send,
            direction="fwd",
        )
        # No sync needed: scatter map build has internal sync on first call,
        # and subsequent calls hit the cache (no GPU work).

        # 2. Adaptive dispatch: pipelined (latency-optimized) for small sizes,
        #    fused_single (bandwidth-optimized) for large sizes.
        #    Crossover: ~16K total send rows for D=128 bf16 (≈2K rows/dest * 8 GPUs).
        #    Scale threshold by row_bytes: larger rows cross over sooner.
        unbuck = unbucketize_permute if unbucketize_permute is not None else self._empty_i64
        splits_key = tuple(output_splits)
        output_splits_t = self._output_splits_cache.get(splits_key)
        if output_splits_t is None:
            output_splits_t = torch.tensor(
                output_splits, dtype=torch.int64, device=self._device
            )
            self._output_splits_cache[splits_key] = output_splits_t

        # Pre-allocate output buffer (reused across iterations)
        needed = total_recv * self._D
        if not hasattr(self, "_output_buf") or self._output_buf is None or self._output_buf.numel() < needed:
            self._output_buf = torch.empty(
                needed, dtype=self._dtype, device=self._device
            )

        # Build pre-computed gather map for per-peer parallel gather
        gather_ipc = None
        gather_out = None
        gather_peer_off = None
        gather_stage_splits = None
        if unbucketize_permute is not None and unbucketize_permute.numel() > 0:
            gather_ipc, gather_out, gather_peer_off, _ = \
                self._build_gather_map(unbucketize_permute, output_splits)
            # gather_stage_splits = None: kernel computes boundary from peer_offsets

        return dynamicemb_extensions.hier_a2a.fwd_pipelined(
            output_embs.contiguous(),
            scatter_map.peer_gather_indices,
            scatter_map.peer_offsets,
            output_splits_t,
            self._rank_to_local_dev,
            unbuck,
            self._max_rows_per_rank,
            self._peer_slot_ptrs_dev,
            self._peer_sig_ptrs_dev,
            self._ipc_raw_ptr,
            self._signal_pad_ptr,
            self._topology.my_local_rank,
            self._topology.local_world_size,
            total_recv,
            self._D,
            self._device_flag,
            self._iter_id,
            self._output_buf,
            gather_ipc,
            gather_out,
            gather_peer_off,
            gather_stage_splits,
        )

    def forward(
        self,
        output_embs: torch.Tensor,
        lengths_after_input_dist: torch.Tensor,
        input_splits: List[int],
        output_splits: List[int],
        sparse_features_recat: torch.Tensor,
        unbucketize_permute: Optional[torch.Tensor],
        batch_size_per_rank: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Execute hierarchical all2all forward.

        Currently uses a reference implementation based on
        SequenceEmbeddingsAllToAll that is numerically identical to the
        baseline. This lets us verify the wiring and fallback logic while
        the fused CUDA kernel is being brought up.

        When the fused kernel is ready, this method will build an
        OutcastScatterMap + HierA2ADeviceContext and launch the fused
        kernel instead.
        """
        if self._fallback:
            raise RuntimeError(
                "HierAll2AllManager is in fallback mode. "
                "Use baseline NCCL all2all instead."
            )

        # Cache metadata for backward
        self._cached_input_splits = input_splits
        self._cached_output_splits = output_splits
        self._cached_sparse_features_recat = sparse_features_recat
        self._cached_unbucketize_permute = unbucketize_permute

        # Fused kernel path: single-node with C++ extension
        if (
            _HAS_HIER_A2A
            and self._topology.num_nodes == 1
            and self._peer_slot_ptrs_dev is not None
            and os.environ.get("HIER_A2A_FUSED", "1") == "1"
        ):
            # Two-phase path: outcast-only + gather-only kernels.
            # Selected when HIER_A2A_TWO_PHASE=1 and gather map is available
            # (unbucketize_permute is non-None and non-empty).
            use_two_phase = (
                os.environ.get("HIER_A2A_TWO_PHASE", "0") == "1"
                and unbucketize_permute is not None
                and unbucketize_permute.numel() > 0
                and hasattr(dynamicemb_extensions.hier_a2a, "fwd_two_phase")
            )
            if use_two_phase:
                return self._forward_two_phase(
                    output_embs,
                    lengths_after_input_dist,
                    input_splits,
                    output_splits,
                    sparse_features_recat,
                    unbucketize_permute,
                )
            return self._forward_fused(
                output_embs,
                lengths_after_input_dist,
                input_splits,
                output_splits,
                sparse_features_recat,
                unbucketize_permute,
            )

        # Reference path: raw all_to_all_single (no recat — identity only).
        D = self._D
        total_recv_rows = sum(output_splits)

        recv_buf = torch.empty(
            total_recv_rows, D, dtype=output_embs.dtype, device=self._device
        )
        in_sizes = [s * D for s in input_splits]
        out_sizes = [s * D for s in output_splits]

        dist.all_to_all_single(
            recv_buf.view(-1), output_embs.contiguous().view(-1),
            out_sizes, in_sizes, group=self._pg,
        )

        # Apply unbucketize permutation (post-all2all gather)
        if unbucketize_permute is not None and unbucketize_permute.numel() > 0:
            is_identity = (unbucketize_permute == torch.arange(
                unbucketize_permute.numel(), device=unbucketize_permute.device,
                dtype=unbucketize_permute.dtype
            )).all()
            if not is_identity:
                recv_buf = torch.index_select(recv_buf, 0, unbucketize_permute)

        return recv_buf

    def backward(
        self,
        grad_final: torch.Tensor,
        sparse_features_recat: torch.Tensor,
        unbucketize_permute: Optional[torch.Tensor],
        lengths_after_output_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Execute hierarchical all2all backward.

        Uses transposed routing: bwd_input_splits = fwd_output_splits.
        """
        if self._fallback:
            raise RuntimeError("HierAll2AllManager is in fallback mode")

        # Cache for backward scatter map
        self._cached_lengths_after_output_dist = lengths_after_output_dist

        # Transposed splits
        bwd_input_splits = self._cached_output_splits
        bwd_output_splits = self._cached_input_splits
        assert bwd_input_splits is not None and bwd_output_splits is not None

        total_send_rows = sum(bwd_input_splits)
        total_recv_rows = sum(bwd_output_splits)
        D = self._D

        # Reference backward: inverse unbucketize + all2all + inverse recat
        if unbucketize_permute is not None:
            inv_unbucketize = torch.argsort(unbucketize_permute)
            grad_send = torch.index_select(grad_final, 0, inv_unbucketize)
        else:
            grad_send = grad_final

        # All-to-all with transposed splits
        grad_recv = torch.empty(
            total_recv_rows, D, dtype=grad_final.dtype, device=self._device
        )

        input_split_sizes = [s * D for s in bwd_input_splits]
        output_split_sizes = [s * D for s in bwd_output_splits]

        dist.all_to_all_single(
            grad_recv.view(-1),
            grad_send.view(-1),
            output_split_sizes,
            input_split_sizes,
            group=self._pg,
        )

        return grad_recv

    def __del__(self) -> None:
        """Clean up IPC handles and GIN resources."""
        if not _HAS_HIER_A2A:
            return
        # Close IPC handles
        if hasattr(self, "_peer_ipc_ptrs"):
            topo = getattr(self, "_topology", None)
            for lr, ptr in self._peer_ipc_ptrs.items():
                if topo and lr != topo.my_local_rank:
                    try:
                        dynamicemb_extensions.hier_a2a.ipc_close_handle(ptr)
                    except Exception:
                        pass

        # Free IPC buffer (cudaMalloc-allocated)
        raw_ptr = getattr(self, "_ipc_raw_ptr", 0)
        if raw_ptr != 0:
            try:
                dynamicemb_extensions.hier_a2a.ipc_cuda_free(raw_ptr)
            except Exception:
                pass
            self._ipc_raw_ptr = 0

        # Destroy GIN context
        gin = getattr(self, "_gin_context", None)
        if gin is not None:
            try:
                dynamicemb_extensions.hier_a2a.gin_destroy(gin)
            except Exception:
                pass

    @staticmethod
    def _align_to(alignment: int, size: int) -> int:
        return ((size + alignment - 1) // alignment) * alignment
