#!/usr/bin/env python3
"""Test: use manager's IPC setup with simple uniform data."""
import os, sys, torch, torch.distributed as dist

os.chdir('/home/scratch.aleliu_sw/recsys-examples/corelib/dynamicemb')
sys.path.insert(0, '.')
if "dynamicemb_extensions" in sys.modules:
    del sys.modules["dynamicemb_extensions"]
import dynamicemb_extensions
ha = dynamicemb_extensions.hier_a2a
from dynamicemb.hier_all2all import HierAll2AllManager

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
W = dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.device(f'cuda:{rank}')

D = 64; F = 2; max_seq = 10; dtype = torch.bfloat16
max_rows = max_seq * F * W
rows_per_dest = 10  # simple uniform

# Create manager (sets up IPC)
manager = HierAll2AllManager(pg=dist.group.WORLD, num_features=F,
                             max_rows_per_rank=max_rows, D=D, device=device, dtype=dtype)
print(f"[{rank}] manager created, fallback={manager.fallback}", flush=True)

# Simple data (like test_kernel_minimal)
total_send = rows_per_dest * W
output_embs = torch.randn(total_send, D, device=device, dtype=dtype)
gather_idx = torch.arange(total_send, dtype=torch.int64, device=device)
offsets = torch.arange(0, total_send + 1, rows_per_dest, dtype=torch.int64, device=device)
total_recv = total_send
gather_permute = torch.empty(total_recv, dtype=torch.int64, device=device)
for s in range(W):
    start = s * rows_per_dest
    gather_permute[start:start+rows_per_dest] = torch.arange(
        s * max_rows, s * max_rows + rows_per_dest, dtype=torch.int64, device=device)
device_flag = torch.zeros(1, dtype=torch.int32, device=device)

torch.cuda.synchronize()
dist.barrier()
print(f"[{rank}] launching two_kernel with manager IPC...", flush=True)

result = ha.two_kernel_single_node(
    output_embs, gather_idx, offsets, gather_permute,
    manager._peer_slot_ptrs_dev, manager._peer_sig_ptrs_dev,
    manager._ipc_recv_buf.data_ptr(), manager._signal_pad_ptr,
    rank, W, total_recv, D, device_flag, 1)
torch.cuda.synchronize()
print(f"[{rank}] kernel done! shape={result.shape}", flush=True)

# Verify
expected = torch.empty(total_recv, D, device=device, dtype=dtype)
dist.all_to_all_single(expected.view(-1), output_embs.view(-1),
                        [rows_per_dest*D]*W, [rows_per_dest*D]*W)
diff = (result - expected).abs().max().item()
print(f"[{rank}] diff = {diff} -> {'PASS' if diff < 1e-3 else 'FAIL'}", flush=True)

# --- Test 2: scatter-map-built data ---
print(f"[{rank}] --- Test 2: scatter map built data ---", flush=True)
torch.manual_seed(42 + rank)
lengths = torch.randint(1, max_seq + 1, (F * W,), device=device)
input_splits = [sum(lengths[r*F+f].item() for f in range(F)) for r in range(W)]
total_send2 = sum(input_splits)
output_embs2 = torch.randn(total_send2, D, device=device, dtype=dtype)
sfrecat = torch.arange(F * W, dtype=torch.int32, device=device)
input_splits_t = torch.tensor(input_splits, dtype=torch.int64, device=device)
output_splits_t = torch.empty_like(input_splits_t)
dist.all_to_all_single(output_splits_t, input_splits_t,
                        output_split_sizes=[1]*W, input_split_sizes=[1]*W)
output_splits = output_splits_t.tolist()
total_recv2 = sum(output_splits)

scatter_map = manager._build_scatter_map_from_context(
    lengths, input_splits, sfrecat, total_send2, total_send2, "fwd", None)
torch.cuda.synchronize()
gi = scatter_map.peer_gather_indices.cpu()
off = scatter_map.peer_offsets.cpu()
print(f"[{rank}] scatter_map: offsets={off.tolist()}, "
      f"gi_range=[{gi.min().item()}, {gi.max().item()}], total_send={total_send2}", flush=True)

# Build simple gather_permute (same logic as test 1)
gather_permute2 = torch.empty(total_recv2, dtype=torch.int64, device=device)
offset = 0
for s in range(W):
    count = output_splits[s]
    lr = s
    slot_start = lr * max_rows
    if count > 0:
        gather_permute2[offset:offset+count] = torch.arange(
            slot_start, slot_start + count, dtype=torch.int64, device=device)
        offset += count

device_flag2 = torch.zeros(1, dtype=torch.int32, device=device)
torch.cuda.synchronize()
dist.barrier()
print(f"[{rank}] launching two_kernel with scatter-map data...", flush=True)

result2 = ha.two_kernel_single_node(
    output_embs2, scatter_map.peer_gather_indices, scatter_map.peer_offsets,
    gather_permute2, manager._peer_slot_ptrs_dev, manager._peer_sig_ptrs_dev,
    manager._ipc_recv_buf.data_ptr(), manager._signal_pad_ptr,
    rank, W, total_recv2, D, device_flag2, 2)
torch.cuda.synchronize()
print(f"[{rank}] kernel done!", flush=True)

# Verify
expected2 = torch.empty(total_recv2, D, device=device, dtype=dtype)
dist.all_to_all_single(expected2.view(-1), output_embs2.view(-1),
                        [s*D for s in output_splits], [s*D for s in input_splits])
diff2 = (result2 - expected2).abs().max().item()
print(f"[{rank}] diff2 = {diff2} -> {'PASS' if diff2 < 1e-2 else 'FAIL'}", flush=True)

# --- Test 3: forward_fast C++ call ---
print(f"[{rank}] --- Test 3: forward_fast ---", flush=True)
output_splits_dev = torch.tensor(output_splits, dtype=torch.int64, device=device)
rank_to_local_dev = torch.tensor(list(range(W)), dtype=torch.int32, device=device)
empty_unbuck = torch.empty(0, dtype=torch.int64, device=device)
device_flag3 = torch.zeros(1, dtype=torch.int32, device=device)
torch.cuda.synchronize()
dist.barrier()
print(f"[{rank}] calling forward_fast...", flush=True)
result3 = ha.forward_fast(
    output_embs2, scatter_map.peer_gather_indices, scatter_map.peer_offsets,
    output_splits_dev, rank_to_local_dev, empty_unbuck, max_rows,
    manager._peer_slot_ptrs_dev, manager._peer_sig_ptrs_dev,
    manager._ipc_recv_buf.data_ptr(), manager._signal_pad_ptr,
    rank, W, total_recv2, D, device_flag3, 3)
torch.cuda.synchronize()
diff3 = (result3 - expected2).abs().max().item()
print(f"[{rank}] forward_fast diff = {diff3} -> {'PASS' if diff3 < 1e-2 else 'FAIL'}", flush=True)

dist.destroy_process_group()
