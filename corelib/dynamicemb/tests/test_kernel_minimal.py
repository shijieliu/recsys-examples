#!/usr/bin/env python3
"""Minimal test for outcast+gather kernels, bypassing the full manager."""
import os, sys, torch, torch.distributed as dist

os.chdir('/home/scratch.aleliu_sw/recsys-examples/corelib/dynamicemb')
sys.path.insert(0, '.')

# Force local extension
if "dynamicemb_extensions" in sys.modules:
    del sys.modules["dynamicemb_extensions"]
import dynamicemb_extensions
ha = dynamicemb_extensions.hier_a2a

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
W = dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.device(f'cuda:{rank}')

D = 64
max_rows = 256
L = W
elem = 2  # bf16
rows_per_dest = 32

# 1. Allocate IPC buffer
lr_slots_bytes = max_rows * L * D * elem
signal_pad_offset = lr_slots_bytes
signal_pad_bytes = 128  # aligned
total_bytes = lr_slots_bytes + signal_pad_bytes

ipc_buf = torch.zeros(total_bytes, dtype=torch.uint8, device=device)
if rank == 0:
    print(f"Buffer allocated: {total_bytes} bytes, signal_pad_offset={signal_pad_offset}", flush=True)

# 2. Exchange IPC handles via all_gather_object
my_handle = ha.ipc_get_handle(ipc_buf.data_ptr())
all_handles = [None] * W
dist.all_gather_object(all_handles, my_handle)
if rank == 0:
    print("IPC handles exchanged", flush=True)

# 3. Open peer handles
peer_ptrs = {}
for r in range(W):
    if r == rank:
        peer_ptrs[r] = ipc_buf.data_ptr()
    else:
        peer_ptrs[r] = ha.ipc_open_handle(all_handles[r])
if rank == 0:
    print("IPC handles opened", flush=True)

# 4. Build pointer arrays
slot_ptrs = []
sig_ptrs = []
for lr in range(L):
    peer_buf = peer_ptrs[lr]
    slot_ptrs.append(peer_buf + rank * max_rows * D * elem)
    if lr != rank:
        sig_ptrs.append(peer_buf + signal_pad_offset + rank * 4)
    else:
        sig_ptrs.append(0)

peer_slot_ptrs_dev = torch.tensor(slot_ptrs, dtype=torch.int64, device=device)
peer_sig_ptrs_dev = torch.tensor(sig_ptrs, dtype=torch.int64, device=device)
signal_pad_ptr = ipc_buf.data_ptr() + signal_pad_offset
device_flag = torch.zeros(1, dtype=torch.int32, device=device)

if rank == 0:
    print(f"Pointer arrays built. slot_ptrs={slot_ptrs}, sig_ptrs={sig_ptrs}", flush=True)

# 5. Create test data
total_send = rows_per_dest * W
output_embs = torch.randn(total_send, D, device=device, dtype=torch.bfloat16)
# Each rank sends rows_per_dest rows to each peer
peer_gather_indices = torch.arange(total_send, dtype=torch.int64, device=device)
peer_offsets = torch.arange(0, total_send + 1, rows_per_dest, dtype=torch.int64, device=device)

# Build gather_permute
total_recv = total_send
gather_permute = torch.empty(total_recv, dtype=torch.int64, device=device)
for s in range(W):
    start = s * rows_per_dest
    slot_start = s * max_rows
    gather_permute[start:start+rows_per_dest] = torch.arange(
        slot_start, slot_start+rows_per_dest, dtype=torch.int64, device=device)

if rank == 0:
    print(f"Test data: total_send={total_send}, total_recv={total_recv}", flush=True)
    print(f"peer_offsets={peer_offsets.tolist()}", flush=True)
    print(f"gather_permute[:8]={gather_permute[:8].tolist()}", flush=True)

# 6. Synchronize before kernel launch
torch.cuda.synchronize()
dist.barrier()
if rank == 0:
    print("All ranks synchronized. Launching kernel...", flush=True)

# 7. Launch the two-kernel single-node path
result = ha.two_kernel_single_node(
    output_embs, peer_gather_indices, peer_offsets, gather_permute,
    peer_slot_ptrs_dev, peer_sig_ptrs_dev,
    ipc_buf.data_ptr(), signal_pad_ptr,
    rank, L, total_recv, D, device_flag,
    1)  # iter_id = 1

torch.cuda.synchronize()
if rank == 0:
    print(f"Kernel completed! result shape={result.shape}", flush=True)

# 8. Verify correctness
# Each rank sends its output_embs to all peers. Rank 0 should receive
# from rank 0 (self) rows [0:rows_per_dest] and from rank 1 rows [0:rows_per_dest]
expected = torch.empty(total_recv, D, device=device, dtype=torch.bfloat16)
# Gather reference: all_to_all_single
in_sizes = [rows_per_dest * D] * W
out_sizes = [rows_per_dest * D] * W
dist.all_to_all_single(expected.view(-1), output_embs.view(-1), out_sizes, in_sizes)

if rank == 0:
    # Compare
    diff = (result - expected).abs().max().item()
    print(f"Max diff vs all_to_all_single: {diff}", flush=True)
    if diff < 1e-3:
        print("PASS", flush=True)
    else:
        print("FAIL", flush=True)
        print(f"result[:4,:4]={result[:4,:4]}", flush=True)
        print(f"expected[:4,:4]={expected[:4,:4]}", flush=True)

# Cleanup
for r in range(W):
    if r != rank:
        ha.ipc_close_handle(peer_ptrs[r])
dist.destroy_process_group()
