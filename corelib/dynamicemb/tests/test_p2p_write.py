#!/usr/bin/env python3
"""Test basic P2P IPC write via kernel on this system."""
import os, sys, torch, torch.distributed as dist

sys.path.insert(0, '/home/scratch.aleliu_sw/recsys-examples/corelib/dynamicemb')
if "dynamicemb_extensions" in sys.modules:
    del sys.modules["dynamicemb_extensions"]
import dynamicemb_extensions
ha = dynamicemb_extensions.hier_a2a

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
W = dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.device(f'cuda:{rank}')

D = 4
max_rows = 16
rows_to_send = 4
elem = 2

if rank == 0:
    for i in range(W):
        for j in range(W):
            if i != j:
                print(f"  P2P {i}->{j}: {torch.cuda.can_device_access_peer(i, j)}", flush=True)

# Allocate buffer: just lr_slots (no signal for now)
lr_slots_bytes = max_rows * W * D * elem
total_bytes = lr_slots_bytes + 128  # 128 for signal pad

ipc_buf = torch.zeros(total_bytes, dtype=torch.uint8, device=device)

# Exchange handles
my_handle = ha.ipc_get_handle(ipc_buf.data_ptr())
all_handles = [None] * W
dist.all_gather_object(all_handles, my_handle)
if rank == 0:
    print("Handles exchanged", flush=True)

peer_ptrs = {}
for r in range(W):
    if r == rank:
        peer_ptrs[r] = ipc_buf.data_ptr()
    else:
        peer_ptrs[r] = ha.ipc_open_handle(all_handles[r])
if rank == 0:
    print("Handles opened", flush=True)

# Test 1: Simple write via torch and IPC pointer
# Rank 0 writes to rank 1's buffer using a simple CUDA kernel
if rank == 0 and W >= 2:
    # Write pattern: [1.0, 2.0, 3.0, 4.0] to rank 1's slot 0
    src_data = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, dtype=torch.bfloat16)
    peer1_slot0_ptr = peer_ptrs[1]  # base of rank 1's buffer, slot for lr=0
    peer1_slot0 = torch.tensor([], dtype=torch.bfloat16, device=device)

    # Can't easily create a tensor from a foreign pointer in Python.
    # Let's just test the C++ kernel path directly.
    print("Test 1: Simple memcpy to peer...", flush=True)

# Test 2: Use cudaMemcpyPeer as a sanity check
torch.cuda.synchronize()
dist.barrier()

if rank == 0:
    print("Test 2: Launch outcast kernel (write only, no gather)...", flush=True)

# Create minimal data: rank 0 sends 4 rows of D=4 to rank 1
# The outcast kernel writes to peer's slot
output_embs = torch.arange(rows_to_send * D, device=device, dtype=torch.bfloat16).view(rows_to_send, D)
output_embs = output_embs + rank * 100  # distinguish per rank

# Setup for outcast only: peer_offsets for 2 peers
total_send = rows_to_send * W
# Redefine: each rank sends rows_to_send to each peer
output_embs = torch.arange(total_send * D, device=device, dtype=torch.bfloat16).view(total_send, D)
output_embs = output_embs + rank * 1000

peer_gather_indices = torch.arange(total_send, dtype=torch.int64, device=device)
peer_offsets = torch.arange(0, total_send + 1, rows_to_send, dtype=torch.int64, device=device)

signal_pad_offset = lr_slots_bytes
slot_ptrs = []
sig_ptrs = []
for lr in range(W):
    peer_buf = peer_ptrs[lr]
    slot_ptrs.append(peer_buf + rank * max_rows * D * elem)
    if lr != rank:
        sig_ptrs.append(peer_buf + signal_pad_offset + rank * 4)
    else:
        sig_ptrs.append(0)

peer_slot_ptrs_dev = torch.tensor(slot_ptrs, dtype=torch.int64, device=device)
peer_sig_ptrs_dev = torch.tensor(sig_ptrs, dtype=torch.int64, device=device)

if rank == 0:
    print(f"slot_ptrs={slot_ptrs}", flush=True)
    print(f"sig_ptrs={sig_ptrs}", flush=True)
    print(f"output_embs shape={output_embs.shape}", flush=True)
    print(f"peer_offsets={peer_offsets.tolist()}", flush=True)

torch.cuda.synchronize()
dist.barrier()

# Launch ONLY the outcast kernel (no gather)
# This tests if IPC writes work
if rank == 0:
    print("Launching outcast kernel...", flush=True)

# We need to call the C++ function that launches just the outcast kernel.
# Unfortunately, the pybind only exposes two_kernel_single_node which includes gather.
# Let's test with a manual kernel call via the two_kernel path but with a timeout.
# Actually, let's test the complete two_kernel path with a watchdog.

signal_pad_ptr = ipc_buf.data_ptr() + signal_pad_offset
device_flag = torch.zeros(1, dtype=torch.int32, device=device)
gather_permute = torch.empty(total_send, dtype=torch.int64, device=device)
for s in range(W):
    start = s * rows_to_send
    slot_start = s * max_rows
    gather_permute[start:start+rows_to_send] = torch.arange(
        slot_start, slot_start+rows_to_send, dtype=torch.int64, device=device)

if rank == 0:
    print("Calling two_kernel_single_node...", flush=True)

result = ha.two_kernel_single_node(
    output_embs, peer_gather_indices, peer_offsets, gather_permute,
    peer_slot_ptrs_dev, peer_sig_ptrs_dev,
    ipc_buf.data_ptr(), signal_pad_ptr,
    rank, W, total_send, D, device_flag, 1)

torch.cuda.synchronize()
if rank == 0:
    print(f"SUCCESS! result shape={result.shape}", flush=True)
    print(f"result[:4]={result[:4].tolist()}", flush=True)

# Cleanup
for r in range(W):
    if r != rank:
        ha.ipc_close_handle(peer_ptrs[r])
dist.destroy_process_group()
