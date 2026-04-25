#!/usr/bin/env python3
"""Test: outcast kernel only, then manually check signals."""
import os, sys, ctypes, torch, torch.distributed as dist

os.chdir('/home/scratch.aleliu_sw/recsys-examples/corelib/dynamicemb')
sys.path.insert(0, '.')
if "dynamicemb_extensions" in sys.modules:
    del sys.modules["dynamicemb_extensions"]
import dynamicemb_extensions
ha = dynamicemb_extensions.hier_a2a

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
W = dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.device(f'cuda:{rank}')
cuda_rt = ctypes.CDLL("libcudart.so")

D = 64; max_rows = 256; L = W; elem = 2; rows_per_dest = 32

# Allocate IPC buffer
lr_slots_bytes = max_rows * L * D * elem
signal_pad_offset = lr_slots_bytes
total_bytes = lr_slots_bytes + 128
ipc_buf = torch.zeros(total_bytes, dtype=torch.uint8, device=device)

# Exchange IPC handles
my_handle = ha.ipc_get_handle(ipc_buf.data_ptr())
all_handles = [None] * W
dist.all_gather_object(all_handles, my_handle)
peer_ptrs = {}
for r in range(W):
    peer_ptrs[r] = ipc_buf.data_ptr() if r == rank else ha.ipc_open_handle(all_handles[r])

slot_ptrs = [peer_ptrs[lr] + rank * max_rows * D * elem for lr in range(L)]
sig_ptrs = [peer_ptrs[lr] + signal_pad_offset + rank * 4 if lr != rank else 0 for lr in range(L)]

peer_slot_ptrs_dev = torch.tensor(slot_ptrs, dtype=torch.int64, device=device)
peer_sig_ptrs_dev = torch.tensor(sig_ptrs, dtype=torch.int64, device=device)
signal_pad_ptr = ipc_buf.data_ptr() + signal_pad_offset

# Data
total_send = rows_per_dest * W
output_embs = torch.randn(total_send, D, device=device, dtype=torch.bfloat16)
gather_idx = torch.arange(total_send, dtype=torch.int64, device=device)
offsets = torch.arange(0, total_send + 1, rows_per_dest, dtype=torch.int64, device=device)

torch.cuda.synchronize()
dist.barrier()

# Launch outcast kernel ONLY
if rank == 0:
    print("Launching outcast_only...", flush=True)

ha.outcast_only(
    output_embs, gather_idx, offsets,
    peer_slot_ptrs_dev, peer_sig_ptrs_dev,
    rank, L, D, 1)  # iter_id=1

torch.cuda.synchronize()
if rank == 0:
    print("Outcast kernel completed!", flush=True)

dist.barrier()

# Check signals
signal_view = torch.zeros(L, dtype=torch.int32, device=device)
cuda_rt.cudaMemcpy(
    ctypes.c_void_p(signal_view.data_ptr()),
    ctypes.c_void_p(signal_pad_ptr),
    ctypes.c_size_t(L * 4), ctypes.c_int(3))
torch.cuda.synchronize()
print(f"Rank {rank}: signals after outcast = {signal_view.cpu().tolist()}", flush=True)

# Check data was written to IPC buffer
# Rank 0's data should be in rank 1's buffer at slot 0 (rank 0's slot)
# Rank 1's data should be in rank 0's buffer at slot 1 (rank 1's slot)
if rank == 0:
    # Read from own IPC buffer, slot 1 (from rank 1)
    slot1_offset = 1 * max_rows * D * elem
    slot1_data = torch.zeros(rows_per_dest, D, dtype=torch.bfloat16, device=device)
    cuda_rt.cudaMemcpy(
        ctypes.c_void_p(slot1_data.data_ptr()),
        ctypes.c_void_p(ipc_buf.data_ptr() + slot1_offset),
        ctypes.c_size_t(rows_per_dest * D * elem), ctypes.c_int(3))
    torch.cuda.synchronize()
    print(f"Rank 0: received data[:2,:4] = {slot1_data[:2,:4].float()}", flush=True)

dist.barrier()

# Verify: all_to_all_single reference
expected = torch.empty(total_send, D, device=device, dtype=torch.bfloat16)
dist.all_to_all_single(
    expected.view(-1), output_embs.view(-1),
    [rows_per_dest * D] * W, [rows_per_dest * D] * W)

if rank == 0:
    # Check data from rank 1 (second half of expected)
    rank1_data_expected = expected[rows_per_dest:2*rows_per_dest]
    slot1_data_check = torch.zeros(rows_per_dest, D, dtype=torch.bfloat16, device=device)
    cuda_rt.cudaMemcpy(
        ctypes.c_void_p(slot1_data_check.data_ptr()),
        ctypes.c_void_p(ipc_buf.data_ptr() + 1 * max_rows * D * elem),
        ctypes.c_size_t(rows_per_dest * D * elem), ctypes.c_int(3))
    torch.cuda.synchronize()
    diff = (slot1_data_check - rank1_data_expected).abs().max().item()
    print(f"Rank 0: data from rank 1 diff = {diff} -> {'PASS' if diff < 1e-3 else 'FAIL'}", flush=True)

dist.barrier()
for r in range(W):
    if r != rank:
        ha.ipc_close_handle(peer_ptrs[r])
dist.destroy_process_group()
if rank == 0:
    print("Done.", flush=True)
