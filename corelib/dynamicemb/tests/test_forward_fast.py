#!/usr/bin/env python3
"""Test: forward_fast C++ path directly."""
import os, sys, torch, torch.distributed as dist

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

D = 64; max_rows = 256; L = W; elem = 2; rows_per_dest = 32

lr_slots_bytes = max_rows * L * D * elem
signal_pad_offset = lr_slots_bytes
total_bytes = lr_slots_bytes + 128
ipc_buf = torch.zeros(total_bytes, dtype=torch.uint8, device=device)

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
device_flag = torch.zeros(1, dtype=torch.int32, device=device)

# Data
total_send = rows_per_dest * W
output_embs = torch.randn(total_send, D, device=device, dtype=torch.bfloat16)
gather_idx = torch.arange(total_send, dtype=torch.int64, device=device)
offsets = torch.arange(0, total_send + 1, rows_per_dest, dtype=torch.int64, device=device)

# Forward_fast needs: output_splits, rank_to_local, unbucketize_permute
output_splits = torch.tensor([rows_per_dest] * W, dtype=torch.int64, device=device)
rank_to_local = torch.tensor(list(range(W)), dtype=torch.int32, device=device)
unbucketize_permute = torch.arange(total_send, dtype=torch.int64, device=device)
total_recv = total_send

torch.cuda.synchronize()
dist.barrier()

if rank == 0:
    print(f"Testing forward_fast with W={W}, D={D}...", flush=True)

result = ha.forward_fast(
    output_embs, gather_idx, offsets,
    output_splits, rank_to_local, unbucketize_permute,
    max_rows,
    peer_slot_ptrs_dev, peer_sig_ptrs_dev,
    ipc_buf.data_ptr(), signal_pad_ptr,
    rank, L, total_recv, D, device_flag, 1)
torch.cuda.synchronize()

if rank == 0:
    print(f"forward_fast done! shape={result.shape}", flush=True)

# Verify
expected = torch.empty(total_send, D, device=device, dtype=torch.bfloat16)
dist.all_to_all_single(expected.view(-1), output_embs.view(-1),
                        [rows_per_dest * D] * W, [rows_per_dest * D] * W)
diff = (result - expected).abs().max().item()
if rank == 0:
    print(f"Diff = {diff} -> {'PASS' if diff < 1e-3 else 'FAIL'}", flush=True)

# Test 2: multiple iterations
for i in range(5):
    result = ha.forward_fast(
        output_embs, gather_idx, offsets,
        output_splits, rank_to_local, unbucketize_permute,
        max_rows,
        peer_slot_ptrs_dev, peer_sig_ptrs_dev,
        ipc_buf.data_ptr(), signal_pad_ptr,
        rank, L, total_recv, D, device_flag, i + 2)
torch.cuda.synchronize()
diff2 = (result - expected).abs().max().item()
if rank == 0:
    print(f"5 iterations done. Diff = {diff2} -> {'PASS' if diff2 < 1e-3 else 'FAIL'}", flush=True)

for r in range(W):
    if r != rank:
        ha.ipc_close_handle(peer_ptrs[r])
dist.destroy_process_group()
