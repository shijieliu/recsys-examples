#!/usr/bin/env python3
"""Test: outcast kernel, sync, then custom gather kernel."""
import os, sys, torch, torch.distributed as dist

os.chdir('/home/scratch.aleliu_sw/recsys-examples/corelib/dynamicemb')
sys.path.insert(0, '.')
if "dynamicemb_extensions" in sys.modules:
    del sys.modules["dynamicemb_extensions"]
import dynamicemb_extensions
ha = dynamicemb_extensions.hier_a2a

from torch.utils.cpp_extension import load
test_gather_ext = load('test_gather_ext', ['tests/test_gather_ext.cu'],
                       extra_cuda_cflags=['-O3', '-gencode', 'arch=compute_90a,code=sm_90a'],
                       verbose=False)

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

total_send = rows_per_dest * W
output_embs = torch.randn(total_send, D, device=device, dtype=torch.bfloat16)
gather_idx = torch.arange(total_send, dtype=torch.int64, device=device)
offsets = torch.arange(0, total_send + 1, rows_per_dest, dtype=torch.int64, device=device)
total_recv = total_send
gather_permute = torch.empty(total_recv, dtype=torch.int64, device=device)
for s in range(W):
    start = s * rows_per_dest
    gather_permute[start:start+rows_per_dest] = torch.arange(
        s * max_rows, s * max_rows + rows_per_dest, dtype=torch.int64, device=device)

torch.cuda.synchronize()
dist.barrier()

# Step 1: outcast only
if rank == 0:
    print("Step 1: outcast_only(iter_id=1)...", flush=True)
ha.outcast_only(output_embs, gather_idx, offsets,
                peer_slot_ptrs_dev, peer_sig_ptrs_dev, rank, L, D, 1)
torch.cuda.synchronize()
dist.barrier()
if rank == 0:
    print("Step 1: done.", flush=True)

# Step 2: gather using test extension (signals are already set)
if rank == 0:
    print("Step 2: test_gather(iter_id=1)...", flush=True)
result = test_gather_ext.launch_test_gather(
    gather_permute, ipc_buf.data_ptr(), D,
    total_recv, L, rank, signal_pad_ptr, device_flag, 1)
torch.cuda.synchronize()
if rank == 0:
    print(f"Step 2: done! result shape={result.shape}", flush=True)

# Verify
expected = torch.empty(total_send, D, device=device, dtype=torch.bfloat16)
dist.all_to_all_single(expected.view(-1), output_embs.view(-1),
                        [rows_per_dest * D] * W, [rows_per_dest * D] * W)
diff = (result - expected).abs().max().item()
if rank == 0:
    print(f"Diff = {diff} -> {'PASS' if diff < 1e-3 else 'FAIL'}", flush=True)

# Step 3: now try the full two_kernel path
if rank == 0:
    print("\nStep 3: two_kernel(iter_id=2)...", flush=True)
device_flag.zero_()
result2 = ha.two_kernel_single_node(
    output_embs, gather_idx, offsets, gather_permute,
    peer_slot_ptrs_dev, peer_sig_ptrs_dev,
    ipc_buf.data_ptr(), signal_pad_ptr,
    rank, L, total_recv, D, device_flag, 2)
torch.cuda.synchronize()
if rank == 0:
    print(f"Step 3: done! shape={result2.shape}", flush=True)
diff2 = (result2 - expected).abs().max().item()
if rank == 0:
    print(f"Diff = {diff2} -> {'PASS' if diff2 < 1e-3 else 'FAIL'}", flush=True)

for r in range(W):
    if r != rank:
        ha.ipc_close_handle(peer_ptrs[r])
dist.destroy_process_group()
