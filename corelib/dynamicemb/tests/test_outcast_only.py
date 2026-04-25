#!/usr/bin/env python3
"""Test: launch outcast kernel only, check signals arrived."""
import os, sys, torch, torch.distributed as dist, ctypes

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

D = 64
max_rows = 256
L = W
elem = 2  # bf16
rows_per_dest = 32

# Allocate IPC buffer
lr_slots_bytes = max_rows * L * D * elem
signal_pad_offset = lr_slots_bytes
signal_pad_bytes = 128
total_bytes = lr_slots_bytes + signal_pad_bytes

ipc_buf = torch.zeros(total_bytes, dtype=torch.uint8, device=device)

# Exchange IPC handles
my_handle = ha.ipc_get_handle(ipc_buf.data_ptr())
all_handles = [None] * W
dist.all_gather_object(all_handles, my_handle)

peer_ptrs = {}
for r in range(W):
    if r == rank:
        peer_ptrs[r] = ipc_buf.data_ptr()
    else:
        peer_ptrs[r] = ha.ipc_open_handle(all_handles[r])

# Build pointer arrays
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

if rank == 0:
    print(f"Signal pad ptr: {signal_pad_ptr}", flush=True)
    print(f"sig_ptrs: {sig_ptrs}", flush=True)

# Create test data
total_send = rows_per_dest * W
output_embs = torch.randn(total_send, D, device=device, dtype=torch.bfloat16)
peer_gather_indices = torch.arange(total_send, dtype=torch.int64, device=device)
peer_offsets = torch.arange(0, total_send + 1, rows_per_dest, dtype=torch.int64, device=device)

# Read signal pad before
signal_view = torch.tensor(
    [0] * L, dtype=torch.int32, device=device
)
cuda_rt = ctypes.CDLL("libcudart.so")
cuda_rt.cudaMemcpy(
    ctypes.c_void_p(signal_view.data_ptr()),
    ctypes.c_void_p(signal_pad_ptr),
    ctypes.c_size_t(L * 4),
    ctypes.c_int(3)  # cudaMemcpyDefault
)
torch.cuda.synchronize()
print(f"Rank {rank}: signals before outcast: {signal_view.cpu().tolist()}", flush=True)

dist.barrier()
torch.cuda.synchronize()

# Launch JUST the outcast kernel via two_kernel but with a modified approach
# Actually, let's use the fused_single_node path which is a cooperative kernel
# Or better: manually launch outcast kernel only

# We don't have a Python binding for outcast-only, so let's use two_kernel
# and just check if it hangs. But first, let's verify the signal addresses by
# doing a manual P2P write to the signal location.

# Manual signal test: rank 0 writes value 99 to rank 1's signal pad at rank 0's slot
if rank == 0 and W >= 2:
    val_tensor = torch.tensor([99], dtype=torch.int32, device=device)
    target = sig_ptrs[1]  # rank 0's signal entry in rank 1's signal pad
    print(f"Rank 0: writing 99 to rank 1's signal at {target}", flush=True)
    cuda_rt.cudaMemcpy(
        ctypes.c_void_p(target), ctypes.c_void_p(val_tensor.data_ptr()),
        ctypes.c_size_t(4), ctypes.c_int(1)  # DeviceToDevice
    )
    torch.cuda.synchronize()
    print(f"Rank 0: write done", flush=True)

dist.barrier()
torch.cuda.synchronize()

# Rank 1 reads its signal pad
cuda_rt.cudaMemcpy(
    ctypes.c_void_p(signal_view.data_ptr()),
    ctypes.c_void_p(signal_pad_ptr),
    ctypes.c_size_t(L * 4),
    ctypes.c_int(3)
)
torch.cuda.synchronize()
print(f"Rank {rank}: signals after manual write: {signal_view.cpu().tolist()}", flush=True)

if rank == 1:
    # signal_view[0] should be 99 (written by rank 0)
    val = signal_view[0].item()
    if val == 99:
        print("SIGNAL P2P WRITE: PASS", flush=True)
    else:
        print(f"SIGNAL P2P WRITE: FAIL (got {val})", flush=True)

dist.barrier()

# Reset signals for kernel test
zero_tensor = torch.zeros(L, dtype=torch.int32, device=device)
cuda_rt.cudaMemcpy(
    ctypes.c_void_p(signal_pad_ptr),
    ctypes.c_void_p(zero_tensor.data_ptr()),
    ctypes.c_size_t(L * 4),
    ctypes.c_int(3)
)
torch.cuda.synchronize()
dist.barrier()

# Now try two_kernel
print(f"Rank {rank}: launching two_kernel...", flush=True)
total_recv = total_send
gather_permute = torch.empty(total_recv, dtype=torch.int64, device=device)
for s in range(W):
    start = s * rows_per_dest
    slot_start = s * max_rows
    gather_permute[start:start+rows_per_dest] = torch.arange(
        slot_start, slot_start+rows_per_dest, dtype=torch.int64, device=device)
device_flag = torch.zeros(1, dtype=torch.int32, device=device)

result = ha.two_kernel_single_node(
    output_embs, peer_gather_indices, peer_offsets, gather_permute,
    peer_slot_ptrs_dev, peer_sig_ptrs_dev,
    ipc_buf.data_ptr(), signal_pad_ptr,
    rank, L, total_recv, D, device_flag, 1)
torch.cuda.synchronize()
print(f"Rank {rank}: kernel completed!", flush=True)

# Cleanup
for r in range(W):
    if r != rank:
        ha.ipc_close_handle(peer_ptrs[r])
dist.destroy_process_group()
