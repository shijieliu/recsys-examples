#!/usr/bin/env python3
"""Minimal test: write a value via IPC and read it back to verify P2P works."""
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

# 1. Check P2P access
for i in range(W):
    for j in range(W):
        if i != j:
            can = torch.cuda.can_device_access_peer(i, j)
            if rank == 0:
                print(f"  P2P {i}->{j}: {can}", flush=True)

# 2. Simple IPC test - allocate buffer, exchange, write, verify
buf = torch.zeros(1024, dtype=torch.int32, device=device)
my_handle = ha.ipc_get_handle(buf.data_ptr())
all_handles = [None] * W
dist.all_gather_object(all_handles, my_handle)

# Open peer handles
peer_ptrs = {}
for r in range(W):
    if r == rank:
        peer_ptrs[r] = buf.data_ptr()
    else:
        peer_ptrs[r] = ha.ipc_open_handle(all_handles[r])

if rank == 0:
    print(f"Rank {rank}: my buf={buf.data_ptr()}, peer_ptrs={peer_ptrs}", flush=True)

dist.barrier()

# 3. Write to peer's buffer via CUDA kernel
# Simple: rank 0 writes value 42 to rank 1's buffer[0]
if rank == 0 and W >= 2:
    # Use a simple copy via torch
    peer_tensor = torch.tensor([42], dtype=torch.int32, device=device)
    # We can't directly memcpy via torch to IPC ptr, so use CUDA
    import ctypes
    # Actually, let's use cudaMemcpy
    cuda_rt = ctypes.CDLL("libcudart.so")
    peer_ptr = peer_ptrs[1]
    src_ptr = peer_tensor.data_ptr()
    ret = cuda_rt.cudaMemcpy(ctypes.c_void_p(peer_ptr), ctypes.c_void_p(src_ptr),
                              ctypes.c_size_t(4), ctypes.c_int(1))  # cudaMemcpyDeviceToDevice
    print(f"cudaMemcpy to peer returned: {ret}", flush=True)
    torch.cuda.synchronize()

dist.barrier()

# 4. Rank 1 checks its buffer
if rank == 1:
    val = buf[0].item()
    print(f"Rank 1: buf[0] = {val} (expected 42)", flush=True)
    if val == 42:
        print("P2P IPC WRITE: PASS", flush=True)
    else:
        print("P2P IPC WRITE: FAIL", flush=True)

dist.barrier()

# Cleanup
for r in range(W):
    if r != rank:
        ha.ipc_close_handle(peer_ptrs[r])
dist.destroy_process_group()
if rank == 0:
    print("Done.", flush=True)
