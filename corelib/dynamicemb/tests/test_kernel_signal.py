#!/usr/bin/env python3
"""Test: kernel-based IPC signal write+read (no copies, just signal)."""
import os, sys, torch, torch.distributed as dist
from torch.utils.cpp_extension import load_inline

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

# Compile inline test kernels
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void write_signal_kernel(int32_t* remote_signal, int32_t value) {
    if (threadIdx.x == 0) {
        __threadfence_system();
        // st.release.sys
        asm volatile("st.release.sys.global.s32 [%0], %1;\n"
                     ::"l"(remote_signal), "r"(value) : "memory");
    }
}

__global__ void wait_signal_kernel(int32_t* my_signal, int32_t expected, int32_t* out) {
    if (threadIdx.x == 0) {
        int32_t val;
        int iters = 0;
        while (true) {
            asm volatile("ld.acquire.sys.global.s32 %0, [%1];\n"
                         : "=r"(val) : "l"(my_signal) : "memory");
            iters++;
            if (val >= expected) break;
            if (iters > 100000000) { val = -1; break; }  // timeout
        }
        out[0] = val;
        out[1] = iters;
    }
}

__global__ void simple_write_kernel(int32_t* ptr, int32_t value) {
    if (threadIdx.x == 0) {
        *ptr = value;
        __threadfence_system();
    }
}

void launch_write_signal(int64_t remote_signal_ptr, int32_t value) {
    int32_t* sp = reinterpret_cast<int32_t*>(remote_signal_ptr);
    write_signal_kernel<<<1, 32>>>(sp, value);
}

void launch_simple_write(int64_t ptr, int32_t value) {
    int32_t* sp = reinterpret_cast<int32_t*>(ptr);
    simple_write_kernel<<<1, 32>>>(sp, value);
}

torch::Tensor launch_wait_signal(int64_t signal_ptr, int32_t expected) {
    int32_t* sp = reinterpret_cast<int32_t*>(signal_ptr);
    auto out = torch::zeros({2}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    wait_signal_kernel<<<1, 32>>>(sp, expected, out.data_ptr<int32_t>());
    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
void launch_write_signal(int64_t remote_signal_ptr, int32_t value);
void launch_simple_write(int64_t ptr, int32_t value);
torch::Tensor launch_wait_signal(int64_t signal_ptr, int32_t expected);
"""

if rank == 0:
    print("Compiling test kernels...", flush=True)
test_ext = load_inline(
    name="test_signal",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["launch_write_signal", "launch_simple_write", "launch_wait_signal"],
    verbose=False,
)

# Allocate IPC buffer (just for signals)
buf_size = 1024
ipc_buf = torch.zeros(buf_size, dtype=torch.uint8, device=device)

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

dist.barrier()

# Test 1: simple kernel write to peer's IPC memory
if rank == 0:
    print("\n--- Test 1: simple_write_kernel to peer ---", flush=True)
if rank == 0:
    target = peer_ptrs[1]  # rank 1's buffer
    test_ext.launch_simple_write(target, 77)
    torch.cuda.synchronize()
    print("Rank 0: wrote 77 to rank 1's buffer[0]", flush=True)

dist.barrier()

if rank == 1:
    val = torch.tensor([0], dtype=torch.int32, device=device)
    import ctypes
    ctypes.CDLL("libcudart.so").cudaMemcpy(
        ctypes.c_void_p(val.data_ptr()),
        ctypes.c_void_p(ipc_buf.data_ptr()),
        ctypes.c_size_t(4), ctypes.c_int(3))
    torch.cuda.synchronize()
    v = val[0].item()
    print(f"Rank 1: read {v} (expected 77) -> {'PASS' if v==77 else 'FAIL'}", flush=True)

dist.barrier()

# Reset
ipc_buf.zero_()
torch.cuda.synchronize()
dist.barrier()

# Test 2: st.release.sys write to peer, then ld.acquire.sys read
if rank == 0:
    print("\n--- Test 2: st.release.sys + ld.acquire.sys ---", flush=True)

if rank == 0:
    # Write signal to rank 1
    target = peer_ptrs[1]
    test_ext.launch_write_signal(target, 42)
    torch.cuda.synchronize()
    print("Rank 0: signal 42 written to rank 1", flush=True)

dist.barrier()

if rank == 1:
    # Read signal with ld.acquire.sys
    out = test_ext.launch_wait_signal(ipc_buf.data_ptr(), 42)
    torch.cuda.synchronize()
    vals = out.cpu().tolist()
    print(f"Rank 1: wait result: val={vals[0]}, iters={vals[1]} -> {'PASS' if vals[0]==42 else 'FAIL'}", flush=True)

dist.barrier()

# Reset
ipc_buf.zero_()
torch.cuda.synchronize()
dist.barrier()

# Test 3: Concurrent - rank 0 signals rank 1, rank 1 spins
if rank == 0:
    print("\n--- Test 3: concurrent signal+wait ---", flush=True)

dist.barrier()

if rank == 1:
    # Launch wait kernel FIRST (will spin until rank 0 signals)
    out = test_ext.launch_wait_signal(ipc_buf.data_ptr(), 1)
    # Don't sync yet - let it run

if rank == 0:
    # Signal rank 1 after a brief delay
    target = peer_ptrs[1]
    test_ext.launch_write_signal(target, 1)
    torch.cuda.synchronize()

dist.barrier()

if rank == 1:
    torch.cuda.synchronize()
    vals = out.cpu().tolist()
    print(f"Rank 1: concurrent wait result: val={vals[0]}, iters={vals[1]} -> {'PASS' if vals[0]==1 else 'FAIL'}", flush=True)

dist.barrier()

# Cleanup
for r in range(W):
    if r != rank:
        ha.ipc_close_handle(peer_ptrs[r])
dist.destroy_process_group()
if rank == 0:
    print("\nAll tests done.", flush=True)
