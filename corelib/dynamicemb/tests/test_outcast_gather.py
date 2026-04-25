#!/usr/bin/env python3
"""Test: minimal outcast+gather kernels compiled inline."""
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

cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int kVecBytes = 16;
constexpr int kThreadsPerCTA = 256;

__device__ __forceinline__ void copy_row_bf16(
    const __nv_bfloat16* src, __nv_bfloat16* dst, int64_t D) {
    const int lane = threadIdx.x % 32;
    const int row_bytes = D * sizeof(__nv_bfloat16);
    const char* src_b = reinterpret_cast<const char*>(src);
    char* dst_b = reinterpret_cast<char*>(dst);
    for (int off = lane * kVecBytes; off < row_bytes; off += 32 * kVecBytes) {
        if (off + kVecBytes <= row_bytes) {
            float4 v = *reinterpret_cast<const float4*>(src_b + off);
            *reinterpret_cast<float4*>(dst_b + off) = v;
        }
    }
}

__global__ void mini_outcast(
    const __nv_bfloat16* input,
    const int64_t* gather_idx,
    const int64_t* offsets,
    const uintptr_t* slot_ptrs,
    const uintptr_t* sig_ptrs,
    int64_t D, int32_t iter_id) {
    constexpr int kWarps = kThreadsPerCTA / 32;
    const int warp_id = threadIdx.x / 32;
    const int dest_lr = (int)blockIdx.x;
    const int64_t start = offsets[dest_lr];
    const int64_t nrows = offsets[dest_lr + 1] - start;
    auto* remote = reinterpret_cast<__nv_bfloat16*>(slot_ptrs[dest_lr]);

    for (int64_t r = warp_id; r < nrows; r += kWarps) {
        int64_t gi = gather_idx[start + r];
        copy_row_bf16(input + gi * D, remote + r * D, D);
    }

    __syncthreads();
    __threadfence_system();

    if (threadIdx.x == 0) {
        uintptr_t sig = sig_ptrs[dest_lr];
        if (sig != 0) {
            int32_t* sp = reinterpret_cast<int32_t*>(sig);
            asm volatile("st.release.sys.global.s32 [%0], %1;\n"
                         ::"l"(sp), "r"(iter_id) : "memory");
        }
    }
}

__global__ void mini_gather(
    const int64_t* gather_perm,
    const __nv_bfloat16* ipc_buf,
    __nv_bfloat16* output,
    int64_t D, int64_t total_out,
    int local_world_size, int my_lr,
    int32_t* signals, int32_t* flag, int32_t iter_id) {
    constexpr int kWarps = kThreadsPerCTA / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Wait for signals
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int s = 0; s < local_world_size; ++s) {
            if (s == my_lr) continue;
            int iters = 0;
            while (true) {
                int32_t val;
                asm volatile("ld.acquire.sys.global.s32 %0, [%1];\n"
                             : "=r"(val) : "l"(&signals[s]) : "memory");
                if (val >= iter_id) break;
                iters++;
                if (iters > 500000000) {
                    // timeout - write error marker and exit
                    output[0] = __float2bfloat16(-999.0f);
                    return;
                }
            }
        }
        __threadfence();
        atomicExch(flag, iter_id);
    }

    if (!(blockIdx.x == 0 && threadIdx.x == 0)) {
        if (lane_id == 0) {
            while (true) {
                int32_t val;
                asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n"
                             : "=r"(val) : "l"(flag) : "memory");
                if (val >= iter_id) break;
            }
        }
        __syncwarp();
    }
    __syncthreads();

    // Gather
    const int gwarp = (int)blockIdx.x * kWarps + warp_id;
    const int total_warps = (int)gridDim.x * kWarps;
    for (int64_t i = gwarp; i < total_out; i += total_warps) {
        int64_t pos = gather_perm[i];
        copy_row_bf16(ipc_buf + pos * D, output + i * D, D);
    }
}

torch::Tensor launch_mini(
    torch::Tensor input, torch::Tensor gather_idx, torch::Tensor offsets,
    torch::Tensor slot_ptrs_dev, torch::Tensor sig_ptrs_dev,
    torch::Tensor gather_perm, int64_t ipc_buf_ptr, int64_t sig_pad_ptr,
    int my_lr, int lws, int64_t total_out, int64_t D, torch::Tensor dflag,
    int32_t iter_id) {

    auto stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::empty({total_out, D}, input.options());

    // Launch outcast
    mini_outcast<<<lws, kThreadsPerCTA, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        gather_idx.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(),
        reinterpret_cast<const uintptr_t*>(slot_ptrs_dev.data_ptr<int64_t>()),
        reinterpret_cast<const uintptr_t*>(sig_ptrs_dev.data_ptr<int64_t>()),
        D, iter_id);

    // Launch gather
    mini_gather<<<lws, kThreadsPerCTA, 0, stream>>>(
        gather_perm.data_ptr<int64_t>(),
        reinterpret_cast<const __nv_bfloat16*>(ipc_buf_ptr),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        D, total_out, lws, my_lr,
        reinterpret_cast<int32_t*>(sig_pad_ptr),
        dflag.data_ptr<int32_t>(), iter_id);

    return out;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor launch_mini(
    torch::Tensor input, torch::Tensor gather_idx, torch::Tensor offsets,
    torch::Tensor slot_ptrs_dev, torch::Tensor sig_ptrs_dev,
    torch::Tensor gather_perm, int64_t ipc_buf_ptr, int64_t sig_pad_ptr,
    int my_lr, int lws, int64_t total_out, int64_t D, torch::Tensor dflag,
    int32_t iter_id);
"""

if rank == 0:
    print("Compiling mini kernels...", flush=True)
mini = load_inline("mini_a2a", cpp_src, cuda_src, ["launch_mini"], verbose=False)

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
device_flag = torch.zeros(1, dtype=torch.int32, device=device)

total_send = rows_per_dest * W
output_embs = torch.randn(total_send, D, device=device, dtype=torch.bfloat16)
gather_idx = torch.arange(total_send, dtype=torch.int64, device=device)
offsets = torch.arange(0, total_send + 1, rows_per_dest, dtype=torch.int64, device=device)
total_recv = total_send
gather_perm = torch.empty(total_recv, dtype=torch.int64, device=device)
for s in range(W):
    start = s * rows_per_dest
    gather_perm[start:start+rows_per_dest] = torch.arange(
        s * max_rows, s * max_rows + rows_per_dest, dtype=torch.int64, device=device)

torch.cuda.synchronize()
dist.barrier()
if rank == 0:
    print("Launching mini outcast+gather...", flush=True)

result = mini.launch_mini(
    output_embs, gather_idx, offsets,
    peer_slot_ptrs_dev, peer_sig_ptrs_dev,
    gather_perm, ipc_buf.data_ptr(), signal_pad_ptr,
    rank, L, total_recv, D, device_flag, 1)
torch.cuda.synchronize()
if rank == 0:
    print(f"Kernel done! result shape={result.shape}", flush=True)

# Verify
expected = torch.empty(total_recv, D, device=device, dtype=torch.bfloat16)
dist.all_to_all_single(expected.view(-1), output_embs.view(-1),
                        [rows_per_dest*D]*W, [rows_per_dest*D]*W)
diff = (result - expected).abs().max().item()
if rank == 0:
    print(f"Max diff: {diff} -> {'PASS' if diff < 1e-3 else 'FAIL'}", flush=True)

for r in range(W):
    if r != rank:
        ha.ipc_close_handle(peer_ptrs[r])
dist.destroy_process_group()
