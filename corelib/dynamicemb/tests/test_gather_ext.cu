#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int kVecBytes = 16;
constexpr int kTPC = 256; // threads per CTA

__device__ __forceinline__ void copy_row(
    const __nv_bfloat16* src, __nv_bfloat16* dst, int64_t D) {
    const int lane = threadIdx.x % 32;
    const int row_bytes = D * sizeof(__nv_bfloat16);
    const char* sb = reinterpret_cast<const char*>(src);
    char* db = reinterpret_cast<char*>(dst);
    for (int off = lane * kVecBytes; off < row_bytes; off += 32 * kVecBytes) {
        if (off + kVecBytes <= row_bytes) {
            float4 v = *reinterpret_cast<const float4*>(sb + off);
            *reinterpret_cast<float4*>(db + off) = v;
        }
    }
}

__global__ void test_gather(
    const int64_t* __restrict__ gather_perm,
    const __nv_bfloat16* __restrict__ ipc_buf,
    __nv_bfloat16* __restrict__ output,
    int64_t D, int64_t total_out,
    int lws, int my_lr,
    int32_t* __restrict__ signals,
    int32_t* __restrict__ flag,
    int32_t iter_id) {
    constexpr int kW = kTPC / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Wait for signals
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("gather: my_lr=%d, lws=%d, iter_id=%d, signals_ptr=%p\n",
               my_lr, lws, iter_id, signals);
        for (int s = 0; s < lws; ++s) {
            if (s == my_lr) continue;
            // First just read with volatile pointer
            int32_t peek = *reinterpret_cast<volatile int32_t*>(&signals[s]);
            printf("gather: checking signal[%d] at %p, peek=%d (need >=%d)\n",
                   s, &signals[s], peek, iter_id);
            if (peek >= iter_id) continue;
            int iters = 0;
            while (true) {
                int32_t val;
                asm volatile("ld.acquire.sys.global.s32 %0, [%1];\n"
                             : "=r"(val) : "l"(&signals[s]) : "memory");
                if (val >= iter_id) {
                    printf("gather: signal[%d] = %d after %d iters\n", s, val, iters);
                    break;
                }
                iters++;
                if (iters > 100000000) {
                    printf("gather: TIMEOUT waiting for signal[%d], last val=%d\n", s, val);
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
    const int gw = (int)blockIdx.x * kW + warp_id;
    const int tw = (int)gridDim.x * kW;
    for (int64_t i = gw; i < total_out; i += tw) {
        int64_t pos = gather_perm[i];
        copy_row(ipc_buf + pos * D, output + i * D, D);
    }
}

torch::Tensor launch_test_gather(
    torch::Tensor gather_perm, int64_t ipc_buf_ptr, int64_t D,
    int64_t total_out, int lws, int my_lr,
    int64_t sig_ptr, torch::Tensor dflag, int32_t iter_id) {

    auto stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::empty({total_out, D},
        torch::dtype(torch::kBFloat16).device(torch::kCUDA));

    test_gather<<<lws, kTPC, 0, stream>>>(
        gather_perm.data_ptr<int64_t>(),
        reinterpret_cast<const __nv_bfloat16*>(ipc_buf_ptr),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        D, total_out, lws, my_lr,
        reinterpret_cast<int32_t*>(sig_ptr),
        dflag.data_ptr<int32_t>(), iter_id);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_test_gather", &launch_test_gather);
}
