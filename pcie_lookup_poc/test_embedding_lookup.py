import torch
from pcie_lookup_poc import embedding_lookup_cuda
# Set random seeds for reproducibility
import random
import numpy as np

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Test parameters
num_embeddings = 10000000
embedding_dim = 512
batch_size = 65536
num_iterations = 100
warmup_iterations = 10

indices_dtype = torch.int64
embedding_table_dtype = torch.float
output_dtype = torch.float
# Create test tensors
indices_list = [
    torch.randperm(num_embeddings, device="cuda", dtype=indices_dtype)[:batch_size] for _ in range(num_iterations)
]
embedding_table = torch.randn(num_embeddings, embedding_dim, dtype=embedding_table_dtype, device="cuda")
host_embedding_table = embedding_table.cpu().pin_memory()
output = torch.empty(batch_size, embedding_dim, dtype=output_dtype, device="cuda")

# Get device properties
device_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
num_sms = device_properties.multi_processor_count
max_threads_per_sm = device_properties.max_threads_per_multi_processor
def test_embedding_lookup():
    for indices in indices_list:
        # Run embedding lookup
        embedding_lookup_cuda(indices, embedding_table, output, num_sms, max_threads_per_sm)

        # Verify results against PyTorch's embedding
        expected = torch.nn.functional.embedding(indices, embedding_table).to(output_dtype)
        torch.testing.assert_close(output, expected)
    print("Test passed")

def benchmark_embedding_lookup():
    # Warmup
    for _ in range(warmup_iterations):
        embedding_lookup_cuda(indices_list[0], embedding_table, output, num_sms, max_threads_per_sm)
        torch.nn.functional.embedding(indices_list[0], embedding_table)

    # Benchmark custom implementation
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iterations):
        if _ == 0:
            with torch.cuda.nvtx.range("custom_embedding_lookup"):
                embedding_lookup_cuda(indices_list[_], embedding_table, output, num_sms, max_threads_per_sm)
        else:
            embedding_lookup_cuda(indices_list[_], embedding_table, output, num_sms, max_threads_per_sm)
    end.record()
    torch.cuda.synchronize()
    custom_time = start.elapsed_time(end) / num_iterations  

    start.record()
    for _ in range(num_iterations):
        if _ == 0:
            with torch.cuda.nvtx.range("custom_embedding_lookup_on_host_table"):
                embedding_lookup_cuda(indices_list[_], host_embedding_table, output, num_sms, max_threads_per_sm)
        else:
            embedding_lookup_cuda(indices_list[_], host_embedding_table, output, num_sms, max_threads_per_sm)
    end.record()
    torch.cuda.synchronize()
    custom_time_for_host_table = start.elapsed_time(end) / num_iterations  

    start.record()
    for _ in range(num_iterations):
        if _ == 0:
            with torch.cuda.nvtx.range("custom_embedding_lookup_on_host_table_with_5sms"):
                embedding_lookup_cuda(indices_list[_], host_embedding_table, output, 5, max_threads_per_sm)
        else:
            embedding_lookup_cuda(indices_list[_], host_embedding_table, output, 5, max_threads_per_sm)
    end.record()
    torch.cuda.synchronize()
    custom_time_for_host_table_with_5sms = start.elapsed_time(end) / num_iterations  

    # Benchmark PyTorch implementation
    start.record()
    for _ in range(num_iterations):
        torch.nn.functional.embedding(indices_list[_], embedding_table)
    end.record()
    torch.cuda.synchronize()
    pytorch_time = start.elapsed_time(end) / num_iterations

    print(embedding_table_dtype.itemsize)
    memory_traffic = batch_size * embedding_dim * (embedding_table_dtype.itemsize + output_dtype.itemsize)
    print(f"\nBenchmark Results (avg over {num_iterations} iterations):")
    print(f"Custom Implementation: {custom_time:.3f} ms, BW: {memory_traffic * 1e-6 / custom_time:.2f} GB/s")
    print(f"Custom Implementation for Host Table: {custom_time_for_host_table:.3f} ms, BW: {memory_traffic * 1e-6 / custom_time_for_host_table:.2f} GB/s")
    print(f"Custom Implementation for Host Table with 5 SMs: {custom_time_for_host_table_with_5sms:.3f} ms, BW: {memory_traffic * 1e-6 / custom_time_for_host_table_with_5sms:.2f} GB/s")
    print(f"PyTorch Implementation: {pytorch_time:.3f} ms, BW: {memory_traffic * 1e-6 / pytorch_time:.2f} GB/s")
    print(f"Speedup: {pytorch_time/custom_time:.2f}x")

if __name__ == "__main__":
    test_embedding_lookup()
    benchmark_embedding_lookup()