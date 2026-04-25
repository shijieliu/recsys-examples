#!/usr/bin/env python3
"""Quick correctness test: manager.forward() vs raw all2all_single."""
import os, sys, torch, torch.distributed as dist

os.chdir('/home/scratch.aleliu_sw/recsys-examples/corelib/dynamicemb')
sys.path.insert(0, '.')
if "dynamicemb_extensions" in sys.modules:
    del sys.modules["dynamicemb_extensions"]

from dynamicemb.hier_all2all import HierAll2AllManager
import dynamicemb_extensions
ha = dynamicemb_extensions.hier_a2a

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
W = dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.device(f'cuda:{rank}')

D = 64; F = 2; max_seq = 5; dtype = torch.bfloat16
torch.manual_seed(42 + rank)

lengths = torch.randint(1, max_seq + 1, (F * W,), device=device)
input_splits = [sum(lengths[r*F+f].item() for f in range(F)) for r in range(W)]
total_send = sum(input_splits)
output_embs = torch.randn(total_send, D, device=device, dtype=dtype)
sfrecat = torch.arange(F * W, dtype=torch.int32, device=device)

input_splits_t = torch.tensor(input_splits, dtype=torch.int64, device=device)
output_splits_t = torch.empty_like(input_splits_t)
dist.all_to_all_single(output_splits_t, input_splits_t,
                        output_split_sizes=[1]*W, input_split_sizes=[1]*W)
output_splits = output_splits_t.tolist()
total_recv = sum(output_splits)
# Use empty tensor (numel=0) to skip index_select in forward_fast
unbuck = torch.empty(0, dtype=torch.int64, device=device)

# Reference: raw all2all_single
ref = torch.empty(total_recv, D, dtype=dtype, device=device)
dist.all_to_all_single(ref.view(-1), output_embs.view(-1),
                        [s*D for s in output_splits], [s*D for s in input_splits])
torch.cuda.synchronize()
dist.barrier()

# Hier manager
max_rows = max_seq * F * W
manager = HierAll2AllManager(pg=dist.group.WORLD, num_features=F,
                             max_rows_per_rank=max_rows, D=D, device=device,
                             dtype=dtype)

print(f"[{rank}] fallback={manager.fallback}", flush=True)
print(f"[{rank}] total_send={total_send}, total_recv={total_recv}", flush=True)
print(f"[{rank}] input_splits={input_splits}, output_splits={output_splits}", flush=True)
print(f"[{rank}] calling ha.two_kernel_single_node directly...", flush=True)

# Build scatter map through manager internals, then call C++ directly
manager._iter_id += 1
scatter_map = manager._build_scatter_map_from_context(
    lengths, input_splits, sfrecat, total_send, total_send, "fwd", None)
torch.cuda.synchronize()
gi = scatter_map.peer_gather_indices.cpu()
off = scatter_map.peer_offsets.cpu()
print(f"[{rank}] scatter_map: offsets={off.tolist()}, gi_shape={gi.shape}, "
      f"gi_min={gi.min().item()}, gi_max={gi.max().item()}, "
      f"gi[:10]={gi[:10].tolist()}, total_send={total_send}", flush=True)

# Build gather_permute manually (like bench_kernel_only)
gather_permute = torch.empty(total_recv, dtype=torch.int64, device=device)
offset = 0
for s in range(W):
    count = output_splits[s]
    lr = s  # single node: local_rank = rank
    slot_start = lr * max_rows
    if count > 0:
        gather_permute[offset:offset+count] = torch.arange(
            slot_start, slot_start + count, dtype=torch.int64, device=device)
        offset += count

device_flag = torch.zeros(1, dtype=torch.int32, device=device)
torch.cuda.synchronize()
dist.barrier()

print(f"[{rank}] launching two_kernel_single_node...", flush=True)
hier = ha.two_kernel_single_node(
    output_embs.clone(), scatter_map.peer_gather_indices, scatter_map.peer_offsets,
    gather_permute, manager._peer_slot_ptrs_dev, manager._peer_sig_ptrs_dev,
    manager._ipc_recv_buf.data_ptr(), manager._signal_pad_ptr,
    rank, W, total_recv, D, device_flag, 1)
torch.cuda.synchronize()
print(f"[{rank}] kernel done!", flush=True)

diff = (ref - hier).abs().max().item()
if rank == 0:
    print(f"diff = {diff} -> {'PASS' if diff < 1e-5 else 'FAIL'}", flush=True)
    if diff > 1e-5:
        print(f"ref[:3,:4]  = {ref[:3,:4]}", flush=True)
        print(f"hier[:3,:4] = {hier[:3,:4]}", flush=True)

dist.barrier()
dist.destroy_process_group()
