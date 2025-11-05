# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2024, NVIDIA Corporation & AFFILIATES.
import torch

sm_major_version = torch.cuda.get_device_properties(0).major
sm_minor_version = torch.cuda.get_device_properties(0).minor
from hstu_attn import hstu_attn_varlen_func, hstu_attn_qkvpacked_func

import math
from typing import Optional, Tuple

import pytest
import torch.nn.functional as F
from einops import rearrange

debug = False
example = False


def pad_input(unpadded_input, cu_seqlen, batch, seqlen):
    indices = []
    for i in range(batch):
        indices.append(
            torch.arange(seqlen * i, seqlen * i + cu_seqlen[i + 1] - cu_seqlen[i])
        )
    indices = torch.cat(indices)
    output = torch.zeros(
        (batch * seqlen),
        *unpadded_input.shape[1:],
        device=unpadded_input.device,
        dtype=unpadded_input.dtype,
    )
    output[indices] = unpadded_input
    return rearrange(output, "(b s) ... -> b s ...", b=batch)

def unpad_input(padded_input, cu_seqlen):
    padded_input.reshape(padded_input.size(0), padded_input.size(1), -1)
    output = []
    for i in range(len(cu_seqlen) - 1):
        output.append(padded_input[i, : (cu_seqlen[i + 1] - cu_seqlen[i]), :])
    return torch.cat(output, dim=0)

def construct_mask(
    batch_size,
    max_seqlen,
    cu_seqlens,
    num_contexts,
    num_targets,
    target_group_size,
    mask_type,
    device=torch.device("cuda"),
):
    mask = torch.zeros((batch_size, max_seqlen, max_seqlen), device=device, dtype=torch.bool)
    if mask_type == "target_no_mask":
        for b in range(batch_size):
            context_seqlen = num_contexts[b].item()
            target_seqlen = num_targets[b].item()
            history_seqlen = cu_seqlens[b + 1] - cu_seqlens[b] - context_seqlen - target_seqlen
            for i in range(context_seqlen):
                mask[b, i, :context_seqlen + history_seqlen] = True
            for i in range(history_seqlen):
                mask[b, context_seqlen + i, :context_seqlen + i] = True
            for i in range(target_seqlen):
                mask[b, context_seqlen + history_seqlen + i, :context_seqlen + history_seqlen + target_seqlen] = True
    elif mask_type == "target_group_no_mask":
        for b in range(batch_size):
            context_seqlen = num_contexts[b].item()
            target_seqlen = num_targets[b].item()
            history_seqlen = cu_seqlens[b + 1] - cu_seqlens[b] - context_seqlen - target_seqlen
            for i in range(context_seqlen):
                mask[b, i, :context_seqlen + history_seqlen] = True
            for i in range(history_seqlen):
                mask[b, context_seqlen + i, :context_seqlen + i] = True
            for i in range(target_seqlen):
                mask[b, context_seqlen + history_seqlen + i, :context_seqlen + history_seqlen] = True
                for j in range(target_seqlen // target_group_size):
                    start = context_seqlen + history_seqlen + j * target_group_size
                    end = min(context_seqlen + history_seqlen + target_seqlen, context_seqlen + history_seqlen + (j + 1) * target_group_size)
                    mask[b, start: end, start:end] = True
    else:
        raise ValueError(f"Invalid mask type: {mask_type}")
    return mask

def generate_input(
    batch_size: int,
    heads: int,
    max_seq_len_h: int,
    max_context_len: int,
    max_target_len: int,
    target_group_size: int,
    attn_dim: int,
    hidden_dim: int,
    dtype: torch.dtype,
    full_batch: bool,
    mask_type: str,
):
    has_context = max_context_len > 0
    has_target = max_target_len > 0
    # Generate lengths for context
    if max_context_len > 0:
        if full_batch:
            num_contexts = (
                torch.ones(
                    (batch_size,), device=torch.device("cuda"), dtype=torch.int32
                )
                * max_context_len
            )
        else:
            num_contexts = torch.randint(
                0,
                max_context_len + 1,
                size=(batch_size,),
                dtype=torch.int32,
                device=torch.device("cuda"),
            )
    else:
        num_contexts = torch.zeros(
            (batch_size,), dtype=torch.int32, device=torch.device("cuda")
        )
    cu_seqlens_c = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device("cuda")
    )
    cu_seqlens_c[1:] = torch.cumsum(num_contexts, dim=0)

    # Generate lengths for historial qkv
    if full_batch:
        lengths_h = (
            torch.ones((batch_size,), device=torch.device("cuda"), dtype=torch.int32)
            * max_seq_len_h
        )
    else:
        lengths_h = torch.randint(
            1, max_seq_len_h + 1, size=(batch_size,), device=torch.device("cuda")
        )
    cu_seqlens_h = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device("cuda")
    )
    cu_seqlens_h[1:] = torch.cumsum(lengths_h, dim=0)

    # Generate lengths for target qkv
    if has_target:
        if full_batch:
            num_targets = (
                torch.ones(
                    (batch_size,), device=torch.device("cuda"), dtype=torch.int32
                )
                * max_target_len
            )
        else:
            num_targets = torch.randint(
                0,
                max_target_len + 1,
                size=(batch_size,),
                dtype=torch.int32,
                device=torch.device("cuda"),
            )
    else:
        num_targets = torch.zeros(
            (batch_size,), dtype=torch.int32, device=torch.device("cuda")
        )
    cu_seqlens_t = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device("cuda")
    )
    cu_seqlens_t[1:] = torch.cumsum(num_targets, dim=0)

    cu_seqlens = cu_seqlens_c + cu_seqlens_h + cu_seqlens_t

    L = int(cu_seqlens[-1].item())

    # Generate q, k, v for history + target
    q = (
        torch.empty(
            (L, heads, attn_dim), dtype=dtype, device=torch.device("cuda")
        )
        .uniform_(-1, 1)
        .requires_grad_()
    )
    k = (
        torch.empty(
            (L, heads, attn_dim), dtype=dtype, device=torch.device("cuda")
        )
        .uniform_(-1, 1)
        .requires_grad_()
    )
    v = (
        torch.empty(
            (L, heads, hidden_dim), dtype=dtype, device=torch.device("cuda")
        )
        .uniform_(-1, 1)
        .requires_grad_()
    )

    assert mask_type in ["target_no_mask", "target_group_no_mask"]
    L_func = L + 256
    var_func = torch.empty(
        (1, 3, L_func), dtype=torch.int32, device=torch.device("cuda")
    )
    if mask_type == "target_no_mask":
        seqlen_offset = 0
        for b in range(batch_size):
            context_seqlen = cu_seqlens_c[b + 1] - cu_seqlens_c[b]
            history_seqlen = cu_seqlens_h[b + 1] - cu_seqlens_h[b]
            target_seqlen = cu_seqlens_t[b + 1] - cu_seqlens_t[b]
            for i in range(context_seqlen):
                var_func[0, :, seqlen_offset + i] = context_seqlen + history_seqlen
            for i in range(history_seqlen):
                var_func[0, :, seqlen_offset + context_seqlen + i] = context_seqlen + i
            for i in range(target_seqlen):
                var_func[0, :, seqlen_offset + context_seqlen + history_seqlen + i] = context_seqlen + history_seqlen + target_seqlen
            seqlen_offset += context_seqlen + history_seqlen + target_seqlen
    elif mask_type == "target_group_no_mask":
        seqlen_offset = 0
        for b in range(batch_size):
            context_seqlen = cu_seqlens_c[b + 1] - cu_seqlens_c[b]
            history_seqlen = cu_seqlens_h[b + 1] - cu_seqlens_h[b]
            target_seqlen = cu_seqlens_t[b + 1] - cu_seqlens_t[b]
            for i in range(context_seqlen):
                var_func[0, 0, seqlen_offset + i] = context_seqlen + history_seqlen
                var_func[0, 1, seqlen_offset + i] = context_seqlen + history_seqlen + target_seqlen
                var_func[0, 2, seqlen_offset + i] = context_seqlen + history_seqlen + target_seqlen
            for i in range(history_seqlen):
                var_func[0, 0, seqlen_offset + context_seqlen + i] = context_seqlen + i
                var_func[0, 1, seqlen_offset + context_seqlen + i] = context_seqlen + target_seqlen
                var_func[0, 2, seqlen_offset + context_seqlen + i] = context_seqlen + target_seqlen
            for i in range(target_seqlen):
                var_func[0, 0, seqlen_offset + context_seqlen + history_seqlen + i] = context_seqlen + history_seqlen
                var_func[0, 1, seqlen_offset + context_seqlen + history_seqlen + i] = context_seqlen + history_seqlen + (i // target_group_size) * target_group_size
                var_func[0, 2, seqlen_offset + context_seqlen + history_seqlen + i] = context_seqlen + history_seqlen + (i // target_group_size + 1) * target_group_size
    else:
        raise ValueError(f"Invalid mask type: {mask_type}")

    attn_mask = (
        construct_mask(
            batch_size=batch_size,
            max_seqlen=max_seq_len_h + max_context_len + max_target_len,
            cu_seqlens=cu_seqlens,
            num_contexts=num_contexts,
            num_targets=num_targets,
            target_group_size=target_group_size,
            mask_type=mask_type,
        )
        .cuda()
        .to(torch.float32)
    )
    print(attn_mask.to(torch.int32).squeeze().cpu().numpy())
    return (
        L,
        cu_seqlens,
        num_contexts if has_context else None,
        num_targets if has_target else None,
        q,
        k,
        v,
        attn_mask,
        var_func,
    )


# @torch.compile
def _hstu_attention_maybe_from_cache(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    max_seqlen: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    invalid_attn_mask: torch.Tensor,
    alpha: float,
    upcast: bool = True,
):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    B: int = q_offsets.size(0) - 1
    dtype_out = q.dtype
    padded_q = pad_input(q, q_offsets, B, max_seqlen)
    padded_k = pad_input(k, k_offsets, B, max_seqlen)
    padded_v = pad_input(v, k_offsets, B, max_seqlen)

    padded_q = padded_q.view(B, max_seqlen, num_heads, attention_dim)
    padded_k = padded_k.view(B, max_seqlen, num_heads, attention_dim)
    padded_v = padded_v.view(B, max_seqlen, num_heads, linear_dim)
    if upcast:
        padded_q, padded_k, padded_v = (
            padded_q.float(),
            padded_k.float(),
            padded_v.float(),
        )
    qk_attn = torch.einsum(
        "bnhd,bmhd->bhnm",
        padded_q,
        padded_k,
    )

    qk_attn = qk_attn * alpha
    qk_attn = F.silu(qk_attn)
    masked_qk_attn = qk_attn / max_seqlen
    if invalid_attn_mask is not None:
        if invalid_attn_mask.ndim == 2:
            invalid_attn_mask = invalid_attn_mask.unsqueeze(0).unsqueeze(0)
        elif invalid_attn_mask.ndim == 3:
            invalid_attn_mask = invalid_attn_mask.unsqueeze(1)
        masked_qk_attn = masked_qk_attn * invalid_attn_mask.type(masked_qk_attn.dtype)

    attn_output = torch.einsum(
        "bhnm,bmhd->bnhd",
        masked_qk_attn,
        padded_v,
    )

    attn_output = attn_output.reshape(B, max_seqlen, num_heads * linear_dim)
    attn_output = unpad_input(attn_output, q_offsets)
    attn_output = attn_output.reshape(-1, num_heads, linear_dim)

    return attn_output.to(dtype_out)

def test_fused_attn(
    batch_size: int,
    heads: int,
    max_seq_len_h: int,
    max_context_len: int,
    max_target_len: int,
    target_group_size: int,
    attn_dim: int,
    hidden_dim: int,
    alpha: float,
    dtype: torch.dtype,
    mask_type: str,
    full_batch: bool,
) -> None:
    has_context = max_context_len > 0
    has_target = max_target_len > 0
    group_target = target_group_size > 1
    if group_target and not has_target:
        raise ValueError("group_target is True but has_target is False")

    torch.cuda.synchronize()
    
    (
        L,
        cu_seqlens,
        num_contexts,
        num_targets,
        q,
        k,
        v,
        attn_mask,
        func,
    ) = generate_input(
        batch_size=batch_size,
        heads=heads,
        max_seq_len_h=max_seq_len_h,
        max_context_len=max_context_len,
        max_target_len=max_target_len,
        target_group_size=target_group_size,
        attn_dim=attn_dim,
        hidden_dim=hidden_dim,
        dtype=dtype,
        full_batch=full_batch,
        mask_type=mask_type,
    )
    out_ref = _hstu_attention_maybe_from_cache(
        num_heads=heads,
        attention_dim=attn_dim,
        linear_dim=hidden_dim,
        max_seqlen=max_context_len + max_seq_len_h + max_target_len,
        q=q.view(L, -1),
        k=k.view(L, -1),
        v=v.view(L, -1),
        q_offsets=cu_seqlens,
        k_offsets=cu_seqlens,
        invalid_attn_mask=attn_mask.to(torch.float32)
        if attn_mask is not None
        else None,
        alpha=alpha,
        upcast=True,
    )

    torch_out = _hstu_attention_maybe_from_cache(
        num_heads=heads,
        attention_dim=attn_dim,
        linear_dim=hidden_dim,
        max_seqlen=max_context_len + max_seq_len_h + max_target_len,
        q=q.view(L, -1),
        k=k.view(L, -1),
        v=v.view(L, -1),
        q_offsets=cu_seqlens,
        k_offsets=cu_seqlens,
        invalid_attn_mask=attn_mask.to(torch.float32)
        if attn_mask is not None
        else None,
        alpha=alpha,
        upcast=False,
    )
    hstu_out = hstu_attn_varlen_func(
        q=q.to(dtype),
        k=k.to(dtype),
        v=v.to(dtype),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_context_len + max_seq_len_h + max_target_len,
        max_seqlen_k=max_context_len + max_seq_len_h + max_target_len,
        num_contexts=None,
        num_targets=None,
        target_group_size=target_group_size,
        window_size=(-1, -1),
        alpha=alpha,
        rab=None,
        has_drab=False,
        func=func,
    )

    print(f"Output max diff: {(hstu_out - out_ref).abs().max().item()}")
    print(f"Pytorch max diff: {(torch_out - out_ref).abs().max().item()}")

    print(f"Output mean diff: {(hstu_out - out_ref).abs().mean().item()}")
    print(f"Pytorch mean diff: {(torch_out - out_ref).abs().mean().item()}")

    assert (hstu_out - out_ref).abs().max().item() <= 2 * (
        torch_out - out_ref
    ).abs().max().item()
    g = torch.rand_like(torch_out)
    (dq_ref, dk_ref, dv_ref) = torch.autograd.grad(
        out_ref, (q, k, v), g, retain_graph=True
    )
    (dq_torch, dk_torch, dv_torch) = torch.autograd.grad(
        torch_out, (q, k, v), g, retain_graph=True
    )
    (dq_hstu, dk_hstu, dv_hstu) = torch.autograd.grad(
        hstu_out,
        (q, k, v),
        g.view(-1, heads, hidden_dim),
        retain_graph=True,
    )
    
    print(f"dV max diff: {(dv_hstu - dv_ref).abs().max().item()}")
    print(f"dV Pytorch max diff: {(dv_torch - dv_ref).abs().max().item()}")

    print(f"dK max diff: {(dk_hstu - dk_ref).abs().max().item()}")
    print(f"dK Pytorch max diff: {(dk_torch - dk_ref).abs().max().item()}")

    print(f"dQ max diff: {(dq_hstu - dq_ref).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(dq_torch - dq_ref).abs().max().item()}")

    assert (dv_hstu - dv_ref).abs().max().item() <= 5 * (
        dv_torch - dv_ref
    ).abs().max().item()
    assert (dk_hstu - dk_ref).abs().max().item() <= 5 * (
        dk_torch - dk_ref
    ).abs().max().item()
    assert (dq_hstu - dq_ref).abs().max().item() <= 5 * (
        dq_torch - dq_ref
    ).abs().max().item()
    torch.cuda.synchronize()


if __name__ == "__main__":
    test_fused_attn(
        batch_size=32,
        heads=4,
        max_seq_len_h=32,
        max_context_len=10,
        max_target_len=10,
        target_group_size=1,
        attn_dim=32,
        hidden_dim=32,
        alpha=1.0,
        dtype=torch.bfloat16,
        full_batch=True,
        mask_type="target_no_mask",
    )
    test_fused_attn(
        batch_size=1,
        heads=1,
        max_seq_len_h=32,
        max_context_len=10,
        max_target_len=10,
        target_group_size=2,
        attn_dim=32,
        hidden_dim=32,
        alpha=1.0,
        dtype=torch.bfloat16,
        full_batch=True,
        mask_type="target_group_no_mask",
    )