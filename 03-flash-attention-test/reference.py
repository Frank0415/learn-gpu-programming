from __future__ import annotations

import math

import torch

from contract import validate_public_inputs


def torch_varlen_available() -> bool:
    try:
        from torch.nn.attention.varlen import varlen_attn  # noqa: F401
    except ImportError:
        return False
    return True


def fp32_segment_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    causal: bool,
    scale: float | None = None,
) -> torch.Tensor:
    validate_public_inputs(q, k, v, cu_seqlens, max_seqlen)
    effective_scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    boundaries = cu_seqlens.detach().cpu().tolist()
    group_size = q.shape[1] // k.shape[1]
    chunks: list[torch.Tensor] = []

    for start, end in zip(boundaries, boundaries[1:]):
        q_part = q[start:end].float().transpose(0, 1)
        k_part = k[start:end].float().transpose(0, 1)
        v_part = v[start:end].float().transpose(0, 1)
        if group_size != 1:
            head_map = torch.arange(q.shape[1], device=q.device) // group_size
            k_part = k_part.index_select(0, head_map)
            v_part = v_part.index_select(0, head_map)
        scores = torch.matmul(q_part, k_part.transpose(-1, -2)) * effective_scale
        if causal:
            length = end - start
            mask = torch.ones((length, length), device=q.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, -math.inf)
        probabilities = torch.softmax(scores, dim=-1)
        chunks.append(torch.matmul(probabilities, v_part).transpose(0, 1))
    return torch.cat(chunks, dim=0).to(q.dtype)


def torch_varlen_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    causal: bool,
    scale: float | None = None,
) -> torch.Tensor:
    validate_public_inputs(q, k, v, cu_seqlens, max_seqlen)
    # PyTorch 2.13 varlen returns NaN for scale=0; that case is correctness-only.
    if scale is not None and float(scale) == 0.0:
        return fp32_segment_reference(
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            causal=causal,
            scale=scale,
        )
    try:
        from torch.nn.attention.varlen import varlen_attn
    except ImportError as error:
        raise RuntimeError(
            "core reference unavailable: torch.nn.attention.varlen.varlen_attn is required"
        ) from error
    return varlen_attn(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        scale=scale,
        window_size=(-1, 0) if causal else (-1, -1),
        enable_gqa=q.shape[1] != k.shape[1],
    )
