"""PyTorch 2.13 varlen adapter used to self-test the shared harness."""

from __future__ import annotations

import torch

from reference import torch_varlen_reference


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    causal: bool,
    scale: float | None = None,
    **options,
) -> torch.Tensor:
    if options:
        raise ValueError(f"unknown options: {', '.join(sorted(options))}")
    return torch_varlen_reference(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        causal=causal,
        scale=scale,
    )
