"""Adapter loading and input checks for the 03A--03D ABI."""

from __future__ import annotations

import inspect
from pathlib import Path
import runpy
from typing import Any, Protocol, cast

import torch


class AttentionForward(Protocol):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        *,
        causal: bool,
        scale: float | None = None,
        **options: Any,
    ) -> torch.Tensor: ...


def load_attention_forward(path: str | Path) -> AttentionForward:
    """Load an adapter file and check the inspectable part of the ABI."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"adapter file does not exist: {resolved}")
    try:
        namespace = runpy.run_path(str(resolved), run_name="<attention-adapter>")
    except Exception as error:
        raise ValueError(f"failed to load adapter {resolved}: {error}") from error

    candidate = namespace.get("attention_forward")
    if not callable(candidate):
        raise ValueError(
            "adapter must export callable attention_forward(q, k, v, "
            "cu_seqlens, max_seqlen, *, causal, scale=None, **options)"
        )

    try:
        signature = inspect.signature(candidate)
        signature.bind(
            object(), object(), object(), object(), 1, causal=False, scale=None
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "attention_forward does not accept the fixed 03 ABI: "
            "(q, k, v, cu_seqlens, max_seqlen, *, causal, scale=None, **options)"
        ) from error

    if not any(
        item.kind == inspect.Parameter.VAR_KEYWORD
        for item in signature.parameters.values()
    ):
        raise ValueError("attention_forward must accept adapter settings via **options")
    return cast(AttentionForward, candidate)


def validate_public_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> None:
    """Validate the v1 semantic preconditions before a reference call."""

    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 3:
            raise ValueError(f"{name} must have shape [T,H,D], got {tuple(tensor.shape)}")
        if any(int(dimension) <= 0 for dimension in tensor.shape):
            raise ValueError(f"{name} dimensions must be positive, got {tuple(tensor.shape)}")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be on CUDA")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"{name} dtype must be float16 or bfloat16")

    if k.shape != v.shape:
        raise ValueError("k and v must have identical [T,Hkv,D] shapes")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError("q/k/v must share total tokens T and head dimension D")
    if not (q.dtype == k.dtype == v.dtype and q.device == k.device == v.device):
        raise ValueError("q/k/v must share dtype and device")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("Hq must be divisible by Hkv")

    if not isinstance(cu_seqlens, torch.Tensor):
        raise TypeError("cu_seqlens must be a torch.Tensor")
    if not cu_seqlens.is_cuda or cu_seqlens.device != q.device:
        raise ValueError("cu_seqlens must be on the same CUDA device as q")
    if cu_seqlens.dtype != torch.int32 or cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must be contiguous int32 with shape [B+1]")
    if not cu_seqlens.is_contiguous() or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be contiguous and contain at least two entries")

    values = cu_seqlens.detach().cpu().tolist()
    if values[0] != 0 or values[-1] != q.shape[0]:
        raise ValueError("cu_seqlens must start at 0 and end at T")
    lengths = [right - left for left, right in zip(values, values[1:])]
    if any(length <= 0 for length in lengths):
        raise ValueError("cu_seqlens must be strictly increasing; empty requests are invalid")
    if isinstance(max_seqlen, bool) or not isinstance(max_seqlen, int):
        raise TypeError("max_seqlen must be an integer")
    if max_seqlen != max(lengths):
        raise ValueError("max_seqlen must exactly equal max(diff(cu_seqlens))")
