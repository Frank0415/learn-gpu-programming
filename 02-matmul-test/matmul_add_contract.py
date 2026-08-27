from __future__ import annotations

import inspect
from pathlib import Path
import runpy
from typing import Any, Protocol, cast

import torch


class MatmulAdd(Protocol):
    def __call__(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, **options: Any
    ) -> torch.Tensor: ...


def validate_inputs(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
    if any(t.ndim != 2 for t in (A, B, C)):
        raise ValueError("A, B, and C must be rank-2")
    if A.shape[1] != B.shape[0] or C.shape != (A.shape[0], B.shape[1]):
        raise ValueError("expected A[M,K], B[K,N], C[M,N]")
    if not (A.dtype == B.dtype == C.dtype and A.device == B.device == C.device):
        raise ValueError("A, B, and C must share dtype and device")
    if not all(t.is_cuda and t.is_contiguous() for t in (A, B, C)):
        raise ValueError("A, B, and C must be contiguous CUDA tensors")


def load_matmul_add(path: str | Path) -> MatmulAdd:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"adapter does not exist: {resolved}")
    try:
        candidate = runpy.run_path(str(resolved), run_name="<matmul-add-adapter>").get(
            "matmul_add"
        )
    except Exception as error:
        raise ValueError(f"failed to load {resolved}: {error}") from error
    if not callable(candidate):
        raise ValueError("adapter must export matmul_add(A, B, C, **options)")
    try:
        signature = inspect.signature(candidate)
        signature.bind(object(), object(), object())
    except (TypeError, ValueError) as error:
        raise ValueError("matmul_add must accept A, B, and C") from error
    if not any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        raise ValueError("matmul_add must accept **options")
    return cast(MatmulAdd, candidate)
