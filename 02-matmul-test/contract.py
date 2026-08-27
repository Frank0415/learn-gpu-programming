from __future__ import annotations

import inspect
from pathlib import Path
import runpy
from typing import Any, Protocol, cast

import torch


class Matmul(Protocol):
    def __call__(
        self, A: torch.Tensor, B: torch.Tensor, **options: Any
    ) -> torch.Tensor: ...


def load_matmul(path: str | Path) -> Matmul:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"adapter does not exist: {resolved}")
    try:
        candidate = runpy.run_path(str(resolved), run_name="<matmul-adapter>").get(
            "matmul"
        )
    except Exception as error:
        raise ValueError(f"failed to load {resolved}: {error}") from error
    if not callable(candidate):
        raise ValueError("adapter must export matmul(A, B, **options)")
    try:
        signature = inspect.signature(candidate)
        signature.bind(object(), object())
    except (TypeError, ValueError) as error:
        raise ValueError("matmul must accept A and B") from error
    if not any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        raise ValueError("matmul must accept **options")
    return cast(Matmul, candidate)
