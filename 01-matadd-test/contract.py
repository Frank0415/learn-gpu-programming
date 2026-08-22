"""The shared Matrix Addition adapter interface and dynamic loader."""

from __future__ import annotations

import inspect
from pathlib import Path
import runpy
from typing import Any, Protocol, cast

import torch


class Matadd(Protocol):
    """A Matrix Addition adapter accepted by the shared harness."""

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        **options: Any,
    ) -> torch.Tensor: ...


def load_matadd(path: str | Path) -> Matadd:
    """Load the callable named ``matadd`` from an adapter file."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"adapter file does not exist: {resolved}")

    candidate = runpy.run_path(str(resolved)).get("matadd")
    if not callable(candidate):
        raise ValueError(f"adapter must export callable matadd(): {resolved}")

    signature = inspect.signature(candidate)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) < 2:
        raise ValueError(
            "matadd interface must accept at least two positional tensors: matadd(A, B, **options)"
        )

    return cast(Matadd, candidate)
