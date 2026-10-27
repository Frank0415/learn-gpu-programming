"""Deterministic GEMM correctness cases shared by every adapter."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch


Pattern = Literal["zeros", "random", "ramp"]


@dataclass(frozen=True)
class MatmulCase:
    name: str
    M: int
    N: int
    K: int
    pattern: Pattern
    seed: int = 0

    def __post_init__(self) -> None:
        for dimension_name in ("M", "N", "K"):
            dimension = getattr(self, dimension_name)
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise ValueError(f"{dimension_name} must be an integer, got {dimension!r}")
            if dimension <= 0:
                raise ValueError(f"{dimension_name} must be positive, got {dimension}")
        if self.pattern not in ("zeros", "random", "ramp"):
            raise ValueError(
                f"pattern must be one of zeros, random, ramp; got {self.pattern!r}"
            )

def _check_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            "dtype must be torch.float32, torch.float16, or torch.bfloat16; "
            f"got {dtype}"
        )


def _bounded_ramp(
    shape: tuple[int, int],
    *,
    device: torch.device,
    multiplier: float,
    offset: float,
) -> torch.Tensor:
    """Create a deterministic ramp bounded approximately to ``[-1, 1]``."""

    values = torch.arange(math.prod(shape), device=device, dtype=torch.float32)
    values = values.reshape(shape)
    return torch.remainder(values * multiplier + offset, 17.0) / 8.0 - 1.0


def make_inputs(
    case: MatmulCase,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_dtype(dtype)
    target = torch.device(device)
    a_shape = (case.M, case.K)
    b_shape = (case.K, case.N)

    if case.pattern == "zeros":
        A = torch.zeros(a_shape, device=target, dtype=dtype)
        B = torch.zeros(b_shape, device=target, dtype=dtype)
    elif case.pattern == "ramp":
        A = _bounded_ramp(
            a_shape, device=target, multiplier=1.0, offset=0.0
        ).to(dtype=dtype)
        B = _bounded_ramp(
            b_shape, device=target, multiplier=3.0, offset=1.0
        ).to(dtype=dtype)
    else:
        generator = torch.Generator(device=target).manual_seed(case.seed)
        A = torch.rand(a_shape, generator=generator, device=target) * 2.0 - 1.0
        B = torch.rand(b_shape, generator=generator, device=target) * 2.0 - 1.0
        A = A.to(dtype=dtype)
        B = B.to(dtype=dtype)

    return A.contiguous(), B.contiguous()
