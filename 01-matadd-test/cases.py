"""Central correctness cases for every Matrix Addition adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch


Pattern = Literal["zeros", "normal", "ramp"]


@dataclass(frozen=True)
class MataddCase:
    name: str
    shape: tuple[int, int]
    pattern: Pattern
    seed: int = 0


BASE_CASES = (
    MataddCase("scalar-zero", (1, 1), "zeros"),
    MataddCase("single-row", (1, 127), "normal", seed=1),
    MataddCase("small-rectangle", (7, 19), "ramp"),
    MataddCase("fixed-tail", (37, 53), "normal", seed=2),
)


def _positive_int(options: dict[str, Any], name: str, default: int) -> int:
    value = int(options.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def cases_for(options: dict[str, Any]) -> tuple[MataddCase, ...]:
    """Return fixed cases plus tails derived from the selected block shape."""
    block_m = _positive_int(options, "block_m", 8)
    block_n = _positive_int(options, "block_n", 16)
    derived = (
        MataddCase(
            "one-past-block",
            (block_m + 1, block_n + 3),
            "normal",
            seed=3,
        ),
        MataddCase(
            "two-block-tail",
            (2 * block_m - 1, 2 * block_n + 1),
            "normal",
            seed=4,
        ),
    )

    unique: dict[tuple[tuple[int, int], Pattern], MataddCase] = {}
    for case in (*BASE_CASES, *derived):
        unique[(case.shape, case.pattern)] = case
    return tuple(unique.values())


def make_inputs(
    case: MataddCase,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic contiguous FP32 inputs for one case."""
    if case.pattern == "zeros":
        A = torch.zeros(case.shape, device=device, dtype=torch.float32)
        B = torch.zeros_like(A)
    elif case.pattern == "ramp":
        elements = case.shape[0] * case.shape[1]
        A = torch.arange(elements, device=device, dtype=torch.float32).reshape(case.shape)
        B = torch.flip(A, dims=(0, 1)).contiguous() - elements / 2
    else:
        generator = torch.Generator(device=device).manual_seed(case.seed)
        A = torch.randn(case.shape, generator=generator, device=device, dtype=torch.float32)
        B = torch.randn(case.shape, generator=generator, device=device, dtype=torch.float32)

    return A.contiguous(), B.contiguous()
