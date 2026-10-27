from __future__ import annotations

import torch


def matmul_add(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, **options
) -> torch.Tensor:
    return torch.addmm(C, A, B)
