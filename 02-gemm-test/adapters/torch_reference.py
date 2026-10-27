"""PyTorch reference adapter used to validate the GEMM harness itself."""

from typing import Any

import torch


def matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    **options: Any,
) -> torch.Tensor:
    known_settings = {"block_m", "block_n", "block_k", "threads", "num_stages"}
    unknown = set(options) - known_settings
    if unknown:
        raise ValueError(f"unknown options: {sorted(unknown)}")
    return A @ B
