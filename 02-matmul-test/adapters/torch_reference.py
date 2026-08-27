"""PyTorch reference adapter used to validate the GEMM harness itself."""

from typing import Any

import torch


def matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    **options: Any,
) -> torch.Tensor:
    allowed = {
        "matmul_1", "matmul_shared", "matmul_gemm", "matmul_pipeline", "matmul_final"
    }
    variant = options.pop("variant", None)
    if variant is not None and variant not in allowed:
        raise ValueError(f"unknown variant: {variant}")
    # Launch/tile settings are meaningful to the student adapter, not PyTorch.
    known_settings = {"block_m", "block_n", "block_k", "threads", "num_stages"}
    unknown = set(options) - known_settings
    if unknown:
        raise ValueError(f"unknown options: {sorted(unknown)}")
    return A @ B
