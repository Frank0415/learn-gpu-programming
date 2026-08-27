from __future__ import annotations

import torch

from matmul_add_contract import validate_inputs


VARIANTS = {"matmul_add_1", "matmul_add_fused", "matmul_add_final"}


def matmul_add(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, **options
) -> torch.Tensor:
    validate_inputs(A, B, C)
    variant = options.pop("variant", None)
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    allowed = {"block_m", "block_n", "block_k", "threads", "num_stages"}
    if set(options) - allowed:
        raise ValueError(f"unknown options: {sorted(set(options) - allowed)}")
    return torch.addmm(C, A, B)
