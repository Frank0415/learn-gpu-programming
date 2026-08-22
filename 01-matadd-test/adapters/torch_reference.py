"""Reference adapter used to validate the shared harness itself."""

from typing import Any

import torch


def matadd(
    A: torch.Tensor,
    B: torch.Tensor,
    **options: Any,
) -> torch.Tensor:
    del options
    return A + B
