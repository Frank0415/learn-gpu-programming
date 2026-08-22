from typing import Any

import tilelang
import tilelang.language as T
import torch


@tilelang.jit
def matadd_1(
    A,
    B,
    block_m: int = 8,
    block_n: int = 16,
    threads: int = 128,
):
    M, N = T.const("M, N")
    A: T.Tensor((M, N), T.float32)
    B: T.Tensor((M, N), T.float32)
    C = T.empty((M, N), T.float32)

    with T.Kernel(
        T.ceildiv(N, block_n),
        T.ceildiv(M, block_m),
        threads=threads,
    ) as (block_x, block_y):
        for local_row, local_col in T.Parallel(block_m, block_n):
            row = block_y * block_m + local_row
            col = block_x * block_n + local_col

            if row < M and col < N:
                C[row, col] = A[row, col] + B[row, col]

    return C


def matadd(
    A: torch.Tensor,
    B: torch.Tensor,
    **options: Any,
) -> torch.Tensor:
    return matadd_1(A, B, **options)
