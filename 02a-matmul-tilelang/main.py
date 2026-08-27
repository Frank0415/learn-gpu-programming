import tilelang
from typing import Any
import tilelang.language as T
import torch


@tilelang.jit
def matmul_1(
    A,
    B,
    block_m: int = 8,
    block_n: int = 16,
    threads: int = 128,
):
    M, N, P = T.const("M,N,P")
    A: T.Tensor((M, P), T.float32)
    B: T.Tensor((P, N), T.float32)
    C = T.empty((M, N), T.float32)

    with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=threads) as (
        block_x,
        block_y,
    ):
        for curr_row, curr_col in T.Parallel(block_m, block_n):
            # Index of the target matrix
            col = block_x * block_n + curr_col
            row = block_y * block_m + curr_row

            if col < N and row < M:
                acc = T.alloc_var(T.float32, init=0.0)
                for i in range(P):
                    acc += A[row, i] * B[i, col]
                C[row,col] = acc
    return C

def matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    **options: Any,
) -> torch.Tensor:
    return matmul_1(
        A,
        B,
        block_m=options["block_m"],
        block_n=options["block_n"],
        threads=options["threads"],
    )

# Testing Code
if __name__ == "__main__":
    A = torch.arange(6, device="cuda", dtype=torch.float32).reshape(2, 3)
    B = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
    expected = A @ B
    for run in range(3):
        out = matmul_1(A, B)
        print(run, torch.equal(out, expected), out)
