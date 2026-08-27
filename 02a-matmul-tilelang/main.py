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
    M, N, K = T.const("M,N,K")
    A: T.Tensor((M, K), T.float16)
    B: T.Tensor((K, N), T.float16)
    C = T.empty((M, N), T.float16)

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
                for i in range(K):
                    acc += A[row, i] * B[i, col]
                C[row, col] = acc
    return C


@tilelang.jit
def matmul_2(
    A,
    B,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 16,
    threads: int = 64,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    M, N, K = T.const("M,N,K")
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=threads) as (
        block_x,
        block_y,
    ):
        A_shared = T.alloc_shared((block_m, block_k),dtype)
        B_shared = T.alloc_shared((block_k, block_n),dtype)
        C_local = T.alloc_fragment((block_m, block_n), accum_dtype)

        T.clear(C_local)

        # Keep the correctness-oriented kernel synchronous: skinny shapes can
        # lower to a 2-byte copy, while cp.async requires 4, 8, or 16 bytes.
        for k in T.serial(T.ceildiv(K, block_k)):
            T.copy(A[block_y * block_m, k * block_k], A_shared)
            T.copy(B[k * block_k, block_x * block_n], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[block_y * block_m, block_x * block_n])

    return C

def matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    **options: Any,
) -> torch.Tensor:
    variant = options["variant"]
    common = {
        "block_m": options["block_m"],
        "block_n": options["block_n"],
        "block_k": options["block_k"],
        "threads": options["threads"],
    }
    if variant == "matmul_2":
        return matmul_2(A, B, **common)
    if variant == "matmul_3":
        return matmul_3(A, B, num_stages=options["num_stages"], **common)
    raise ValueError(f"unknown variant: {variant}")

# Testing Code
if __name__ == "__main__":
    A = torch.arange(6, device="cuda", dtype=torch.float16).reshape(2, 3)
    B = torch.arange(12, device="cuda", dtype=torch.float16).reshape(3, 4)
    expected = A @ B
    for run in range(3):
        out = matmul_1(A, B)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
        print(run, "PASS", out)
