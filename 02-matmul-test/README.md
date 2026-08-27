# 02 Matmul harness

Implement one function:

```python
def matmul(A: torch.Tensor, B: torch.Tensor, **options) -> torch.Tensor:
    ...
```

The contract is `A[M,K] @ B[K,N] -> C[M,N]`. Inputs and output are contiguous
CUDA tensors with the same dtype and device. There is no transpose, batch,
broadcast, or `out` parameter. Unknown variants must fail instead of falling
back to another implementation.

## Run

The student CLI accepts only stage, mode, and adapter. Shapes, dtype, options,
tolerances, and timing come from `manifests/matmul_v1.json`.

```bash
uv run python 02-matmul-test/student_runner.py \
  --stage naive-fp32 \
  --mode correctness \
  --adapter 02a-matmul-tilelang/main.py
```

Stages are `naive-fp32`, `shared-fp32`, `tensorcore`, `pipeline`, and `final`.
Modes are `correctness`, `performance`, and `all`. The hidden `--case-id` is
only a debugging filter.

Harness self-test:

```bash
uv run python 02-matmul-test/student_runner.py \
  --stage naive-fp32 \
  --mode all \
  --adapter 02-matmul-test/adapters/torch_reference.py
```

## What the gate checks

- Fixed zero, rectangular, skinny, and three-axis tail cases.
- Output type, shape, dtype, device, non-aliasing, numerical result, repeated
  result, input immutability, and unknown-variant rejection.
- Prepare/JIT outside steady-state timing.
- Fixed 20 warmups, 5 trials, and 50 CUDA Event samples per trial.
- Alternating student/PyTorch order on identical A/B objects and pointers.
- p20/median/p80, `2*M*N*K` TFLOP/s, latency ratio, peak-memory delta, and
  GPU/software metadata.

The manifest targets SM120. Tensor Core stages must be confirmed from generated
code; `T.gemm` alone is not proof. SM100 TCGen05/Tensor Memory paths and B200
numbers do not apply to this harness.

## 02B: A@B+C

02B keeps the original Matmul ABI unchanged and adds a separate function:

```python
def matmul_add(A, B, C, **options) -> D:
    ...
```

`D[M,N] = A[M,K] @ B[K,N] + C[M,N]`. C is a complete matrix; this stage does
not support bias broadcasting or alpha/beta parameters.

```bash
uv run python 02-matmul-test/matmul_add_runner.py \
  --stage 02b-baseline-fp32 \
  --mode correctness \
  --adapter 02b-matmul-add-tilelang/main.py
```

Later stages are `02b-fused-fp16` and `02b-final`. The reference is
`torch.addmm(C,A,B)`. The fixed gate verifies that C is observable, rejects a
broadcast C, checks all inputs for mutation, and times the whole fused call.
