# 02 Matmul harness

Implement one function:

```python
def matmul(A: torch.Tensor, B: torch.Tensor, **options) -> torch.Tensor:
    ...
```

The contract is `A[M,K] @ B[K,N] -> C[M,N]`. The testbench always calls the
exported `matmul`; that wrapper selects the current `matmul_1/2/3`. Inputs are
FP16. Accumulation dtype is entirely owned by the kernel implementation.

## Run

Shortest command: the first file is the testbench; the second file exports
`matmul` and is the adapter under test. It runs correctness and performance:

```bash
CXX=/usr/bin/g++-15 uv run 02-matmul-test/main.py 02a-matmul-tilelang/main.py
```

This defaults to correctness plus performance. Optional `--mode` can select one
part. Shapes, FP16 input dtype, block options, tolerances, and timing come from
`manifests/matmul_v1.json`. The testbench never selects an internal kernel and
never injects TileLang dtype objects.

```bash
CXX=/usr/bin/g++-15 uv run 02-matmul-test/main.py 02a-matmul-tilelang/main.py \
  --mode correctness
```

Modes are `correctness`, `performance`, and `all`. The hidden `--case-id` is a
debugging filter.

Harness self-test:

```bash
CXX=/usr/bin/g++-15 uv run 02-matmul-test/main.py \
  02-matmul-test/adapters/torch_reference.py
```

## What the gate checks

- Fixed zero, rectangular, skinny, and three-axis tail cases.
- Numerical equality with `A @ B` across fixed zero, rectangular, skinny, and
  three-axis tail cases.
- Prepare/JIT outside steady-state timing.
- gau-nernst-style `triton.testing.do_bench`: 25 ms warmup, 100 ms measurement,
  median result, and L2 cleared before each sample.
- Reference first, then student, on identical A/B objects and pointers.
- Markdown tables with duration, `2*M*N*K` TFLOP/s, percent of cuBLAS,
  peak-memory delta, and GPU/software metadata.

The harness does not fabricate a bandwidth value from tensor sizes. The
bandwidth column in gau-nernst's SIMT table comes from Nsight Compute hardware
counters; collect it separately when profiler access is available.

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
CXX=/usr/bin/g++-15 uv run python 02-matmul-test/matmul_add_runner.py \
  --stage 02b-baseline-fp32 \
  --mode correctness \
  --adapter 02b-matmul-add-tilelang/main.py
```

Later stages are `02b-fused-fp16` and `02b-final`. The reference is
`torch.addmm(C,A,B)`. The fixed gate compares only numerical results for cases
where C is observable, then times the whole fused call.
