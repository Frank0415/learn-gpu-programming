# 03 FlashAttention harness

Implement one function:

```python
def attention_forward(
    q, k, v, cu_seqlens, max_seqlen,
    *, causal, scale=None, **options,
):
    ...
```

For 03A--03D, `q=[T,Hq,D]`, `k/v=[T,Hkv,D]`, and output is `[T,Hq,D]`.
Inputs are contiguous CUDA FP16/BF16 tensors on one device. `cu_seqlens` is
contiguous CUDA int32 `[B+1]`, starts at 0, ends at T, and is strictly
increasing. Empty requests are invalid. `max_seqlen` equals the largest segment.

This is packed self-attention with one boundary vector for Q/K/V, not cross
attention. Causal indices are segment-local. GQA requires `Hq % Hkv == 0` and
maps `q_head // (Hq/Hkv)` without copying K/V. `scale=None` means `D**-0.5`;
`scale=0.0` is a correctness edge case.

## Run

```bash
uv run python 03-flash-attention-test/runner.py \
  --stage 03a \
  --mode correctness \
  --adapter 03a-flash-attention-tilelang/main.py
```

The CLI accepts only stage, mode, and adapter. Shapes, tolerances, options, and
timing come from `manifests/flash_attention_v1.json`; hidden `--case-id` is only
for debugging. Repeat `--adapter NAME=PATH` to compare variants on the same
Q/K/V/cu objects and pointers.

03A/03B are single-request MHA. 03C adds packed GQA on SM120. 03D reuses the
03C case IDs but requires an H200/SM90 device.

## Gate

Correctness uses an FP32 reference per segment and checks tails, causal masking,
GQA ratios 1/2/4/8, segment isolation, head mapping, scale 0, invalid inputs,
input immutability, output metadata, and cached-output behavior.

Performance uses PyTorch 2.13 `varlen_attn` with identical inputs. A core case
is blocked if that backend is unavailable. Prepare/JIT is separate. Steady
state uses gau-nernst-style `triton.testing.do_bench`: 25 ms warmup, 100 ms
measurement, median result, and L2 clearing before every sample. Results use
Markdown tables with duration, effective TFLOP/s, and percent of PyTorch
varlen. The 03C gate requires median ours/library <= 1.10 and worst core ratio
<= 1.25. Variants within 1% of the best tested score are tied.

## 03E

`manifests/flash_attention_precision_v2.json` records the deferred low-precision
contract: FP8, MXFP8, MXFP4, NVFP4, scale granularity/layout, output dtype, and
explicit native/reference/emulation support. Runtime Python types and validators
will be added with the first 03E implementation instead of being guessed now.
