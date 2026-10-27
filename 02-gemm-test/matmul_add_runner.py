from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cases import MatmulCase, make_inputs
from common import DTYPES, TOLERANCES, benchmark, environment, prepare, print_table
from matmul_add_manifest import AddCase, load_matmul_add_manifest


STAGES = ("02b-baseline-fp32", "02b-fused-fp16", "02b-final")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed 02B A@B+C gate")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--mode", required=True, choices=("correctness", "performance", "all"))
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--case-id", help=argparse.SUPPRESS)
    return parser.parse_args()


def _c_tensor(case: AddCase, dtype: torch.dtype) -> torch.Tensor:
    shape = (case.M, case.N)
    if case.c_pattern == "zeros":
        return torch.zeros(shape, device="cuda", dtype=dtype)
    if case.c_pattern == "ramp":
        values = torch.arange(case.M * case.N, device="cuda", dtype=torch.float32)
        return ((values.reshape(shape) % 17) / 8 - 1).to(dtype).contiguous()
    generator = torch.Generator(device="cuda").manual_seed(case.seed + 1000)
    return (torch.rand(shape, generator=generator, device="cuda") * 2 - 1).to(dtype).contiguous()


def _inputs(case: AddCase, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A, B = make_inputs(
        MatmulCase(case.case_id, case.M, case.N, case.K, case.ab_pattern, case.seed),
        dtype=dtype,
    )
    return A, B, _c_tensor(case, dtype)


def _check(
    case: AddCase,
    output: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    if case.ab_pattern == case.c_pattern == "zeros":
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    else:
        rtol, atol = TOLERANCES[expected.dtype]
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


def correctness(matmul_add, cases: list[AddCase], dtype: torch.dtype, options: dict) -> None:
    for case in cases:
        inputs = _inputs(case, dtype)
        expected = torch.addmm(inputs[2], inputs[0], inputs[1])
        _check(case, matmul_add(*inputs, **options), expected)
        print(f"PASS correctness {case.case_id} {case.M}x{case.N}x{case.K}")


def performance(matmul_add, cases: list[AddCase], dtype: torch.dtype, options: dict) -> None:
    metadata = environment("torch.addmm")
    for case in cases:
        inputs = _inputs(case, dtype)
        expected = torch.addmm(inputs[2], inputs[0], inputs[1])
        ours = lambda: matmul_add(*inputs, **options)
        reference = lambda: torch.addmm(inputs[2], inputs[0], inputs[1])
        prepared, prepare_ms, peak_delta = prepare(ours, inputs[0].device)
        _check(case, prepared, expected)
        reference_ms = benchmark(
            reference, warmup_ms=case.warmup_ms, rep_ms=case.rep_ms
        )
        ours_ms = benchmark(ours, warmup_ms=case.warmup_ms, rep_ms=case.rep_ms)

        operations = 2 * case.M * case.N * case.K + case.M * case.N
        print(
            f"Environment: {metadata['gpu']} {metadata['compute_capability']}, "
            f"Torch {metadata['torch']}, TileLang {metadata['tilelang']}"
        )
        print(f"Prepare: {prepare_ms:.3f} ms; peak allocation delta: {peak_delta} bytes")
        print_table(
            f"case={case.case_id} shape={case.M}x{case.N}x{case.K} dtype={dtype}",
            "torch.addmm",
            reference_ms,
            operations / (reference_ms * 1e-3) / 1e12,
            [("student", ours_ms, operations / (ours_ms * 1e-3) / 1e12)],
        )


def main() -> None:
    args = _args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise SystemExit("02B requires an SM120 CUDA GPU")
    stages, all_cases = load_matmul_add_manifest(HERE / "manifests" / "matmul_add_v1.json")
    stage = stages[args.stage]
    cases = [case for case in all_cases if args.stage in case.stages]
    if args.case_id:
        cases = [case for case in cases if case.case_id == args.case_id]
    if not cases:
        raise SystemExit("no manifest cases matched")
    matmul_add = runpy.run_path(
        str(args.adapter.expanduser().resolve()), run_name="<student-matmul-add>"
    )["matmul_add"]
    dtype = DTYPES[stage.dtype]
    if args.mode in ("correctness", "all"):
        correctness(matmul_add, [c for c in cases if c.mode == "correctness"], dtype, stage.options)
    if args.mode in ("performance", "all"):
        performance(matmul_add, [c for c in cases if c.mode == "performance"], dtype, stage.options)


if __name__ == "__main__":
    main()
