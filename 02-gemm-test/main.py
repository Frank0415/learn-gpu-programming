from __future__ import annotations

import argparse
from pathlib import Path
import runpy

import torch

from cases import MatmulCase, make_inputs
from common import DTYPES, TOLERANCES, benchmark, environment, prepare, print_table
from manifest import FixedCase, load_manifest


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check matmul numerics and compare its FLOP/s with cuBLAS"
    )
    parser.add_argument("implementation", type=Path)
    parser.add_argument(
        "--mode", default="all", choices=("correctness", "performance", "all")
    )
    parser.add_argument("--case-id", help=argparse.SUPPRESS)
    return parser.parse_args()


def _inputs(case: FixedCase, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    return make_inputs(
        MatmulCase(case.case_id, case.M, case.N, case.K, case.pattern, case.seed),
        dtype=dtype,
    )


def _assert_result(
    case: FixedCase,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    performance_case: bool = False,
) -> None:
    if case.pattern == "zeros":
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        rtol, atol = TOLERANCES[expected.dtype]
        if performance_case and expected.dtype == torch.float16:
            rtol = atol = 3e-2
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _correctness(matmul, cases: list[FixedCase], dtype: torch.dtype, options: dict) -> None:
    for case in cases:
        A, B = _inputs(case, dtype)
        _assert_result(case, matmul(A, B, **options), A @ B)
        print(f"PASS {case.case_id} {case.M}x{case.N}x{case.K}")


def _performance(
    matmul, kernel_name: str, cases: list[FixedCase], dtype: torch.dtype, options: dict
) -> None:
    metadata = environment("torch.matmul")
    for case in cases:
        A, B = _inputs(case, dtype)
        ours = lambda: matmul(A, B, **options)
        reference = lambda: torch.matmul(A, B)
        prepared, prepare_ms, _ = prepare(ours, A.device)
        _assert_result(case, prepared, reference(), performance_case=True)
        reference_ms = benchmark(
            reference, warmup_ms=case.warmup_ms, rep_ms=case.rep_ms
        )
        ours_ms = benchmark(ours, warmup_ms=case.warmup_ms, rep_ms=case.rep_ms)
        flops = 2 * case.M * case.N * case.K
        print(
            f"Environment: {metadata['gpu']} {metadata['compute_capability']}, "
            f"Torch {metadata['torch']}, TileLang {metadata['tilelang']}"
        )
        print(f"Prepare: {prepare_ms:.3f} ms")
        print_table(
            f"case={case.case_id} shape={case.M}x{case.N}x{case.K} dtype={dtype}",
            "cuBLAS",
            reference_ms,
            flops / (reference_ms * 1e-3) / 1e12,
            [(kernel_name, ours_ms, flops / (ours_ms * 1e-3) / 1e12)],
        )


def main() -> None:
    args = _args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    namespace = runpy.run_path(
        str(args.implementation.expanduser().resolve()), run_name="<student-matmul>"
    )
    manifest = load_manifest(Path(__file__).parent / "manifests" / "matmul_v1.json")
    matmul = namespace["matmul"]
    config = manifest.config
    dtype = DTYPES[config.input_dtype]
    options = dict(config.options)
    actual_cc = f"sm{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"
    if actual_cc != config.compute_capability:
        raise SystemExit(f"test requires {config.compute_capability}")
    cases = list(manifest.cases)
    if args.case_id:
        cases = [case for case in cases if case.case_id == args.case_id]
    if args.mode in ("correctness", "all"):
        _correctness(matmul, [c for c in cases if c.mode == "correctness"], dtype, options)
    if args.mode in ("performance", "all"):
        _performance(
            matmul,
            "matmul",
            [c for c in cases if c.mode == "performance"],
            dtype,
            options,
        )


if __name__ == "__main__":
    main()
