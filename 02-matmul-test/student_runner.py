"""Manifest-locked student entrypoint for the staged Matmul course."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cases import MatmulCase, make_inputs
from common import DTYPES, TOLERANCES, benchmark_pair, environment, prepare
from contract import Matmul, load_matmul
from manifest import FixedCase, load_manifest


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed Matmul learning-stage gate")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("naive-fp32", "shared-fp32", "tensorcore", "pipeline", "final"),
    )
    parser.add_argument("--mode", required=True, choices=("correctness", "performance", "all"))
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--case-id", help=argparse.SUPPRESS)
    return parser.parse_args()


def _materialize(case: FixedCase, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    return make_inputs(
        MatmulCase(case.case_id, case.M, case.N, case.K, case.pattern, case.seed),
        device="cuda",
        dtype=dtype,
    )


def _check(
    case: FixedCase,
    A: torch.Tensor,
    B: torch.Tensor,
    output: object,
    expected: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise AssertionError(f"{case.case_id}: output is not a Tensor")
    if output.shape != (case.M, case.N):
        raise AssertionError(f"{case.case_id}: wrong output shape {tuple(output.shape)}")
    if output.dtype != A.dtype or output.device != A.device:
        raise AssertionError(f"{case.case_id}: output must preserve dtype and device")
    if output.data_ptr() in (A.data_ptr(), B.data_ptr()):
        raise AssertionError(f"{case.case_id}: output aliases an input")
    if case.pattern == "zeros":
        if not torch.equal(output, expected):
            raise AssertionError(f"{case.case_id}: zero case must match exactly")
    else:
        rtol, atol = TOLERANCES[A.dtype]
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
    return output


def _unknown_variant_probe(
    adapter: Matmul,
    case: FixedCase,
    *,
    dtype: torch.dtype,
    options: dict,
) -> None:
    """Reject adapters that silently fall back for an unknown variant."""

    A, B = _materialize(case, dtype)
    bad = dict(options)
    bad["variant"] = "__manifest_unknown_variant__"
    try:
        adapter(A, B, **bad)
    except Exception:
        print("PASS unknown-variant-probe")
    else:
        raise AssertionError("adapter silently accepted an unknown variant")


def correctness(
    adapter: Matmul,
    cases: list[FixedCase],
    *,
    dtype: torch.dtype,
    options: dict,
) -> None:
    for case in cases:
        A, B = _materialize(case, dtype)
        before_A, before_B = A.clone(), B.clone()
        expected = A @ B
        first = _check(case, A, B, adapter(A, B, **options), expected).clone()
        second = _check(case, A, B, adapter(A, B, **options), expected)
        if not torch.equal(first, second):
            raise AssertionError(f"{case.case_id}: repeated outputs differ")
        if not torch.equal(A, before_A) or not torch.equal(B, before_B):
            raise AssertionError(f"{case.case_id}: adapter mutated A or B")
        print(f"PASS correctness {case.case_id} {case.M}x{case.N}x{case.K}")

def performance(
    adapter: Matmul,
    cases: list[FixedCase],
    *,
    dtype: torch.dtype,
    options: dict,
) -> None:
    metadata = environment("torch.matmul")
    for case in cases:
        A, B = _materialize(case, dtype)
        before_A, before_B = A.clone(), B.clone()
        expected = A @ B
        ours = lambda: adapter(A, B, **options)
        library = lambda: torch.matmul(A, B)

        prepared, prepare_ms, peak_memory_delta = prepare(ours, A.device)
        _check(case, A, B, prepared, expected)
        if not torch.equal(A, before_A) or not torch.equal(B, before_B):
            raise AssertionError(f"{case.case_id}: adapter mutated A or B during prepare")

        timing = benchmark_pair(
            ours,
            library,
            warmup=case.warmup,
            trials=case.trials,
            samples=case.samples,
        )
        _check(case, A, B, timing.last, expected)
        if not torch.equal(A, before_A) or not torch.equal(B, before_B):
            raise AssertionError(f"{case.case_id}: adapter mutated A or B during timing")

        changed_A = A.clone()
        changed_A.view(-1)[0] += 0.25
        _check(case, changed_A, B, adapter(changed_A, B, **options), changed_A @ B)

        result = {
            "case": case.case_id,
            "shape_mnk": [case.M, case.N, case.K],
            "dtype": str(dtype).removeprefix("torch."),
            "prepare_ms": prepare_ms,
            "ours_peak_memory_delta_bytes": peak_memory_delta,
            "p20_ms": timing.p20_ms,
            "median_ms": timing.median_ms,
            "p80_ms": timing.p80_ms,
            "torch_median_ms": timing.reference_median_ms,
            "ratio_ours_over_torch": timing.median_ms / timing.reference_median_ms,
            "tflops": (2 * case.M * case.N * case.K) / (timing.median_ms / 1000) / 1e12,
            **metadata,
        }
        print(json.dumps(result, sort_keys=True))


def main() -> None:
    args = _args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    # The student command has no manifest override: shape, dtype, tile
    # options, and timing are part of the checked-in course protocol.
    manifest = load_manifest(HERE / "manifests" / "matmul_v1.json")
    stage = manifest.stages[args.stage]
    capability = f"sm{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"
    if capability != stage.compute_capability:
        raise SystemExit(f"stage {stage.name} requires {stage.compute_capability}; current GPU is {capability}")
    cases = [case for case in manifest.cases if args.stage in case.stages]
    if args.case_id:
        cases = [case for case in cases if case.case_id == args.case_id]
    if not cases:
        raise SystemExit("no manifest cases matched")
    adapter = load_matmul(args.adapter)
    dtype = DTYPES[stage.dtype]
    if args.mode in ("correctness", "all"):
        correctness_cases = [case for case in cases if case.mode == "correctness"]
        correctness(adapter, correctness_cases, dtype=dtype, options=stage.options)
        if correctness_cases:
            _unknown_variant_probe(
                adapter,
                min(correctness_cases, key=lambda item: item.M * item.N * item.K),
                dtype=dtype,
                options=stage.options,
            )
    if args.mode in ("performance", "all"):
        performance(adapter, [case for case in cases if case.mode == "performance"], dtype=dtype, options=stage.options)


if __name__ == "__main__":
    main()
