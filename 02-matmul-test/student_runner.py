"""Manifest-locked student entrypoint for the staged Matmul course."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Callable

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cases import MatmulCase, make_inputs
from contract import Matmul, load_matmul
from manifest import FixedCase, load_manifest


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
TOLERANCES = {
    torch.float32: (1e-4, 1e-5),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (2e-2, 2e-2),
}


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



def _event(call: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = call()
    end.record()
    end.synchronize()
    return output, float(start.elapsed_time(end))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] if low == high else ordered[low] * (high - position) + ordered[high] * (position - low)


def _environment() -> dict[str, object]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    driver: str | None = None
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                f"--id={torch.cuda.current_device()}",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        driver = query.stdout.strip().splitlines()[0] or None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError):
        pass
    try:
        tilelang_version: str | None = version("tilelang")
    except PackageNotFoundError:
        tilelang_version = None
    return {
        "gpu": props.name,
        "compute_capability": f"sm{props.major}{props.minor}",
        "torch": torch.__version__,
        "tilelang": tilelang_version,
        "cuda_runtime": torch.version.cuda,
        "driver": driver,
        "backend": "torch.matmul",
    }


def performance(
    adapter: Matmul,
    cases: list[FixedCase],
    *,
    dtype: torch.dtype,
    options: dict,
) -> None:
    environment = _environment()
    for case in cases:
        A, B = _materialize(case, dtype)
        before_A, before_B = A.clone(), B.clone()
        expected = A @ B
        ours = lambda: adapter(A, B, **options)
        library = lambda: torch.matmul(A, B)

        torch.cuda.reset_peak_memory_stats(A.device)
        memory_before = torch.cuda.memory_allocated(A.device)
        torch.cuda.synchronize(A.device)
        prepare_start = time.perf_counter()
        prepared = ours()
        torch.cuda.synchronize(A.device)
        prepare_ms = (time.perf_counter() - prepare_start) * 1_000.0
        peak_memory_delta = max(
            0, torch.cuda.max_memory_allocated(A.device) - memory_before
        )
        _check(case, A, B, prepared, expected)
        if not torch.equal(A, before_A) or not torch.equal(B, before_B):
            raise AssertionError(f"{case.case_id}: adapter mutated A or B during prepare")

        for _ in range(case.warmup):
            ours()
            library()
        torch.cuda.synchronize()

        ours_trials: list[float] = []
        lib_trials: list[float] = []
        last = prepared
        for trial in range(case.trials):
            ours_samples: list[float] = []
            lib_samples: list[float] = []
            for _ in range(case.samples):
                ordered = ((ours, ours_samples, True), (library, lib_samples, False))
                if trial % 2:
                    ordered = tuple(reversed(ordered))
                for call, sink, is_ours in ordered:
                    output, elapsed = _event(call)
                    sink.append(elapsed)
                    if is_ours:
                        last = output
            ours_trials.append(statistics.median(ours_samples))
            lib_trials.append(statistics.median(lib_samples))
        _check(case, A, B, last, expected)
        if not torch.equal(A, before_A) or not torch.equal(B, before_B):
            raise AssertionError(f"{case.case_id}: adapter mutated A or B during timing")

        changed_A = A.clone()
        changed_A.view(-1)[0] += 0.25
        _check(case, changed_A, B, adapter(changed_A, B, **options), changed_A @ B)

        ours_ms = statistics.median(ours_trials)
        lib_ms = statistics.median(lib_trials)
        result = {
            "case": case.case_id,
            "shape_mnk": [case.M, case.N, case.K],
            "dtype": str(dtype).removeprefix("torch."),
            "prepare_ms": prepare_ms,
            "ours_peak_memory_delta_bytes": peak_memory_delta,
            "p20_ms": _percentile(ours_trials, 0.2),
            "median_ms": ours_ms,
            "p80_ms": _percentile(ours_trials, 0.8),
            "torch_median_ms": lib_ms,
            "ratio_ours_over_torch": ours_ms / lib_ms,
            "tflops": (2 * case.M * case.N * case.K) / (ours_ms / 1000) / 1e12,
            **environment,
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
