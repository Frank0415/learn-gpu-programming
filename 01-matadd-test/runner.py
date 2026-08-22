"""Run shared correctness cases and optional benchmarks for any matadd adapter."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
from typing import Any

import torch

from cases import MataddCase, cases_for, make_inputs
from contract import Matadd, load_matadd


@dataclass(frozen=True)
class Configuration:
    name: str
    options: dict[str, Any]


def parse_shape(value: str) -> tuple[int, int]:
    normalized = value.lower().replace("×", "x")
    pieces = normalized.split("x")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("shape must look like 4096x4096")
    try:
        shape = (int(pieces[0]), int(pieces[1]))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape dimensions must be integers") from error
    if shape[0] <= 0 or shape[1] <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return shape


def parse_options(value: str) -> dict[str, Any]:
    try:
        options = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON options: {error}") from error
    if not isinstance(options, dict):
        raise argparse.ArgumentTypeError("options must be one JSON object")
    return options


def load_suite(path: Path) -> tuple[Configuration, ...]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("suite must be a non-empty JSON list")

    configurations = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"suite item {index} must be an object")
        name = item.get("name")
        options = item.get("options")
        if not isinstance(name, str) or not name:
            raise ValueError(f"suite item {index} requires a non-empty name")
        if not isinstance(options, dict):
            raise ValueError(f"suite item {index} requires an options object")
        configurations.append(Configuration(name=name, options=options))
    return tuple(configurations)


def validate_result(
    *,
    case_name: str,
    A: torch.Tensor,
    B: torch.Tensor,
    actual: object,
) -> torch.Tensor:
    if not isinstance(actual, torch.Tensor):
        raise AssertionError(f"{case_name}: adapter returned {type(actual).__name__}, not Tensor")
    if actual.shape != A.shape:
        raise AssertionError(f"{case_name}: output shape {actual.shape} != input shape {A.shape}")
    if actual.dtype != A.dtype:
        raise AssertionError(f"{case_name}: output dtype {actual.dtype} != {A.dtype}")
    if actual.device != A.device:
        raise AssertionError(f"{case_name}: output device {actual.device} != {A.device}")

    torch.testing.assert_close(actual, A + B, rtol=1e-5, atol=1e-6)
    return actual


def run_correctness(
    matadd: Matadd,
    configuration: Configuration,
    *,
    device: torch.device,
    consistency: int,
    selected_case: str | None,
) -> int:
    cases = cases_for(configuration.options)
    if selected_case is not None:
        cases = tuple(case for case in cases if case.name == selected_case)
        if not cases:
            names = ", ".join(case.name for case in cases_for(configuration.options))
            raise ValueError(f"unknown case {selected_case!r}; choose one of: {names}")

    for case in cases:
        A, B = make_inputs(case, device=device)
        A_before = A.clone()
        B_before = B.clone()
        expected_first: torch.Tensor | None = None

        for _ in range(consistency):
            actual = validate_result(
                case_name=case.name,
                A=A,
                B=B,
                actual=matadd(A, B, **configuration.options),
            )
            if expected_first is None:
                expected_first = actual.clone()
            else:
                torch.testing.assert_close(actual, expected_first, rtol=0, atol=0)

        if not torch.equal(A, A_before) or not torch.equal(B, B_before):
            raise AssertionError(f"{case.name}: adapter mutated an input tensor")

        print(f"  PASS {case.name:<18} shape={case.shape}")

    return len(cases)


def benchmark(
    matadd: Matadd,
    configuration: Configuration,
    *,
    A: torch.Tensor,
    B: torch.Tensor,
    warmup: int,
    repeat: int,
) -> tuple[float, float]:
    validate_result(
        case_name="benchmark",
        A=A,
        B=B,
        actual=matadd(A, B, **configuration.options),
    )

    for _ in range(warmup):
        matadd(A, B, **configuration.options)
    torch.cuda.synchronize(A.device)

    elapsed_ms = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        start.record()
        matadd(A, B, **configuration.options)
        end.record()
        end.synchronize()
        elapsed_ms.append(start.elapsed_time(end))

    median_ms = median(elapsed_ms)
    requested_bytes = 3 * A.numel() * A.element_size()
    effective_gb_s = requested_bytes / (median_ms * 1e-3) / 1e9
    return median_ms, effective_gb_s


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test any adapter exporting matadd(A, B, **options) -> C"
    )
    parser.add_argument("--adapter", required=True, type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--options", type=parse_options, default={})
    group.add_argument("--suite", type=Path)
    parser.add_argument("--case", dest="selected_case")
    parser.add_argument("--consistency", type=int, default=3)
    parser.add_argument("--benchmark", type=parse_shape)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.consistency <= 0 or args.warmup < 0 or args.repeat <= 0:
        raise ValueError("consistency and repeat must be positive; warmup cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("the current test harness requires a CUDA device")

    matadd = load_matadd(args.adapter)
    configurations = (
        load_suite(args.suite)
        if args.suite is not None
        else (Configuration(name="command-line", options=args.options),)
    )
    device = torch.device("cuda")
    total_cases = 0
    benchmark_inputs: tuple[torch.Tensor, torch.Tensor] | None = None
    if args.benchmark is not None:
        benchmark_case = MataddCase("benchmark", args.benchmark, "normal", seed=17)
        benchmark_inputs = make_inputs(benchmark_case, device=device)

    print(f"adapter: {args.adapter.resolve()}")
    if benchmark_inputs is not None:
        benchmark_A, benchmark_B = benchmark_inputs
        print(
            f"benchmark inputs: shape={tuple(benchmark_A.shape)} "
            f"A_ptr=0x{benchmark_A.data_ptr():x} B_ptr=0x{benchmark_B.data_ptr():x} "
            "(reused by every configuration)"
        )
    for configuration in configurations:
        print(f"\n[{configuration.name}] options={configuration.options}")
        total_cases += run_correctness(
            matadd,
            configuration,
            device=device,
            consistency=args.consistency,
            selected_case=args.selected_case,
        )
        if benchmark_inputs is not None:
            benchmark_A, benchmark_B = benchmark_inputs
            median_ms, effective_gb_s = benchmark(
                matadd,
                configuration,
                A=benchmark_A,
                B=benchmark_B,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            print(
                f"  BENCH shape={tuple(benchmark_A.shape)} median={median_ms:.4f} ms "
                f"effective={effective_gb_s:.1f} GB/s"
            )

    print(f"\nALL PASS: {total_cases} correctness cases")


if __name__ == "__main__":
    main()
