from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cases import MatmulCase, make_inputs
from common import DTYPES, TOLERANCES, benchmark, environment, prepare, print_table
from matmul_add_contract import MatmulAdd, load_matmul_add
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
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output: object,
    expected: torch.Tensor,
) -> torch.Tensor:
    A, B, C = inputs
    if not isinstance(output, torch.Tensor):
        raise AssertionError(f"{case.case_id}: output is not a Tensor")
    if output.shape != C.shape or output.dtype != A.dtype or output.device != A.device:
        raise AssertionError(f"{case.case_id}: wrong output metadata")
    if output.data_ptr() in {A.data_ptr(), B.data_ptr(), C.data_ptr()}:
        raise AssertionError(f"{case.case_id}: output aliases an input")
    if case.ab_pattern == case.c_pattern == "zeros":
        if not torch.equal(output, expected):
            raise AssertionError(f"{case.case_id}: zero case must match exactly")
    else:
        rtol, atol = TOLERANCES[A.dtype]
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
    return output


def _probes(adapter: MatmulAdd, case: AddCase, dtype: torch.dtype, options: dict) -> None:
    A, B, C = _inputs(case, dtype)
    bad = dict(options, variant="__unknown_variant__")
    for label, call in (
        ("unknown variant", lambda: adapter(A, B, C, **bad)),
        ("broadcast C", lambda: adapter(A, B, C[:1], **options)),
    ):
        try:
            call()
        except Exception:
            continue
        raise AssertionError(f"adapter accepted {label}")
    print("PASS contract-probes")


def correctness(adapter: MatmulAdd, cases: list[AddCase], dtype: torch.dtype, options: dict) -> None:
    for case in cases:
        inputs = _inputs(case, dtype)
        before = tuple(t.clone() for t in inputs)
        expected = torch.addmm(inputs[2], inputs[0], inputs[1])
        first = _check(case, inputs, adapter(*inputs, **options), expected).clone()
        second = _check(case, inputs, adapter(*inputs, **options), expected)
        if not torch.equal(first, second):
            raise AssertionError(f"{case.case_id}: repeated outputs differ")
        if any(not torch.equal(now, saved) for now, saved in zip(inputs, before)):
            raise AssertionError(f"{case.case_id}: adapter mutated an input")
        print(f"PASS correctness {case.case_id} {case.M}x{case.N}x{case.K}")
    if cases:
        probe_case = next((c for c in cases if c.M > 1), cases[0])
        _probes(adapter, probe_case, dtype, options)


def performance(adapter: MatmulAdd, cases: list[AddCase], dtype: torch.dtype, options: dict) -> None:
    metadata = environment("torch.addmm")
    for case in cases:
        inputs = _inputs(case, dtype)
        before = tuple(t.clone() for t in inputs)
        expected = torch.addmm(inputs[2], inputs[0], inputs[1])
        ours = lambda: adapter(*inputs, **options)
        reference = lambda: torch.addmm(inputs[2], inputs[0], inputs[1])
        prepared, prepare_ms, peak_delta = prepare(ours, inputs[0].device)
        _check(case, inputs, prepared, expected)
        reference_ms = benchmark(
            reference, warmup_ms=case.warmup_ms, rep_ms=case.rep_ms
        )
        ours_ms = benchmark(ours, warmup_ms=case.warmup_ms, rep_ms=case.rep_ms)
        _check(case, inputs, ours(), expected)
        if any(not torch.equal(now, saved) for now, saved in zip(inputs, before)):
            raise AssertionError(f"{case.case_id}: adapter mutated an input")

        changed_c = inputs[2].clone()
        changed_c.view(-1)[0] += 0.25
        changed_inputs = (inputs[0], inputs[1], changed_c)
        changed_expected = torch.addmm(changed_c, inputs[0], inputs[1])
        _check(case, changed_inputs, adapter(*changed_inputs, **options), changed_expected)

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
    adapter = load_matmul_add(args.adapter)
    dtype = DTYPES[stage.dtype]
    if args.mode in ("correctness", "all"):
        correctness(adapter, [c for c in cases if c.mode == "correctness"], dtype, stage.options)
    if args.mode in ("performance", "all"):
        performance(adapter, [c for c in cases if c.mode == "performance"], dtype, stage.options)


if __name__ == "__main__":
    main()
