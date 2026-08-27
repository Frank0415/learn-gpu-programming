"""Manifest-locked correctness and gau-style median benchmark runner.

Student-facing choices are intentionally limited to stage, mode, and adapter.
``--case-id`` exists only as an explicit debugging escape hatch.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

import torch
from triton.testing import do_bench

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cases import AttentionCase, load_manifest, make_inputs
from contract import AttentionForward, load_attention_forward
from reference import fp32_segment_reference, torch_varlen_available, torch_varlen_reference


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed FlashAttention 03 correctness/performance gate"
    )
    parser.add_argument("--stage", required=True, choices=("03a", "03b", "03c", "03d"))
    parser.add_argument("--mode", required=True, choices=("correctness", "performance", "all"))
    parser.add_argument(
        "--adapter",
        required=True,
        action="append",
        metavar="[NAME=]PATH",
        help="adapter path; repeat to compare eligible variants on identical inputs",
    )
    parser.add_argument("--case-id", help=argparse.SUPPRESS)
    return parser.parse_args()


def _load_adapters(values: list[str]) -> list[tuple[str, AttentionForward]]:
    adapters: list[tuple[str, AttentionForward]] = []
    seen: set[str] = set()
    for value in values:
        if "=" in value:
            label, path_text = value.split("=", 1)
            if not label or not path_text:
                raise ValueError("--adapter NAME=PATH requires both a name and a path")
            path = Path(path_text)
        else:
            path = Path(value)
            label = path.parent.name if path.stem == "main" else path.stem
        if label in seen:
            raise ValueError(
                f"duplicate adapter name {label!r}; use distinct NAME=PATH labels"
            )
        seen.add(label)
        adapters.append((label, load_attention_forward(path)))
    return adapters


def _invoke(
    adapter: AttentionForward,
    case: AttentionCase,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu: torch.Tensor,
) -> torch.Tensor:
    return adapter(
        q,
        k,
        v,
        cu,
        case.max_seqlen,
        causal=case.causal,
        scale=case.scale,
        **case.options,
    )


def _validate_output(
    case: AttentionCase,
    output: torch.Tensor,
    expected: torch.Tensor,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    if not isinstance(output, torch.Tensor):
        raise AssertionError(f"{case.case_id}: output must be a Tensor")
    wanted = (case.total_tokens, case.hq, case.head_dim)
    if tuple(output.shape) != wanted:
        raise AssertionError(f"{case.case_id}: shape {tuple(output.shape)} != {wanted}")
    if output.dtype != inputs[0].dtype or output.device != inputs[0].device:
        raise AssertionError(f"{case.case_id}: output must preserve q dtype and device")
    if any(output.data_ptr() == tensor.data_ptr() for tensor in inputs):
        raise AssertionError(f"{case.case_id}: output aliases an input")
    if not torch.isfinite(output).all():
        raise AssertionError(f"{case.case_id}: output contains NaN or Inf")
    torch.testing.assert_close(output, expected, rtol=case.rtol, atol=case.atol)


def _assert_inputs_unchanged(
    case: AttentionCase,
    current: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    before: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    for actual, saved, name in zip(current, before, ("q", "k", "v", "cu_seqlens")):
        if not torch.equal(actual, saved):
            raise AssertionError(f"{case.case_id}: adapter mutated {name}")


def _run_relational_checks(
    adapter: AttentionForward,
    case: AttentionCase,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu: torch.Tensor,
    baseline: torch.Tensor,
) -> None:
    boundaries = cu.cpu().tolist()
    if case.causal:
        changed_v = v.clone()
        for start, end in zip(boundaries, boundaries[1:]):
            changed_v[end - 1].add_(7)
        changed = _invoke(adapter, case, q, k, changed_v, cu)
        for start, end in zip(boundaries, boundaries[1:]):
            if end - start > 1:
                torch.testing.assert_close(
                    changed[start : end - 1],
                    baseline[start : end - 1],
                    rtol=0,
                    atol=0,
                )
    if "segment_isolation" in case.checks and len(boundaries) > 2:
        changed_q, changed_k, changed_v = q.clone(), k.clone(), v.clone()
        start, end = boundaries[1], boundaries[2]
        changed_q[start:end].add_(3)
        changed_k[start:end].sub_(2)
        changed_v[start:end].mul_(-1)
        changed = _invoke(adapter, case, changed_q, changed_k, changed_v, cu)
        first_end = boundaries[1]
        torch.testing.assert_close(
            changed[:first_end], baseline[:first_end], rtol=0, atol=0
        )
    if "kv_head_sentinel" in case.checks and case.hq != case.hkv:
        sentinel_v = torch.zeros_like(v)
        for kv_head in range(case.hkv):
            sentinel_v[:, kv_head, :].fill_(kv_head + 1)
        output = _invoke(adapter, case, q, k, sentinel_v, cu)
        group = case.hq // case.hkv
        for q_head in range(case.hq):
            expected_value = q_head // group + 1
            expected = torch.full_like(output[:, q_head], expected_value)
            torch.testing.assert_close(output[:, q_head], expected, rtol=0, atol=0)


def _expect_error(label: str, call: Callable[[], Any]) -> None:
    try:
        call()
    except Exception:
        return
    raise AssertionError(f"negative contract probe did not fail: {label}")


def _negative_contract_probes(adapter: AttentionForward, case: AttentionCase) -> None:
    q, k, v, cu = make_inputs(case)
    base = dict(causal=case.causal, scale=case.scale)
    _expect_error(
        "rank",
        lambda: adapter(q[:, 0], k, v, cu, case.max_seqlen, **base),
    )
    _expect_error(
        "k/v head mismatch",
        lambda: adapter(q, k[:, :1], v, cu, case.max_seqlen, **base),
    )
    invalid_q = q[:, : max(1, q.shape[1] - 1)].contiguous()
    _expect_error(
        "non-divisible GQA head ratio",
        lambda: adapter(invalid_q, k, v, cu, case.max_seqlen, **base),
    )
    _expect_error(
        "q token-count mismatch",
        lambda: adapter(q[:-1], k, v, cu, case.max_seqlen, **base),
    )
    _expect_error(
        "dtype mismatch",
        lambda: adapter(q, k.float(), v, cu, case.max_seqlen, **base),
    )
    _expect_error(
        "q on CPU / device mismatch",
        lambda: adapter(q.cpu(), k, v, cu, case.max_seqlen, **base),
    )
    padded_q = torch.empty(
        (*q.shape[:-1], q.shape[-1] * 2), device=q.device, dtype=q.dtype
    )
    noncontiguous_q = padded_q[..., ::2]
    noncontiguous_q.copy_(q)
    _expect_error(
        "non-contiguous q",
        lambda: adapter(noncontiguous_q, k, v, cu, case.max_seqlen, **base),
    )
    _expect_error(
        "cu_seqlens on CPU",
        lambda: adapter(q, k, v, cu.cpu(), case.max_seqlen, **base),
    )
    _expect_error(
        "cu_seqlens dtype",
        lambda: adapter(q, k, v, cu.to(torch.int64), case.max_seqlen, **base),
    )
    _expect_error(
        "cu_seqlens rank",
        lambda: adapter(q, k, v, cu.unsqueeze(0), case.max_seqlen, **base),
    )
    padded_cu = torch.empty(
        (cu.numel(), 2), device=cu.device, dtype=cu.dtype
    )
    noncontiguous_cu = padded_cu[:, 0]
    noncontiguous_cu.copy_(cu)
    _expect_error(
        "non-contiguous cu_seqlens",
        lambda: adapter(q, k, v, noncontiguous_cu, case.max_seqlen, **base),
    )
    bad_start = cu.clone()
    bad_start[0] = 1
    _expect_error(
        "cu_seqlens start",
        lambda: adapter(q, k, v, bad_start, case.max_seqlen, **base),
    )
    empty_request = torch.cat((cu[:1], cu[:1], cu[1:])).contiguous()
    _expect_error(
        "empty request / non-strict cu_seqlens",
        lambda: adapter(q, k, v, empty_request, case.max_seqlen, **base),
    )
    bad_cu = cu.clone()
    bad_cu[-1] -= 1
    _expect_error(
        "cu_seqlens end",
        lambda: adapter(q, k, v, bad_cu, case.max_seqlen, **base),
    )
    _expect_error(
        "max_seqlen",
        lambda: adapter(q, k, v, cu, case.max_seqlen + 1, **base),
    )
    _expect_error(
        "unknown option",
        lambda: adapter(
            q,
            k,
            v,
            cu,
            case.max_seqlen,
            **base,
            variant="__manifest_unknown_variant__",
        ),
    )


def run_correctness(adapter: AttentionForward, cases: list[AttentionCase]) -> None:
    if not cases:
        print("NO correctness cases selected")
        return
    for case in cases:
        q, k, v, cu = make_inputs(case)
        before = tuple(tensor.clone() for tensor in (q, k, v, cu))
        expected = fp32_segment_reference(
            q,
            k,
            v,
            cu,
            case.max_seqlen,
            causal=case.causal,
            scale=case.scale,
        )
        output = _invoke(adapter, case, q, k, v, cu)
        _validate_output(case, output, expected, (q, k, v))
        _assert_inputs_unchanged(case, (q, k, v, cu), before)
        _run_relational_checks(adapter, case, q, k, v, cu, output)
        print(f"PASS correctness {case.case_id}")

    # Run malformed-ABI probes once using the smallest manifest case.
    smallest = min(cases, key=lambda item: item.total_tokens * item.hq * item.head_dim)
    _negative_contract_probes(adapter, smallest)
    print("PASS negative-contract-probes")


def _bench(call: Callable[[], torch.Tensor], case: AttentionCase) -> float:
    return float(
        do_bench(
            call,
            warmup=case.warmup_ms,
            rep=case.rep_ms,
            return_mode="median",
        )
    )


def _environment() -> dict[str, Any]:
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
    }


def run_performance(
    adapters: list[tuple[str, AttentionForward]],
    cases: list[AttentionCase],
    policy: dict[str, Any],
    requested_stage: str,
) -> None:
    if not cases:
        print("NO performance cases selected")
        return
    # The baseline is part of the performance contract. A missing PyTorch
    # varlen backend blocks every core comparison; it never authorizes a dense
    # or padded substitute.
    if not torch_varlen_available():
        core_blocked = False
        for case in cases:
            status = "blocked" if case.eligibility == "core" else "N/A"
            core_blocked = core_blocked or status == "blocked"
            print(
                json.dumps(
                    {
                        "case": case.case_id,
                        "eligibility": case.eligibility,
                        "status": status,
                        "reason": "torch.nn.attention.varlen.varlen_attn unavailable",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        if core_blocked:
            raise RuntimeError(
                "core performance gate blocked: PyTorch 2.13 varlen baseline unavailable"
            )
        return

    rows: list[dict[str, Any]] = []
    environment = _environment()
    backend_name = "PyTorch varlen"
    for case in cases:
        q, k, v, cu = make_inputs(case)
        before = tuple(tensor.clone() for tensor in (q, k, v, cu))
        expected = fp32_segment_reference(
            q, k, v, cu, case.max_seqlen, causal=case.causal, scale=case.scale
        )
        library_call = lambda: torch_varlen_reference(
            q,
            k,
            v,
            cu,
            case.max_seqlen,
            causal=case.causal,
            scale=case.scale,
        )
        _validate_output(case, library_call(), expected, (q, k, v))
        library_ms = _bench(library_call, case)
        case_rows: list[dict[str, Any]] = []

        for adapter_name, adapter in adapters:
            ours_call = lambda adapter=adapter: _invoke(adapter, case, q, k, v, cu)
            torch.cuda.reset_peak_memory_stats(q.device)
            memory_before = torch.cuda.memory_allocated(q.device)
            torch.cuda.synchronize(q.device)
            prepare_start = time.perf_counter()
            prepared = ours_call()
            torch.cuda.synchronize(q.device)
            prepare_ms = (time.perf_counter() - prepare_start) * 1_000
            peak_memory_delta = max(
                0, torch.cuda.max_memory_allocated(q.device) - memory_before
            )
            _validate_output(case, prepared, expected, (q, k, v))
            _assert_inputs_unchanged(case, (q, k, v, cu), before)

            ours_ms = _bench(ours_call, case)
            _validate_output(case, ours_call(), expected, (q, k, v))
            _assert_inputs_unchanged(case, (q, k, v, cu), before)

            changed_q = q.clone()
            changed_q.view(-1)[0] += 0.25
            changed_expected = fp32_segment_reference(
                changed_q, k, v, cu, case.max_seqlen,
                causal=case.causal, scale=case.scale,
            )
            changed = _invoke(adapter, case, changed_q, k, v, cu)
            _validate_output(case, changed, changed_expected, (changed_q, k, v))

            row = {
                "adapter": adapter_name,
                "case": case.case_id,
                "eligibility": case.eligibility,
                "duration_ms": ours_ms,
                "library_ms": library_ms,
                "ratio_ours_over_library": ours_ms / library_ms,
                "effective_tflops": case.flops / (ours_ms * 1e-3) / 1e12,
                "prepare_ms": prepare_ms,
                "peak_memory_delta": peak_memory_delta,
            }
            rows.append(row)
            case_rows.append(row)

        print(
            f"\ncase={case.case_id} lengths={list(case.lengths)} "
            f"Hq/Hkv={case.hq}/{case.hkv} D={case.head_dim} "
            f"dtype={case.dtype_name} causal={case.causal}"
        )
        print(f"Environment: {environment['gpu']} {environment['compute_capability']}")
        print(f"| Kernel | Duration (ms) | TFLOP/s | % of {backend_name} |")
        print("|---|---:|---:|---:|")
        library_tflops = case.flops / (library_ms * 1e-3) / 1e12
        print(f"| {backend_name} | {library_ms:.4f} | {library_tflops:.4f} | 100.00% |")
        for row in case_rows:
            percent = library_ms / row["duration_ms"] * 100
            print(
                f"| {row['adapter']} | {row['duration_ms']:.4f} | "
                f"{row['effective_tflops']:.4f} | {percent:.2f}% |"
            )

    thresholds = policy["thresholds"]
    summaries: list[dict[str, Any]] = []
    for adapter_name, _adapter in adapters:
        core = [
            row["ratio_ours_over_library"]
            for row in rows
            if row["adapter"] == adapter_name and row["eligibility"] == "core"
        ]
        if core:
            summaries.append(
                {
                    "adapter": adapter_name,
                    "core_score": statistics.median(core),
                    "core_worst": max(core),
                }
            )

    if not summaries:
        return
    best_score = min(item["core_score"] for item in summaries)
    tie_fraction = float(thresholds["tie_fraction"])
    gate_applied = requested_stage == "03c"
    failed: list[str] = []
    ranked = list(enumerate(
        sorted(summaries, key=lambda item: (item["core_score"], item["adapter"])),
        start=1,
    ))
    print(f"\n{requested_stage} core ranking")
    print("| Rank | Adapter | Median ours/library | Worst | Tied best | Gate |")
    print("|---:|---|---:|---:|---|---|")
    for rank, summary in ranked:
        summary.update(
            {
                "type": "adapter_ranking",
                "rank": rank,
                "stage": requested_stage,
                "tied_best": summary["core_score"] <= best_score * (1 + tie_fraction),
                "tie_fraction": tie_fraction,
                "near_library_gate_applied": gate_applied,
                "environment": environment,
                "backend": backend_name,
            }
        )
        gate = "applied" if gate_applied else "not applied"
        print(
            f"| {rank} | {summary['adapter']} | {summary['core_score']:.4f} | "
            f"{summary['core_worst']:.4f} | {summary['tied_best']} | {gate} |"
        )
        if gate_applied and (
            summary["core_score"] > thresholds["median_ratio_max"]
            or summary["core_worst"] > thresholds["worst_ratio_max"]
        ):
            failed.append(summary["adapter"])
    if failed:
        raise AssertionError(
            "performance gate failed for adapter(s): " + ", ".join(failed)
        )


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required by the fixed attention contract")
    # Packed shapes, tolerances, and do_bench time budgets are fixed by manifest.
    manifest = load_manifest(HERE / "manifests" / "flash_attention_v1.json")
    expected_cc = manifest.stage_rules[args.stage]["compute_capability"]
    actual_cc = f"sm{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"
    if expected_cc != actual_cc:
        raise SystemExit(f"stage {args.stage} requires {expected_cc}; current GPU is {actual_cc}")
    device_name = torch.cuda.get_device_properties(torch.cuda.current_device()).name
    if args.stage == "03d" and "H200" not in device_name.upper():
        raise SystemExit(
            f"stage 03d is the H200 migration gate; current GPU is {device_name}"
        )
    selected = [
        case
        for case in manifest.cases
        if case.stage == args.stage or args.stage in case.also_run_on
    ]
    if args.case_id:
        selected = [case for case in selected if case.case_id == args.case_id]
    if not selected:
        raise SystemExit("no manifest cases matched stage/case-id")
    adapters = _load_adapters(args.adapter)
    if args.mode in ("correctness", "all"):
        correctness_cases = [case for case in selected if "correctness" in case.modes]
        for adapter_name, adapter in adapters:
            print(f"[adapter {adapter_name}] correctness")
            run_correctness(adapter, correctness_cases)
    if args.mode in ("performance", "all"):
        run_performance(
            adapters,
            [case for case in selected if "performance" in case.modes],
            manifest.performance_policy,
            args.stage,
        )


if __name__ == "__main__":
    main()
