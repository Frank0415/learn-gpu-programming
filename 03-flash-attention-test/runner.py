"""Manifest-locked correctness and paired-performance runner.

Student-facing choices are intentionally limited to stage, mode, and adapter.
``--case-id`` exists only as an explicit debugging escape hatch.
"""

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
from typing import Any, Callable

import torch

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


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _event_call(call: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = call()
    end.record()
    end.synchronize()
    return output, float(start.elapsed_time(end))


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
    backend_name = "torch.nn.attention.varlen.varlen_attn"
    for case_index, case in enumerate(cases):
        # Materialize once. Every tested adapter and the library baseline see
        # these exact objects, data pointers, strides, dtype, device, and stream.
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
        library_prepared = library_call()
        _validate_output(case, library_prepared, expected, (q, k, v))

        # Reverse variant order every other case to reduce systematic thermal
        # or frequency bias while preserving CLI order in the final ranking.
        variant_order = adapters if case_index % 2 == 0 else list(reversed(adapters))
        for adapter_name, adapter in variant_order:
            ours_call = lambda adapter=adapter: _invoke(adapter, case, q, k, v, cu)

            # Host wall time plus synchronization includes host-side JIT/lowering
            # and the first GPU work, but remains separate from steady-state data.
            torch.cuda.reset_peak_memory_stats(q.device)
            memory_before = torch.cuda.memory_allocated(q.device)
            torch.cuda.synchronize(q.device)
            prepare_start = time.perf_counter()
            prepared = ours_call()
            torch.cuda.synchronize(q.device)
            prepare_ms = (time.perf_counter() - prepare_start) * 1_000.0
            peak_memory_delta = max(
                0, torch.cuda.max_memory_allocated(q.device) - memory_before
            )
            _validate_output(case, prepared, expected, (q, k, v))
            _assert_inputs_unchanged(case, (q, k, v, cu), before)

            for _ in range(case.warmup):
                ours_call()
                library_call()
            torch.cuda.synchronize()

            ours_trial_medians: list[float] = []
            library_trial_medians: list[float] = []
            last_output = prepared
            for trial in range(case.trials):
                ours_samples: list[float] = []
                library_samples: list[float] = []
                for _ in range(case.samples):
                    ordered = ((ours_call, ours_samples), (library_call, library_samples))
                    if trial % 2:
                        ordered = tuple(reversed(ordered))
                    for call, sink in ordered:
                        result, elapsed = _event_call(call)
                        sink.append(elapsed)
                        if call is ours_call:
                            last_output = result
                ours_trial_medians.append(statistics.median(ours_samples))
                library_trial_medians.append(statistics.median(library_samples))

            _validate_output(case, last_output, expected, (q, k, v))
            _assert_inputs_unchanged(case, (q, k, v, cu), before)

            # A new input after timing catches adapters that cache an output.
            changed_q = q.clone()
            changed_q.view(-1)[0] += 0.25
            changed_expected = fp32_segment_reference(
                changed_q,
                k,
                v,
                cu,
                case.max_seqlen,
                causal=case.causal,
                scale=case.scale,
            )
            changed = _invoke(adapter, case, changed_q, k, v, cu)
            _validate_output(case, changed, changed_expected, (changed_q, k, v))

            ours_ms = statistics.median(ours_trial_medians)
            library_ms = statistics.median(library_trial_medians)
            row = {
                "adapter": adapter_name,
                "case": case.case_id,
                "stage": requested_stage,
                "manifest_stage": case.stage,
                "lengths": list(case.lengths),
                "max_seqlen": case.max_seqlen,
                "hq": case.hq,
                "hkv": case.hkv,
                "head_dim": case.head_dim,
                "dtype": case.dtype_name,
                "causal": case.causal,
                "scale": case.scale,
                "eligibility": case.eligibility,
                "backend": backend_name,
                **environment,
                "prepare_ms": prepare_ms,
                "ours_peak_memory_delta_bytes": peak_memory_delta,
                "ours_p20_ms": _percentile(ours_trial_medians, 0.2),
                "ours_median_ms": ours_ms,
                "ours_p80_ms": _percentile(ours_trial_medians, 0.8),
                "library_median_ms": library_ms,
                "ratio_ours_over_library": ours_ms / library_ms,
                "tokens_per_s": case.total_tokens / (ours_ms / 1000),
                "effective_tflops": case.flops / (ours_ms / 1000) / 1e12,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))

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
    for rank, summary in enumerate(
        sorted(summaries, key=lambda item: (item["core_score"], item["adapter"])),
        start=1,
    ):
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
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
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
    # The student command has no manifest override: packed shapes, GQA cases,
    # tolerances, and the 20x5x50 timing protocol are fixed in the repository.
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
