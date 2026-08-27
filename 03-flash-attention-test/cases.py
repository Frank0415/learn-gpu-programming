from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch


DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass(frozen=True)
class AttentionCase:
    case_id: str
    stage: str
    lengths: tuple[int, ...]
    hq: int
    hkv: int
    head_dim: int
    dtype_name: str
    seed: int
    causal: bool
    scale: float | None
    options: dict[str, Any]
    rtol: float
    atol: float
    eligibility: str
    modes: tuple[str, ...]
    warmup_ms: int
    rep_ms: int
    input_multiplier: float
    checks: tuple[str, ...]
    also_run_on: tuple[str, ...]

    @property
    def total_tokens(self) -> int:
        return sum(self.lengths)

    @property
    def max_seqlen(self) -> int:
        return max(self.lengths)

    @property
    def dtype(self) -> torch.dtype:
        return DTYPES[self.dtype_name]

    @property
    def flops(self) -> int:
        return 4 * self.hq * sum(n * n for n in self.lengths) * self.head_dim


@dataclass(frozen=True)
class Manifest:
    performance_policy: dict[str, Any]
    stage_rules: dict[str, Any]
    cases: tuple[AttentionCase, ...]


def load_manifest(path: str | Path) -> Manifest:
    raw = json.loads(Path(path).read_text())
    if raw["schema_version"] != 1:
        raise ValueError("unsupported FlashAttention manifest")
    policy = raw["performance_policy"]
    timing = policy["timing"]
    fixed_timing = (timing["warmup_ms"], timing["rep_ms"])
    if fixed_timing != (25, 100):
        raise ValueError("timing must remain 25 ms warmup and 100 ms measurement")

    cases = tuple(
        AttentionCase(
            case_id=item["id"],
            stage=item["stage"].lower(),
            lengths=tuple(item["lengths"]),
            hq=item["hq"],
            hkv=item["hkv"],
            head_dim=item["head_dim"],
            dtype_name=item["dtype"],
            seed=item["seed"],
            causal=item["causal"],
            scale=item.get("scale"),
            options=item.get("options", {}),
            rtol=item["tolerance"]["rtol"],
            atol=item["tolerance"]["atol"],
            eligibility=item["eligibility"],
            modes=tuple(item["modes"]),
            warmup_ms=fixed_timing[0],
            rep_ms=fixed_timing[1],
            input_multiplier=item.get("input_multiplier", 1.0),
            checks=tuple(item.get("checks", [])),
            also_run_on=tuple(item.get("also_run_on", [])),
        )
        for item in raw["cases"]
    )
    return Manifest(policy, raw["stage_rules"], cases)


def make_inputs(
    case: AttentionCase, device: str | torch.device = "cuda"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.device(device)
    generator = torch.Generator(device=target).manual_seed(case.seed)

    def values(heads: int) -> torch.Tensor:
        shape = (case.total_tokens, heads, case.head_dim)
        return (
            torch.randn(shape, generator=generator, device=target)
            * case.input_multiplier
        ).to(case.dtype).contiguous()

    cu = torch.tensor(
        [0, *torch.tensor(case.lengths).cumsum(0).tolist()],
        device=target,
        dtype=torch.int32,
    )
    return values(case.hq), values(case.hkv), values(case.hkv), cu
