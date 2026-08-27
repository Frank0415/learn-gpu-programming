from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Stage:
    name: str
    compute_capability: str
    dtype: str
    options: dict[str, Any]


@dataclass(frozen=True)
class FixedCase:
    case_id: str
    stages: tuple[str, ...]
    mode: str
    M: int
    N: int
    K: int
    pattern: str
    seed: int
    warmup_ms: int
    rep_ms: int


@dataclass(frozen=True)
class MatmulManifest:
    stages: dict[str, Stage]
    cases: tuple[FixedCase, ...]
    performance_policy: dict[str, Any]


def load_manifest(path: str | Path) -> MatmulManifest:
    raw = json.loads(Path(path).read_text())
    if raw["schema_version"] != 1:
        raise ValueError("unsupported Matmul manifest")
    policy = raw["performance_policy"]
    timing = policy["timing"]
    fixed_timing = (timing["warmup_ms"], timing["rep_ms"])
    if fixed_timing != (25, 100):
        raise ValueError("timing must remain 25 ms warmup and 100 ms measurement")

    stages = {
        name: Stage(name, item["compute_capability"], item["dtype"], item["options"])
        for name, item in raw["stages"].items()
    }
    cases = tuple(
        FixedCase(
            item["id"],
            tuple(item["stages"]),
            item["mode"],
            item["M"],
            item["N"],
            item["K"],
            item["pattern"],
            item["seed"],
            *fixed_timing,
        )
        for item in raw["cases"]
    )
    return MatmulManifest(stages, cases, policy)
