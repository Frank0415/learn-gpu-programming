from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    compute_capability: str
    input_dtype: str
    options: dict[str, Any]


@dataclass(frozen=True)
class FixedCase:
    case_id: str
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
    config: Config
    cases: tuple[FixedCase, ...]
    performance_policy: dict[str, Any]


def load_manifest(path: str | Path) -> MatmulManifest:
    raw = json.loads(Path(path).read_text())
    timing = raw["performance_policy"]["timing"]
    fixed_timing = (timing["warmup_ms"], timing["rep_ms"])
    config = raw["config"]
    cases = tuple(
        FixedCase(
            item["id"], item["mode"], item["M"], item["N"], item["K"],
            item["pattern"], item["seed"], *fixed_timing,
        )
        for item in raw["cases"]
    )
    return MatmulManifest(
        Config(config["compute_capability"], config["input_dtype"], config["options"]),
        cases,
        raw["performance_policy"],
    )
