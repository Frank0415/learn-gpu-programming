from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AddStage:
    name: str
    dtype: str
    options: dict[str, Any]


@dataclass(frozen=True)
class AddCase:
    case_id: str
    stages: tuple[str, ...]
    mode: str
    M: int
    N: int
    K: int
    ab_pattern: str
    c_pattern: str
    seed: int
    warmup_ms: int
    rep_ms: int


def load_matmul_add_manifest(path: str | Path) -> tuple[dict[str, AddStage], tuple[AddCase, ...]]:
    raw = json.loads(Path(path).read_text())
    if raw["schema_version"] != 1:
        raise ValueError("unsupported MatmulAdd manifest")
    timing = raw["performance_policy"]["timing"]
    fixed = (timing["warmup_ms"], timing["rep_ms"])
    if fixed != (25, 100):
        raise ValueError("timing must remain 25 ms warmup and 100 ms measurement")
    stages = {
        name: AddStage(name, item["dtype"], item["options"])
        for name, item in raw["stages"].items()
    }
    cases = tuple(
        AddCase(
            item["id"], tuple(item["stages"]), item["mode"],
            item["M"], item["N"], item["K"],
            item["ab_pattern"], item["c_pattern"], item["seed"], *fixed,
        )
        for item in raw["cases"]
    )
    return stages, cases
