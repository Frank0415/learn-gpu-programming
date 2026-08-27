from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import math
import statistics
import subprocess
import time
from typing import Callable

import torch


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


@dataclass(frozen=True)
class Timing:
    last: torch.Tensor
    p20_ms: float
    median_ms: float
    p80_ms: float
    reference_median_ms: float


def prepare(call: Callable[[], torch.Tensor], device: torch.device) -> tuple[torch.Tensor, float, int]:
    torch.cuda.reset_peak_memory_stats(device)
    memory_before = torch.cuda.memory_allocated(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    output = call()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1_000
    peak_delta = max(0, torch.cuda.max_memory_allocated(device) - memory_before)
    return output, elapsed_ms, peak_delta


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
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def benchmark_pair(
    ours: Callable[[], torch.Tensor],
    reference: Callable[[], torch.Tensor],
    *,
    warmup: int,
    trials: int,
    samples: int,
) -> Timing:
    for _ in range(warmup):
        ours()
        reference()
    torch.cuda.synchronize()

    ours_trials: list[float] = []
    reference_trials: list[float] = []
    last: torch.Tensor | None = None
    for trial in range(trials):
        ours_samples: list[float] = []
        reference_samples: list[float] = []
        for _ in range(samples):
            ordered = ((ours, ours_samples, True), (reference, reference_samples, False))
            if trial % 2:
                ordered = tuple(reversed(ordered))
            for call, sink, is_ours in ordered:
                output, elapsed = _event(call)
                sink.append(elapsed)
                if is_ours:
                    last = output
        ours_trials.append(statistics.median(ours_samples))
        reference_trials.append(statistics.median(reference_samples))
    assert last is not None
    return Timing(
        last,
        _percentile(ours_trials, 0.2),
        statistics.median(ours_trials),
        _percentile(ours_trials, 0.8),
        statistics.median(reference_trials),
    )


def environment(backend: str) -> dict[str, object]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    driver: str | None = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={torch.cuda.current_device()}",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        driver = result.stdout.strip().splitlines()[0]
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
        "backend": backend,
    }
