from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import subprocess
import time
from typing import Callable

import torch
from triton.testing import do_bench


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


def benchmark(call: Callable[[], torch.Tensor], *, warmup_ms: int, rep_ms: int) -> float:
    return float(
        do_bench(call, warmup=warmup_ms, rep=rep_ms, return_mode="median")
    )


def print_table(
    title: str,
    reference_name: str,
    reference_ms: float,
    reference_tflops: float,
    rows: list[tuple[str, float, float]],
) -> None:
    print(f"\n{title}")
    print(f"| Kernel | Duration (ms) | TFLOP/s | % of {reference_name} |")
    print("|---|---:|---:|---:|")
    print(
        f"| {reference_name} | {reference_ms:.4f} | "
        f"{reference_tflops:.4f} | 100.00% |"
    )
    for name, duration_ms, tflops in rows:
        percent = reference_ms / duration_ms * 100
        print(f"| {name} | {duration_ms:.4f} | {tflops:.4f} | {percent:.2f}% |")


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
