from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy

import torch
from triton.testing import do_bench


HERE = Path(__file__).resolve().parent
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=HERE / "performance_configs.json")
    parser.add_argument("--config")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--mode", choices=("benchmark", "profile"), default="benchmark")
    return parser.parse_args()


def main() -> None:
    parsed = args()
    table = json.loads(parsed.table.read_text())
    workload = table["workload"]
    dtype = DTYPES[workload["dtype"]]
    if parsed.mode == "profile":
        # Avoid initialization kernels so launch-skip=1 selects the second
        # TileLang Matmul launch, not a torch.randn kernel.
        A = torch.empty((workload["M"], workload["K"]), device="cuda", dtype=dtype)
        B = torch.empty((workload["K"], workload["N"]), device="cuda", dtype=dtype)
    else:
        torch.manual_seed(workload["seed"])
        A = torch.randn((workload["M"], workload["K"]), device="cuda", dtype=dtype)
        B = torch.randn((workload["K"], workload["N"]), device="cuda", dtype=dtype)

    if parsed.reference:
        call = lambda: torch.matmul(A, B)
        name = "cuBLAS"
        options = None
    else:
        config = next(item for item in table["configs"] if item["name"] == parsed.config)
        matmul = runpy.run_path(str(HERE / "main.py"), run_name="<performance-kernel>")["matmul"]
        options = config["options"]
        call = lambda: matmul(A, B, **options)
        name = config["name"]

    if parsed.mode == "profile":
        call()
        torch.cuda.synchronize()
        call()
        torch.cuda.synchronize()
        print("PROFILE_DONE", name)
        return

    call()
    torch.cuda.synchronize()
    timing = table["benchmark"]
    duration_ms = float(
        do_bench(
            call,
            warmup=timing["warmup_ms"],
            rep=timing["rep_ms"],
            return_mode="median",
        )
    )
    flops = 2 * workload["M"] * workload["N"] * workload["K"]
    result = {
        "name": name,
        "status": "ok",
        "group": None if parsed.reference else config.get("group", "initial"),
        "duration_ms": duration_ms,
        "tflops": flops / (duration_ms * 1e-3) / 1e12,
        "options": options,
    }
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
