from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=HERE / "performance_configs.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--ncu-set", default="full")
    parser.add_argument("--ncu-timeout-seconds", type=int)
    parser.add_argument("--sudo-ncu", action="store_true")
    parser.add_argument("--group")
    parser.add_argument("--only", action="append")
    return parser.parse_args()


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    log_path.write_text(
        "$ " + " ".join(command) + "\n\n" + result.stdout + result.stderr
    )
    return result


def extract_result(output: str) -> dict | None:
    for line in reversed(output.splitlines()):
        if line.startswith("RESULT_JSON="):
            return json.loads(line.removeprefix("RESULT_JSON="))
    return None


def markdown(rows: list[dict], reference: dict, output_dir: Path) -> str:
    def verified_nsight(row: dict, ncu_log_path: Path, report_path: Path) -> bool:
        if row.get("nsight_status") != "ok" or not report_path.exists():
            return False
        expected_kernel = {
            "matmul_2": "matmul_2_kernel",
            "matmul_3": "matmul_3_kernel",
        }.get(row["options"].get("variant"))
        if expected_kernel is None:
            return False
        evidence_paths = [
            ncu_log_path,
            ncu_log_path.with_suffix(".details.log"),
        ]
        return any(
            path.is_file() and expected_kernel in path.read_text(errors="replace")
            for path in evidence_paths
        )

    formal_nsight_count = 0
    for row in rows:
        ncu_log_path = output_dir / "nsight" / f"{row['name']}.log"
        report_path = output_dir / "nsight" / f"{row['name']}.ncu-rep"
        if verified_nsight(row, ncu_log_path, report_path):
            formal_nsight_count += 1
    not_run_count = len(rows) - formal_nsight_count
    lines = [
        "# Matmul（矩阵乘法）性能扫描",
        "",
        "先看结论：这是 performance-only（仅性能）报告。本轮没有运行 correctness（正确性）测试。",
        "",
        f"本轮包含 {len(rows)} 个配置：性能状态为 `ok` 的有 "
        f"{sum(row['status'] == 'ok' for row in rows)} 个；正式 Nsight Compute（NVIDIA GPU 性能分析器）报告有 "
        f"{formal_nsight_count} 个，其余 {not_run_count} 个显示为 `not_run`。",
        "",
        f"cuBLAS（CUDA 的矩阵乘法库）基线：{reference['duration_ms']:.4f} ms，"
        f"{reference['tflops']:.4f} TFLOP/s（每秒万亿次浮点运算）。",
        "这条基线只属于本轮；两轮基线请分别查看各自的 `results.json`（原始结果文件）。",
        "",
        "`BM`、`BN`、`BK` 是 M、N、K 维度的 tile（分块）大小；`Threads` 是线程数；`Stages` 是流水级数。",
        "正式 Nsight 报告只接受已核对 `Kernel Name`（内核名称）为 `matmul_2_kernel` 或 `matmul_3_kernel` 的记录。",
        "未核对的 profiling（性能分析）产物显示为 `not_run`，并隔离在 `nsight/invalid-random-kernel/`。",
        "",
        "| 配置 | 变体 | BM | BN | BK | Threads | Stages | 性能状态 | ms | TFLOP/s | % cuBLAS | Nsight 状态 | 计算日志 | Nsight 日志 | 指标 CSV | Nsight 报告 |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in rows:
        options = row["options"]
        if row["status"] == "ok":
            duration = f"{row['duration_ms']:.4f}"
            tflops = f"{row['tflops']:.4f}"
            percent = f"{reference['duration_ms'] / row['duration_ms'] * 100:.2f}%"
        else:
            duration = tflops = percent = "—"
        compute_path = output_dir / "compute_logs" / f"{row['name']}.log"
        ncu_log_path = output_dir / "nsight" / f"{row['name']}.log"
        report_path = output_dir / "nsight" / f"{row['name']}.ncu-rep"
        metrics_path = output_dir / "nsight" / f"{row['name']}.metrics.csv"
        nsight_ok = verified_nsight(row, ncu_log_path, report_path)
        compute = f"[log](compute_logs/{row['name']}.log)" if compute_path.exists() else "—"
        nsight_status = "ok" if nsight_ok else "not_run"
        ncu_log = f"[log](nsight/{row['name']}.log)" if nsight_ok else "—"
        report = f"[report](nsight/{row['name']}.ncu-rep)" if nsight_ok else "—"
        metrics = f"[csv](nsight/{row['name']}.metrics.csv)" if nsight_ok and metrics_path.exists() else "—"
        lines.append(
            f"| {row['name']} | {options['variant']} | {options['block_m']} | "
            f"{options['block_n']} | {options['block_k']} | {options['threads']} | "
            f"{options['num_stages']} | {row['status']} | {duration} | {tflops} | "
            f"{percent} | {nsight_status} | {compute} | {ncu_log} | {metrics} | {report} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parsed = args()
    table_path = parsed.table.resolve()
    table = json.loads(table_path.read_text())
    configs = [
        config
        for config in table["configs"]
        if parsed.group is None or config.get("group") == parsed.group
        if parsed.only is None or config["name"] in parsed.only
    ]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        parsed.output_dir.resolve()
        if parsed.output_dir
        else REPO / "Trash" / "profiles" / "02a-matmul" / timestamp
    )
    compute_dir = output_dir / "compute_logs"
    nsight_dir = output_dir / "nsight"
    compute_dir.mkdir(parents=True, exist_ok=True)
    nsight_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(table_path, output_dir / "performance_configs.json")

    env = os.environ.copy()
    env["CXX"] = "/usr/bin/g++-15"
    one = HERE / "performance_one.py"
    if parsed.profile_only or parsed.summarize_only:
        if parsed.output_dir is None or not (output_dir / "results.json").is_file():
            raise ValueError("--profile-only requires --output-dir with results.json")
        saved = json.loads((output_dir / "results.json").read_text())
        reference = saved["reference"]
        rows = saved["results"]
        profile_rows = (
            [row for row in rows if row["name"] in parsed.only]
            if parsed.only
            else rows
        )
    else:
        reference_proc = run_logged(
            [sys.executable, str(one), "--table", str(table_path), "--reference"],
            compute_dir / "cuBLAS.log",
            env,
        )
        reference = extract_result(reference_proc.stdout)
        if reference is None:
            raise RuntimeError(f"cuBLAS benchmark failed; see {compute_dir / 'cuBLAS.log'}")

        rows: list[dict] = []
        for config in configs:
            name = config["name"]
            print("benchmark", name, flush=True)
            proc = run_logged(
                [sys.executable, str(one), "--table", str(table_path), "--config", name],
                compute_dir / f"{name}.log",
                env,
            )
            result = extract_result(proc.stdout)
            if result is None:
                result = {
                    "name": name,
                    "status": "error",
                    "group": config.get("group", "initial"),
                    "options": config["options"],
                }
            rows.append(result)

        (output_dir / "results.json").write_text(
            json.dumps({"reference": reference, "results": rows}, indent=2, sort_keys=True)
        )
        profile_rows = rows

    if parsed.profile or parsed.profile_only:
        ncu = shutil.which("ncu")
        if ncu is None:
            (nsight_dir / "README.txt").write_text("ncu was not found\n")
        else:
            for row in profile_rows:
                if row["status"] != "ok":
                    continue
                name = row["name"]
                print("profile", name, flush=True)
                command = [
                    ncu,
                    "--set", parsed.ncu_set,
                    "--target-processes", "all",
                    "--launch-skip", "1",
                    "--launch-count", "1",
                    "--force-overwrite",
                    "--export", str(nsight_dir / name),
                    sys.executable,
                    str(one),
                    "--table", str(table_path),
                    "--config", name,
                    "--mode", "profile",
                ]
                if parsed.ncu_timeout_seconds is not None:
                    command = [
                        "timeout",
                        "--signal=INT",
                        "--kill-after=10s",
                        f"{parsed.ncu_timeout_seconds}s",
                        *command,
                    ]
                if parsed.sudo_ncu:
                    command = ["sudo", "-A", "-E", *command]
                profile_result = run_logged(command, nsight_dir / f"{name}.log", env)
                timeout_marker = nsight_dir / f"{name}.timeout.txt"
                error_marker = nsight_dir / f"{name}.error.txt"
                timeout_marker.unlink(missing_ok=True)
                error_marker.unlink(missing_ok=True)
                if profile_result.returncode in (124, 137):
                    timeout_marker.write_text(
                        f"Nsight Compute exceeded {parsed.ncu_timeout_seconds} seconds\n"
                    )
                    reset_command = [
                        ncu,
                        "--clock-control", "reset",
                        "--pipeline-boost-state", "dynamic",
                    ]
                    if parsed.sudo_ncu:
                        reset_command = ["sudo", "-A", "-E", *reset_command]
                    reset_result = run_logged(
                        reset_command,
                        nsight_dir / f"{name}.clock-reset.log",
                        env,
                    )
                    if reset_result.returncode != 0:
                        error_marker.write_text(
                            "Nsight Compute timed out and GPU clock reset failed with "
                            f"status {reset_result.returncode}\n"
                        )
                    print("timeout", name, flush=True)
                elif profile_result.returncode != 0:
                    error_marker.write_text(
                        f"Nsight Compute exited with status {profile_result.returncode}\n"
                    )
                    print("profile_error", name, profile_result.returncode, flush=True)

    ncu_import = shutil.which("ncu")
    for row in rows:
        report_path = nsight_dir / f"{row['name']}.ncu-rep"
        metrics_path = nsight_dir / f"{row['name']}.metrics.csv"
        if not report_path.exists():
            continue
        if report_path.stat().st_uid != os.getuid():
            user_copy = report_path.with_suffix(".ncu-rep.user-copy")
            shutil.copy2(report_path, user_copy)
            user_copy.replace(report_path)
        if ncu_import and (
            not metrics_path.exists()
            or metrics_path.stat().st_mtime < report_path.stat().st_mtime
        ):
            subprocess.run(
                [
                    ncu_import,
                    "--import", str(report_path),
                    "--page", "raw",
                    "--csv",
                    "--log-file", str(metrics_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

    for row in rows:
        log_path = nsight_dir / f"{row['name']}.log"
        report_path = nsight_dir / f"{row['name']}.ncu-rep"
        if report_path.exists():
            row["nsight_status"] = "ok"
        elif log_path.exists() and "ERR_NVGPUCTRPERM" in log_path.read_text():
            row["nsight_status"] = "blocked: ERR_NVGPUCTRPERM"
        elif log_path.exists():
            row["nsight_status"] = "error"
        else:
            row["nsight_status"] = "not_run"
    (output_dir / "results.json").write_text(
        json.dumps({"reference": reference, "results": rows}, indent=2, sort_keys=True)
    )
    (output_dir / "results.md").write_text(markdown(rows, reference, output_dir))
    print("RESULT_DIR=" + str(output_dir))


if __name__ == "__main__":
    main()
