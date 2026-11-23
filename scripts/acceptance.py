#!/usr/bin/env python3
"""Offline CLI checks, with explicitly requested and separately reported GPU checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from fixtures import ROOT, checked_count, generate, verify


class CheckFailure(RuntimeError):
    pass


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def choose_hardware(adapters: object, requested: int | None) -> dict | None:
    if not isinstance(adapters, list) or not all(isinstance(a, dict) for a in adapters):
        raise CheckFailure("devices --json must return an AdapterInfo array")
    hardware = []
    for adapter in adapters:
        kind = str(adapter.get("device_type", "")).lower().replace("_", "").replace("-", "")
        if kind in ("discretegpu", "integratedgpu"):
            if not isinstance(adapter.get("index"), int):
                raise CheckFailure("hardware adapter is missing its numeric index")
            hardware.append(adapter)
    if requested is not None:
        for adapter in hardware:
            if adapter["index"] == requested:
                return adapter
        raise CheckFailure(f"adapter {requested} is absent or is not a discrete/integrated GPU")
    return hardware[0] if hardware else None


def check_timings(payload: object, iterations: int) -> dict:
    if not isinstance(payload, dict) or not isinstance(payload.get("timings"), dict):
        raise CheckFailure("bench must return a JSON object containing timings")
    timings = payload["timings"]
    for field in ("initialization_ms", "pipeline_ms", "upload_ms", "readback_ms", "total_ms"):
        value = timings.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise CheckFailure(f"invalid bench timing: {field}")
    for field in ("dispatch_wall_ms", "kernel_ms"):
        values = timings.get(field)
        if field == "kernel_ms" and values is None:
            continue
        if not isinstance(values, list) or len(values) != iterations:
            raise CheckFailure(f"{field}: expected {iterations} measurement samples")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
            raise CheckFailure(f"{field}: invalid measurement")
    return timings


def no_adapter_reply(code: int, stdout: bytes, stderr: bytes) -> bool:
    return code == 1 and stdout.strip() == b"[]" and stderr.startswith(b"zaccel: no GPU adapters found;")


def invoke(command: list[str], cases: list[dict], name: str, timeout: float, stdin: bytes | None = None,
           allow_no_adapter: bool = False) -> bytes:
    """Run only the owned process; terminate its process group on an outer timeout."""
    started = time.perf_counter()
    process = subprocess.Popen(command, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=os.name == "posix")
    timed_out = False
    try:
        stdout, stderr = process.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            try:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            raise CheckFailure(f"{name}: process cleanup did not finish; inspect the runner before another GPU run") from None
    unavailable = allow_no_adapter and not timed_out and no_adapter_reply(process.returncode, stdout, stderr)
    record = {"name": name, "command": command, "exit_code": process.returncode,
              "cli_wall_ms": (time.perf_counter() - started) * 1000,
              "stdout_bytes": len(stdout), "stderr": stderr.decode("utf-8", errors="replace")[:8192],
              "status": "unavailable" if unavailable else "failed" if timed_out or process.returncode else "passed"}
    cases.append(record)
    if timed_out:
        raise CheckFailure(f"{name}: outer timeout; killing a process does not reset or guarantee interruption of a GPU driver")
    if process.returncode and not unavailable:
        raise CheckFailure(f"{name}: exit {process.returncode}: {record['stderr']}")
    return stdout


def run_checks(args: argparse.Namespace, report: dict) -> int:
    checked_count(args.count)
    if not 1 <= args.iterations <= 10_000 or not 0 <= args.warmup <= 10_000:
        raise CheckFailure("iterations must be 1..10000 and warmup 0..10000")
    if not 1 <= args.timeout_ms <= 3_600_000:
        raise CheckFailure("timeout-ms must be between 1 and 3600000")
    resolved = shutil.which(args.bin)
    if resolved is None:
        candidate = Path(args.bin)
        if not candidate.is_file():
            raise CheckFailure(f"binary not found: {args.bin}")
        resolved = str(candidate.resolve())
    binary = Path(resolved).resolve()
    report["binary"] = {"path": str(binary), "sha256": digest(binary)}
    cases = report["cases"]
    for shader in ("vector_add.wgsl", "vector_mul.wgsl", "affine.wgsl"):
        path = str(ROOT / "examples" / shader)
        invoke([str(binary), "validate", path], cases, f"validate:{shader}", 15)
        raw = invoke([str(binary), "inspect", path, "--entry", "main", "--json"], cases, f"inspect:{shader}", 15)
        info = json.loads(raw)
        if not isinstance(info, dict) or not isinstance(info.get("bindings"), list) or not isinstance(info.get("entry_points"), list):
            raise CheckFailure(f"inspect:{shader}: missing shader metadata")
        if not any(entry.get("name") == "main" and entry.get("workgroup_size") == [64, 1, 1] for entry in info["entry_points"]):
            raise CheckFailure(f"inspect:{shader}: wrong workgroup metadata")
    if not (args.gpu or args.require_gpu):
        report["status"] = "passed"
        return 0
    report["gpu"]["status"] = "checking_adapter"
    adapters = json.loads(invoke([str(binary), "devices", "--json"], cases, "devices", 45, allow_no_adapter=True))
    report["adapters"] = adapters
    adapter = choose_hardware(adapters, args.adapter)
    if adapter is None:
        report["gpu"].update(status="unavailable", reason="no discrete or integrated GPU adapter was returned")
        report["status"] = "failed" if args.require_gpu else "skipped"
        return 1 if args.require_gpu else 77
    report["gpu"].update(status="running", adapter=adapter)
    timeout = args.timeout_ms / 1000 + 15
    options = ["--adapter", str(adapter["index"]), "--timeout-ms", str(args.timeout_ms)]
    with tempfile.TemporaryDirectory(prefix="zaccel-acceptance-") as temporary:
        directory = generate(Path(temporary), args.count)
        pairs = directory / "data" / "pairs.bin"
        common = ["--entry", "main", "--dispatch", f"{(args.count + 63) // 64},1,1", "--size", str(args.count * 4)]
        for operation, shader in (("add", "vector_add.wgsl"), ("mul", "vector_mul.wgsl")):
            output = directory / "data" / f"{operation}.bin"
            invoke([str(binary), "run", str(directory / shader), *common, "--input", str(pairs), "--output", str(output), *options],
                   cases, f"gpu:{operation}", timeout)
            report["results"].append(verify(directory, operation, output))
        invoke([str(binary), "run", "--manifest", str(directory / "affine.json"), *options], cases, "gpu:uniform_manifest", timeout)
        report["results"].append(verify(directory, "affine"))
        raw = invoke([str(binary), "run", str(directory / "vector_add.wgsl"), *common, "--input", "-", "--output", "-", *options],
                     cases, "gpu:binary_stdin_stdout", timeout, pairs.read_bytes())
        stream_output = directory / "data" / "stream.bin"
        stream_output.write_bytes(raw)
        report["results"].append(verify(directory, "add", stream_output))
        eval_output = directory / "data" / "eval.bin"
        source = (directory / "vector_mul.wgsl").read_text(encoding="utf-8")
        invoke([str(binary), "eval", source, *common, "--input", str(pairs), "--output", str(eval_output), *options],
               cases, "gpu:eval", timeout)
        report["results"].append(verify(directory, "mul", eval_output))
        bench_output = directory / "data" / "bench.bin"
        raw = invoke([str(binary), "bench", str(directory / "vector_mul.wgsl"), *common, "--input", str(pairs), "--output", str(bench_output),
                      "--iterations", str(args.iterations), "--warmup", str(args.warmup), *options], cases, "gpu:bench", timeout)
        report["bench"] = json.loads(raw)
        check_timings(report["bench"], args.iterations)
        report["results"].append(verify(directory, "mul", bench_output))
    report["gpu"]["status"] = "passed"
    report["status"] = "passed"
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", default="zaccel", help="path to the already-built zaccel binary")
    parser.add_argument("--gpu", action="store_true", help="run real GPU cases; no hardware exits 77")
    parser.add_argument("--require-gpu", action="store_true", help="run GPU cases and fail if hardware is absent")
    parser.add_argument("--adapter", type=int)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--report", type=Path, help="save the exact JSON report in addition to stdout")
    args = parser.parse_args()
    report = {"schema": "zaccel.acceptance.v1", "status": "running",
              "scope": "offline_and_gpu" if args.gpu or args.require_gpu else "offline",
              "host": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()},
              "gpu": {"requested": args.gpu or args.require_gpu, "required": args.require_gpu, "status": "not_requested"},
              "cases": [], "results": []}
    try:
        code = run_checks(args, report)
    except (CheckFailure, OSError, ValueError, KeyError, TypeError) as error:
        code = 1
        report["status"] = "failed"
        report["error"] = str(error)
        if report["gpu"]["status"] in ("running", "checking_adapter"):
            report["gpu"]["status"] = "failed"
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.report:
        try:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(encoded, encoding="utf-8")
        except OSError as error:
            print(f"acceptance: could not save report: {error}", file=sys.stderr)
            code = 1
    sys.stdout.write(encoded)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
