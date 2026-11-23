#!/usr/bin/env python3
"""Measure an already-built zaccel on real hardware; never builds or installs it.

All generated binary fixtures use OS temporary directories. Reports and textual
command logs go to a new output directory. No GPU command runs in --self-test.
"""

from __future__ import annotations

import argparse
from array import array
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import tempfile
import time

from acceptance import CheckFailure, check_timings, choose_hardware

ROOT = Path(__file__).resolve().parents[1]
MIB = 1024 * 1024
DEFAULT_BYTES = 256 * MIB
REQUIRED_COUNT = 1_048_576
OPTIONAL_COUNT = 4_194_304
CHUNK_ELEMENTS = 16_384
TAIPEI = timezone(timedelta(hours=8))

INCREMENT = """@group(0) @binding(0) var<storage, read_write> values: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&values)) { values[id.x] += 1u; }
}
"""
ZERO_TAIL = INCREMENT.replace("+= 1u", "+= 10u")
GENERATE = INCREMENT.replace("values[id.x] += 1u", "values[id.x] = id.x + 7u")
GAPPED = """@group(0) @binding(0) var<storage, read_write> first: array<u32>;
@group(2) @binding(3) var<storage, read_write> second: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&first)) { first[id.x] = id.x + 100u; }
    if (id.x < arrayLength(&second)) { second[id.x] = id.x * 2u + 3u; }
}
"""


class MeasurementFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def summary(values: list[float]) -> dict | None:
    if not values:
        return None
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise MeasurementFailure("summary requires finite nonnegative observations")
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max": ordered[-1],
    }


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def memory_snapshot() -> dict:
    result = {}
    text = read_text(Path("/proc/meminfo"))
    if text is not None:
        for line in text.splitlines():
            parts = line.split()
            if parts and parts[0] in ("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:"):
                result[parts[0][:-1] + "_bytes"] = int(parts[1]) * 1024
    return {
        "timestamp_utc": now(),
        "memory": result if result else None,
        "loadavg": (read_text(Path("/proc/loadavg")) or "").strip() or None,
    }


def memory_available() -> int | None:
    memory = memory_snapshot()["memory"]
    return memory.get("MemAvailable_bytes") if memory else None


def process_tree_rss(pid: int) -> dict:
    """Observed parent and descendants; disappearing processes remain missing."""
    pending, seen, rows = [pid], set(), []
    while pending and len(seen) < 64:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        base = Path("/proc") / str(current)
        status = read_text(base / "status")
        if status is not None:
            fields = {}
            for line in status.splitlines():
                if line.startswith(("VmRSS:", "VmHWM:")):
                    parts = line.split()
                    fields[parts[0][:-1] + "_bytes"] = int(parts[1]) * 1024
            stat = read_text(base / "stat")
            start_ticks = None
            if stat and ") " in stat:
                fields_after_name = stat.rsplit(") ", 1)[1].split()
                if len(fields_after_name) > 19:
                    start_ticks = int(fields_after_name[19])
            if "VmRSS_bytes" in fields:
                rows.append({"pid": current, "start_ticks": start_ticks, **fields})
        children = read_text(base / "task" / str(current) / "children")
        if children:
            pending.extend(int(value) for value in children.split() if value.isdigit())
    return {
        "processes": rows,
        # Sum of process RSS can double-count shared pages. It is not PSS/USS.
        "sum_rss_bytes": sum(row["VmRSS_bytes"] for row in rows) if rows else None,
        "traversal_capped": bool(pending),
    }


class Runner:
    def __init__(self, directory: Path, report: dict, sample_interval: float):
        self.directory = directory
        self.report = report
        self.sample_interval = sample_interval
        (directory / "logs").mkdir()

    def checkpoint(self) -> None:
        write_json(self.directory / "report.json", self.report)

    def run(self, label: str, command: list[str], timeout: float,
            expected: tuple[int, ...] = (0,), sample_rss: bool = False) -> tuple[dict, bytes, bytes]:
        if not 0 < timeout <= 600:
            raise ValueError("every subprocess requires a timeout in (0, 600] seconds")
        ordinal = len(self.report["commands"])
        stem = f"{ordinal:03d}-{label}"
        record = {
            "id": ordinal, "label": label, "command": command, "cwd": str(ROOT),
            "started_utc": now(), "timeout_seconds": timeout,
            "exit_code": None, "timed_out": False, "cleanup_confirmed": None,
            "status": "starting",
        }
        self.report["commands"].append(record)
        self.checkpoint()
        stdout, stderr = b"", b""
        samples = []
        started = time.perf_counter()
        process = None
        try:
            process = subprocess.Popen(
                command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
            record["pid"] = process.pid
            deadline = started + timeout
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                if sample_rss and self.sample_interval:
                    observed = process_tree_rss(process.pid)
                    observed["elapsed_ms"] = (time.perf_counter() - started) * 1000
                    samples.append(observed)
                interval = min(remaining, self.sample_interval) if sample_rss and self.sample_interval else remaining
                try:
                    stdout, stderr = process.communicate(timeout=interval)
                    record["cleanup_confirmed"] = True
                    break
                except subprocess.TimeoutExpired:
                    if time.perf_counter() >= deadline:
                        raise
        except subprocess.TimeoutExpired:
            record["timed_out"] = True
            if process is not None:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                try:
                    stdout, stderr = process.communicate(timeout=5)
                    record["cleanup_confirmed"] = True
                except subprocess.TimeoutExpired:
                    record["cleanup_confirmed"] = False
        except OSError as error:
            record["spawn_error"] = str(error)
        finally:
            record["cli_wall_ms"] = (time.perf_counter() - started) * 1000
            record["finished_utc"] = now()
            if process is not None:
                record["exit_code"] = process.returncode
            record["stdout_log"] = f"logs/{stem}.stdout.txt"
            record["stderr_log"] = f"logs/{stem}.stderr.txt"
            (self.directory / record["stdout_log"]).write_bytes(stdout)
            (self.directory / record["stderr_log"]).write_bytes(stderr)
            record["stdout_bytes"] = len(stdout)
            record["stderr_bytes"] = len(stderr)
            values = [sample["sum_rss_bytes"] for sample in samples if sample["sum_rss_bytes"] is not None]
            record["rss"] = {
                "requested": sample_rss and bool(self.sample_interval),
                "sampling_interval_ms": self.sample_interval * 1000 if sample_rss else None,
                "observations": samples,
                "peak_sampled_sum_rss_bytes": max(values) if values else None,
                "scope": "CLI parent plus observed descendants; shared pages may be counted twice",
                "limit": "sampled observation, not exact peak, PSS, USS, or exclusive GPU memory",
            }
            ok = (
                not record["timed_out"]
                and "spawn_error" not in record
                and record["exit_code"] in expected
            )
            record["status"] = "passed" if ok else "failed"
            self.checkpoint()
        if not ok:
            suffix = stderr.decode("utf-8", errors="replace")[-2000:]
            raise MeasurementFailure(
                f"{label}: exit={record['exit_code']} timed_out={record['timed_out']} "
                f"cleanup_confirmed={record['cleanup_confirmed']}; {record.get('spawn_error', '')} {suffix}"
            )
        return record, stdout, stderr

    def optional(self, label: str, command: list[str], timeout: float = 10) -> dict:
        try:
            record, stdout, _ = self.run(label, command, timeout)
            return {"status": "observed", "command_id": record["id"],
                    "text": stdout.decode("utf-8", errors="replace").strip()}
        except MeasurementFailure as error:
            observed = self.report["commands"][-1]
            if observed["timed_out"] or observed["cleanup_confirmed"] is False:
                # Even optional metadata must not hide a process whose outcome
                # is uncertain. Stop; do not begin or repeat GPU work.
                raise
            return {"status": "unknown", "reason": str(error), "command_id": len(self.report["commands"]) - 1}


def parse_execution(raw: bytes) -> dict:
    """Allow backend diagnostic lines before the single compact JSON report."""
    decoder = json.JSONDecoder()
    text = raw.decode("utf-8", errors="replace")
    found = []
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and isinstance(candidate.get("timings"), dict):
            found.append(candidate)
    if len(found) != 1:
        raise MeasurementFailure(f"expected one execution report, found {len(found)}")
    return found[0]


def validate_execution(payload: dict, iterations: int, warmup: int, adapter: dict) -> None:
    try:
        check_timings(payload, iterations)
    except CheckFailure as error:
        raise MeasurementFailure(str(error)) from error
    if payload.get("iterations") != iterations or payload.get("warmup") != warmup:
        raise MeasurementFailure("execution report changed the requested repetitions")
    if payload.get("buffer_state") != "retained_between_dispatches":
        raise MeasurementFailure("missing explicit retained-buffer-state declaration")
    actual = payload.get("adapter", {})
    for field in ("name", "vendor", "device", "device_type", "backend"):
        if actual.get(field) != adapter.get(field):
            raise MeasurementFailure(f"selected adapter changed in field {field}")
    for field in ("preparation_ms", "output_write_ms", "worker_total_ms"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise MeasurementFailure(f"invalid stage timing {field}")
    allocation = payload.get("gpu_allocated_bytes")
    if not isinstance(allocation, int) or not 0 < allocation <= DEFAULT_BYTES:
        raise MeasurementFailure("execution allocation exceeds the default 256 MiB budget")


def pair(index: int) -> tuple[float, float]:
    # Binary fractions are exact in f32 over this bounded range.
    return ((index % 257) - 128) * 0.25, ((index % 31) - 15) * 0.5


def generate_pairs(path: Path, count: int) -> dict:
    """At most CHUNK_ELEMENTS pairs in an array, never a count-sized list."""
    if not 1 <= count <= OPTIONAL_COUNT:
        raise ValueError("fixture count outside the bounded measurement range")
    sha = hashlib.sha256()
    with path.open("xb") as stream:
        for start in range(0, count, CHUNK_ELEMENTS):
            values = array("f")
            for index in range(start, min(count, start + CHUNK_ELEMENTS)):
                a, b = pair(index)
                values.append(a)
                values.append(b)
            if values.itemsize != 4:
                raise MeasurementFailure("platform float array is not 32 bits")
            if sys.byteorder != "little":
                values.byteswap()
            raw = values.tobytes()
            stream.write(raw)
            sha.update(raw)
    return {"bytes": count * 8, "sha256": sha.hexdigest(), "chunk_elements": CHUNK_ELEMENTS}


def verify_add(path: Path, count: int) -> dict:
    if path.stat().st_size != count * 4:
        raise MeasurementFailure("vector_add output byte count differs from the requested size")
    sha = hashlib.sha256()
    largest = 0.0
    observed = 0
    started = time.perf_counter()
    with path.open("rb") as stream:
        while raw := stream.read(CHUNK_ELEMENTS * 4):
            if len(raw) % 4:
                raise MeasurementFailure("partial f32 in vector_add output")
            sha.update(raw)
            for (actual,) in struct.iter_unpack("<f", raw):
                a, b = pair(observed)
                expected = a + b
                if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
                    raise MeasurementFailure(f"vector_add element {observed}: expected {expected}, observed {actual}")
                largest = max(largest, abs(actual - expected))
                observed += 1
    if observed != count:
        raise MeasurementFailure("output changed while being checked")
    return {
        "status": "passed", "elements": observed, "bytes": count * 4,
        "sha256": sha.hexdigest(), "max_abs_error": largest,
        "verification_wall_ms": (time.perf_counter() - started) * 1000,
        "reference_role": "correctness only; not a CPU performance baseline",
    }


def check_u32(path: Path, expected: list[int]) -> dict:
    raw = path.read_bytes()
    if len(raw) != 4 * len(expected):
        raise MeasurementFailure(f"{path.name}: wrong output length")
    actual = list(struct.unpack("<" + "I" * len(expected), raw))
    if actual != expected:
        raise MeasurementFailure(f"{path.name}: expected {expected}, observed {actual}")
    return {"status": "passed", "actual": actual, "expected": expected, "sha256": hashlib.sha256(raw).hexdigest()}


def nvidia_snapshot(runner: Runner, label: str) -> dict:
    return runner.optional(
        label,
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,pstate,temperature.gpu",
         "--format=csv,noheader,nounits"],
        10,
    )


def environment(runner: Runner, binary: Path) -> dict:
    _, version, _ = runner.run("binary-version", [str(binary), "--version"], 10)
    cpuinfo = read_text(Path("/proc/cpuinfo")) or ""
    models = sorted({line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith("model name")})
    files = [
        "Cargo.toml", "Cargo.lock", "rust-toolchain.toml",
        "src/main.rs", "src/lib.rs", "src/gpu.rs", "src/plan.rs", "src/shader.rs",
        "examples/vector_add.wgsl", "scripts/measure.py", "scripts/acceptance.py", "scripts/fixtures.py",
    ]
    hashes = {path: digest(ROOT / path) if (ROOT / path).is_file() else None for path in files}
    compiler = runner.optional("installed-rustc-path", ["rustup", "which", "rustc"], 10)
    if compiler["status"] == "observed" and Path(compiler["text"]).is_file():
        rust = runner.optional("installed-rustc-version", [compiler["text"], "--version", "--verbose"], 10)
    else:
        rust = {"status": "unknown", "reason": "no installed rustc path was resolved"}
    return {
        "binary": {"path": str(binary), "sha256": digest(binary), "version": version.decode().strip()},
        "source": {
            "git_head": runner.optional("git-head", ["git", "rev-parse", "HEAD"]),
            "git_status": runner.optional("git-status", ["git", "status", "--short"]),
            "file_sha256": hashes,
        },
        "host": {
            "uname": dict(zip(("system", "node", "release", "version", "machine", "processor"), platform.uname())),
            "python": platform.python_version(), "cpu_models": models or None,
            "logical_cpus": os.cpu_count(), "rustc": rust,
            "rustc_scope": "installed compiler observation; binary identity is its SHA-256",
            "before": memory_snapshot(), "nvidia_before": nvidia_snapshot(runner, "nvidia-before"),
        },
    }


def boundary_checks(runner: Runner, args: argparse.Namespace, binary: Path,
                    adapter: dict, adapters: list[dict], directory: Path) -> list[dict]:
    records = runner.report["boundaries"]
    options = ["--adapter", str(adapter["index"]), "--timeout-ms", str(args.timeout_ms), "--json"]
    for name, source in (("zero-tail", ZERO_TAIL), ("increment", INCREMENT), ("generate", GENERATE), ("gapped", GAPPED)):
        (directory / f"{name}.wgsl").write_text(source, encoding="utf-8")
    # These textual fixtures are retained for reproduction; no .bin is copied.
    runner.report["boundary_sources"] = {
        "zero_tail": ZERO_TAIL, "increment": INCREMENT, "exact_budget": GENERATE, "gapped": GAPPED,
    }

    short = directory / "short.bin"
    short.write_bytes(struct.pack("<II", 1, 2))
    output = directory / "zero-tail.bin"
    record, _, stderr = runner.run("boundary-zero-tail", [
        str(binary), "run", str(directory / "zero-tail.wgsl"), "--input", str(short),
        "--size", "16", "--output", str(output), "--dispatch", "1,1,1", *options,
    ], args.process_timeout)
    payload = parse_execution(stderr)
    validate_execution(payload, 1, 0, adapter)
    records.append({"name": "short-input-zero-tail", "status": "passed", "command_id": record["id"],
                    "result": check_u32(output, [11, 12, 10, 10]), "execution": payload})
    runner.checkpoint()

    for label, shader, expected, extra in (
        ("stateful-increment", "increment", [5, 5, 5, 5], []),
        ("exact-budget-no-timestamps", "generate", [7, 8, 9, 10], ["--max-bytes", "32"]),
    ):
        output = directory / f"{label}.bin"
        record, stdout, _ = runner.run("boundary-" + label, [
            str(binary), "bench", str(directory / f"{shader}.wgsl"), "--size", "16",
            "--output", str(output), "--dispatch", "1,1,1", "--iterations", "3", "--warmup", "2",
            *extra, *options,
        ], args.process_timeout)
        payload = parse_execution(stdout)
        validate_execution(payload, 3, 2, adapter)
        result = check_u32(output, expected)
        if extra and (payload["gpu_allocated_bytes"] != 32 or payload["timings"]["kernel_ms"] is not None or payload.get("kernel") is not None):
            raise MeasurementFailure("exact resident+readback budget did not suppress optional timestamp buffers")
        records.append({"name": label, "status": "passed", "command_id": record["id"],
                        "result": result, "execution": payload})
        runner.checkpoint()

    first, second = directory / "group0.bin", directory / "group2.bin"
    manifest = {
        "shader": "gapped.wgsl", "entry": "main", "dispatch": [1, 1, 1],
        "buffers": [
            {"group": 0, "binding": 0, "kind": "storage_read_write", "size": 16, "output": first.name},
            {"group": 2, "binding": 3, "kind": "storage_read_write", "size": 16, "output": second.name},
        ],
    }
    write_json(directory / "gapped.json", manifest)
    runner.report["boundary_manifest"] = manifest
    record, _, stderr = runner.run("boundary-gapped-groups", [
        str(binary), "run", "--manifest", str(directory / "gapped.json"), *options,
    ], args.process_timeout)
    payload = parse_execution(stderr)
    validate_execution(payload, 1, 0, adapter)
    results = [check_u32(first, [100, 101, 102, 103]), check_u32(second, [3, 5, 7, 9])]
    records.append({"name": "group0-group2-and-two-outputs", "status": "passed",
                    "command_id": record["id"], "results": results, "execution": payload})
    runner.checkpoint()

    software = [
        item for item in adapters
        if str(item.get("device_type", "")).lower() == "cpu" or "(software)" in str(item.get("name", "")).lower()
    ]
    if not software:
        records.append({"name": "software-index-requires-opt-in", "status": "unsupported",
                        "reason": "no CPU/software adapter was enumerated"})
    else:
        selected = software[0]
        command = [
            str(binary), "run", str(directory / "generate.wgsl"), "--size", "16",
            "--output", str(directory / "software-rejected.bin"), "--dispatch", "1,1,1",
            "--adapter", str(selected["index"]), "--timeout-ms", str(args.timeout_ms), "--json",
        ]
        record, _, stderr = runner.run("boundary-software-rejection", command, args.process_timeout, expected=(1,))
        diagnostic = stderr.decode("utf-8", errors="replace")
        if "software" not in diagnostic.lower() or "--allow-software" not in diagnostic:
            raise MeasurementFailure("software adapter failed for an unrelated reason")
        records.append({"name": "software-index-requires-opt-in", "status": "passed",
                        "adapter": selected, "command_id": record["id"], "exit_code": record["exit_code"],
                        "diagnostic": diagnostic.strip()})
    runner.checkpoint()
    return records


def aggregate(samples: list[dict]) -> list[dict]:
    groups = []
    for count, mode in sorted({(sample["count"], sample["mode"]) for sample in samples}):
        selected = [sample for sample in samples if sample["count"] == count and sample["mode"] == mode]
        fields = {
            "whole_cli_wall_ms": [sample["cli_wall_ms"] for sample in selected],
            "preparation_ms": [sample["execution"]["preparation_ms"] for sample in selected],
            "output_write_ms": [sample["execution"]["output_write_ms"] for sample in selected],
            "worker_total_ms": [sample["execution"]["worker_total_ms"] for sample in selected],
        }
        for field in ("initialization_ms", "pipeline_ms", "upload_ms", "readback_ms", "total_ms"):
            fields[field] = [sample["execution"]["timings"][field] for sample in selected]
        host = [value for sample in selected for value in sample["execution"]["timings"]["dispatch_wall_ms"]]
        kernels = [sample["execution"]["timings"]["kernel_ms"] for sample in selected]
        observed_kernels = [values for values in kernels if values is not None]
        fields["per_process_dispatch_median_ms"] = [
            statistics.median(sample["execution"]["timings"]["dispatch_wall_ms"]) for sample in selected
        ]
        fields["per_process_kernel_median_ms"] = [statistics.median(values) for values in observed_kernels]
        groups.append({
            "count": count, "mode": mode, "independent_cli_processes": len(selected),
            "units": "milliseconds", "p95_rule": "nearest rank: sorted[ceil(0.95*n)-1]",
            "process_summaries": {field: summary(values) for field, values in fields.items()},
            "pooled_dispatch_wall_ms": summary(host),
            "pooled_kernel_ms": summary([value for values in observed_kernels for value in values]),
            "processes_with_kernel_timestamps": len(observed_kernels),
            "dispatch_sample_scope": "dispatches within one process share resources; not independent process replicates",
            "correct_outputs": sum(sample["verification"]["status"] == "passed" for sample in selected),
        })
    return groups


def finish_report(runner: Runner, binary: Path, binary_hash: str) -> None:
    report = runner.report
    report["binary"]["sha256_after"] = digest(binary)
    if report["binary"]["sha256_after"] != binary_hash:
        raise MeasurementFailure("binary changed during measurement; results cannot be attributed to one executable")
    report["host"]["after"] = memory_snapshot()
    report["host"]["nvidia_after"] = nvidia_snapshot(runner, "nvidia-after")
    report["status"] = "passed"


def run_measurements(args: argparse.Namespace, runner: Runner, binary: Path) -> None:
    report = runner.report
    report.update(environment(runner, binary))
    binary_hash = report["binary"]["sha256"]
    _, raw, _ = runner.run("devices", [str(binary), "devices", "--json"], args.process_timeout)
    adapters = json.loads(raw)
    adapter = choose_hardware(adapters, args.adapter)
    if args.adapter_name and args.adapter is None:
        candidates = [item for item in adapters if args.adapter_name.lower() in str(item.get("name", "")).lower()]
        adapter = choose_hardware(candidates, None)
    if adapter is None:
        raise MeasurementFailure("no requested discrete/integrated hardware adapter is available")
    if args.adapter_name and args.adapter_name.lower() not in str(adapter.get("name", "")).lower():
        raise MeasurementFailure("selected adapter does not match --adapter-name")
    report["adapters"], report["adapter"] = adapters, adapter
    acceptance = runner.directory / "acceptance.json"
    record, stdout, _ = runner.run("strict-acceptance", [
        sys.executable, "-B", str(ROOT / "scripts" / "acceptance.py"), "--bin", str(binary),
        "--require-gpu", "--adapter", str(adapter["index"]), "--count", "4096",
        "--iterations", str(args.iterations), "--warmup", str(args.warmup),
        "--timeout-ms", str(args.timeout_ms), "--report", str(acceptance),
    ], 600)
    accepted = json.loads(stdout)
    if accepted.get("status") != "passed" or accepted.get("gpu", {}).get("status") != "passed":
        raise MeasurementFailure("strict GPU acceptance did not pass")
    if accepted.get("binary", {}).get("sha256") != binary_hash:
        raise MeasurementFailure("acceptance observed a different binary")
    report["acceptance"] = {
        "status": "passed", "command_id": record["id"], "path": "acceptance.json",
        "sha256": digest(acceptance), "case_count": len(accepted.get("cases", [])),
        "numerical_results": accepted.get("results"),
    }
    runner.checkpoint()

    with tempfile.TemporaryDirectory(prefix="zaccel-measure-") as temporary:
        directory = Path(temporary)
        boundary_checks(runner, args, binary, adapter, adapters, directory)
        if args.functional_only:
            report["optional_4m"] = {"status": "not_requested", "reason": "functional-only scope"}
            finish_report(runner, binary, binary_hash)
            return
        counts = [REQUIRED_COUNT]
        if args.include_4m:
            available = memory_available()
            required_groups = (OPTIONAL_COUNT + 63) // 64
            adapter_limit = adapter.get("max_compute_workgroups_per_dimension")
            if not isinstance(adapter_limit, int) or required_groups > adapter_limit:
                report["optional_4m"] = {
                    "status": "unsupported", "MemAvailable_bytes": available,
                    "reason": "unchanged vector_add shader requires 65,536 X workgroups",
                    "required_x_workgroups": required_groups, "adapter_x_limit": adapter_limit,
                }
            elif available is not None and available >= 2 * 1024 * MIB:
                counts.append(OPTIONAL_COUNT)
                report["optional_4m"] = {"status": "admitted", "MemAvailable_bytes": available,
                                         "minimum_available_bytes": 2 * 1024 * MIB}
            else:
                report["optional_4m"] = {"status": "skipped", "MemAvailable_bytes": available,
                                         "reason": "requires an observed MemAvailable of at least 2 GiB"}
        else:
            report["optional_4m"] = {"status": "not_requested"}

        for count in counts:
            accounted = count * 24
            if accounted + 32 > DEFAULT_BYTES:
                raise MeasurementFailure("fixture exceeds default accounted GPU budget")
            required_available = (2 * 1024 * MIB if count == OPTIONAL_COUNT else 512 * MIB + count * 32)
            available = memory_available()
            if available is None or available < required_available:
                raise MeasurementFailure(f"insufficient observed MemAvailable for {count} elements: {available}")
            inputs = directory / f"pairs-{count}.bin"
            fixture = generate_pairs(inputs, count)
            report["fixtures"].append({
                "count": count, "input": fixture, "output_bytes": count * 4,
                "resident_bytes": count * 12, "upload_staging_bytes": count * 8,
                "readback_staging_bytes": count * 4, "accounted_without_timestamps_bytes": accounted,
                "timestamp_optional_bytes": 32, "budget_bytes": DEFAULT_BYTES,
                "MemAvailable_before_generation_bytes": available,
            })
            runner.checkpoint()
            for repetition in range(args.samples):
                for mode in ("run", "bench"):
                    available = memory_available()
                    if available is None or available < required_available:
                        raise MeasurementFailure("MemAvailable fell below the declared per-dataset headroom")
                    output = directory / f"{mode}-{count}-{repetition}.bin"
                    command = [
                        str(binary), mode, str(ROOT / "examples" / "vector_add.wgsl"),
                        "--entry", "main", "--dispatch", f"{(count + 63) // 64},1,1",
                        "--input", str(inputs), "--size", str(count * 4), "--output", str(output),
                        "--adapter", str(adapter["index"]), "--timeout-ms", str(args.timeout_ms), "--json",
                    ]
                    iterations, warmup = (args.iterations, args.warmup) if mode == "bench" else (1, 0)
                    if mode == "bench":
                        command.extend(["--iterations", str(iterations), "--warmup", str(warmup)])
                    record, stdout, stderr = runner.run(
                        f"{mode}-{count}-{repetition:02d}", command, args.process_timeout, sample_rss=True,
                    )
                    payload = parse_execution(stdout if mode == "bench" else stderr)
                    validate_execution(payload, iterations, warmup, adapter)
                    verification = verify_add(output, count)
                    report["samples"].append({
                        "count": count, "mode": mode, "repetition": repetition,
                        "command_id": record["id"], "exit_code": record["exit_code"],
                        "cli_wall_ms": record["cli_wall_ms"], "execution": payload,
                        "verification": verification, "MemAvailable_before_bytes": available,
                    })
                    output.unlink()
                    report["summaries"] = aggregate(report["samples"])
                    runner.checkpoint()
                    print(f"{mode} count={count} sample={repetition + 1}/{args.samples} passed "
                          f"cli_wall_ms={record['cli_wall_ms']:.3f}", flush=True)
            inputs.unlink()
    finish_report(runner, binary, binary_hash)


def self_test() -> int:
    """Small real harness checks, all CPU-only and bounded."""
    import unittest

    class HarnessTests(unittest.TestCase):
        def test_nearest_rank_and_unknown(self):
            self.assertIsNone(summary([]))
            self.assertEqual(summary(list(range(1, 11))), {"count": 10, "min": 1, "median": 5.5, "p95": 10, "max": 10})
            with self.assertRaises(MeasurementFailure):
                summary([float("nan")])

        def test_chunk_fixture_verifier_and_corruption(self):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                count = CHUNK_ELEMENTS + 3
                generated = generate_pairs(root / "pairs.bin", count)
                self.assertEqual(generated["bytes"], count * 8)
                with (root / "pairs.bin").open("rb") as pairs, (root / "out.bin").open("wb") as output:
                    while raw := pairs.read(CHUNK_ELEMENTS * 8):
                        for a, b in struct.iter_unpack("<ff", raw):
                            output.write(struct.pack("<f", a + b))
                self.assertEqual(verify_add(root / "out.bin", count)["max_abs_error"], 0)
                with (root / "out.bin").open("r+b") as output:
                    output.write(struct.pack("<f", float("nan")))
                with self.assertRaises(MeasurementFailure):
                    verify_add(root / "out.bin", count)

        def test_report_with_driver_prefix(self):
            raw = b"backend note\n" + json.dumps({"timings": {"kernel_ms": None}}).encode()
            self.assertIsNone(parse_execution(raw)["timings"]["kernel_ms"])

        def test_owned_subprocess_timeout_has_receipt(self):
            with tempfile.TemporaryDirectory() as temporary:
                report = {"commands": []}
                runner = Runner(Path(temporary), report, 0)
                with self.assertRaises(MeasurementFailure):
                    runner.optional("cpu-timeout", [sys.executable, "-c", "import time; time.sleep(5)"], 0.1)
                record = report["commands"][0]
                self.assertTrue(record["timed_out"])
                self.assertTrue(record["cleanup_confirmed"])
                self.assertIsNotNone(record["exit_code"])
                self.assertTrue((Path(temporary) / "report.json").is_file())

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HarnessTests))
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", default="/var/tmp/project-z-cargo-target/release/zaccel")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile", choices=("debug", "release"), default="release",
                        help="caller-declared build profile; executable identity is its SHA-256")
    parser.add_argument("--functional-only", action="store_true",
                        help="strict GPU correctness/boundaries only; no timing campaign")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--adapter", type=int)
    parser.add_argument("--adapter-name")
    parser.add_argument("--include-4m", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--process-timeout", type=float, default=45)
    parser.add_argument("--sample-interval-ms", type=int, default=100,
                        help="0 disables optional /proc RSS sampling; otherwise 20..1000 ms")
    parser.add_argument("--self-test", action="store_true", help="CPU-only harness regressions; no zaccel execution")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.output_dir is None:
        parser.error("--output-dir is required for measurements")
    if args.profile == "debug" and not args.functional_only:
        parser.error("debug binaries are accepted only with --functional-only")
    if args.functional_only and args.include_4m:
        parser.error("--include-4m belongs to the release timing campaign")
    if not 10 <= args.samples <= 50 or not 20 <= args.iterations <= 1000 or not 3 <= args.warmup <= 100:
        parser.error("samples=10..50, iterations=20..1000, warmup=3..100")
    if not 1000 <= args.timeout_ms <= 60_000 or not args.timeout_ms / 1000 + 5 <= args.process_timeout <= 90:
        parser.error("timeout-ms=1000..60000 and process-timeout must allow 5s cleanup headroom, at most 90s")
    if args.sample_interval_ms != 0 and not 20 <= args.sample_interval_ms <= 1000:
        parser.error("sample-interval-ms must be 0 or 20..1000")
    binary = Path(shutil.which(args.bin) or args.bin).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error("the already-built executable is missing or not executable")
    directory = args.output_dir.resolve()
    if directory.is_relative_to(ROOT) and not directory.is_relative_to(ROOT / "artifacts" / "measurements"):
        parser.error("in-repository reports must be inside ignored artifacts/measurements")
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error(f"output-dir must be new; existing results are never overwritten: {error}")
    report = {
        "schema": "zaccel.measurements.v1", "status": "running",
        "scope": "functional_only" if args.functional_only else "functional_and_measurements",
        "profile": {"declared": args.profile, "source": "caller label; actual binary is hashed"},
        "started_utc": now(), "started_asia_taipei": datetime.now(TAIPEI).isoformat(),
        "arguments": vars(args) | {"output_dir": str(directory)},
        "protocol": {
            "independent_processes_per_count_and_mode": 0 if args.functional_only else args.samples,
            "modes": ["fresh CLI run, one dispatch", "fresh CLI bench, shared device/pipeline/buffers within process"],
            "timing_campaign_requested": not args.functional_only,
            "measured_dispatches_per_bench_process": None if args.functional_only else args.iterations,
            "warmup_dispatches": None if args.functional_only else args.warmup,
            "process_order": "run then bench per repetition",
            "cache_state": "fresh processes; driver shader cache and OS file cache are not flushed",
            "kernel_unavailable": None,
            "cpu_reference": "correctness only; no CPU speedup comparison",
            "gpu_budget_bytes": DEFAULT_BYTES,
            "rss_scope": "sampled parent+descendant RSS sum; may double-count shared pages, can miss short peaks",
            "gpu_memory_scope": "nvidia-smi snapshots are device-wide and confounded by other clients",
            "concurrency": "shared development host; no CPU/GPU isolation or clock locking",
            "timeout_rule": "stop on first timeout/failure; never automatically retry an uncertain GPU outcome",
        },
        "commands": [], "boundaries": [], "fixtures": [], "samples": [], "summaries": [],
    }
    runner = Runner(directory, report, args.sample_interval_ms / 1000)
    code = 0
    try:
        run_measurements(args, runner, binary)
    except (MeasurementFailure, CheckFailure, OSError, ValueError, KeyError, TypeError, struct.error) as error:
        code = 1
        report["status"] = "failed"
        report["error"] = str(error)
        print(f"measure: {error}", file=sys.stderr, flush=True)
    finally:
        report["finished_utc"] = now()
        report["finished_asia_taipei"] = datetime.now(TAIPEI).isoformat()
        report["exit_code"] = code
        runner.checkpoint()
    print(json.dumps({"status": report["status"], "exit_code": code, "report": str(directory / "report.json"),
                      "scope": report["scope"], "profile": args.profile,
                      "measured_processes": len(report["samples"])}, indent=2), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
