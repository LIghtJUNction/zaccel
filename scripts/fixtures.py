#!/usr/bin/env python3
"""Generate bounded raw-buffer fixtures outside the source tree and check results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
MAX_COUNT = 1_048_576
OPERATIONS = ("add", "mul", "affine")


def checked_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_COUNT:
        raise ValueError(f"count must be an integer between 1 and {MAX_COUNT}")
    return value


def generate(directory: Path, count: int = 1024) -> Path:
    """Create a new fixture set; never overwrite an existing nonempty directory."""
    checked_count(count)
    directory = directory.resolve()
    if directory.is_relative_to(ROOT):
        raise ValueError("generated binary fixtures must be outside the source tree")
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError(f"fixture destination must be new or empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    data = directory / "data"
    data.mkdir()
    for name in ("vector_add.wgsl", "vector_mul.wgsl", "affine.wgsl"):
        shutil.copyfile(ROOT / "examples" / name, directory / name)
    with (data / "pairs.bin").open("wb") as pairs, (data / "a.bin").open("wb") as a_file, (data / "b.bin").open("wb") as b_file:
        for i in range(count):
            a = ((i % 257) - 128) * 0.25
            b = ((i % 31) - 15) * 0.5
            pairs.write(struct.pack("<ff", a, b))
            a_file.write(struct.pack("<f", a))
            b_file.write(struct.pack("<f", b))
    (data / "params.bin").write_bytes(struct.pack("<Ifff", count, 2.0, -0.5, 1.25))
    manifest = json.loads((ROOT / "examples" / "affine.json").read_text(encoding="utf-8"))
    manifest["dispatch"] = [(count + 63) // 64, 1, 1]
    manifest["buffers"][2]["size"] = count * 4
    (directory / "affine.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    metadata = {"schema": "zaccel.fixture.v1", "count": count, "byte_order": "little", "scalar": "f32"}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return directory


def read_exact(path: Path, size: int) -> bytes:
    actual = path.stat().st_size
    if actual != size:
        raise ValueError(f"{path}: expected {size} bytes, found {actual}")
    with path.open("rb") as stream:
        raw = stream.read(size + 1)
    if len(raw) != size:
        raise ValueError(f"{path}: file size changed while reading")
    return raw


def expected_values(directory: Path, operation: str) -> list[float]:
    if operation not in OPERATIONS:
        raise ValueError(f"unknown operation: {operation}")
    metadata_path = directory / "metadata.json"
    if metadata_path.stat().st_size > 4096:
        raise ValueError("fixture metadata exceeds 4096 bytes")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "zaccel.fixture.v1":
        raise ValueError("unsupported fixture metadata schema")
    count = checked_count(metadata["count"])
    data = directory / "data"
    if operation in ("add", "mul"):
        pairs = struct.iter_unpack("<ff", read_exact(data / "pairs.bin", count * 8))
        values = [a + b if operation == "add" else a * b for a, b in pairs]
    else:
        a_values = struct.iter_unpack("<f", read_exact(data / "a.bin", count * 4))
        b_values = struct.iter_unpack("<f", read_exact(data / "b.bin", count * 4))
        uniform_count, scale_a, scale_b, bias = struct.unpack("<Ifff", read_exact(data / "params.bin", 16))
        if uniform_count != count:
            raise ValueError("uniform count differs from fixture metadata")
        values = [scale_a * a[0] + scale_b * b[0] + bias for a, b in zip(a_values, b_values, strict=True)]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("the fixture reference contains a non-finite result")
    return [struct.unpack("<f", struct.pack("<f", v))[0] for v in values]


def verify(directory: Path, operation: str, output: Path | None = None) -> dict:
    expected = expected_values(directory, operation)
    output = output if output is not None else directory / "data" / f"{operation}.bin"
    raw = read_exact(output, len(expected) * 4)
    max_abs_error = 0.0
    for i, ((actual,), wanted) in enumerate(zip(struct.iter_unpack("<f", raw), expected, strict=True)):
        if not math.isfinite(actual) or not math.isclose(actual, wanted, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"{operation}: element {i}: expected {wanted}, found {actual}")
        max_abs_error = max(max_abs_error, abs(actual - wanted))
    return {"status": "passed", "operation": operation, "elements": len(expected), "bytes": len(raw), "max_abs_error": max_abs_error}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("generate", help="create binary inputs and a manifest outside the repository")
    create.add_argument("directory", type=Path)
    create.add_argument("--count", type=int, default=1024)
    check = commands.add_parser("verify", help="compare a GPU result with the scalar reference")
    check.add_argument("directory", type=Path)
    check.add_argument("--operation", choices=OPERATIONS, required=True)
    check.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "generate":
            directory = generate(args.directory, args.count)
            result = {"directory": str(directory), "count": args.count, "dispatch": [(args.count + 63) // 64, 1, 1], "output_bytes": args.count * 4}
        else:
            result = verify(args.directory.resolve(), args.operation, args.output)
    except (OSError, ValueError, KeyError, struct.error) as error:
        print(f"fixtures: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
