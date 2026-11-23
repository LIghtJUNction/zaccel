#!/usr/bin/env python3
"""Package an already-built binary and create SHA-256 release checksums."""

from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu", "x86_64-pc-windows-msvc", "x86_64-apple-darwin", "aarch64-apple-darwin")


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def version_for_tag(tag: str, root: Path = ROOT) -> str:
    with (root / "Cargo.toml").open("rb") as stream:
        version = tomllib.load(stream)["package"]["version"]
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?", version):
        raise ValueError("unsupported Cargo release version")
    if tag != f"v{version}":
        raise ValueError(f"tag {tag!r} does not match Cargo version v{version}")
    return version


def package(binary: Path, target: str, version: str, output: Path, root: Path = ROOT) -> Path:
    if target not in TARGETS:
        raise ValueError(f"unsupported release target: {target}")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?", version):
        raise ValueError("invalid release version")
    if not binary.is_file() or binary.stat().st_size == 0:
        raise ValueError(f"built binary is missing or empty: {binary}")
    output = output.resolve()
    if output.is_relative_to(root.resolve()):
        raise ValueError("release archives must be written outside the source tree")
    output.mkdir(parents=True, exist_ok=True)
    stem = f"zaccel-v{version}-{target}"
    windows = "windows" in target
    archive = output / (stem + (".zip" if windows else ".tar.gz"))
    if archive.exists():
        raise ValueError(f"refusing to overwrite existing archive: {archive}")
    files = [(binary, "zaccel.exe" if windows else "zaccel", 0o755)]
    required = ["README.md", "LICENSE", "scripts/fixtures.py", "scripts/acceptance.py"]
    for name in required:
        if not (root / name).is_file():
            raise ValueError(f"required release file missing: {name}")
    sources = [root / name for name in required]
    sources += sorted((root / "docs").glob("*.md"))
    sources += sorted((root / "examples").glob("*.wgsl"))
    sources += sorted((root / "examples").glob("*.json"))
    files.extend((path, path.relative_to(root).as_posix(), 0o644) for path in sources)
    if windows:
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path, name, mode in files:
                entry = zipfile.ZipInfo(f"{stem}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
                entry.create_system = 3
                entry.external_attr = (0o100000 | mode) << 16
                entry.compress_type = zipfile.ZIP_DEFLATED
                with path.open("rb") as source, bundle.open(entry, "w") as destination:
                    shutil.copyfileobj(source, destination)
    else:
        with archive.open("xb") as destination, gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as bundle:
                for path, name, mode in files:
                    entry = tarfile.TarInfo(f"{stem}/{name}")
                    entry.size = path.stat().st_size
                    entry.mode = mode
                    entry.mtime = 0
                    with path.open("rb") as source:
                        bundle.addfile(entry, source)
    return archive


def checksums(directory: Path, expected_count: int | None = None) -> Path:
    archives = sorted([*directory.glob("zaccel-v*.tar.gz"), *directory.glob("zaccel-v*.zip")])
    if not archives:
        raise ValueError("no release archives found")
    if expected_count is not None and len(archives) != expected_count:
        raise ValueError(f"expected {expected_count} archives, found {len(archives)}")
    destination = directory / "SHA256SUMS"
    destination.write_text("".join(f"{sha256(path)}  {path.name}\n" for path in archives), encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-version")
    check.add_argument("--tag", required=True)
    build = commands.add_parser("package")
    build.add_argument("--binary", type=Path, required=True)
    build.add_argument("--target", choices=TARGETS, required=True)
    build.add_argument("--tag", required=True)
    build.add_argument("--output", type=Path, required=True)
    sums = commands.add_parser("checksums")
    sums.add_argument("directory", type=Path)
    sums.add_argument("--expect-count", type=int)
    args = parser.parse_args()
    try:
        if args.command == "check-version":
            print(version_for_tag(args.tag))
        elif args.command == "package":
            print(package(args.binary, args.target, version_for_tag(args.tag), args.output))
        else:
            print(checksums(args.directory, args.expect_count))
    except (OSError, ValueError, KeyError) as error:
        print(f"package_release: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
