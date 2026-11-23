"""CPU-only checks for fixture integrity, acceptance outcomes and archive contents."""

import argparse
import json
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import acceptance
import fixtures
import package_release


class FixtureTests(unittest.TestCase):
    def test_tail_dispatch_and_all_reference_results(self):
        with tempfile.TemporaryDirectory(prefix="zaccel-support-") as temporary:
            directory = fixtures.generate(Path(temporary), 65)
            manifest = json.loads((directory / "affine.json").read_text())
            self.assertEqual(manifest["dispatch"], [2, 1, 1])
            self.assertEqual(manifest["buffers"][2]["size"], 260)
            self.assertEqual((directory / "data/params.bin").stat().st_size, 16)
            for operation in fixtures.OPERATIONS:
                output = directory / "data" / f"{operation}.bin"
                values = fixtures.expected_values(directory, operation)
                output.write_bytes(b"".join(struct.pack("<f", v) for v in values))
                self.assertEqual(fixtures.verify(directory, operation)["elements"], 65)

    def test_truncation_nan_and_wrong_values_fail(self):
        with tempfile.TemporaryDirectory(prefix="zaccel-support-") as temporary:
            directory = fixtures.generate(Path(temporary), 1)
            output = directory / "data/add.bin"
            for raw in (b"", struct.pack("<f", float("nan")), struct.pack("<f", 999.0)):
                output.write_bytes(raw)
                with self.assertRaises(ValueError):
                    fixtures.verify(directory, "add")

    def test_generation_preserves_existing_files_and_bounds_size(self):
        with tempfile.TemporaryDirectory(prefix="zaccel-support-") as temporary:
            directory = Path(temporary)
            (directory / "keep.txt").write_text("keep")
            with self.assertRaises(ValueError):
                fixtures.generate(directory)
            self.assertEqual((directory / "keep.txt").read_text(), "keep")
        for count in (0, -1, fixtures.MAX_COUNT + 1, True):
            with self.assertRaises(ValueError):
                fixtures.checked_count(count)


class AcceptanceTests(unittest.TestCase):
    def test_only_the_explicit_empty_device_result_can_be_skipped(self):
        self.assertTrue(acceptance.no_adapter_reply(1, b"[]\n", b"zaccel: no GPU adapters found; check driver\n"))
        self.assertFalse(acceptance.no_adapter_reply(1, b"[]", b"zaccel: device lost\n"))
        self.assertFalse(acceptance.no_adapter_reply(124, b"[]", b"zaccel: no GPU adapters found; check driver\n"))
        self.assertFalse(acceptance.no_adapter_reply(1, b"", b"zaccel: no GPU adapters found; check driver\n"))

    def test_software_is_not_counted_as_hardware(self):
        adapters = [{"index": 0, "device_type": "Cpu", "name": "software"},
                    {"index": 1, "device_type": "DiscreteGpu", "name": "hardware"}]
        self.assertEqual(acceptance.choose_hardware(adapters, None)["index"], 1)
        self.assertIsNone(acceptance.choose_hardware(adapters[:1], None))
        with self.assertRaises(acceptance.CheckFailure):
            acceptance.choose_hardware(adapters, 0)

    def test_missing_hardware_is_skip_or_required_failure(self):
        info = {"entry_points": [{"name": "main", "workgroup_size": [64, 1, 1]}], "bindings": []}

        def fake_invoke(command, *unused, **kwargs):
            if command[1] == "inspect":
                return json.dumps(info).encode()
            return b"[]" if command[1] == "devices" else b""

        with tempfile.TemporaryDirectory(prefix="zaccel-support-") as temporary:
            binary = Path(temporary) / "fake-zaccel"
            binary.write_bytes(b"fixture only, not an executable")
            for required, code, status in ((False, 77, "skipped"), (True, 1, "failed")):
                args = argparse.Namespace(bin=str(binary), count=65, iterations=2, warmup=1,
                                          timeout_ms=1000, gpu=True, require_gpu=required, adapter=None)
                report = {"cases": [], "results": [], "gpu": {"status": "not_requested"}}
                with patch.object(acceptance, "invoke", side_effect=fake_invoke):
                    self.assertEqual(acceptance.run_checks(args, report), code)
                self.assertEqual(report["status"], status)
                self.assertEqual(report["gpu"]["status"], "unavailable")

    def test_benchmark_distinguishes_optional_gpu_samples(self):
        timings = {name: 0.0 for name in ("initialization_ms", "pipeline_ms", "upload_ms", "readback_ms", "total_ms")}
        timings.update(dispatch_wall_ms=[1.0, 2.0], kernel_ms=None)
        acceptance.check_timings({"timings": timings}, 2)
        timings["kernel_ms"] = [0.5]
        with self.assertRaises(acceptance.CheckFailure):
            acceptance.check_timings({"timings": timings}, 2)
        timings["kernel_ms"] = [0.5, float("nan")]
        with self.assertRaises(acceptance.CheckFailure):
            acceptance.check_timings({"timings": timings}, 2)


class PackageTests(unittest.TestCase):
    def test_archives_contain_binary_sources_and_checksums(self):
        with tempfile.TemporaryDirectory(prefix="zaccel-packaging-") as temporary:
            base = Path(temporary)
            source = base / "source"
            (source / "scripts").mkdir(parents=True)
            for name in ("README.md", "LICENSE", "scripts/fixtures.py", "scripts/acceptance.py"):
                (source / name).write_text("packaging fixture\n")
            binary = base / "binary"
            binary.write_bytes(b"not a compiled binary; archive fixture\n")
            destination = base / "artifacts"
            linux = package_release.package(binary, "x86_64-unknown-linux-gnu", "0.1.0", destination, source)
            windows = package_release.package(binary, "x86_64-pc-windows-msvc", "0.1.0", destination, source)
            with tarfile.open(linux) as archive:
                members = archive.getmembers()
                executable = next(m for m in members if m.name.endswith("/zaccel"))
                self.assertEqual(executable.mode, 0o755)
                self.assertEqual(archive.extractfile(executable).read(), binary.read_bytes())
                self.assertTrue(any(m.name.endswith("/LICENSE") for m in members))
            with zipfile.ZipFile(windows) as archive:
                executable = next(n for n in archive.namelist() if n.endswith("/zaccel.exe"))
                self.assertEqual(archive.read(executable), binary.read_bytes())
            sums = package_release.checksums(destination, 2).read_text()
            self.assertIn(package_release.sha256(linux) + "  " + linux.name, sums)
            self.assertIn(package_release.sha256(windows) + "  " + windows.name, sums)
            with self.assertRaises(ValueError):
                package_release.package(binary, "x86_64-unknown-linux-gnu", "0.1.0", destination, source)
            with self.assertRaises(ValueError):
                package_release.checksums(destination, 5)

    def test_tag_must_match_manifest(self):
        with tempfile.TemporaryDirectory(prefix="zaccel-packaging-") as temporary:
            source = Path(temporary)
            (source / "Cargo.toml").write_text('[package]\nversion = "0.1.0"\n')
            self.assertEqual(package_release.version_for_tag("v0.1.0", source), "0.1.0")
            with self.assertRaises(ValueError):
                package_release.version_for_tag("v0.2.0", source)


if __name__ == "__main__":
    unittest.main()
