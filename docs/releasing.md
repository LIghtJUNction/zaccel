# Local checks and releases

## Daily work

Daily development uses local checks. [ci.yml](../.github/workflows/ci.yml) has only a `workflow_dispatch` trigger, so ordinary pushes and pull requests do not start routine CI. A maintainer can request the Linux, Windows and macOS matrix explicitly. It runs formatting, Clippy, Rust tests, Python support tests and offline validation of the compiled CLI.

[gpu-acceptance.yml](../.github/workflows/gpu-acceptance.yml) is a separate manual workflow. It requires a self-hosted Linux x86-64 runner labelled `zaccel-gpu`, a working native GPU driver, Python 3.11 or newer, Rust and `flock`. It acquires `/var/tmp/project-z-cargo-target/.project-z-cargo-owner.lock` once around the build and acceptance batch. `--require-gpu` makes unavailable hardware fail the job. The workflow preserves its actual report and does not silently substitute a software adapter.

## Prepare a version

1. Finish local validation for the exact source and committed lockfile. Retain the actual hardware acceptance report when claiming GPU correctness for a release.
2. Update the package version and ensure the working tree is ready for publication. Preserve text evidence and links; keep compiled binaries, generated shader data and archives outside Git history.
3. Create and push the matching `vMAJOR.MINOR.PATCH` tag only when publishing that version is authorized. The tag workflow is the formal build and publication trigger.

The [release workflow](../.github/workflows/release.yml) rejects a tag that differs from `Cargo.toml`. It uses the checked-in Rust toolchain and `--locked` resolution. Each native runner tests and builds its target, runs offline acceptance, and packages the executable, license, documentation, shaders and Python fixture/acceptance helpers.

| Target | Runner | Archive |
| --- | --- | --- |
| `x86_64-unknown-linux-gnu` | Ubuntu 24.04 x86-64 | `.tar.gz` |
| `aarch64-unknown-linux-gnu` | Ubuntu 24.04 Arm64 | `.tar.gz` |
| `x86_64-pc-windows-msvc` | Windows x86-64 | `.zip` |
| `x86_64-apple-darwin` | macOS 15 Intel | `.tar.gz` |
| `aarch64-apple-darwin` | macOS Apple Silicon | `.tar.gz` |

The runner labels follow the [GitHub-hosted runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners). A native build or offline test on a hosted runner is separate from real GPU execution. Linux archives use the GNU target and are not static musl builds; system-library compatibility follows the actual build environment.

## Archives and checksums

`scripts/package_release.py` packages an already-built executable. It never invokes Cargo, rejects an existing archive name, writes archives outside the checkout and uses fixed archive metadata. Unix archives retain an executable mode on the binary. The combined checksum file is created only after all five expected archives are present.

To exercise packaging locally after a build:

```sh
python3 scripts/package_release.py check-version --tag v0.1.1
python3 scripts/package_release.py package \
  --binary /path/to/built/zaccel \
  --target x86_64-unknown-linux-gnu --tag v0.1.1 \
  --output /tmp/zaccel-release
python3 scripts/package_release.py checksums /tmp/zaccel-release
```

Use the actual package version in place of the example tag. The formal publish job uses `--expect-count 5`, downloads the build artifacts, writes `SHA256SUMS` and creates the GitHub release for the verified existing tag. It does not overwrite existing release assets on a rerun. Inspect any partially published release before deciding how to resume it.

On Linux, run `sha256sum -c SHA256SUMS` after downloading the complete asset set. For one archive on macOS use `shasum -a 256 ARCHIVE`; on Windows use `Get-FileHash ARCHIVE -Algorithm SHA256` and compare the displayed value with its checksum entry. Checksums detect changed bytes relative to the checksum file; they are not a separate signing identity.

## Credentials and publication

Build and test jobs have `contents: read`. Only the GitHub-release job has `contents: write`, and checkout does not persist credentials. Official actions are pinned to exact commits, with version comments. Their releases are [checkout v7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1), [setup-python v7.1.0](https://github.com/actions/setup-python/releases/tag/v7.1.0), [upload-artifact v7.0.2](https://github.com/actions/upload-artifact/releases/tag/v7.0.2) and [download-artifact v8.0.2](https://github.com/actions/download-artifact/releases/tag/v8.0.2). Keep a self-hosted runner compatible with these actions' runtime requirements.

crates.io publication is disabled by default. Enabling repository variable `ZACCEL_PUBLISH_CRATE=true` adds the tagged-release publish job; it requires secret `CARGO_REGISTRY_TOKEN` and runs `cargo publish --locked`. Missing credentials fail that explicitly enabled job. The GitHub token is not a crates.io token.

Authoring these workflows does not execute them. A release claim should name the actual tag, completed build jobs, produced assets and checksums, and any separately collected GPU evidence.
