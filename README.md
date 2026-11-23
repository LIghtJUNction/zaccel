# zaccel

Run WGSL compute programs from the terminal. Give zaccel a shader, explicit buffer bytes and workgroup counts; it validates the program with Naga, executes it through native wgpu, and writes raw result bytes.

The host still reads files, builds the GPU pipeline and moves data. The GPU performs the numeric work written in the shader. This is useful for independent array operations, transforms and other workloads that fit a compute kernel. Measure the complete command before deciding whether a particular workload benefits.

## Build

From a source checkout with Rust installed:

```sh
cargo install --path . --locked
zaccel --version
```

The repository selects its Rust toolchain in [rust-toolchain.toml](rust-toolchain.toml) and uses edition 2024. Native execution requires a usable graphics driver: Vulkan on Linux, DX12 or Vulkan on Windows, and Metal on macOS. The [release workflow](.github/workflows/release.yml) packages Linux x86-64/Arm64, Windows x86-64 and macOS Intel/Apple Silicon executables when a matching version tag is published.

## First run

Validation and inspection work without a GPU:

```sh
zaccel validate examples/vector_add.wgsl
zaccel inspect examples/affine.wgsl --entry main --json
```

Before execution, list the adapters with `zaccel devices --json`. A missing adapter is reported as an error by the CLI.

The examples use Python 3.11 or newer and only its standard library. Generate inputs in a fresh directory outside the checkout. This Bash example creates 1,024 pairs of little-endian `f32` values:

```sh
fixture_dir=$(mktemp -d)
python3 scripts/fixtures.py generate "$fixture_dir" --count 1024

zaccel run "$fixture_dir/vector_add.wgsl" \
  --entry main --dispatch 16,1,1 \
  --input "$fixture_dir/data/pairs.bin" \
  --size 4096 --output "$fixture_dir/data/add.bin"

python3 scripts/fixtures.py verify "$fixture_dir" --operation add
```

Each input element contains `(a, b)` in eight bytes. Each output element is `a + b` in four bytes, so `--size 4096` deliberately allocates a smaller output than the 8,192-byte input. The shader uses 64 invocations per workgroup; 16 workgroups cover 1,024 elements. Both the input and output lengths are checked inside the shader. See [the source](examples/vector_add.wgsl).

Run the uniform and multiple-binding example from the same generated directory:

```sh
zaccel run --manifest "$fixture_dir/affine.json"
python3 scripts/fixtures.py verify "$fixture_dir" --operation affine
```

This computes `2*a - 0.5*b + 1.25` using two input buffers, one output buffer and a 16-byte uniform. All paths in the manifest resolve relative to that manifest, so the command works from any current directory.

## Commands

| Command | Purpose |
| --- | --- |
| `devices [--json]` | List adapter indices, backends, limits and timestamp support. |
| `validate SHADER` | Parse and validate WGSL without initializing a GPU. Use `-` for shader stdin. |
| `inspect SHADER [--entry NAME] [--json]` | Report compute entries, workgroup sizes and buffer bindings. |
| `run SHADER [OPTIONS]` | Execute one dispatch using simple input/output bindings. |
| `run --manifest JOB.json [OPTIONS]` | Execute an explicit storage/uniform binding layout. |
| `eval 'WGSL' [OPTIONS]` | Execute source passed as one command argument. |
| `bench SHADER [OPTIONS]` | Warm up, repeat dispatches and report separate timing stages. A manifest is also accepted. |

Use `--adapter INDEX` to select an adapter from the current `devices` result. Software adapters require `--allow-software`. zaccel has no automatic CPU implementation for arbitrary WGSL.

Raw buffers have no file header or inferred element type. Match the shader's byte order, layout and size explicitly. `--input -` and `--output -` use binary stdin/stdout. Execution diagnostics go to stderr. Bench JSON goes to stdout when a result buffer is not using stdout; otherwise it goes to stderr. [Usage and binary layouts](docs/usage.md) cover all three examples.

## Limits and timing

The default limits are 256 MiB for accounted GPU buffers, 1,048,576 workgroups per dispatch and a 30,000 ms GPU worker deadline. The buffer budget covers resident buffers, uploaded data staging and readback staging; it is not a cap on the complete driver or operating system's memory use. Adapter-specific limits apply as well. [Architecture and limits](docs/architecture.md) explain the checks and their boundaries.

The parent requests termination when the worker exceeds its deadline and exits with code 124. A GPU kernel or a driver call can outlive that process. The watchdog does not reset the GPU or provide a hostile-shader isolation boundary.

Bench reuses one device, pipeline, binding layout and buffer set. Warmups and measured dispatches retain buffer state. Host submit/wait samples and optional GPU timestamp samples measure different intervals; initialization, upload, readback, output writing and whole-command costs must remain visible when comparing with a CPU program. [Validation and measurement](docs/validation.md) describes the report fields and acceptance procedure.

## Check a built binary

```sh
# Offline checks only; does not enumerate or submit work to a GPU.
python3 scripts/acceptance.py --bin /path/to/zaccel

# Explicit real-hardware acceptance; absent hardware is a failure.
python3 scripts/acceptance.py --bin /path/to/zaccel \
  --require-gpu --report /tmp/zaccel-gpu.json
```

The GPU suite verifies vector addition, elementwise multiplication, a uniform manifest, raw binary stdin/stdout, inline source and bench outputs against a scalar reference. `--gpu` permits an explicit skip with exit 77 when no hardware adapter is available; `--require-gpu` makes that condition fail. Driver errors, validation errors, timeouts and incorrect numerical results fail in both modes. These commands produce evidence when executed; this README makes no performance claim.

Daily CI is [manually dispatched](.github/workflows/ci.yml). [Hardware acceptance](.github/workflows/gpu-acceptance.yml) is a separate manual workflow on a deliberately labelled GPU runner. [Release instructions](docs/releasing.md) describe version tags, checksums and optional crates.io publication.

## License

[MIT](LICENSE).
