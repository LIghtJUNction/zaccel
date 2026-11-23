# Validation and measurement

## Four separate checks

| Check | Command or source | What a successful result establishes |
| --- | --- | --- |
| Support scripts | `python3 -m unittest discover -s scripts -p 'test_*.py' -v` | Fixture layout, corruption detection, skip/fail classification, timing report checks and archive packaging behave as tested. No GPU is used. |
| CLI analysis | `cargo test --locked --test cli` | The compiled CLI validates and inspects shipped WGSL, accepts shader stdin and rejects invalid input without a GPU. |
| Offline acceptance | `python3 scripts/acceptance.py --bin /path/to/zaccel` | Every shipped shader validates and its reported entry metadata is consistent. The script does not request adapter enumeration or a dispatch. |
| Hardware acceptance | `python3 scripts/acceptance.py --bin /path/to/zaccel --require-gpu` | A selected physical adapter executed the actual examples and its readback matched the reference checks. |

Run Rust commands under the shared target lock when using the Project Z development machine, as specified in [AGENTS.md](../AGENTS.md). The coordinator owns that Cargo batch during concurrent work. The Python tests need only the standard library and can run separately on a sandbox or an ordinary development host.

The development coordinator reported an NVIDIA GeForce RTX 4060 Laptop GPU with 8,188 MiB and driver 620.73.04 for this integration. That is an inventory observation. Executable correctness and performance results belong in a report from the specific built binary and selected adapter; this document supplies no measured throughput or speedup figure.

## Explicit hardware outcomes

The acceptance script emits a JSON report even when a check fails. Its `scope`, `status`, `gpu.status`, individual command exit codes and numerical checks retain their different meanings:

| Invocation and outcome | Exit | Report |
| --- | --- | --- |
| Default invocation, offline checks pass | 0 | `scope: offline`, `status: passed`, `gpu.status: not_requested` |
| `--gpu`, hardware cases pass | 0 | `scope: offline_and_gpu`, `gpu.status: passed` |
| `--gpu`, enumeration finds no discrete/integrated adapter | 77 | `status: skipped`, `gpu.status: unavailable` |
| `--require-gpu`, hardware absent | 1 | `status: failed`, `gpu.status: unavailable` |
| Invalid shader, wrong result, driver error or timeout | 1 | `status: failed`, plus the original failing command and its exit code |

The CLI's explicit empty-adapter result is `[]` together with exit 1 and its `no GPU adapters found` diagnostic. The acceptance script recognizes that precise result as unavailable hardware. An arbitrary enumeration error, empty stdout, device-loss diagnostic or exit 124 remains a failure. Software-only adapters are not counted as hardware acceptance, and selecting an absent or software adapter explicitly fails.

The GPU suite generates its own data in an OS temporary directory and removes that directory when finished. It covers addition, elementwise multiplication, a uniform with two binding groups, binary stdin/stdout, inline WGSL and repeated dispatches. Comparisons reject truncated files, non-finite values and numerical mismatches. The test kernels check both input and output bounds and include no deliberately hanging shader.

Save a report outside Git when running the suite:

```sh
python3 scripts/acceptance.py --bin /path/to/zaccel \
  --require-gpu --count 4096 --iterations 5 --warmup 2 \
  --report /tmp/zaccel-acceptance.json
```

The report records the executable SHA-256, platform, actual adapter, invoked commands, command exit codes, elapsed CLI times and comparison results. Keep the source commit and whether it had local changes alongside that report. GPU command stderr includes the program's execution report; raw stdout is checked independently as binary data.

## Read a benchmark

Use a complete fixture and preserve both the output and JSON report:

```sh
fixture_dir=$(mktemp -d)
python3 scripts/fixtures.py generate "$fixture_dir" --count 65536
zaccel bench "$fixture_dir/vector_mul.wgsl" \
  --input "$fixture_dir/data/pairs.bin" --size 262144 \
  --output "$fixture_dir/data/mul.bin" --dispatch 1024,1,1 \
  --iterations 20 --warmup 3 > "$fixture_dir/bench.json"
python3 scripts/fixtures.py verify "$fixture_dir" --operation mul
```

| Field | Interpretation |
| --- | --- |
| `preparation_ms` | Host job planning and input preparation before native execution. |
| `timings.initialization_ms` | Native adapter/device setup measured by the execution layer. |
| `timings.pipeline_ms` | Pipeline preparation measured by the execution layer. |
| `timings.upload_ms` | Initial data transfer stage. |
| `timings.dispatch_wall_ms` | One host-observed submit/completion sample per measured dispatch. |
| `timings.kernel_ms` | Optional GPU timestamp samples. Null means this measurement was unavailable. |
| `timings.readback_ms` | Final result readback. |
| `timings.total_ms` | Execution-layer wall time across its measured stages. |
| `output_write_ms` | Writing returned bytes to the requested files or stdout. |
| `worker_total_ms` | The worker's total time for planning, execution and output writing. |
| Acceptance `cases[].cli_wall_ms` | External process startup, supervision, output collection and command completion. |

The CLI also reports sample summaries in `dispatch_wall` and `kernel`. Keep the raw samples when calculating a different percentile or comparing runs. Timestamp support is optional; do not relabel a host sample as hardware kernel time when timestamps are unavailable.

Warmups and measurements share buffer state. An in-place shader can compound its transformation across all dispatches, including warmups. For an independent repetition, design an idempotent kernel or arrange separate input and output buffers. The shipped pair-input kernels read unchanged inputs and replace their output each time.

For a speed comparison, first verify equal output and equivalent numeric precision. Report data size, operations, adapter, driver, build profile, repetitions and whether allocation/transfers are included. A one-shot CLI should be compared with the CPU's complete one-shot path; a resident GPU workload should be described as resident. The fixture reference prioritizes correctness and is not an optimized CPU performance baseline.

## Reproduce failures

Retain the failing report, shader and text manifest. Copy only the smallest required binary inputs into an artifact destination outside Git. A missing driver, a rejected allocation, a process timeout and an incorrect result are separate failure modes; record the actual message and exit code before deciding what to change. After a timeout, inspect the device state before retrying. The [architecture guide](architecture.md) explains the driver-recovery limit.
