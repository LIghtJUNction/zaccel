# Architecture and limits

## Separate planning from GPU execution

The CPU prepares a finite description of the work before the GPU receives it. WGSL provides the operations, the host provides buffer bytes and bindings, and dispatch counts determine how many workgroups are submitted. zaccel performs no data-type inference from input filenames or file contents.

| Component | Responsibility | Source |
| --- | --- | --- |
| CLI | Parse arguments, supervise GPU commands, separate binary output and reports. | [src/main.rs](../src/main.rs) |
| Shader inspection | Naga parsing and validation; entry-point, binding and layout metadata. | [src/shader.rs](../src/shader.rs) |
| Job planning | Bounded source/data reads, manifest paths, binding consistency and output targets. | [src/plan.rs](../src/plan.rs) |
| Native execution | Adapter and device selection, buffer allocation, pipeline creation, dispatch, readback and timings. | [src/gpu.rs](../src/gpu.rs) |
| Public data model | Explicit buffers, options, adapter information and timing fields. | [src/lib.rs](../src/lib.rs) |

`validate` and `inspect` finish after CPU-side analysis. `devices`, `run`, `eval` and `bench` use a supervised child for native GPU interaction. A new CLI invocation creates a new worker; there is no persistent GPU daemon or model runtime.

## What the checks bound

Shader input is limited to 1 MiB. Plans check sizes, arithmetic, duplicate slots, output destinations, workgroup products and compatible storage/uniform bindings. The selected adapter supplies additional per-buffer, workgroup and feature limits. Naga validation checks WGSL structure and types; the compute pipeline still needs to be accepted by the actual driver.

The default byte budget is 256 MiB. Accounting includes resident buffers, staging for supplied upload data and readback staging. The output buffer and its readback allocation both consume budget. Driver bookkeeping, compiler memory, device-wide contention and unrelated processes remain outside this application accounting. Increasing the budget does not reserve VRAM or guarantee allocation success.

The workgroup product limit applies to each submitted dispatch. Bench performs the requested warmups and measured repetitions on the same buffers. Neither workgroup count nor buffer size is a proof of shader execution time: a shader can contain long-running loops or otherwise expensive work.

## Process timeout and driver behavior

The parent observes the GPU child against `--timeout-ms`. At the deadline it requests termination, performs a bounded cleanup attempt and reports exit 124. This gives the calling application a bounded supervision path. It does not guarantee interruption of a kernel already submitted to the GPU, cancellation of a stuck kernel-space driver call or recovery of the display device.

Treat untrusted shaders as untrusted native GPU workloads. Review loop bounds and memory access patterns before using a shared display GPU. After a timeout or device-loss error, inspect the actual driver and device state before repeating the job. A process exit alone is not a device-reset receipt. The acceptance suite deliberately uses finite, bounds-checked kernels; it never tests recovery by hanging the GPU.

## Hardware and software adapters

Adapter enumeration records the name, backend, device type, limits and timestamp capability. The CLI requires an explicit opt-in for software adapters. A software Vulkan implementation can execute WGSL through wgpu when available, but zaccel does not translate arbitrary WGSL into a separate CPU implementation or silently substitute a CPU algorithm.

The application remains limited to the native backends configured in [Cargo.toml](../Cargo.toml): Vulkan, DX12 and Metal. The presence of a GPU or its advertised VRAM is only part of eligibility; driver support and the selected shader's requirements also matter.

## Timing boundaries

Bench keeps the device, pipeline, bind groups and resident buffers alive across repetitions. Upload happens before repeated dispatches and final readback happens afterward. This makes dispatch measurements useful for workloads that can keep data on the device, while the separate setup and transfer stages show the cost of a one-shot command.

Warmups modify the same buffers used by measurements. An in-place shader therefore changes its input on each dispatch; benchmark it with that progression in mind. The supplied pair-input examples preserve their read-only inputs and overwrite the output on every repetition.

`dispatch_wall_ms` measures host-observed dispatch submission/completion intervals. `kernel_ms` contains optional hardware timestamp intervals and stays absent or null when unavailable. They are separate observations, with different overheads and clock domains. The [measurement guide](validation.md) explains how to preserve both and report whole-command latency.
