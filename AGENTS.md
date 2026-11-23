# zaccel

Read README.md and docs/architecture.md. Use Rust 2024, native wgpu and Naga. Keep raw buffer I/O independent of diagnostics. All allocation and dispatch arithmetic must be checked before GPU submission. WGSL validation must work without a GPU. Software adapters require explicit opt-in; never promise arbitrary CPU fallback. A process timeout does not reset a hung GPU driver.

Do not start additional agents. Preserve other workers' file ownership. Keep binary outputs, shader data and benchmarks' generated files outside Git; source shaders and textual evidence belong in the repository. Use existing cached dependencies where possible.

On the shared development machine use CARGO_TARGET_DIR=/var/tmp/project-z-cargo-target and hold /var/tmp/project-z-cargo-target/.project-z-cargo-owner.lock once around the outer Cargo batch. The coordinator owns the build lock. Do not start concurrent Cargo batches. Local validation precedes source publication; retain formal cross-platform release builds. Report exact tested platforms and measured results.
