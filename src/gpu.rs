//! Native compute execution. All budgets below count logical buffer allocations;
//! driver heaps, shader compiler memory and command bookkeeping remain driver-owned.

use std::{
    borrow::Cow,
    collections::BTreeMap,
    future::Future,
    sync::{Arc, Mutex, mpsc},
    task::{Context as TaskContext, Poll, Wake, Waker},
    thread,
    time::{Duration, Instant},
};

use crate::{
    AdapterInfo, BufferData, BufferKind, BufferOutput, Result, RunPlan, RunReport, ShaderInfo,
    Timings,
};

const MAX_REPETITIONS: u32 = 10_000;
const TIMESTAMP_BYTES: u64 = 16;
const TIMESTAMP_ALLOCATION: u64 = TIMESTAMP_BYTES * 2;

struct Deadline {
    end: Instant,
}

impl Deadline {
    fn new(start: Instant, milliseconds: u64) -> Result<Self> {
        if milliseconds == 0 {
            return Err("GPU timeout must be greater than zero milliseconds".into());
        }
        let end = start
            .checked_add(Duration::from_millis(milliseconds))
            .ok_or("GPU timeout exceeds the monotonic clock range")?;
        Ok(Self { end })
    }

    fn remaining(&self, phase: &str) -> Result<Duration> {
        self.end
            .checked_duration_since(Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .ok_or_else(|| format!("GPU timeout during {phase}"))
    }
}

struct ThreadWake(thread::Thread);

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

/// A small executor for the native wgpu futures, with no runtime dependency.
/// Native driver calls can still block inside a poll; the CLI watchdog handles
/// that process boundary. This deadline does not claim to reset a GPU.
fn wait_future<F: Future>(
    future: F,
    deadline: &Deadline,
    phase: &str,
    device: Option<&wgpu::Device>,
    errors: Option<&ErrorState>,
) -> Result<F::Output> {
    let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
    let mut context = TaskContext::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        deadline.remaining(phase)?;
        if let Some(errors) = errors {
            errors.check(phase)?;
        }
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            deadline.remaining(phase)?;
            return Ok(value);
        }
        if let Some(device) = device {
            device
                .poll(wgpu::PollType::Poll)
                .map_err(|error| format!("GPU poll failed during {phase}: {error}"))?;
        }
        thread::park_timeout(deadline.remaining(phase)?.min(Duration::from_millis(2)));
    }
}

#[derive(Clone, Default)]
struct ErrorState(Arc<Mutex<Option<String>>>);

impl ErrorState {
    fn record(&self, message: String) {
        let mut slot = self.0.lock().unwrap_or_else(|error| error.into_inner());
        if slot.is_none() {
            *slot = Some(message);
        }
    }

    fn install(device: &wgpu::Device) -> Self {
        let state = Self::default();
        let uncaptured = state.clone();
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            uncaptured.record(describe_gpu_error(&error));
        }));
        let lost = state.clone();
        device.set_device_lost_callback(move |reason, message| {
            lost.record(format!("GPU device lost ({reason:?}): {message}"));
        });
        state
    }

    fn check(&self, phase: &str) -> Result<()> {
        let slot = self.0.lock().unwrap_or_else(|error| error.into_inner());
        match slot.as_ref() {
            Some(error) => Err(format!("{phase}: {error}")),
            None => Ok(()),
        }
    }
}

fn describe_gpu_error(error: &wgpu::Error) -> String {
    match error {
        wgpu::Error::OutOfMemory { .. } => format!("GPU out of memory: {error}"),
        wgpu::Error::Validation { .. } => format!("GPU validation error: {error}"),
        wgpu::Error::Internal { .. } => format!("GPU internal/driver error: {error}"),
    }
}

struct GpuContext<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    errors: &'a ErrorState,
    deadline: &'a Deadline,
}

impl GpuContext<'_> {
    fn checked<T>(&self, phase: &str, work: impl FnOnce() -> Result<T>) -> Result<T> {
        self.deadline.remaining(phase)?;
        self.errors.check(phase)?;
        let internal = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
        let memory = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = work();

        // Pop all guards in reverse order before awaiting any result. This also
        // keeps the scope stack balanced if a future reaches the deadline.
        let validation_future = validation.pop();
        let memory_future = memory.pop();
        let internal_future = internal.pop();
        let validation_result = wait_future(
            validation_future,
            self.deadline,
            phase,
            Some(self.device),
            Some(self.errors),
        );
        let memory_result = wait_future(
            memory_future,
            self.deadline,
            phase,
            Some(self.device),
            Some(self.errors),
        );
        let internal_result = wait_future(
            internal_future,
            self.deadline,
            phase,
            Some(self.device),
            Some(self.errors),
        );
        self.errors.check(phase)?;
        // OOM and driver failures explain the invalid objects they may leave.
        for captured in [memory_result, internal_result, validation_result] {
            if let Some(error) = captured? {
                return Err(format!("{phase}: {}", describe_gpu_error(&error)));
            }
        }
        self.deadline.remaining(phase)?;
        result
    }

    fn wait_submission(&self, index: wgpu::SubmissionIndex, phase: &str) -> Result<()> {
        self.errors.check(phase)?;
        let result = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: Some(self.deadline.remaining(phase)?),
        });
        self.errors.check(phase)?;
        match result {
            Ok(_) => {
                self.deadline.remaining(phase)?;
                Ok(())
            }
            Err(wgpu::PollError::Timeout) => Err(format!("GPU timeout during {phase}")),
            Err(error) => Err(format!(
                "GPU submission wait failed during {phase}: {error}"
            )),
        }
    }

    fn read_mapping<T>(
        &self,
        buffer: &wgpu::Buffer,
        phase: &str,
        read: impl FnOnce(&[u8]) -> Result<T>,
    ) -> Result<T> {
        self.deadline.remaining(phase)?;
        self.errors.check(phase)?;
        let (sender, receiver) = mpsc::sync_channel(1);
        buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send(result);
        });
        let mapping = UnmapOnDrop(buffer);
        loop {
            self.deadline.remaining(phase)?;
            self.device
                .poll(wgpu::PollType::Poll)
                .map_err(|error| format!("GPU poll failed during {phase}: {error}"))?;
            self.errors.check(phase)?;
            match receiver.try_recv() {
                Ok(Ok(())) => break,
                Ok(Err(error)) => {
                    return Err(format!("GPU buffer mapping failed during {phase}: {error}"));
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err(format!(
                        "GPU buffer mapping callback was lost during {phase}"
                    ));
                }
                Err(mpsc::TryRecvError::Empty) => {
                    thread::park_timeout(
                        self.deadline
                            .remaining(phase)?
                            .min(Duration::from_millis(2)),
                    );
                }
            }
        }
        let view = buffer
            .get_mapped_range(..)
            .map_err(|error| format!("GPU mapped range failed during {phase}: {error}"))?;
        let result = read(&view);
        drop(view);
        drop(mapping);
        self.errors.check(phase)?;
        self.deadline.remaining(phase)?;
        result
    }
}

struct UnmapOnDrop<'a>(&'a wgpu::Buffer);

impl Drop for UnmapOnDrop<'_> {
    fn drop(&mut self) {
        self.0.unmap();
    }
}

fn native_backends() -> wgpu::Backends {
    wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL
}

fn native_instance() -> Result<wgpu::Instance> {
    let backends = native_backends() & wgpu::Instance::enabled_backend_features();
    if backends.is_empty() {
        return Err(
            "this build has no native Vulkan, DX12 or Metal backend for this platform".into(),
        );
    }
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = backends;
    Ok(wgpu::Instance::new(descriptor))
}

fn software_adapter(info: &wgpu::AdapterInfo) -> bool {
    if info.device_type == wgpu::DeviceType::Cpu {
        return true;
    }
    let identity =
        format!("{} {} {}", info.name, info.driver, info.driver_info).to_ascii_lowercase();
    [
        "llvmpipe",
        "lavapipe",
        "softpipe",
        "swiftshader",
        "microsoft basic render driver",
    ]
    .iter()
    .any(|name| identity.contains(name))
}

fn adapter_info(index: usize, adapter: &wgpu::Adapter) -> AdapterInfo {
    let info = adapter.get_info();
    let limits = adapter.limits();
    let device_type = if software_adapter(&info) {
        format!("{:?} (software)", info.device_type)
    } else {
        format!("{:?}", info.device_type)
    };
    AdapterInfo {
        index,
        name: info.name,
        vendor: info.vendor,
        device: info.device,
        device_type,
        backend: format!("{:?}", info.backend),
        driver: info.driver,
        driver_info: info.driver_info,
        timestamp_query: adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY),
        max_buffer_size: limits.max_buffer_size,
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
    }
}

/// Enumerate every native adapter, including explicitly labelled software ones.
/// Indices are taken from this same enumeration order when selecting a device.
pub fn devices() -> Result<Vec<AdapterInfo>> {
    let start = Instant::now();
    let deadline = Deadline::new(start, 30_000)?;
    let instance = native_instance()?;
    let adapters = wait_future(
        instance.enumerate_adapters(native_backends()),
        &deadline,
        "adapter enumeration",
        None,
        None,
    )?;
    Ok(adapters
        .iter()
        .enumerate()
        .map(|(index, adapter)| adapter_info(index, adapter))
        .collect())
}

fn select_adapter(adapters: &[wgpu::Adapter], plan: &RunPlan) -> Result<usize> {
    if let Some(index) = plan.options.adapter {
        let adapter = adapters.get(index).ok_or_else(|| {
            format!(
                "adapter index {index} does not exist; found {} adapters (run zaccel devices)",
                adapters.len()
            )
        })?;
        if software_adapter(&adapter.get_info()) && !plan.options.allow_software {
            return Err(format!(
                "adapter {index} is a CPU/software adapter; explicitly use --allow-software to select it",
            ));
        }
        return Ok(index);
    }
    adapters
        .iter()
        .enumerate()
        .filter_map(|(index, adapter)| {
            let info = adapter.get_info();
            if software_adapter(&info) {
                return plan.options.allow_software.then_some((index, 5));
            }
            let rank = match info.device_type {
                wgpu::DeviceType::DiscreteGpu => 0,
                wgpu::DeviceType::IntegratedGpu => 1,
                wgpu::DeviceType::VirtualGpu => 2,
                _ => 3,
            };
            Some((index, rank))
        })
        .min_by_key(|(index, rank)| (*rank, *index))
        .map(|(index, _)| index)
        .ok_or_else(|| {
            if adapters.is_empty() {
                "no native GPU adapter is available; check the Vulkan, DX12 or Metal driver".into()
            } else {
                "no hardware GPU adapter is available; CPU/software adapters require explicit --allow-software".into()
            }
        })
}

fn product(dimensions: [u32; 3], label: &str) -> Result<u64> {
    dimensions.into_iter().try_fold(1_u64, |total, value| {
        if value == 0 {
            return Err(format!("{label} dimensions must all be greater than zero"));
        }
        total
            .checked_mul(u64::from(value))
            .ok_or_else(|| format!("{label} product overflows u64"))
    })
}

fn allocation_bytes(buffers: &[BufferData]) -> Result<u64> {
    let mut total = 0_u64;
    for buffer in buffers {
        if buffer.size == 0 || buffer.size % wgpu::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(format!(
                "buffer @group({}) @binding({}) size must be a nonzero multiple of 4 bytes",
                buffer.group, buffer.binding,
            ));
        }
        let upload =
            u64::try_from(buffer.data.len()).map_err(|_| "input byte length exceeds u64")?;
        if upload > buffer.size || upload % wgpu::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(format!(
                "buffer @group({}) @binding({}) input must fit its allocation and have a multiple-of-4 byte length",
                buffer.group, buffer.binding,
            ));
        }
        if buffer.readback {
            usize::try_from(buffer.size)
                .map_err(|_| "readback buffer exceeds host address space")?;
        }
        for amount in [
            buffer.size,
            upload,
            if buffer.readback { buffer.size } else { 0 },
        ] {
            total = total
                .checked_add(amount)
                .ok_or("GPU buffer and staging allocation sum overflows u64")?;
        }
    }
    Ok(total)
}

struct PlanSummary {
    shader: ShaderInfo,
    groups: u32,
    workgroup: [u32; 3],
    bytes: u64,
}

fn validate_plan(plan: &RunPlan, repetitions: u32, warmup: u32) -> Result<PlanSummary> {
    if repetitions == 0 || repetitions > MAX_REPETITIONS || warmup > MAX_REPETITIONS {
        return Err(format!(
            "iterations must be 1..={MAX_REPETITIONS}; warmup must be 0..={MAX_REPETITIONS}"
        ));
    }
    repetitions
        .checked_add(warmup)
        .ok_or("iteration count overflows u32")?;
    if plan.options.max_gpu_bytes == 0 || plan.options.max_workgroups == 0 {
        return Err("GPU byte and workgroup budgets must be greater than zero".into());
    }
    if u64::try_from(plan.source.len()).map_err(|_| "shader length exceeds u64")?
        > crate::plan::MAX_SHADER_BYTES
    {
        return Err(format!(
            "shader exceeds {} bytes",
            crate::plan::MAX_SHADER_BYTES
        ));
    }
    let workgroups = product(plan.dispatch, "dispatch")?;
    if workgroups > plan.options.max_workgroups {
        return Err(format!(
            "dispatch needs {workgroups} workgroups, exceeding --max-workgroups {}",
            plan.options.max_workgroups,
        ));
    }
    let bytes = allocation_bytes(&plan.buffers)?;
    if bytes > plan.options.max_gpu_bytes {
        return Err(format!(
            "GPU buffers plus upload/readback staging need {bytes} bytes, exceeding --max-bytes {}",
            plan.options.max_gpu_bytes,
        ));
    }
    let shader = crate::shader::inspect(&plan.source, Some(&plan.entry))?;
    let workgroup = shader
        .entry_points
        .iter()
        .find(|entry| entry.name == plan.entry)
        .ok_or_else(|| format!("compute entry point {:?} does not exist", plan.entry))?
        .workgroup_size;
    product(workgroup, "shader workgroup")?;
    let mut slots = BTreeMap::new();
    let mut groups = 0_u32;
    for buffer in &plan.buffers {
        if slots
            .insert((buffer.group, buffer.binding), buffer)
            .is_some()
        {
            return Err(format!(
                "duplicate buffer @group({}) @binding({})",
                buffer.group, buffer.binding
            ));
        }
        groups = groups.max(
            buffer
                .group
                .checked_add(1)
                .ok_or("bind group index overflows u32")?,
        );
    }
    if slots.len() != shader.bindings.len() {
        return Err(format!(
            "entry {:?} uses {} buffer bindings, but {} were supplied",
            plan.entry,
            shader.bindings.len(),
            slots.len(),
        ));
    }
    for binding in &shader.bindings {
        let buffer = slots
            .get(&(binding.group, binding.binding))
            .ok_or_else(|| {
                format!(
                    "missing buffer @group({}) @binding({})",
                    binding.group, binding.binding
                )
            })?;
        if buffer.kind != binding.kind {
            return Err(format!(
                "buffer @group({}) @binding({}) has kind {:?}; shader requires {:?}",
                binding.group, binding.binding, buffer.kind, binding.kind,
            ));
        }
        if buffer.size < binding.min_size {
            return Err(format!(
                "buffer @group({}) @binding({}) has {} bytes; shader requires at least {}",
                binding.group, binding.binding, buffer.size, binding.min_size,
            ));
        }
    }
    Ok(PlanSummary {
        shader,
        groups,
        workgroup,
        bytes,
    })
}

fn validate_adapter_limits(
    plan: &RunPlan,
    summary: &PlanSummary,
    limits: &wgpu::Limits,
) -> Result<()> {
    if summary.groups > limits.max_bind_groups
        || summary.groups > limits.max_bind_groups_plus_vertex_buffers
    {
        return Err(format!(
            "shader needs {} bind group slots; adapter supports {}",
            summary.groups,
            limits
                .max_bind_groups
                .min(limits.max_bind_groups_plus_vertex_buffers),
        ));
    }
    let mut storage = 0_u32;
    let mut uniforms = 0_u32;
    for buffer in &plan.buffers {
        if buffer.binding >= limits.max_bindings_per_bind_group {
            return Err(format!(
                "binding {} reaches adapter binding-index limit {}",
                buffer.binding, limits.max_bindings_per_bind_group,
            ));
        }
        if buffer.size > limits.max_buffer_size {
            return Err(format!(
                "buffer size {} exceeds adapter max_buffer_size {}",
                buffer.size, limits.max_buffer_size
            ));
        }
        let maximum = match buffer.kind {
            BufferKind::Uniform => {
                uniforms = uniforms
                    .checked_add(1)
                    .ok_or("uniform binding count overflow")?;
                limits.max_uniform_buffer_binding_size
            }
            BufferKind::StorageRead | BufferKind::StorageReadWrite => {
                storage = storage
                    .checked_add(1)
                    .ok_or("storage binding count overflow")?;
                limits.max_storage_buffer_binding_size
            }
        };
        if buffer.size > maximum {
            return Err(format!(
                "buffer @group({}) @binding({}) size {} exceeds adapter {:?} binding size {}",
                buffer.group, buffer.binding, buffer.size, buffer.kind, maximum,
            ));
        }
    }
    if storage > limits.max_storage_buffers_per_shader_stage {
        return Err(format!(
            "shader uses {storage} storage buffers; adapter supports {}",
            limits.max_storage_buffers_per_shader_stage
        ));
    }
    if uniforms > limits.max_uniform_buffers_per_shader_stage {
        return Err(format!(
            "shader uses {uniforms} uniform buffers; adapter supports {}",
            limits.max_uniform_buffers_per_shader_stage
        ));
    }
    let bindings = storage
        .checked_add(uniforms)
        .ok_or("total binding count overflow")?;
    if bindings > limits.max_buffers_and_acceleration_structures_per_shader_stage
        || bindings > limits.max_non_sampler_bindings
    {
        return Err(format!(
            "shader buffer count {bindings} exceeds the adapter's combined binding limit"
        ));
    }
    for (axis, count) in plan.dispatch.into_iter().enumerate() {
        if count > limits.max_compute_workgroups_per_dimension {
            return Err(format!(
                "dispatch axis {axis} has {count} workgroups; adapter maximum is {}",
                limits.max_compute_workgroups_per_dimension,
            ));
        }
    }
    let dimensions = [
        limits.max_compute_workgroup_size_x,
        limits.max_compute_workgroup_size_y,
        limits.max_compute_workgroup_size_z,
    ];
    for (axis, (requested, maximum)) in summary.workgroup.into_iter().zip(dimensions).enumerate() {
        if requested > maximum {
            return Err(format!(
                "shader workgroup axis {axis} size {requested} exceeds adapter limit {maximum}"
            ));
        }
    }
    let invocations = product(summary.workgroup, "shader workgroup")?;
    if invocations > u64::from(limits.max_compute_invocations_per_workgroup) {
        return Err(format!(
            "shader workgroup needs {invocations} invocations; adapter supports {}",
            limits.max_compute_invocations_per_workgroup,
        ));
    }
    Ok(())
}

/// Request compute limits for this job rather than allocating the default
/// million-entry D3D12 descriptor heap. No optional shader features are enabled.
fn requested_limits(
    plan: &RunPlan,
    summary: &PlanSummary,
    available: &wgpu::Limits,
    timestamps: bool,
) -> Result<wgpu::Limits> {
    let mut limits = wgpu::Limits::downlevel_defaults().using_alignment(available.clone());
    limits.max_texture_dimension_1d = 1;
    limits.max_texture_dimension_2d = 1;
    limits.max_texture_dimension_3d = 1;
    limits.max_texture_array_layers = 1;
    limits.max_bind_groups = summary.groups;
    limits.max_bind_groups_plus_vertex_buffers = summary.groups;
    limits.max_bindings_per_bind_group = plan
        .buffers
        .iter()
        .map(|buffer| buffer.binding)
        .max()
        .map(|value| value.checked_add(1).ok_or("binding index overflow"))
        .transpose()?
        .unwrap_or(0);
    limits.max_dynamic_uniform_buffers_per_pipeline_layout = 0;
    limits.max_dynamic_storage_buffers_per_pipeline_layout = 0;
    limits.max_sampled_textures_per_shader_stage = 0;
    limits.max_samplers_per_shader_stage = 0;
    limits.max_storage_textures_per_shader_stage = 0;
    limits.max_storage_buffers_per_shader_stage = u32::try_from(
        plan.buffers
            .iter()
            .filter(|buffer| buffer.kind != BufferKind::Uniform)
            .count(),
    )
    .map_err(|_| "storage binding count exceeds u32")?;
    limits.max_uniform_buffers_per_shader_stage = u32::try_from(
        plan.buffers
            .iter()
            .filter(|buffer| buffer.kind == BufferKind::Uniform)
            .count(),
    )
    .map_err(|_| "uniform binding count exceeds u32")?;
    limits.max_uniform_buffer_binding_size = plan
        .buffers
        .iter()
        .filter(|buffer| buffer.kind == BufferKind::Uniform)
        .map(|buffer| buffer.size)
        .max()
        .unwrap_or(0);
    limits.max_storage_buffer_binding_size = plan
        .buffers
        .iter()
        .filter(|buffer| buffer.kind != BufferKind::Uniform)
        .map(|buffer| buffer.size)
        .max()
        .unwrap_or(0);
    limits.max_buffer_size = plan
        .buffers
        .iter()
        .map(|buffer| buffer.size)
        .max()
        .unwrap_or(0)
        .max(if timestamps { TIMESTAMP_BYTES } else { 0 });
    limits.max_vertex_buffers = 0;
    limits.max_vertex_attributes = 0;
    limits.max_vertex_buffer_array_stride = 0;
    limits.max_inter_stage_shader_variables = 0;
    limits.max_color_attachments = 0;
    limits.max_color_attachment_bytes_per_sample = 0;
    limits.max_compute_workgroup_storage_size = available.max_compute_workgroup_storage_size;
    limits.max_compute_invocations_per_workgroup =
        u32::try_from(product(summary.workgroup, "shader workgroup")?)
            .map_err(|_| "shader workgroup invocation count exceeds u32")?;
    limits.max_compute_workgroup_size_x = summary.workgroup[0];
    limits.max_compute_workgroup_size_y = summary.workgroup[1];
    limits.max_compute_workgroup_size_z = summary.workgroup[2];
    limits.max_compute_workgroups_per_dimension = plan.dispatch.into_iter().max().unwrap_or(0);
    limits.max_immediate_size = 0;
    let binding_count =
        u32::try_from(plan.buffers.len()).map_err(|_| "buffer binding count exceeds u32")?;
    limits.max_non_sampler_bindings = binding_count.max(1).min(available.max_non_sampler_bindings);
    limits.max_buffers_and_acceleration_structures_per_shader_stage = binding_count;
    if !limits.check_limits(available) {
        return Err("adapter cannot provide the compute limits required by this job".into());
    }
    Ok(limits)
}

fn make_pipeline(
    context: &GpuContext<'_>,
    plan: &RunPlan,
    summary: &PlanSummary,
) -> Result<(Vec<wgpu::BindGroupLayout>, wgpu::ComputePipeline)> {
    context.checked("compute pipeline creation", || {
        let count = usize::try_from(summary.groups)
            .map_err(|_| "bind group count exceeds host address space")?;
        let mut layouts = Vec::new();
        layouts
            .try_reserve_exact(count)
            .map_err(|error| format!("host out of memory for layouts: {error}"))?;
        for group in 0..summary.groups {
            let entries: Vec<_> = summary
                .shader
                .bindings
                .iter()
                .filter(|binding| binding.group == group)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding: binding.binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: match binding.kind {
                            BufferKind::StorageRead => {
                                wgpu::BufferBindingType::Storage { read_only: true }
                            }
                            BufferKind::StorageReadWrite => {
                                wgpu::BufferBindingType::Storage { read_only: false }
                            }
                            BufferKind::Uniform => wgpu::BufferBindingType::Uniform,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(binding.min_size),
                    },
                    count: None,
                })
                .collect();
            layouts.push(context.device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("zaccel buffer layout"),
                    entries: &entries,
                },
            ));
        }
        let refs: Vec<_> = layouts.iter().map(Some).collect();
        let layout = context
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("zaccel pipeline layout"),
                bind_group_layouts: &refs,
                immediate_size: 0,
            });
        let module = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("zaccel WGSL"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&plan.source)),
            });
        let pipeline = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("zaccel compute"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(&plan.entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Ok((layouts, pipeline))
    })
}

struct TimestampBuffers {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
}

struct ResidentBuffers {
    buffers: Vec<wgpu::Buffer>,
    readbacks: Vec<Option<wgpu::Buffer>>,
    timestamps: Option<TimestampBuffers>,
}

fn create_and_upload(
    context: &GpuContext<'_>,
    plan: &RunPlan,
    timestamps: bool,
) -> Result<ResidentBuffers> {
    context.checked("buffer allocation and upload", || {
        let mut buffers = Vec::new();
        let mut readbacks = Vec::new();
        let mut uploads = Vec::new();
        for (label, vector) in [("buffers", &mut buffers), ("uploads", &mut uploads)] {
            vector
                .try_reserve_exact(plan.buffers.len())
                .map_err(|error| format!("host out of memory for {label}: {error}"))?;
        }
        readbacks
            .try_reserve_exact(plan.buffers.len())
            .map_err(|error| format!("host out of memory for readback descriptors: {error}"))?;
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zaccel upload"),
            });
        for buffer in &plan.buffers {
            context.deadline.remaining("buffer allocation and upload")?;
            let mut usage = match buffer.kind {
                BufferKind::Uniform => wgpu::BufferUsages::UNIFORM,
                _ => wgpu::BufferUsages::STORAGE,
            } | wgpu::BufferUsages::COPY_DST;
            if buffer.readback {
                usage |= wgpu::BufferUsages::COPY_SRC;
            }
            let resident = context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("zaccel resident data"),
                size: buffer.size,
                usage,
                mapped_at_creation: false,
            });
            if !buffer.data.is_empty() {
                let size =
                    u64::try_from(buffer.data.len()).map_err(|_| "upload size exceeds u64")?;
                let staging = context.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("zaccel upload staging"),
                    size,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                });
                {
                    let mut view = staging
                        .get_mapped_range_mut(..)
                        .map_err(|error| format!("GPU upload mapping failed: {error}"))?;
                    view.copy_from_slice(&buffer.data);
                }
                staging.unmap();
                encoder.copy_buffer_to_buffer(&staging, 0, &resident, 0, size);
                uploads.push(staging);
            }
            // wgpu initializes untouched bytes to zero; upload only the actual
            // input, preserving the explicitly sized zero-filled tail.
            let readback = buffer.readback.then(|| {
                context.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("zaccel output staging"),
                    size: buffer.size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            });
            buffers.push(resident);
            readbacks.push(readback);
        }
        let timestamps = timestamps.then(|| TimestampBuffers {
            queries: context.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("zaccel compute timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }),
            resolve: context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("zaccel timestamp resolve"),
                size: TIMESTAMP_BYTES,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("zaccel timestamp staging"),
                size: TIMESTAMP_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        });
        if !uploads.is_empty() {
            let submission = context.queue.submit([encoder.finish()]);
            context.wait_submission(submission, "input upload")?;
        }
        // Upload buffers are retained until their submission is complete.
        drop(uploads);
        Ok(ResidentBuffers {
            buffers,
            readbacks,
            timestamps,
        })
    })
}

fn make_bind_groups(
    context: &GpuContext<'_>,
    plan: &RunPlan,
    layouts: &[wgpu::BindGroupLayout],
    resident: &ResidentBuffers,
) -> Result<Vec<wgpu::BindGroup>> {
    context.checked("bind group creation", || {
        let mut groups = Vec::new();
        groups
            .try_reserve_exact(layouts.len())
            .map_err(|error| format!("host out of memory for bind groups: {error}"))?;
        for (group, layout) in layouts.iter().enumerate() {
            let group = u32::try_from(group).map_err(|_| "bind group index exceeds u32")?;
            let entries: Vec<_> = plan
                .buffers
                .iter()
                .zip(&resident.buffers)
                .filter(|(buffer, _)| buffer.group == group)
                .map(|(buffer, resident)| wgpu::BindGroupEntry {
                    binding: buffer.binding,
                    resource: resident.as_entire_binding(),
                })
                .collect();
            groups.push(
                context
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("zaccel buffer bindings"),
                        layout,
                        entries: &entries,
                    }),
            );
        }
        Ok(groups)
    })
}

fn dispatch_once(
    context: &GpuContext<'_>,
    pipeline: &wgpu::ComputePipeline,
    groups: &[wgpu::BindGroup],
    dispatch: [u32; 3],
    timestamps: Option<&TimestampBuffers>,
) -> Result<f64> {
    context.checked("compute dispatch", || {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zaccel dispatch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zaccel compute pass"),
                timestamp_writes: timestamps.map(|timing| wgpu::ComputePassTimestampWrites {
                    query_set: &timing.queries,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            pass.set_pipeline(pipeline);
            for (index, group) in groups.iter().enumerate() {
                pass.set_bind_group(
                    u32::try_from(index).map_err(|_| "bind group index exceeds u32")?,
                    group,
                    &[],
                );
            }
            pass.dispatch_workgroups(dispatch[0], dispatch[1], dispatch[2]);
        }
        if let Some(timing) = timestamps {
            encoder.resolve_query_set(&timing.queries, 0..2, &timing.resolve, 0);
            encoder.copy_buffer_to_buffer(&timing.resolve, 0, &timing.readback, 0, TIMESTAMP_BYTES);
        }
        let command = encoder.finish();
        // Encoding and timestamp mapping are excluded. Host samples include
        // submission, completion wait, and timestamp copies when instrumented.
        let start = Instant::now();
        let submission = context.queue.submit([command]);
        context.wait_submission(submission, "compute dispatch")?;
        Ok(start.elapsed().as_secs_f64() * 1_000.0)
    })
}

fn timestamp_ms(bytes: &[u8], period_ns: f32) -> Result<Option<f64>> {
    if bytes.len()
        != usize::try_from(TIMESTAMP_BYTES).map_err(|_| "timestamp size exceeds usize")?
    {
        return Err("GPU timestamp readback has an unexpected size".into());
    }
    let first: [u8; 8] = bytes[..8]
        .try_into()
        .map_err(|_| "invalid beginning timestamp")?;
    let last: [u8; 8] = bytes[8..]
        .try_into()
        .map_err(|_| "invalid ending timestamp")?;
    let Some(ticks) = u64::from_le_bytes(last).checked_sub(u64::from_le_bytes(first)) else {
        // Counter wrapping makes this optional sample unavailable; never emit
        // a huge fabricated duration or discard successfully computed data.
        return Ok(None);
    };
    if !period_ns.is_finite() || period_ns <= 0.0 {
        return Ok(None);
    }
    let milliseconds = ticks as f64 * f64::from(period_ns) / 1_000_000.0;
    Ok(milliseconds.is_finite().then_some(milliseconds))
}

fn read_outputs(
    context: &GpuContext<'_>,
    plan: &RunPlan,
    resident: &ResidentBuffers,
) -> Result<Vec<BufferOutput>> {
    context.checked("output readback", || {
        let count = plan.buffers.iter().filter(|buffer| buffer.readback).count();
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zaccel readback"),
            });
        for ((buffer, source), target) in plan
            .buffers
            .iter()
            .zip(&resident.buffers)
            .zip(&resident.readbacks)
        {
            if let Some(target) = target {
                encoder.copy_buffer_to_buffer(source, 0, target, 0, buffer.size);
            }
        }
        let submission = context.queue.submit([encoder.finish()]);
        context.wait_submission(submission, "output copy")?;
        let mut outputs = Vec::new();
        outputs
            .try_reserve_exact(count)
            .map_err(|error| format!("host out of memory for output descriptors: {error}"))?;
        for (buffer, staging) in plan.buffers.iter().zip(&resident.readbacks) {
            if let Some(staging) = staging {
                let size = usize::try_from(buffer.size)
                    .map_err(|_| "readback exceeds host address space")?;
                let data = context.read_mapping(staging, "output mapping", |bytes| {
                    if bytes.len() != size {
                        return Err("GPU output mapping has an unexpected byte length".into());
                    }
                    let mut data = Vec::new();
                    data.try_reserve_exact(size)
                        .map_err(|error| format!("host out of memory for output bytes: {error}"))?;
                    data.extend_from_slice(bytes);
                    Ok(data)
                })?;
                outputs.push(BufferOutput {
                    group: buffer.group,
                    binding: buffer.binding,
                    data,
                });
            }
        }
        Ok(outputs)
    })
}

/// Execute one resident job repeatedly. Warmups and measured iterations both
/// mutate the same buffers; only the measured iterations produce samples.
pub fn execute(plan: &RunPlan, repetitions: u32, warmup: u32) -> Result<RunReport> {
    let start = Instant::now();
    let deadline = Deadline::new(start, plan.options.timeout_ms)?;
    let summary = validate_plan(plan, repetitions, warmup)?;
    deadline.remaining("plan validation")?;
    let instance = native_instance()?;
    let adapters = wait_future(
        instance.enumerate_adapters(native_backends()),
        &deadline,
        "adapter enumeration",
        None,
        None,
    )?;
    let index = select_adapter(&adapters, plan)?;
    let adapter = &adapters[index];
    let info = adapter_info(index, adapter);
    let available = adapter.limits();
    validate_adapter_limits(plan, &summary, &available)?;

    // Timestamp instrumentation is optional, including when the byte budget
    // has no room for its two small buffers.
    let timestamp_total = summary.bytes.checked_add(TIMESTAMP_ALLOCATION);
    let timestamps = info.timestamp_query
        && available.max_buffer_size >= TIMESTAMP_BYTES
        && timestamp_total.is_some_and(|bytes| bytes <= plan.options.max_gpu_bytes);
    let allocated_bytes = if timestamps {
        timestamp_total.ok_or("GPU timestamp allocation sum overflow")?
    } else {
        summary.bytes
    };
    let limits = requested_limits(plan, &summary, &available, timestamps)?;
    let features = if timestamps {
        wgpu::Features::TIMESTAMP_QUERY
    } else {
        wgpu::Features::empty()
    };
    let descriptor = wgpu::DeviceDescriptor {
        label: Some("zaccel compute device"),
        required_features: features,
        required_limits: limits,
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        ..Default::default()
    };
    let (device, queue) = wait_future(
        adapter.request_device(&descriptor),
        &deadline,
        "device initialization",
        None,
        None,
    )?
    .map_err(|error| format!("GPU device initialization failed: {error}"))?;
    let errors = ErrorState::install(&device);
    let context = GpuContext {
        device: &device,
        queue: &queue,
        errors: &errors,
        deadline: &deadline,
    };
    let initialization_ms = start.elapsed().as_secs_f64() * 1_000.0;

    let pipeline_start = Instant::now();
    let (layouts, pipeline) = make_pipeline(&context, plan, &summary)?;
    let mut pipeline_ms = pipeline_start.elapsed().as_secs_f64() * 1_000.0;
    let upload_start = Instant::now();
    let resident = create_and_upload(&context, plan, timestamps)?;
    let upload_ms = upload_start.elapsed().as_secs_f64() * 1_000.0;
    let bindings_start = Instant::now();
    let groups = make_bind_groups(&context, plan, &layouts, &resident)?;
    pipeline_ms += bindings_start.elapsed().as_secs_f64() * 1_000.0;

    let sample_count =
        usize::try_from(repetitions).map_err(|_| "iteration count exceeds host address space")?;
    let mut wall_samples = Vec::new();
    wall_samples
        .try_reserve_exact(sample_count)
        .map_err(|error| format!("host out of memory for timing samples: {error}"))?;
    let period_ns = queue.get_timestamp_period();
    let mut kernel_samples = if timestamps && period_ns.is_finite() && period_ns > 0.0 {
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(sample_count)
            .map_err(|error| format!("host out of memory for GPU timing samples: {error}"))?;
        Some(samples)
    } else {
        None
    };
    for _ in 0..warmup {
        dispatch_once(&context, &pipeline, &groups, plan.dispatch, None)?;
    }
    for _ in 0..repetitions {
        let timing = if kernel_samples.is_some() {
            resident.timestamps.as_ref()
        } else {
            None
        };
        wall_samples.push(dispatch_once(
            &context,
            &pipeline,
            &groups,
            plan.dispatch,
            timing,
        )?);
        if let Some(timing) = timing {
            let sample = context.checked("timestamp readback", || {
                context.read_mapping(&timing.readback, "timestamp mapping", |bytes| {
                    timestamp_ms(bytes, period_ns)
                })
            })?;
            match (kernel_samples.as_mut(), sample) {
                (Some(samples), Some(sample)) => samples.push(sample),
                _ => kernel_samples = None,
            }
        }
    }
    let readback_start = Instant::now();
    let outputs = read_outputs(&context, plan, &resident)?;
    let readback_ms = readback_start.elapsed().as_secs_f64() * 1_000.0;
    errors.check("execution completion")?;
    deadline.remaining("execution completion")?;
    Ok(RunReport {
        adapter: info,
        outputs,
        gpu_allocated_bytes: allocated_bytes,
        timings: Timings {
            initialization_ms,
            pipeline_ms,
            upload_ms,
            dispatch_wall_ms: wall_samples,
            kernel_ms: kernel_samples,
            readback_ms,
            total_ms: start.elapsed().as_secs_f64() * 1_000.0,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(size: u64, input: usize, readback: bool) -> BufferData {
        BufferData {
            group: 0,
            binding: 0,
            kind: BufferKind::StorageReadWrite,
            size,
            data: vec![0; input],
            readback,
        }
    }

    #[test]
    fn budget_counts_upload_and_readback_staging() {
        assert_eq!(allocation_bytes(&[data(64, 16, true)]).unwrap(), 144);
        assert_eq!(allocation_bytes(&[data(64, 0, false)]).unwrap(), 64);
    }

    #[test]
    fn allocation_overflow_and_invalid_lengths_are_rejected() {
        assert!(allocation_bytes(&[data(u64::MAX - 3, 0, true)]).is_err());
        assert!(allocation_bytes(&[data(12, 16, false)]).is_err());
        assert!(allocation_bytes(&[data(12, 3, false)]).is_err());
        assert!(allocation_bytes(&[data(0, 0, false)]).is_err());
        assert!(allocation_bytes(&[data(6, 0, false)]).is_err());
    }

    #[test]
    fn dispatch_product_is_checked_in_every_dimension() {
        assert_eq!(product([17, 3, 2], "dispatch").unwrap(), 102);
        assert!(product([u32::MAX; 3], "dispatch").is_err());
        assert!(product([1, 0, 1], "dispatch").is_err());
    }

    #[test]
    fn host_deadline_rejects_zero_and_expiration() {
        let start = Instant::now();
        assert!(Deadline::new(start, 0).is_err());
        assert!(Deadline { end: start }.remaining("test").is_err());
    }

    #[test]
    fn timestamp_units_and_wrap_are_explicit() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&10_u64.to_le_bytes());
        bytes.extend_from_slice(&1_000_010_u64.to_le_bytes());
        assert_eq!(timestamp_ms(&bytes, 2.0).unwrap(), Some(2.0));
        assert_eq!(timestamp_ms(&bytes, 0.0).unwrap(), None);
        bytes[..8].copy_from_slice(&2_000_000_u64.to_le_bytes());
        assert_eq!(timestamp_ms(&bytes, 1.0).unwrap(), None);
        assert!(timestamp_ms(&bytes[..8], 1.0).is_err());
    }
}
