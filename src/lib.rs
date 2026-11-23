//! Bounded WGSL compute execution for the zaccel CLI.
pub mod gpu;
pub mod plan;
pub mod shader;

use serde::{Deserialize, Serialize};
pub type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BufferKind {
    StorageRead,
    StorageReadWrite,
    Uniform,
}

#[derive(Clone, Debug, Serialize)]
pub struct ShaderBinding {
    pub group: u32,
    pub binding: u32,
    pub kind: BufferKind,
    pub min_size: u64,
}
#[derive(Clone, Debug, Serialize)]
pub struct EntryPoint {
    pub name: String,
    pub workgroup_size: [u32; 3],
}
#[derive(Clone, Debug, Serialize)]
pub struct ShaderInfo {
    pub entry_points: Vec<EntryPoint>,
    pub bindings: Vec<ShaderBinding>,
}

#[derive(Clone, Debug)]
pub struct BufferData {
    pub group: u32,
    pub binding: u32,
    pub kind: BufferKind,
    pub size: u64,
    pub data: Vec<u8>,
    pub readback: bool,
}
#[derive(Clone, Debug)]
pub struct GpuOptions {
    pub adapter: Option<usize>,
    pub allow_software: bool,
    pub max_gpu_bytes: u64,
    pub max_workgroups: u64,
    pub timeout_ms: u64,
}
impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            adapter: None,
            allow_software: false,
            max_gpu_bytes: 256 * 1024 * 1024,
            max_workgroups: 1_048_576,
            timeout_ms: 30_000,
        }
    }
}
#[derive(Clone, Debug)]
pub struct RunPlan {
    pub source: String,
    pub entry: String,
    pub dispatch: [u32; 3],
    pub buffers: Vec<BufferData>,
    pub options: GpuOptions,
}
#[derive(Clone, Debug, Serialize)]
pub struct AdapterInfo {
    pub index: usize,
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: String,
    pub backend: String,
    pub driver: String,
    pub driver_info: String,
    pub timestamp_query: bool,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_compute_workgroups_per_dimension: u32,
}
#[derive(Debug)]
pub struct BufferOutput {
    pub group: u32,
    pub binding: u32,
    pub data: Vec<u8>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Timings {
    pub initialization_ms: f64,
    pub pipeline_ms: f64,
    pub upload_ms: f64,
    pub dispatch_wall_ms: Vec<f64>,
    pub kernel_ms: Option<Vec<f64>>,
    pub readback_ms: f64,
    pub total_ms: f64,
}
#[derive(Debug)]
pub struct RunReport {
    pub adapter: AdapterInfo,
    pub outputs: Vec<BufferOutput>,
    pub gpu_allocated_bytes: u64,
    pub timings: Timings,
}
