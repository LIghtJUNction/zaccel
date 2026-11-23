//! Checked job preparation and raw output delivery; no GPU calls live here.

use crate::{
    BufferData, BufferKind, BufferOutput, GpuOptions, Result, RunPlan, ShaderBinding, shader,
};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const MAX_SHADER_BYTES: u64 = 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Default)]
pub struct JobSpec {
    pub shader: Option<String>,
    pub source: Option<String>,
    pub manifest: Option<String>,
    pub entry: Option<String>,
    pub dispatch: Option<[u32; 3]>,
    pub input: Option<String>,
    pub output: Option<String>,
    pub size: Option<u64>,
    pub options: GpuOptions,
}

#[derive(Debug)]
pub struct PreparedJob {
    pub plan: RunPlan,
    pub outputs: Vec<OutputTarget>,
}

#[derive(Debug)]
pub struct OutputTarget {
    pub group: u32,
    pub binding: u32,
    pub path: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    shader: String,
    entry: Option<String>,
    dispatch: Option<[u32; 3]>,
    buffers: Vec<ManifestBuffer>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestBuffer {
    #[serde(default)]
    group: u32,
    binding: u32,
    kind: BufferKind,
    input: Option<String>,
    output: Option<String>,
    size: Option<u64>,
}

struct Request {
    group: u32,
    binding: u32,
    kind: BufferKind,
    input: Option<String>,
    output: Option<String>,
    size: Option<u64>,
    follows: Option<usize>,
}

struct Pending {
    request: Request,
    minimum: u64,
    size: Option<u64>,
    input_len: Option<u64>,
}

/// Read at most `limit` bytes, plus one byte to detect an oversized stream.
/// Incremental fallible reservation avoids reserving a user-supplied maximum.
fn read_limited(mut reader: impl Read, limit: u64, label: &str) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    let mut chunk = [0_u8; 8192];
    loop {
        let remaining = limit.saturating_sub(data.len() as u64);
        let count = remaining.saturating_add(1).min(chunk.len() as u64) as usize;
        let read = match reader.read(&mut chunk[..count]) {
            Ok(read) => read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(format!("cannot read {label}: {error}")),
        };
        if read == 0 {
            return Ok(data);
        }
        if read as u64 > remaining {
            return Err(format!("{label} exceeds its {limit}-byte limit"));
        }
        let required = data
            .len()
            .checked_add(read)
            .ok_or_else(|| "input allocation size overflow".to_owned())?;
        if required > data.capacity() {
            let maximum = usize::try_from(limit)
                .unwrap_or(isize::MAX as usize)
                .min(isize::MAX as usize);
            let capacity = data
                .capacity()
                .checked_mul(2)
                .unwrap_or(maximum)
                .max(required)
                .min(maximum);
            if capacity < required {
                return Err(format!("{label} exceeds the host addressable size"));
            }
            data.try_reserve_exact(capacity - data.len())
                .map_err(|error| format!("cannot allocate input for {label}: {error}"))?;
        }
        data.extend_from_slice(&chunk[..read]);
    }
}

fn regular_file_size(path: &str) -> Result<u64> {
    let metadata =
        fs::metadata(path).map_err(|error| format!("cannot inspect {path:?}: {error}"))?;
    if !metadata.is_file() {
        return Err(format!(
            "input {path:?} is not a regular file; use - for stdin"
        ));
    }
    Ok(metadata.len())
}

fn read_path(path: &str, limit: u64) -> Result<Vec<u8>> {
    if path == "-" {
        return read_limited(io::stdin().lock(), limit, "stdin");
    }
    let size = regular_file_size(path)?;
    if size > limit {
        return Err(format!("{path:?} exceeds its {limit}-byte limit"));
    }
    let file = File::open(path).map_err(|error| format!("cannot open {path:?}: {error}"))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("cannot inspect {path:?}: {error}"))?;
    if !metadata.is_file() || metadata.len() > limit {
        return Err(format!(
            "{path:?} changed or exceeds its {limit}-byte limit"
        ));
    }
    read_limited(file, limit, path)
}

pub fn read_shader(path: &str) -> Result<String> {
    String::from_utf8(read_path(path, MAX_SHADER_BYTES)?)
        .map_err(|error| format!("shader {path:?} is not UTF-8: {error}"))
}

fn path_string(path: PathBuf) -> Result<String> {
    path.into_os_string()
        .into_string()
        .map_err(|_| "resolved path is not UTF-8".into())
}

fn resolve_path(base: &Path, path: String) -> Result<String> {
    if path.is_empty() {
        return Err("file paths must not be empty".into());
    }
    if path == "-" {
        return Ok(path);
    }
    let path = PathBuf::from(path);
    path_string(if path.is_absolute() {
        path
    } else {
        base.join(path)
    })
}

/// Canonicalize the parent, without following a destination symlink which an
/// atomic rename will replace. This also detects ./ and parent-directory aliases.
fn output_path(path: &str) -> Result<String> {
    if path == "-" {
        return Ok(path.to_owned());
    }
    let path = Path::new(path);
    let name = path
        .file_name()
        .ok_or_else(|| "output must name a file".to_owned())?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = parent
        .canonicalize()
        .map_err(|error| format!("cannot access output directory {parent:?}: {error}"))?;
    if !parent.is_dir() {
        return Err(format!("output parent {parent:?} is not a directory"));
    }
    let destination = parent.join(name);
    match fs::symlink_metadata(&destination) {
        Ok(metadata) if metadata.is_dir() => {
            return Err(format!("output {destination:?} is a directory"));
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("cannot inspect output {destination:?}: {error}")),
    }
    path_string(destination)
}

fn validate_dispatch(dispatch: [u32; 3], options: &GpuOptions) -> Result<()> {
    if options.max_gpu_bytes == 0 || options.max_workgroups == 0 || options.timeout_ms == 0 {
        return Err("byte budget, workgroup budget and timeout must be positive".into());
    }
    let count = dispatch
        .iter()
        .try_fold(1_u64, |total, &axis| {
            if axis == 0 {
                None
            } else {
                total.checked_mul(u64::from(axis))
            }
        })
        .ok_or_else(|| {
            "dispatch dimensions must be positive and their product must fit u64".to_owned()
        })?;
    if count > options.max_workgroups {
        return Err(format!(
            "dispatch requests {count} workgroups, exceeding budget {}",
            options.max_workgroups
        ));
    }
    Ok(())
}

fn size_valid(size: u64, minimum: u64, group: u32, binding: u32) -> Result<()> {
    if size == 0 || !size.is_multiple_of(4) {
        return Err(format!(
            "buffer {group}:{binding} size must be nonzero and a multiple of 4 bytes"
        ));
    }
    if size < minimum {
        return Err(format!(
            "buffer {group}:{binding} has {size} bytes; WGSL layout requires at least {minimum}"
        ));
    }
    if usize::try_from(size).is_err() || size > isize::MAX as u64 {
        return Err(format!(
            "buffer {group}:{binding} exceeds the host addressable size"
        ));
    }
    Ok(())
}

fn copies(request: &Request) -> u64 {
    if request.output.is_some() { 2 } else { 1 }
}

fn allocation_cost(size: u64, copies: u64) -> Result<u64> {
    size.checked_mul(copies)
        .ok_or_else(|| "buffer and readback allocation size overflow".into())
}

fn budget(pending: &[Pending], maximum: u64) -> Result<(u64, u64)> {
    let mut fixed = 0_u64;
    let mut variable_copies = 0_u64;
    for item in pending {
        if let Some(size) = item.size {
            size_valid(size, item.minimum, item.request.group, item.request.binding)?;
            fixed = fixed
                .checked_add(allocation_cost(size, copies(&item.request))?)
                .ok_or_else(|| "total GPU allocation size overflow".to_owned())?;
        } else {
            variable_copies = variable_copies
                .checked_add(copies(&item.request))
                .ok_or_else(|| "buffer count overflow".to_owned())?;
        }
        // queue.write_buffer needs upload staging for the actual input bytes.
        if item.request.input.is_some() {
            if let Some(length) = item.input_len {
                fixed = fixed
                    .checked_add(length)
                    .ok_or_else(|| "upload allocation size overflow".to_owned())?;
            } else {
                variable_copies = variable_copies
                    .checked_add(1)
                    .ok_or_else(|| "upload count overflow".to_owned())?;
            }
        }
    }
    if fixed > maximum {
        return Err(format!(
            "buffers, uploads and readback staging require {fixed} bytes, exceeding budget {maximum}"
        ));
    }
    Ok((fixed, variable_copies))
}

fn simple_requests(spec: &JobSpec, bindings: &[ShaderBinding]) -> Result<Vec<Request>> {
    let request = |binding, kind, input, output, size, follows| Request {
        group: 0,
        binding,
        kind,
        input,
        output,
        size,
        follows,
    };
    match bindings {
        [] if spec.input.is_none() && spec.output.is_none() && spec.size.is_none() => Ok(Vec::new()),
        [single] if single.group == 0 && single.binding == 0 && single.kind == BufferKind::StorageReadWrite => {
            Ok(vec![request(0, single.kind, spec.input.clone(), spec.output.clone(), spec.size, None)])
        }
        [input, output] if input.group == 0 && input.binding == 0 && input.kind == BufferKind::StorageRead
            && output.group == 0 && output.binding == 1 && output.kind == BufferKind::StorageReadWrite => {
            let path = spec.input.clone().ok_or_else(|| "two-buffer mode requires --input".to_owned())?;
            Ok(vec![
                request(0, input.kind, Some(path), None, None, None),
                request(1, output.kind, None, spec.output.clone(), spec.size, Some(0)),
            ])
        }
        _ => Err("simple mode needs storage read_write 0:0, or storage read 0:0 plus read_write 0:1; use --manifest for other bindings".into()),
    }
}

pub fn prepare(spec: JobSpec) -> Result<PreparedJob> {
    // CLI constraints are repeated here because callers can construct JobSpec.
    validate_dispatch(spec.dispatch.unwrap_or([1, 1, 1]), &spec.options)?;
    if spec.source.is_some() && spec.shader.is_some() {
        return Err("choose a shader file or eval source, not both".into());
    }
    let cwd = std::env::current_dir()
        .map_err(|error| format!("cannot read current directory: {error}"))?;
    let mut shader_path = spec.shader.clone();
    let mut entry = spec.entry.clone();
    let mut dispatch = spec.dispatch;
    let mut manifest_requests = None;
    let mut stdin_consumers = 0_usize;
    if let Some(path) = spec.manifest.as_deref() {
        if spec.source.is_some()
            || spec.shader.is_some()
            || spec.input.is_some()
            || spec.output.is_some()
            || spec.size.is_some()
        {
            return Err("--manifest cannot be combined with a shader argument, eval, --input, --output or --size".into());
        }
        stdin_consumers += usize::from(path == "-");
        let bytes = read_path(path, MAX_MANIFEST_BYTES)?;
        let manifest: Manifest =
            serde_json::from_slice(&bytes).map_err(|error| format!("invalid manifest: {error}"))?;
        let base = if path == "-" {
            cwd.clone()
        } else {
            let absolute = PathBuf::from(resolve_path(&cwd, path.to_owned())?);
            absolute
                .parent()
                .ok_or_else(|| "manifest must name a file".to_owned())?
                .to_path_buf()
        };
        shader_path = Some(resolve_path(&base, manifest.shader)?);
        entry = entry.or(manifest.entry);
        dispatch = dispatch.or(manifest.dispatch);
        let mut requests = Vec::new();
        for buffer in manifest.buffers {
            let input = buffer
                .input
                .map(|path| resolve_path(&base, path))
                .transpose()?;
            let output = buffer
                .output
                .map(|path| resolve_path(&base, path))
                .transpose()?;
            stdin_consumers += usize::from(input.as_deref() == Some("-"));
            requests.push(Request {
                group: buffer.group,
                binding: buffer.binding,
                kind: buffer.kind,
                input,
                output,
                size: buffer.size,
                follows: None,
            });
        }
        manifest_requests = Some(requests);
    } else {
        stdin_consumers += usize::from(spec.input.as_deref() == Some("-"));
    }
    stdin_consumers += usize::from(shader_path.as_deref() == Some("-"));
    if stdin_consumers > 1 {
        return Err("stdin may be consumed by only one manifest, shader or input buffer".into());
    }
    let dispatch = dispatch.unwrap_or([1, 1, 1]);
    validate_dispatch(dispatch, &spec.options)?;
    let source = if let Some(source) = spec.source.as_ref() {
        if source.len() as u64 > MAX_SHADER_BYTES {
            return Err(format!("WGSL source exceeds {MAX_SHADER_BYTES} bytes"));
        }
        source.clone()
    } else {
        read_shader(
            shader_path
                .as_deref()
                .ok_or_else(|| "a shader file or eval source is required".to_owned())?,
        )?
    };
    let entry = entry.unwrap_or_else(|| "main".into());
    let info = shader::inspect(&source, Some(&entry))?;
    let requests = match manifest_requests {
        Some(requests) => requests,
        None => simple_requests(&spec, &info.bindings)?,
    };
    let expected: BTreeMap<_, _> = info
        .bindings
        .iter()
        .map(|binding| ((binding.group, binding.binding), binding))
        .collect();
    let mut slots = BTreeSet::new();
    let mut destinations = BTreeSet::new();
    let mut pending: Vec<Pending> = Vec::new();
    let mut outputs = Vec::new();
    for mut request in requests {
        let key = (request.group, request.binding);
        if !slots.insert(key) {
            return Err(format!("duplicate buffer binding {}:{}", key.0, key.1));
        }
        let reflection = expected
            .get(&key)
            .ok_or_else(|| format!("buffer {}:{} is not used by entry {entry:?}", key.0, key.1))?;
        if request.kind != reflection.kind {
            return Err(format!(
                "buffer {}:{} kind does not match the shader",
                key.0, key.1
            ));
        }
        if let Some(path) = request.output.as_ref() {
            let path = output_path(path)?;
            if !destinations.insert(path.clone()) {
                return Err(format!(
                    "duplicate output destination {path:?}; stdout may have only one producer"
                ));
            }
            outputs.push(OutputTarget {
                group: key.0,
                binding: key.1,
                path: path.clone(),
            });
            request.output = Some(path);
        }
        if let Some(size) = request.size {
            size_valid(size, reflection.min_size, key.0, key.1)?;
        }
        if request.size.is_none() && request.input.is_none() && request.follows.is_none() {
            return Err(format!(
                "buffer {}:{} requires input bytes or an explicit size",
                key.0, key.1
            ));
        }
        let size = request.size;
        pending.push(Pending {
            request,
            minimum: reflection.min_size,
            size,
            input_len: None,
        });
    }
    if slots.len() != expected.len() {
        let missing: Vec<_> = expected
            .keys()
            .filter(|slot| !slots.contains(*slot))
            .map(|(g, b)| format!("{g}:{b}"))
            .collect();
        return Err(format!(
            "manifest is missing shader bindings: {}",
            missing.join(", ")
        ));
    }
    // Reject oversized declared allocations before inspecting raw input files.
    budget(&pending, spec.options.max_gpu_bytes)?;
    for index in 0..pending.len() {
        let input_len = match pending[index].request.input.as_deref() {
            Some("-") => None,
            Some(path) => Some(regular_file_size(path)?),
            None => Some(0),
        };
        if input_len.is_some_and(|length| length % 4 != 0) {
            return Err("raw input files must contain a multiple of 4 bytes".into());
        }
        pending[index].input_len = input_len;
        if pending[index].size.is_none() {
            pending[index].size = if let Some(parent) = pending[index].request.follows {
                pending[parent].size
            } else {
                input_len
            };
        }
        if let (Some(length), Some(size)) = (input_len, pending[index].size)
            && length > size
        {
            return Err(format!(
                "input for buffer {}:{} is larger than its {size}-byte allocation",
                pending[index].request.group, pending[index].request.binding
            ));
        }
    }
    let (fixed, variable_copies) = budget(&pending, spec.options.max_gpu_bytes)?;
    let stdin_index = pending
        .iter()
        .position(|item| item.request.input.as_deref() == Some("-"));
    let mut stdin_data = None;
    if let Some(index) = stdin_index {
        if variable_copies == 0 {
            return Err("missing stdin allocation accounting".into());
        }
        let available = (spec.options.max_gpu_bytes - fixed) / variable_copies;
        let limit = available
            .min(pending[index].size.unwrap_or(available))
            .min(isize::MAX as u64)
            & !3;
        for item in pending.iter().filter(|item| item.size.is_none()) {
            size_valid(
                limit,
                item.minimum,
                item.request.group,
                item.request.binding,
            )?;
        }
        let data = read_limited(io::stdin().lock(), limit, "stdin buffer")?;
        if data.len() % 4 != 0 {
            return Err("stdin buffer must contain a multiple of 4 bytes".into());
        }
        pending[index].input_len = Some(data.len() as u64);
        if pending[index].size.is_none() {
            pending[index].size = Some(data.len() as u64);
        }
        for next in 0..pending.len() {
            if pending[next].size.is_none() {
                let parent = pending[next]
                    .request
                    .follows
                    .ok_or_else(|| "buffer size is unresolved".to_owned())?;
                pending[next].size = pending[parent].size;
            }
        }
        stdin_data = Some(data);
    }
    let (_, unresolved) = budget(&pending, spec.options.max_gpu_bytes)?;
    if unresolved != 0 {
        return Err("buffer sizes remain unresolved".into());
    }
    // All GPU allocations, staging allocations and dispatch counts are now
    // checked. Input file contents are allocated only after this preflight.
    let mut buffers = Vec::new();
    for item in pending {
        let request = item.request;
        let size = item
            .size
            .ok_or_else(|| "buffer size is unresolved".to_owned())?;
        let data = match request.input.as_deref() {
            None => Vec::new(),
            Some("-") => stdin_data
                .take()
                .ok_or_else(|| "stdin buffer was already consumed".to_owned())?,
            Some(path) => {
                let expected_len = item
                    .input_len
                    .ok_or_else(|| "input size is unresolved".to_owned())?;
                let bytes = read_path(path, expected_len)?;
                if bytes.len() as u64 != expected_len {
                    return Err(format!(
                        "input {path:?} changed size while preparing the job"
                    ));
                }
                bytes
            }
        };
        buffers.push(BufferData {
            group: request.group,
            binding: request.binding,
            kind: request.kind,
            size,
            data,
            readback: request.output.is_some(),
        });
    }
    Ok(PreparedJob {
        plan: RunPlan {
            source,
            entry,
            dispatch,
            buffers,
            options: spec.options,
        },
        outputs,
    })
}

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct StagedOutput {
    temporary: PathBuf,
    destination: PathBuf,
}

impl Drop for StagedOutput {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.temporary);
    }
}

fn stage_output(destination: &Path, data: &[u8]) -> Result<StagedOutput> {
    let parent = destination
        .parent()
        .ok_or_else(|| "output must have a parent directory".to_owned())?;
    for _ in 0..64 {
        let serial = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(".zaccel-{}-{serial}.tmp", std::process::id()));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = match options.open(&temporary) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("cannot stage output {destination:?}: {error}")),
        };
        let staged = StagedOutput {
            temporary,
            destination: destination.to_path_buf(),
        };
        let written = file.write_all(data).and_then(|()| file.sync_all());
        drop(file);
        written.map_err(|error| format!("cannot write output {destination:?}: {error}"))?;
        return Ok(staged);
    }
    Err(format!(
        "cannot reserve a temporary output for {destination:?}"
    ))
}

/// Call only after a successful GPU result. Each file replacement is atomic;
/// multiple destinations are not a filesystem-wide transaction. Validate and
/// stage every file before publishing files, and write raw stdout last.
pub fn write_outputs(targets: &[OutputTarget], outputs: &[BufferOutput]) -> Result<()> {
    let mut results = BTreeMap::new();
    for output in outputs {
        if results
            .insert((output.group, output.binding), &output.data)
            .is_some()
        {
            return Err("GPU returned duplicate output bindings".into());
        }
    }
    let mut slots = BTreeSet::new();
    let mut destinations = BTreeSet::new();
    let mut resolved = Vec::new();
    for target in targets {
        let slot = (target.group, target.binding);
        if !slots.insert(slot) {
            return Err("duplicate output binding".into());
        }
        let data = results
            .get(&slot)
            .ok_or_else(|| format!("GPU result is missing output {}:{}", slot.0, slot.1))?;
        let path = output_path(&target.path)?;
        if !destinations.insert(path.clone()) {
            return Err("duplicate output destination, including stdout".into());
        }
        resolved.push((path, data.as_slice()));
    }
    if results.len() != slots.len() {
        return Err("GPU returned unexpected output bindings".into());
    }
    let mut staged = Vec::new();
    let mut stdout = None;
    for (path, data) in resolved {
        if path == "-" {
            stdout = Some(data);
        } else {
            staged.push(stage_output(Path::new(&path), data)?);
        }
    }
    for output in &staged {
        fs::rename(&output.temporary, &output.destination)
            .map_err(|error| format!("cannot publish output {:?}: {error}", output.destination))?;
    }
    if let Some(data) = stdout {
        let mut stream = io::stdout().lock();
        stream
            .write_all(data)
            .and_then(|()| stream.flush())
            .map_err(|error| format!("cannot write raw output to stdout: {error}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const IN_PLACE: &str = "@group(0) @binding(0) var<storage, read_write> data: array<u32>; @compute @workgroup_size(1) fn main() { data[0] = data[0] + 1u; }";
    const TWO: &str = "@group(0) @binding(0) var<storage, read> src: array<u32>; @group(0) @binding(1) var<storage, read_write> dst: array<u32>; @compute @workgroup_size(1) fn main() { dst[0] = src[0]; }";

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let serial = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("zaccel-plan-test-{}-{serial}", std::process::id()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn path(&self, name: &str) -> String {
            self.0.join(name).into_os_string().into_string().unwrap()
        }
        fn write(&self, name: &str, bytes: impl AsRef<[u8]>) -> String {
            let path = self.path(name);
            fs::write(&path, bytes).unwrap();
            path
        }
        fn manifest(&self, value: serde_json::Value) -> String {
            self.write("job.json", serde_json::to_vec(&value).unwrap())
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn simple(source: &str) -> JobSpec {
        JobSpec {
            source: Some(source.into()),
            ..JobSpec::default()
        }
    }

    #[test]
    fn bounded_reader_stops_after_one_extra_byte() {
        let mut cursor = io::Cursor::new(vec![0_u8; 100]);
        assert!(
            read_limited(&mut cursor, 8, "fixture")
                .unwrap_err()
                .contains("8-byte")
        );
        assert_eq!(cursor.position(), 9);
        assert_eq!(
            read_limited(&b"12345678"[..], 8, "fixture").unwrap(),
            b"12345678"
        );
    }

    #[test]
    fn dispatch_overflow_and_budget_fail_before_shader_or_data_reads() {
        let mut spec = JobSpec {
            shader: Some("/does-not-exist-zaccel.wgsl".into()),
            dispatch: Some([u32::MAX; 3]),
            ..JobSpec::default()
        };
        spec.options.max_workgroups = u64::MAX;
        assert!(prepare(spec).unwrap_err().contains("product"));
        let mut spec = simple(IN_PLACE);
        spec.size = Some(16);
        spec.input = Some("/does-not-exist-zaccel.bin".into());
        spec.options.max_gpu_bytes = 8;
        assert!(prepare(spec).unwrap_err().contains("exceeding budget"));
    }

    #[test]
    fn in_place_preserves_bytes_and_does_not_allocate_zero_padding_on_host() {
        let fixture = Fixture::new();
        let mut spec = simple(IN_PLACE);
        spec.input = Some(fixture.write("input.bin", [0xff, 0x00, 0x01, 0x80]));
        spec.output = Some(fixture.path("output.bin"));
        spec.size = Some(8);
        spec.options.max_gpu_bytes = 20;
        let job = prepare(spec).unwrap();
        assert_eq!(job.plan.buffers[0].size, 8);
        assert_eq!(job.plan.buffers[0].data, [0xff, 0x00, 0x01, 0x80]);
        assert!(job.plan.buffers[0].readback);
        assert!(!Path::new(&job.outputs[0].path).exists());
    }

    #[test]
    fn two_buffer_default_size_includes_readback_staging_in_budget() {
        let fixture = Fixture::new();
        let input = fixture.write("input.bin", [1_u8; 8]);
        let build = |max| {
            let mut spec = simple(TWO);
            spec.input = Some(input.clone());
            spec.output = Some(fixture.path("out.bin"));
            spec.options.max_gpu_bytes = max;
            spec
        };
        assert!(prepare(build(31)).unwrap_err().contains("32 bytes"));
        let job = prepare(build(32)).unwrap();
        assert_eq!(
            job.plan
                .buffers
                .iter()
                .map(|buffer| buffer.size)
                .collect::<Vec<_>>(),
            [8, 8]
        );
        assert!(!job.plan.buffers[0].readback);
        assert!(job.plan.buffers[1].data.is_empty());
    }

    #[test]
    fn output_only_requires_size_and_absent_output_does_not_read_back() {
        assert!(
            prepare(simple(IN_PLACE))
                .unwrap_err()
                .contains("explicit size")
        );
        let mut spec = simple(IN_PLACE);
        spec.size = Some(4);
        let job = prepare(spec).unwrap();
        assert!(job.outputs.is_empty());
        assert!(!job.plan.buffers[0].readback);
        assert!(job.plan.buffers[0].data.is_empty());
    }

    #[test]
    fn raw_alignment_minimum_and_truncation_are_errors() {
        let fixture = Fixture::new();
        let mut spec = simple(IN_PLACE);
        spec.input = Some(fixture.write("unaligned.bin", [1_u8; 3]));
        assert!(prepare(spec).unwrap_err().contains("multiple of 4"));
        let mut spec = simple(IN_PLACE);
        spec.input = Some(fixture.write("long.bin", [1_u8; 8]));
        spec.size = Some(4);
        assert!(prepare(spec).unwrap_err().contains("larger than"));
        let source = "@group(0) @binding(0) var<storage, read_write> data: vec4<u32>; @compute @workgroup_size(1) fn main() { data.x = 1u; }";
        let mut spec = simple(source);
        spec.size = Some(4);
        assert!(prepare(spec).unwrap_err().contains("at least 16"));
        assert!(allocation_cost(u64::MAX, 2).is_err());
    }

    #[test]
    fn manifest_paths_uniforms_and_explicit_overrides_are_honored() {
        let fixture = Fixture::new();
        fixture.write(
            "shader.wgsl",
            r#"
            struct Params { value: vec4<u32>, }
            @group(0) @binding(0) var<storage, read> src: array<u32>;
            @group(0) @binding(1) var<storage, read_write> dst: array<u32>;
            @group(2) @binding(3) var<uniform> params: Params;
            @compute @workgroup_size(1) fn main() { dst[0] = src[0] + params.value.x; }
        "#,
        );
        fixture.write("data.bin", [1_u8; 4]);
        fixture.write("params.bin", [2_u8; 16]);
        let path = fixture.manifest(
            json!({"shader":"shader.wgsl", "entry":"missing", "dispatch":[0,1,1],
            "buffers":[
                {"binding":0,"kind":"storage_read","input":"data.bin"},
                {"binding":1,"kind":"storage_read_write","size":4,"output":"result.bin"},
                {"group":2,"binding":3,"kind":"uniform","input":"params.bin"}
            ]}),
        );
        let job = prepare(JobSpec {
            manifest: Some(path),
            entry: Some("main".into()),
            dispatch: Some([2, 1, 1]),
            ..JobSpec::default()
        })
        .unwrap();
        assert_eq!(job.plan.dispatch, [2, 1, 1]);
        assert_eq!(job.plan.buffers[2].kind, BufferKind::Uniform);
        assert_eq!(job.plan.buffers[2].size, 16);
        // The planner canonicalizes the existing parent. macOS /var and
        // Windows short/extended paths may name it differently from temp_dir.
        let expected_parent = fs::canonicalize(&fixture.0).unwrap();
        assert_eq!(
            Path::new(&job.outputs[0].path),
            expected_parent.join("result.bin")
        );
    }

    #[test]
    fn manifest_rejects_unknown_fields_duplicate_slots_and_wrong_kinds() {
        assert!(
            serde_json::from_str::<Manifest>(r#"{"shader":"x","buffers":[],"unknown":1}"#).is_err()
        );
        assert!(
            serde_json::from_str::<Manifest>(
                r#"{"shader":"x","buffers":[{"binding":0,"kind":"uniform","size":4,"unknown":1}]}"#
            )
            .is_err()
        );
        let fixture = Fixture::new();
        fixture.write("shader.wgsl", IN_PLACE);
        let duplicate = fixture.manifest(json!({"shader":"shader.wgsl","buffers":[
            {"binding":0,"kind":"storage_read_write","size":4},
            {"binding":0,"kind":"storage_read_write","size":4}
        ]}));
        assert!(
            prepare(JobSpec {
                manifest: Some(duplicate),
                ..JobSpec::default()
            })
            .unwrap_err()
            .contains("duplicate buffer")
        );
        let wrong = fixture.manifest(
            json!({"shader":"shader.wgsl","buffers":[{"binding":0,"kind":"uniform","size":4}]}),
        );
        assert!(
            prepare(JobSpec {
                manifest: Some(wrong),
                ..JobSpec::default()
            })
            .unwrap_err()
            .contains("kind")
        );
    }

    #[test]
    fn manifest_rejects_alias_destinations_and_multiple_stdout_producers() {
        let fixture = Fixture::new();
        fixture.write("shader.wgsl", TWO);
        for (first, second) in [("out.bin", "./out.bin"), ("-", "-")] {
            let path = fixture.manifest(json!({"shader":"shader.wgsl","buffers":[
                {"binding":0,"kind":"storage_read","size":4,"output":first},
                {"binding":1,"kind":"storage_read_write","size":4,"output":second}
            ]}));
            assert!(
                prepare(JobSpec {
                    manifest: Some(path),
                    ..JobSpec::default()
                })
                .unwrap_err()
                .contains("duplicate output")
            );
        }
    }

    #[test]
    fn stdin_conflicts_are_rejected_before_attempting_stdin() {
        let spec = JobSpec {
            shader: Some("-".into()),
            input: Some("-".into()),
            ..JobSpec::default()
        };
        assert!(prepare(spec).unwrap_err().contains("stdin may be consumed"));
        let fixture = Fixture::new();
        let path = fixture.manifest(json!({"shader":"does-not-exist.wgsl","buffers":[
            {"binding":0,"kind":"storage_read","input":"-"},
            {"binding":1,"kind":"storage_read_write","input":"-"}
        ]}));
        assert!(
            prepare(JobSpec {
                manifest: Some(path),
                ..JobSpec::default()
            })
            .unwrap_err()
            .contains("stdin may be consumed")
        );
    }

    #[test]
    fn default_entry_is_main_and_shader_reads_are_bounded_utf8() {
        assert!(
            prepare(simple("@compute @workgroup_size(1) fn custom() {}"))
                .unwrap_err()
                .contains("--entry")
        );
        let mut explicit = simple("@compute @workgroup_size(1) fn custom() {}");
        explicit.entry = Some("custom".into());
        assert!(prepare(explicit).is_ok());
        let fixture = Fixture::new();
        let large = fixture.write("large.wgsl", vec![b' '; MAX_SHADER_BYTES as usize + 1]);
        assert!(read_shader(&large).unwrap_err().contains("exceeds"));
        let invalid = fixture.write("invalid.wgsl", [0xff]);
        assert!(read_shader(&invalid).unwrap_err().contains("UTF-8"));
    }

    #[test]
    fn output_validation_preserves_existing_file_then_success_replaces_bytes() {
        let fixture = Fixture::new();
        let target = fixture.write("out.bin", b"old!");
        let targets = vec![OutputTarget {
            group: 0,
            binding: 0,
            path: target.clone(),
        }];
        assert!(write_outputs(&targets, &[]).is_err());
        assert_eq!(fs::read(&target).unwrap(), b"old!");
        write_outputs(
            &targets,
            &[BufferOutput {
                group: 0,
                binding: 0,
                data: vec![0, 255, 0, 128],
            }],
        )
        .unwrap();
        assert_eq!(fs::read(&target).unwrap(), [0, 255, 0, 128]);
        assert_eq!(fs::read_dir(&fixture.0).unwrap().count(), 1);
    }

    #[test]
    fn missing_later_result_cannot_partially_replace_an_earlier_file() {
        let fixture = Fixture::new();
        let first = fixture.write("first.bin", b"old!");
        let second = fixture.path("second.bin");
        let targets = [
            OutputTarget {
                group: 0,
                binding: 0,
                path: first.clone(),
            },
            OutputTarget {
                group: 0,
                binding: 1,
                path: second.clone(),
            },
        ];
        assert!(
            write_outputs(
                &targets,
                &[BufferOutput {
                    group: 0,
                    binding: 0,
                    data: vec![1; 4]
                }]
            )
            .is_err()
        );
        assert_eq!(fs::read(&first).unwrap(), b"old!");
        assert!(!Path::new(&second).exists());
    }
}
