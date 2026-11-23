use std::{
    collections::HashSet,
    process::{Child, Command, ExitCode, Stdio},
    thread,
    time::{Duration, Instant},
};
use zaccel::{Result, gpu, plan, shader};

const HELP: &str = "\
zaccel — execute bounded WGSL compute scripts

USAGE
  zaccel devices [--json]
  zaccel validate SHADER
  zaccel inspect SHADER [--entry NAME] [--json]
  zaccel run SHADER [OPTIONS]
  zaccel run --manifest JOB.json [OPTIONS]
  zaccel eval 'WGSL' [OPTIONS]
  zaccel bench SHADER [OPTIONS]
  zaccel bench --manifest JOB.json [OPTIONS]

EXECUTION
  --entry NAME             Compute entry point (default: main)
  --dispatch X,Y,Z         Workgroup counts (default: 1,1,1)
  --input PATH             Raw input bytes; '-' reads stdin
  --output PATH            Raw output bytes; '-' writes stdout
  --size BYTES             Output buffer allocation, a multiple of 4
  --manifest PATH          JSON job with explicit storage/uniform bindings
  --adapter INDEX          Adapter index from 'devices'
  --allow-software         Explicitly permit a labelled software adapter
  --max-bytes BYTES        GPU buffers plus staging budget (default: 268435456)
  --max-workgroups N       Dispatch product cap (default: 1048576)
  --timeout-ms MS          Whole GPU child-process deadline (default: 30000)
  --iterations N          Measured bench dispatches (default: 20; maximum: 10000)
  --warmup N              Unmeasured bench dispatches (default: 3; maximum: 10000)
  --json                  Compact machine-readable metadata
  -h, --help              Show this help
  -V, --version           Show the version

Shader and input files are bounded before allocation. Buffers use raw bytes:
match WGSL types, alignment and host byte order explicitly. Diagnostics go to
stderr so a binary stdout remains clean. Bench reuses the device, pipeline and
resident buffers; each iteration sees the previous iteration's buffer state.

A timeout kills the CLI worker. A driver may keep GPU work running or require
an OS reset; this is not a hostile-code sandbox. There is no automatic CPU
implementation of arbitrary WGSL. See README.md for complete examples.
";

#[derive(Debug)]
enum Action {
    Help,
    Version,
    Devices {
        json: bool,
        timeout_ms: u64,
    },
    Validate {
        path: String,
    },
    Inspect {
        path: String,
        entry: Option<String>,
        json: bool,
    },
    Execute {
        spec: plan::JobSpec,
        bench: bool,
        iterations: u32,
        warmup: u32,
        json: bool,
    },
}

fn unsigned(value: &str, label: &str) -> Result<u64> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!(
            "{label} requires an unsigned integer, got {value:?}"
        ));
    }
    value.parse().map_err(|_| format!("{label} is too large"))
}

fn dispatch(value: &str) -> Result<[u32; 3]> {
    let parts: Vec<_> = value.split(',').collect();
    if parts.len() != 3 {
        return Err("--dispatch requires exactly X,Y,Z workgroup counts".into());
    }
    let mut counts = [0; 3];
    for (out, text) in counts.iter_mut().zip(parts) {
        *out = u32::try_from(unsigned(text, "--dispatch")?)
            .map_err(|_| "--dispatch values must fit in 32 bits")?;
        if *out == 0 {
            return Err("--dispatch counts must be positive".into());
        }
    }
    Ok(counts)
}

fn parse(args: &[String]) -> Result<Action> {
    let Some(command) = args.first().map(String::as_str) else {
        return Ok(Action::Help);
    };
    if matches!(command, "-h" | "--help" | "help") {
        return if args.len() == 1 {
            Ok(Action::Help)
        } else {
            Err("help takes no arguments".into())
        };
    }
    if matches!(command, "-V" | "--version") {
        return if args.len() == 1 {
            Ok(Action::Version)
        } else {
            Err("--version takes no arguments".into())
        };
    }
    if !matches!(
        command,
        "devices" | "validate" | "inspect" | "run" | "eval" | "bench"
    ) {
        return Err(format!("unknown command {command:?}; use --help"));
    }
    let executable = matches!(command, "run" | "eval" | "bench");
    let mut spec = plan::JobSpec::default();
    let mut json = false;
    let mut iterations = if command == "bench" { 20 } else { 1 };
    let mut warmup = if command == "bench" { 3 } else { 0 };
    let mut seen = HashSet::new();
    let mut positional = Vec::new();
    let mut index = 1;
    let mut literal = false;
    while index < args.len() {
        let arg = args[index].as_str();
        index += 1;
        if literal || !arg.starts_with('-') || arg == "-" {
            positional.push(arg.to_owned());
            continue;
        }
        if arg == "--" {
            literal = true;
            continue;
        }
        if matches!(arg, "-h" | "--help") {
            return Ok(Action::Help);
        }
        if !seen.insert(arg) {
            return Err(format!("duplicate option {arg}"));
        }
        let allowed = match arg {
            "--json" => command != "validate",
            "--entry" => executable || command == "inspect",
            "--timeout-ms" => executable || command == "devices",
            "--iterations" | "--warmup" => command == "bench",
            "--dispatch" | "--input" | "--output" | "--size" | "--manifest" | "--adapter"
            | "--allow-software" | "--max-bytes" | "--max-workgroups" => executable,
            _ => false,
        };
        if !allowed {
            return Err(format!("option {arg:?} is not valid for {command}"));
        }
        if arg == "--json" {
            json = true;
            continue;
        }
        if arg == "--allow-software" {
            spec.options.allow_software = true;
            continue;
        }
        let value = args
            .get(index)
            .ok_or_else(|| format!("{arg} requires a value"))?;
        index += 1;
        match arg {
            "--entry" => {
                if value.is_empty() {
                    return Err("--entry cannot be empty".into());
                }
                spec.entry = Some(value.clone());
            }
            "--dispatch" => spec.dispatch = Some(dispatch(value)?),
            "--input" => spec.input = Some(value.clone()),
            "--output" => spec.output = Some(value.clone()),
            "--manifest" => spec.manifest = Some(value.clone()),
            "--size" => {
                let bytes = unsigned(value, arg)?;
                if bytes == 0 || !bytes.is_multiple_of(4) {
                    return Err("--size must be a positive multiple of 4 bytes".into());
                }
                spec.size = Some(bytes);
            }
            "--adapter" => {
                spec.options.adapter = Some(
                    usize::try_from(unsigned(value, arg)?)
                        .map_err(|_| "--adapter is too large for this platform")?,
                );
            }
            "--max-bytes" => {
                spec.options.max_gpu_bytes = unsigned(value, arg)?;
                if spec.options.max_gpu_bytes == 0 {
                    return Err("--max-bytes must be positive".into());
                }
            }
            "--max-workgroups" => {
                spec.options.max_workgroups = unsigned(value, arg)?;
                if spec.options.max_workgroups == 0 {
                    return Err("--max-workgroups must be positive".into());
                }
            }
            "--timeout-ms" => {
                spec.options.timeout_ms = unsigned(value, arg)?;
                if !(1..=3_600_000).contains(&spec.options.timeout_ms) {
                    return Err("--timeout-ms must be between 1 and 3600000".into());
                }
            }
            "--iterations" => {
                let count = unsigned(value, arg)?;
                if !(1..=10_000).contains(&count) {
                    return Err("--iterations must be between 1 and 10000".into());
                }
                iterations = count as u32;
            }
            "--warmup" => {
                let count = unsigned(value, arg)?;
                if count > 10_000 {
                    return Err("--warmup must be between 0 and 10000".into());
                }
                warmup = count as u32;
            }
            _ => return Err(format!("unknown option {arg}")),
        }
    }
    if positional.len() > 1 {
        return Err(format!("{command} accepts one shader argument"));
    }
    let path = positional.pop();
    match command {
        "devices" => {
            if path.is_some() {
                return Err("devices takes no shader argument".into());
            }
            Ok(Action::Devices {
                json,
                timeout_ms: spec.options.timeout_ms,
            })
        }
        "validate" => Ok(Action::Validate {
            path: path.ok_or("validate requires a shader path (or '-')")?,
        }),
        "inspect" => Ok(Action::Inspect {
            path: path.ok_or("inspect requires a shader path (or '-')")?,
            entry: spec.entry,
            json,
        }),
        _ => {
            if command == "eval" {
                if spec.manifest.is_some() {
                    return Err("eval does not accept --manifest".into());
                }
                spec.source = Some(path.ok_or("eval requires inline WGSL source")?);
            } else if spec.manifest.is_some() {
                if path.is_some() {
                    return Err("choose a shader path or --manifest, not both".into());
                }
            } else {
                spec.shader = Some(path.ok_or("a shader path or --manifest is required")?);
            }
            if spec.manifest.is_some()
                && (spec.input.is_some() || spec.output.is_some() || spec.size.is_some())
            {
                return Err(
                    "manifest bindings own input/output/size; put those fields in the manifest"
                        .into(),
                );
            }
            Ok(Action::Execute {
                spec,
                bench: command == "bench",
                iterations,
                warmup,
                json,
            })
        }
    }
}

fn json_text(value: &impl serde::Serialize, compact: bool) -> Result<String> {
    (if compact {
        serde_json::to_string(value)
    } else {
        serde_json::to_string_pretty(value)
    })
    .map_err(|error| format!("could not encode report: {error}"))
}

fn summarize(samples: &[f64]) -> serde_json::Value {
    if samples.is_empty() {
        return serde_json::Value::Null;
    }
    let mut values = samples.to_vec();
    values.sort_by(f64::total_cmp);
    let count = values.len();
    let median = if count.is_multiple_of(2) {
        (values[count / 2 - 1] + values[count / 2]) / 2.0
    } else {
        values[count / 2]
    };
    serde_json::json!({
        "count": count,
        "min_ms": values[0],
        "median_ms": median,
        "p95_ms": values[((count * 95).div_ceil(100) - 1).min(count - 1)],
        "max_ms": values[count - 1],
        "mean_ms": values.iter().sum::<f64>() / count as f64
    })
}

fn execute(action: Action) -> Result<()> {
    match action {
        Action::Help => print!("{HELP}"),
        Action::Version => println!("zaccel {}", env!("CARGO_PKG_VERSION")),
        Action::Devices { json, .. } => {
            let adapters = gpu::devices()?;
            println!("{}", json_text(&adapters, json)?);
            if adapters.is_empty() {
                return Err(
                    "no GPU adapters found; check the native graphics driver and device access"
                        .into(),
                );
            }
        }
        Action::Validate { path } => {
            let source = plan::read_shader(&path)?;
            shader::validate(&source)?;
            println!(
                "{}",
                serde_json::json!({"valid": true, "shader": path, "bytes": source.len()})
            );
        }
        Action::Inspect { path, entry, json } => {
            let source = plan::read_shader(&path)?;
            let info = shader::inspect(&source, entry.as_deref())?;
            println!("{}", json_text(&info, json)?);
        }
        Action::Execute {
            spec,
            bench,
            iterations,
            warmup,
            json,
        } => {
            let total_start = Instant::now();
            let prepared = plan::prepare(spec)?;
            let prepared_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            let binary_stdout = prepared.outputs.iter().any(|target| target.path == "-");
            let report = gpu::execute(&prepared.plan, iterations, warmup)?;
            let host = summarize(&report.timings.dispatch_wall_ms);
            let kernel = report.timings.kernel_ms.as_deref().map(summarize);
            let write_start = Instant::now();
            plan::write_outputs(&prepared.outputs, &report.outputs)?;
            let write_ms = write_start.elapsed().as_secs_f64() * 1000.0;
            let value = serde_json::json!({
                "command": if bench { "bench" } else { "run" },
                "entry": prepared.plan.entry,
                "dispatch": prepared.plan.dispatch,
                "adapter": report.adapter,
                "gpu_allocated_bytes": report.gpu_allocated_bytes,
                "iterations": iterations,
                "warmup": warmup,
                "preparation_ms": prepared_ms,
                "output_write_ms": write_ms,
                "worker_total_ms": total_start.elapsed().as_secs_f64() * 1000.0,
                "timings": report.timings,
                "dispatch_wall": host,
                "kernel": kernel,
                "buffer_state": "retained_between_dispatches"
            });
            let rendered = json_text(&value, json)?;
            if bench && !binary_stdout {
                println!("{rendered}");
            } else {
                eprintln!("{rendered}");
            }
        }
    }
    Ok(())
}

fn cleanup_child(child: &mut Child) {
    let _ = child.kill();
    // Do not turn a driver stuck in kernel space into an unbounded parent wait.
    let until = Instant::now() + Duration::from_millis(250);
    while Instant::now() < until {
        match child.try_wait() {
            Ok(Some(_)) | Err(_) => return,
            Ok(None) => thread::sleep(Duration::from_millis(5)),
        }
    }
}

fn wait_child(mut child: Child, timeout: Duration) -> Result<u8> {
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return Ok(status
                    .code()
                    .and_then(|code| u8::try_from(code).ok())
                    .unwrap_or(1));
            }
            Ok(None) => {}
            Err(error) => {
                cleanup_child(&mut child);
                return Err(format!("could not observe GPU worker: {error}"));
            }
        }
        if started.elapsed() >= timeout {
            cleanup_child(&mut child);
            eprintln!(
                "zaccel: GPU worker exceeded {} ms and termination was requested; \
                 a running GPU kernel or hung driver may outlive this process",
                timeout.as_millis()
            );
            return Ok(124);
        }
        thread::sleep(Duration::from_millis(5));
    }
}

fn supervise(args: &[String], timeout_ms: u64) -> Result<u8> {
    let executable =
        std::env::current_exe().map_err(|e| format!("locate zaccel executable: {e}"))?;
    let child = Command::new(executable)
        .arg("__gpu_worker")
        .args(args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("start GPU worker: {e}"))?;
    wait_child(child, Duration::from_millis(timeout_ms))
}

fn main() -> ExitCode {
    let args: std::result::Result<Vec<String>, _> = std::env::args_os()
        .skip(1)
        .map(|arg| arg.into_string())
        .collect();
    let mut args = match args {
        Ok(args) => args,
        Err(_) => {
            eprintln!("zaccel: command arguments must be valid UTF-8");
            return ExitCode::from(2);
        }
    };
    let worker = args.first().is_some_and(|arg| arg == "__gpu_worker");
    if worker {
        args.remove(0);
    }
    let action = match parse(&args) {
        Ok(action) => action,
        Err(error) => {
            eprintln!("zaccel: {error}");
            return ExitCode::from(2);
        }
    };
    let timeout = match &action {
        Action::Devices { timeout_ms, .. } => Some(*timeout_ms),
        Action::Execute { spec, .. } => Some(spec.options.timeout_ms),
        _ => None,
    };
    if !worker && let Some(timeout_ms) = timeout {
        return match supervise(&args, timeout_ms) {
            Ok(code) => ExitCode::from(code),
            Err(error) => {
                eprintln!("zaccel: {error}");
                ExitCode::from(1)
            }
        };
    }
    match execute(action) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("zaccel: {error}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn cli_rejects_ambiguous_or_unbounded_arguments() {
        for arguments in [
            vec!["run", "a.wgsl", "--dispatch", "1,0,1"],
            vec!["run", "a.wgsl", "--dispatch", "1,2"],
            vec!["run", "a.wgsl", "--timeout-ms", "0"],
            vec!["run", "a.wgsl", "--max-bytes", "0"],
            vec!["run", "a.wgsl", "--size", "3"],
            vec!["run", "a.wgsl", "--size", "4", "--size", "8"],
            vec!["run", "--manifest", "a.json", "--output", "b.bin"],
            vec!["run", "a.wgsl", "--iterations", "2"],
            vec!["bench", "a.wgsl", "--iterations", "10001"],
            vec!["devices", "extra"],
            vec!["validate", "a.wgsl", "--adapter", "0"],
        ] {
            assert!(parse(&argv(&arguments)).is_err(), "{arguments:?}");
        }
    }

    #[test]
    fn cli_preserves_raw_io_and_workgroup_values() {
        let action = parse(&argv(&[
            "run",
            "a.wgsl",
            "--dispatch",
            "2,3,4",
            "--input",
            "-",
            "--output",
            "-",
            "--max-bytes",
            "4096",
        ]))
        .unwrap();
        match action {
            Action::Execute {
                spec,
                iterations,
                warmup,
                ..
            } => {
                assert_eq!(spec.dispatch, Some([2, 3, 4]));
                assert_eq!(spec.input.as_deref(), Some("-"));
                assert_eq!(spec.output.as_deref(), Some("-"));
                assert_eq!(spec.options.max_gpu_bytes, 4096);
                assert_eq!((iterations, warmup), (1, 0));
            }
            _ => panic!("expected execution"),
        }
    }

    #[test]
    fn sample_summary_has_correct_median_and_nearest_rank_percentile() {
        let summary = summarize(&[4.0, 1.0, 3.0, 2.0]);
        assert_eq!(summary["median_ms"], 2.5);
        assert_eq!(summary["p95_ms"], 4.0);
        assert_eq!(summary["mean_ms"], 2.5);
        assert!(summarize(&[]).is_null());
    }

    #[test]
    fn watchdog_fixture() {
        if std::env::var_os("ZACCEL_TEST_SLEEP").is_some() {
            thread::sleep(Duration::from_secs(30));
        }
    }

    #[test]
    fn watchdog_terminates_a_real_stalled_child_without_a_gpu() {
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tests::watchdog_fixture", "--nocapture"])
            .env("ZACCEL_TEST_SLEEP", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let start = Instant::now();
        assert_eq!(wait_child(child, Duration::from_millis(40)).unwrap(), 124);
        assert!(start.elapsed() < Duration::from_secs(3));
    }
}
