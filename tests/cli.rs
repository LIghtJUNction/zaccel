//! These integration tests exercise Naga-backed commands without a GPU.
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct FixtureDirectory(PathBuf);

impl FixtureDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "zaccel-cli-{}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).expect("create isolated CLI fixture directory");
        Self(path)
    }

    fn shader(&self, source: &str) -> PathBuf {
        let path = self.0.join("test.wgsl");
        fs::write(&path, source).expect("write source fixture");
        path
    }
}

impl Drop for FixtureDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zaccel"))
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn validate_all_shipped_examples_without_gpu() {
    for name in ["vector_add.wgsl", "vector_mul.wgsl", "affine.wgsl"] {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join(name);
        let output = cli().arg("validate").arg(path).output().unwrap();
        assert_success(&output);
    }
}

#[test]
fn inspect_reports_uniform_and_multiple_binding_groups() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/affine.wgsl");
    let output = cli()
        .arg("inspect")
        .arg(path)
        .args(["--entry", "main", "--json"])
        .output()
        .unwrap();
    assert_success(&output);
    let info: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let entries = info["entry_points"].as_array().unwrap();
    assert!(entries.iter().any(|entry| {
        entry["name"] == "main" && entry["workgroup_size"] == serde_json::json!([64, 1, 1])
    }));
    let bindings = info["bindings"].as_array().unwrap();
    assert_eq!(bindings.len(), 4);
    assert!(bindings.iter().any(|binding| {
        binding["group"] == 1
            && binding["binding"] == 0
            && binding["kind"] == "uniform"
            && binding["min_size"].as_u64().unwrap() >= 16
    }));
}

#[test]
fn validate_reads_wgsl_from_stdin() {
    let mut child = cli()
        .args(["validate", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"@compute @workgroup_size(1) fn main() {}")
        .unwrap();
    assert_success(&child.wait_with_output().unwrap());
}

#[test]
fn validate_rejects_syntax_and_type_errors() {
    let directory = FixtureDirectory::new();
    for source in [
        "@compute fn main(",
        "@compute @workgroup_size(1) fn main() { let bad: u32 = vec2<u32>(1u, 2u); }",
    ] {
        let output = cli()
            .arg("validate")
            .arg(directory.shader(source))
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(!output.stderr.is_empty(), "errors must have a diagnostic");
    }
}

#[test]
fn inspect_rejects_an_unknown_entry_point() {
    let directory = FixtureDirectory::new();
    let output = cli()
        .arg("inspect")
        .arg(directory.shader("@compute @workgroup_size(1) fn main() {}"))
        .args(["--entry", "missing", "--json"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!output.stderr.is_empty());
}

#[test]
fn inspect_rejects_a_missing_file() {
    let directory = FixtureDirectory::new();
    let output = cli()
        .arg("inspect")
        .arg(directory.0.join("absent.wgsl"))
        .arg("--json")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!output.stderr.is_empty());
}
