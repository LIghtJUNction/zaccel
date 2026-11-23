// A 16-byte uniform. Host packing is little-endian <Ifff.
struct Params {
    count: u32,
    scale_a: f32,
    scale_b: f32,
    bias: f32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < params.count && i < arrayLength(&a)
        && i < arrayLength(&b) && i < arrayLength(&result)) {
        result[i] = params.scale_a * a[i] + params.scale_b * b[i] + params.bias;
    }
}
