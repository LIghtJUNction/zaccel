// Input: little-endian pairs (a, b), 8 bytes each. Output: a + b, 4 bytes each.
@group(0) @binding(0) var<storage, read> pairs: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&pairs) && i < arrayLength(&result)) {
        result[i] = pairs[i].x + pairs[i].y;
    }
}
