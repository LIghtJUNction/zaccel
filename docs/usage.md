# Buffers, bindings and dispatch

## Dispatch counts

`--dispatch X,Y,Z` specifies workgroups. The shader's `@workgroup_size` specifies the number of invocations within each group. With `@workgroup_size(64)`, dispatching `2,1,1` creates 128 invocations. A 65-element input therefore needs two groups and a bounds check for the extra invocations.

The default entry is `main` and the default dispatch is `1,1,1`. zaccel does not infer an element count from raw bytes. The fixture generator calculates `(count + 63) // 64` for the included examples.

## Simple mode

Simple mode accepts these layouts in group 0:

| Shader bindings | Input | Output |
| --- | --- | --- |
| Binding 0: storage read-write | `--input` initializes the buffer. | `--output` reads back the same buffer. |
| Binding 0: storage read; binding 1: storage read-write | `--input` fills binding 0. | `--size` selects binding 1's byte allocation; otherwise its size follows the input. |
| Binding 0: storage read-write, with no input | `--size` specifies the allocation. | `--output` reads back the generated values. |

Use a manifest for additional buffers, another group or uniforms. All raw file lengths and explicit sizes must be multiples of four. Storage buffers must be nonempty. The planner also checks the minimum binding sizes discovered by Naga; four-byte divisibility alone does not establish the right WGSL layout.

### Vector addition and multiplication

[vector_add.wgsl](../examples/vector_add.wgsl) and [vector_mul.wgsl](../examples/vector_mul.wgsl) accept the same input layout:

| Buffer | WGSL type | Host packing | Bytes for N elements |
| --- | --- | --- | --- |
| Group 0, binding 0 | `array<vec2<f32>>` | `a0,b0,a1,b1,...`, little-endian `struct.pack('<ff', a, b)` | `8*N` |
| Group 0, binding 1 | `array<f32>` | One little-endian float per result | `4*N` |

The multiply shader computes `a[i]*b[i]` for each element. It does not perform a reduction or a matrix product.

```sh
fixture_dir=$(mktemp -d)
python3 scripts/fixtures.py generate "$fixture_dir" --count 65
zaccel run "$fixture_dir/vector_mul.wgsl" \
  --dispatch 2,1,1 --input "$fixture_dir/data/pairs.bin" \
  --size 260 --output "$fixture_dir/data/mul.bin"
python3 scripts/fixtures.py verify "$fixture_dir" --operation mul
```

The generator accepts 1 through 1,048,576 elements, refuses to overwrite a nonempty destination and rejects destinations inside the source tree. It copies the text shaders and writes a matching manifest, metadata and binary inputs to the chosen directory. Keep that directory until any result comparisons are finished.

## Explicit manifest

[examples/affine.json](../examples/affine.json) is the 1,024-element source template. Generate a complete runnable copy with `scripts/fixtures.py`; the source repository intentionally contains no generated `.bin` files.

```json
{
  "shader": "affine.wgsl",
  "entry": "main",
  "dispatch": [16, 1, 1],
  "buffers": [
    {"group": 0, "binding": 0, "kind": "storage_read", "input": "data/a.bin"},
    {"group": 0, "binding": 1, "kind": "storage_read", "input": "data/b.bin"},
    {"group": 0, "binding": 2, "kind": "storage_read_write", "size": 4096, "output": "data/affine.bin"},
    {"group": 1, "binding": 0, "kind": "uniform", "input": "data/params.bin"}
  ]
}
```

The uniform has four four-byte fields, with offsets 0, 4, 8 and 12:

| Offset | Field | Host type | Generated value |
| --- | --- | --- | --- |
| 0 | `count` | `u32` | Number of elements |
| 4 | `scale_a` | `f32` | `2.0` |
| 8 | `scale_b` | `f32` | `-0.5` |
| 12 | `bias` | `f32` | `1.25` |

Python packs it as `struct.pack('<Ifff', count, 2.0, -0.5, 1.25)`. This layout follows the specific [Params declaration](../examples/affine.wgsl); other structures can require padding or a different array stride. Consult [WGSL alignment and size rules](https://www.w3.org/TR/WGSL/#alignment-and-size) when defining a new host layout.

Paths resolve from the manifest directory. Explicit CLI `--entry` and `--dispatch` override the manifest's values. Adapter choice, budgets, software opt-in and the deadline come from CLI options. A manifest cannot be combined with inline source or simple-mode `--input`, `--output` and `--size`. Unknown fields, duplicate binding slots, duplicate output destinations and conflicting stdin/stdout consumers are rejected.

## Binary streams and inline source

Keep binary stdout separate from diagnostics:

```sh
zaccel run "$fixture_dir/vector_add.wgsl" \
  --dispatch 2,1,1 --input - --size 260 --output - \
  < "$fixture_dir/data/pairs.bin" > "$fixture_dir/data/add.bin"
python3 scripts/fixtures.py verify "$fixture_dir" --operation add
```

The command above uses the 65-element fixture from this page. Do not combine stderr with stdout when stdout carries a buffer. Shader stdin and buffer stdin share one stream, so they cannot both use `-` in the same job.

`eval` accepts the complete shader as one argument. For example, this output-only shader writes the integers 0 through 63:

```sh
zaccel eval '@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < arrayLength(&out)) { out[id.x] = id.x; }
}' --dispatch 1,1,1 --size 256 --output "$fixture_dir/indices.bin"
```

Shell quoting is part of passing a single argument; zaccel does not invoke a shell to interpret WGSL. Use a file when the shader is easier to review and reuse that way.

## Execution options

| Option | Default | Meaning |
| --- | --- | --- |
| `--adapter INDEX` | Automatic eligible hardware selection | Index from the current `devices` response. |
| `--allow-software` | Disabled | Explicitly allow a labelled software adapter. |
| `--max-bytes BYTES` | `268435456` | Accounted resident, upload staging and readback staging bytes. |
| `--max-workgroups N` | `1048576` | Maximum `X*Y*Z` for one dispatch. |
| `--timeout-ms MS` | `30000` | GPU child-process deadline; accepted range 1 to 3,600,000. |
| `--iterations N` | `20` | Measured bench dispatches, from 1 to 10,000. |
| `--warmup N` | `3` | Unmeasured bench dispatches, from 0 to 10,000. |

The byte budget is expressed as an integer, so write `268435456` rather than `256MiB`. GPU limits can reject a job even when its application budget is larger. Errors exit nonzero; argument errors use 2, execution errors use 1, and a worker deadline uses 124. The acceptance script's exit 77 is its own explicit hardware-skip result.
