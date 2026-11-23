//! Offline WGSL validation and entry-point-scoped buffer reflection.

use crate::{BufferKind, EntryPoint, Result, ShaderBinding, ShaderInfo};
use naga::{AddressSpace, ShaderStage, StorageAccess};
use std::collections::BTreeMap;

fn module(source: &str) -> Result<(naga::Module, naga::valid::ModuleInfo)> {
    if source.len() as u64 > crate::plan::MAX_SHADER_BYTES {
        return Err(format!(
            "WGSL source exceeds {} bytes",
            crate::plan::MAX_SHADER_BYTES
        ));
    }
    let module =
        naga::front::wgsl::parse_str(source).map_err(|error| error.emit_to_string(source))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| error.emit_to_string(source))?;
    Ok((module, info))
}

/// Validate syntax and semantics without creating a GPU instance or device.
pub fn validate(source: &str) -> Result<()> {
    module(source).map(|_| ())
}

/// With an entry name, reflect only that compute entry's statically used buffers.
/// Without a name, return the union used by all compute entries.
pub fn inspect(source: &str, entry: Option<&str>) -> Result<ShaderInfo> {
    let (module, info) = module(source)?;
    let compute: Vec<_> = module
        .entry_points
        .iter()
        .enumerate()
        .filter(|(_, point)| point.stage == ShaderStage::Compute)
        .collect();
    if compute.is_empty() {
        return Err("shader has no compute entry point".into());
    }
    let selected: Vec<_> = match entry {
        Some(name) => vec![
            compute
                .iter()
                .copied()
                .find(|(_, point)| point.name == name)
                .ok_or_else(|| {
                    format!("compute entry {name:?} was not found; choose an entry with --entry")
                })?,
        ],
        None => compute.clone(),
    };
    let mut layout = naga::proc::Layouter::default();
    layout
        .update(module.to_ctx())
        .map_err(|error| error.to_string())?;
    let mut bindings: BTreeMap<(u32, u32), ShaderBinding> = BTreeMap::new();
    for (index, point) in selected {
        if point.workgroup_size_overrides.is_some() {
            return Err(format!(
                "entry {:?} uses workgroup-size overrides; zaccel requires a literal or const workgroup size",
                point.name
            ));
        }
        let workgroup_volume = point
            .workgroup_size
            .iter()
            .try_fold(1_u64, |product, &axis| {
                if axis == 0 {
                    None
                } else {
                    product.checked_mul(u64::from(axis))
                }
            });
        if workgroup_volume.is_none() {
            return Err(format!(
                "entry {:?} has an invalid workgroup size",
                point.name
            ));
        }
        let usage = info.get_entry_point(index);
        for (handle, global) in module.global_variables.iter() {
            if usage[handle].is_empty() {
                continue;
            }
            let Some(slot) = global.binding.as_ref() else {
                continue;
            };
            let kind = match global.space {
                AddressSpace::Uniform => BufferKind::Uniform,
                AddressSpace::Storage { access } if access.contains(StorageAccess::STORE) => {
                    BufferKind::StorageReadWrite
                }
                AddressSpace::Storage { .. } => BufferKind::StorageRead,
                _ => {
                    return Err(format!(
                        "@group({}) @binding({}) is not a supported uniform or storage buffer",
                        slot.group, slot.binding
                    ));
                }
            };
            if matches!(
                module.types[global.ty].inner,
                naga::TypeInner::BindingArray { .. }
            ) {
                return Err(format!(
                    "@group({}) @binding({}) uses an unsupported binding array",
                    slot.group, slot.binding
                ));
            }
            let min_size = u64::from(layout[global.ty].size);
            match bindings.entry((slot.group, slot.binding)) {
                std::collections::btree_map::Entry::Vacant(vacant) => {
                    vacant.insert(ShaderBinding {
                        group: slot.group,
                        binding: slot.binding,
                        kind,
                        min_size,
                    });
                }
                std::collections::btree_map::Entry::Occupied(mut occupied) => {
                    if occupied.get().kind != kind {
                        return Err(format!(
                            "@group({}) @binding({}) has different kinds across entries; use --entry",
                            slot.group, slot.binding
                        ));
                    }
                    let minimum = occupied.get().min_size.max(min_size);
                    occupied.get_mut().min_size = minimum;
                }
            }
        }
    }
    Ok(ShaderInfo {
        entry_points: compute
            .into_iter()
            .map(|(_, point)| EntryPoint {
                name: point.name.clone(),
                workgroup_size: point.workgroup_size,
            })
            .collect(),
        bindings: bindings.into_values().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syntax_and_semantic_errors_are_offline_errors() {
        assert!(validate("this is not WGSL").is_err());
        assert!(validate("@compute @workgroup_size(1) fn main() { let x: u32 = true; }").is_err());
        assert!(validate("@compute @workgroup_size(1) fn main() {}").is_ok());
    }

    #[test]
    fn selected_entry_includes_callees_but_excludes_other_entries() {
        let source = r#"
            @group(0) @binding(0) var<storage, read_write> first: array<u32>;
            @group(2) @binding(3) var<storage, read_write> second: array<u32>;
            @group(9) @binding(9) var unused_texture: texture_2d<f32>;
            fn helper() { first[0] = first[0] + 1u; }
            @compute @workgroup_size(8, 2, 1) fn main() { helper(); }
            @compute @workgroup_size(1) fn other() { second[0] = 7u; }
        "#;
        let selected = inspect(source, Some("main")).unwrap();
        assert_eq!(selected.bindings.len(), 1);
        assert_eq!(
            (selected.bindings[0].group, selected.bindings[0].binding),
            (0, 0)
        );
        assert_eq!(selected.bindings[0].min_size, 4);
        assert_eq!(selected.entry_points[0].workgroup_size, [8, 2, 1]);
        assert_eq!(inspect(source, None).unwrap().bindings.len(), 2);
    }

    #[test]
    fn layout_accounts_for_struct_padding_and_runtime_array_element() {
        let source = r#"
            struct Params { scale: vec3<u32>, count: u32, }
            struct Data { header: vec4<u32>, values: array<vec4<u32>>, }
            @group(0) @binding(0) var<uniform> params: Params;
            @group(0) @binding(1) var<storage, read_write> data: Data;
            @compute @workgroup_size(1) fn main() { data.values[0].x = params.count; }
        "#;
        let info = inspect(source, Some("main")).unwrap();
        assert_eq!(info.bindings[0].kind, BufferKind::Uniform);
        assert_eq!(info.bindings[0].min_size, 16);
        assert_eq!(info.bindings[1].min_size, 32);
    }

    #[test]
    fn missing_entry_and_used_texture_are_rejected() {
        let source = "@compute @workgroup_size(1) fn custom() {}";
        assert!(
            inspect(source, Some("main"))
                .unwrap_err()
                .contains("--entry")
        );
        assert!(inspect(source, Some("custom")).is_ok());
        let texture = r#"
            @group(0) @binding(0) var image: texture_2d<f32>;
            @group(0) @binding(1) var<storage, read_write> result: u32;
            @compute @workgroup_size(1) fn main() { result = textureDimensions(image).x; }
        "#;
        assert!(
            inspect(texture, Some("main"))
                .unwrap_err()
                .contains("supported")
        );
    }

    #[test]
    fn workgroup_override_is_not_reported_as_a_known_literal() {
        let source = "override WIDTH: u32 = 4u; @compute @workgroup_size(WIDTH) fn main() {}";
        assert!(validate(source).is_ok());
        assert!(
            inspect(source, Some("main"))
                .unwrap_err()
                .contains("overrides")
        );
    }
}
