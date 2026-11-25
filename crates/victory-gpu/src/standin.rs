//! The upscaler stand-in (`VICTORY_FSR_STANDIN`, on unless `=0`).
//!
//! On `--console series-s` the title draws its world at 1280x720 and upscales
//! it into the 1920x1080 game plane with AMD's FSR 4
//! (`amd_fidelityfx_upscaler_xbox.dll`). Its pipeline cache fails
//! ("DeserializeComputePipelineStateX FAILED ... falling back to NewBE")
//! and the fallback's 20 compute programs come as SM5 DXBC containers with
//! no DXIL part, which do not translate: every FSR dispatch was dropped and
//! the game plane stayed black under the UI plane.
//!
//! In place of such a dispatch (a compute pipeline whose container has no
//! DXIL part, `crate::pipeline::without_dxil`) the textures its descriptor
//! tables hold are looked at ([`choose`]): the largest colour texture it
//! writes through unordered access is FSR's output; the largest colour
//! texture it reads that is smaller, of the same aspect, is the frame to
//! upscale. That frame is stretched (bilinear) into the output by a
//! compute pass of ours ([`stretch`]), once per output and frame (the
//! chain has 20 dispatches; the stretch is the same for each). Float input
//! into a UNORM output is Reinhard-mapped (x / (1 + x)) rather than cut at
//! 1; float into float and UNORM into anything is copied as it is.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// Whether the stand-in runs: `VICTORY_FSR_STANDIN=0` turns it off.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FSR_STANDIN").is_none_or(|v| v != "0"))
}

/// A texture a dropped dispatch's tables hold: whether it is written
/// through unordered access, its size and host format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Seen {
    pub(crate) uav: bool,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: i32,
}

/// Whether a host format holds colour, and if so whether it is float
/// (`Some(true)`) or UNORM (`Some(false)`); depth, integer, one- and
/// two-channel formats (depth, motion, masks) are `None`.
pub(crate) fn colour(format: i32) -> Option<bool> {
    match format {
        1 | 2 | 9 | 10 | 26 | 67 => Some(true),
        11 | 23 | 24 | 27 | 28 | 29 | 87 | 90 | 91 => Some(false),
        _ => None,
    }
}

/// The format the input is read with (typeless as its float or UNORM
/// member).
fn srv_format(format: i32) -> Option<DXGI_FORMAT> {
    colour(format)?;
    Some(DXGI_FORMAT(match format {
        1 => 2,
        9 => 10,
        23 => 24,
        27 => 28,
        90 => 87,
        f => f,
    }))
}

/// The format the output is written with (sRGB through its linear
/// member: unordered access cannot write sRGB); `None` for shared-exponent.
fn uav_format(format: i32) -> Option<DXGI_FORMAT> {
    if format == PACKED {
        return Some(DXGI_FORMAT_R32_UINT);
    }
    colour(format)?;
    Some(DXGI_FORMAT(match format {
        67 => return None,
        1 => 2,
        9 => 10,
        23 => 24,
        27 | 29 => 28,
        90 | 91 => 87,
        f => f,
    }))
}

/// What to stretch: the input's index, the outputs' indices (every colour
/// texture written of the largest written size, usually one) and the tone
/// mode (1: Reinhard, float into UNORM; 0: as it is). `None` when no
/// written colour texture is larger than a read colour texture of its
/// aspect.
pub(crate) fn choose(seen: &[Seen]) -> Option<(usize, Vec<usize>, u32)> {
    let area = |s: &Seen| u64::from(s.width) * u64::from(s.height);
    let written = |s: &Seen| s.uav && colour(s.format).is_some() && uav_format(s.format).is_some();
    let largest = seen.iter().filter(|s| written(s)).max_by_key(|s| area(s))?;
    let (w, h) = (largest.width, largest.height);
    // R32_UINT textures of the output's size too: the title reads FSR's
    // output as R9G9B9E5 through a view at their memory (in a capture, the post
    // chain reads 1920x1080 f67 at the memory FSR writes).
    let packed = |s: &Seen| pack_on() && s.uav && (s.format == PACKED || s.format == SHARED_EXPONENT);
    let outputs: Vec<usize> = (0..seen.len())
        .filter(|&i| (written(&seen[i]) || packed(&seen[i])) && (seen[i].width, seen[i].height) == (w, h))
        .collect();
    // Same aspect within 2%.
    let aspect = |s: &Seen| {
        let (a, b) = (u64::from(s.width) * u64::from(h), u64::from(w) * u64::from(s.height));
        a.abs_diff(b) * 50 <= b
    };
    let input = (0..seen.len())
        .filter(|&i| {
            let s = &seen[i];
            !s.uav && colour(s.format).is_some() && s.width < w && s.height < h && s.width > 0 && aspect(s)
        })
        // The first of the largest (FSR binds its colour input first).
        .min_by_key(|&i| (std::cmp::Reverse(area(&seen[i])), i))?;
    let float_in = colour(seen[input].format) == Some(true);
    let float_out = colour(largest.format) == Some(true);
    let mode = u32::from(float_in && !float_out);
    Some((input, outputs, mode))
}

/// R32_UINT: an output written with light packed as R9G9B9E5.
const PACKED: i32 = 42;

/// R9G9B9E5_SHAREDEXP: an output the host cannot write through unordered
/// access; packed into a texture of ours and copied in as bytes
/// ([`stretch_shared_exponent`]).
pub(crate) const SHARED_EXPONENT: i32 = 67;

/// Whether R32_UINT outputs are written packed (`VICTORY_FSR_STANDIN_PACK`,
/// on unless `=0`).
fn pack_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FSR_STANDIN_PACK").is_none_or(|v| v != "0"))
}

/// The tone mode of one output (input and output host formats): 2 packs
/// R9G9B9E5 into R32_UINT, 1 is Reinhard (float into UNORM), 0 as it is.
pub(crate) fn mode_for(input: i32, output: i32) -> u32 {
    if output == PACKED || output == SHARED_EXPONENT {
        2
    } else {
        u32::from(colour(input) == Some(true) && colour(output) == Some(false))
    }
}

/// Whether `output` (a title texture) has been stretched into this frame
/// already; marks it if not.
pub(crate) fn first_this_frame(output: usize, frame: u64) -> bool {
    static DONE: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);
    let mut done = DONE.lock().unwrap_or_else(|e| e.into_inner());
    let done = done.get_or_insert_with(HashMap::new);
    if done.len() > 64 {
        done.retain(|_, f| *f + 2 >= frame);
    }
    done.insert(output, frame) != Some(frame)
}

/// Logs a pair once: "FSR stand-in: <in> -> <out>".
pub(crate) fn note_pair(input: (usize, Seen), output: (usize, Seen), mode: u32) {
    static PAIRS: Mutex<Option<HashSet<(usize, usize)>>> = Mutex::new(None);
    let mut pairs = PAIRS.lock().unwrap_or_else(|e| e.into_inner());
    if !pairs.get_or_insert_with(HashSet::new).insert((input.0, output.0)) {
        return;
    }
    let (i, o) = (input.1, output.1);
    tracing::info!(
        tone = match mode {
            2 => "packed as R9G9B9E5 (R32_UINT output)",
            1 => "Reinhard (float input, UNORM output)",
            _ => "none (copied as it is)",
        },
        "FSR stand-in: {:#x} {}x{} f{} -> {:#x} {}x{} f{}",
        input.0,
        i.width,
        i.height,
        i.format,
        output.0,
        o.width,
        o.height,
        o.format
    );
}

const STRETCH: &str = r#"
Texture2D<float4> input : register(t0);
RWTexture2D<float4> output : register(u0);
SamplerState clamped : register(s0);
cbuffer Size : register(b0) { uint2 size; uint mode; uint unused; };
[numthreads(8, 8, 1)]
void stretch(uint3 id : SV_DispatchThreadID) {
    if (id.x >= size.x || id.y >= size.y) return;
    float4 c = input.SampleLevel(clamped, (float2(id.xy) + 0.5) / float2(size), 0);
    if (mode == 1) {
        c.rgb = max(c.rgb, 0.0);
        c.rgb = c.rgb / (1.0 + c.rgb);
        c.a = saturate(c.a);
    }
    output[id.xy] = c;
}

RWTexture2D<uint> packed : register(u0);

// R9G9B9E5_SHAREDEXP, as D3D packs it.
uint pack_rgb9e5(float3 c) {
    const float most = 65408.0; // (511 / 512) * 2^16
    c = clamp(c, 0.0, most);
    float m = max(c.r, max(c.g, c.b));
    int e = max(-16, (int)floor(log2(max(m, 1e-30)))) + 1 + 15;
    float scale = exp2((float)(e - 15 - 9));
    if ((uint)floor(m / scale + 0.5) == 512) { e += 1; scale *= 2.0; }
    uint3 v = (uint3)floor(c / scale + 0.5);
    return v.r | (v.g << 9) | (v.b << 18) | ((uint)e << 27);
}

[numthreads(8, 8, 1)]
void pack(uint3 id : SV_DispatchThreadID) {
    if (id.x >= size.x || id.y >= size.y) return;
    float4 c = input.SampleLevel(clamped, (float2(id.xy) + 0.5) / float2(size), 0);
    packed[id.xy] = pack_rgb9e5(max(c.rgb, 0.0));
}
"#;

/// Views in a heap of ours used as a ring, two per stretch.
const VIEWS: u32 = 128;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// The same into R32_UINT, packed as R9G9B9E5 (mode 2).
    pso_pack: ID3D12PipelineState,
    views: ID3D12DescriptorHeap,
    cpu: usize,
    gpu: u64,
    step: u32,
    next: AtomicU32,
    /// Formats unordered access can store to, by format.
    stores: Mutex<HashMap<i32, bool>>,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

fn pass() -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let made = || -> windows::core::Result<Pass> {
            let host = crate::host::host().ok_or_else(windows::core::Error::empty)?;
            let cs = crate::host::compile(STRETCH, "stretch", "cs_5_1")?;
            let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            let ranges = [D3D12_DESCRIPTOR_RANGE_TYPE_SRV, D3D12_DESCRIPTOR_RANGE_TYPE_UAV]
                .map(|range_type| D3D12_DESCRIPTOR_RANGE {
                    RangeType: range_type,
                    NumDescriptors: 1,
                    OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
                    ..Default::default()
                });
            let params = [
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                            NumDescriptorRanges: ranges.len() as u32,
                            pDescriptorRanges: ranges.as_ptr(),
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            Num32BitValues: 4,
                            ..Default::default()
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
            ];
            let sampler = D3D12_STATIC_SAMPLER_DESC {
                Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
                AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
                MaxLOD: f32::MAX,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                ..Default::default()
            };
            // SAFETY: the ranges and root inputs live through serialization,
            // and the pipeline descriptor references live shader bytecode.
            unsafe {
                let root =
                    crate::host::root_signature_from_params(&host.device, &params, &[sampler])?;
                let make = |code: &[u8]| -> windows::core::Result<ID3D12PipelineState> {
                    host.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: code.as_ptr().cast(),
                            BytecodeLength: code.len(),
                        },
                        ..Default::default()
                    })
                };
                let pso = make(&cs[..])?;
                let packing = crate::host::compile(STRETCH, "pack", "cs_5_1")?;
                let pso_pack = make(&packing[..])?;
                let views: ID3D12DescriptorHeap = host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: VIEWS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?;
                Ok(Pass {
                    cpu: views.GetCPUDescriptorHandleForHeapStart().ptr,
                    gpu: views.GetGPUDescriptorHandleForHeapStart().ptr,
                    step: host.device.GetDescriptorHandleIncrementSize(kind),
                    root,
                    pso,
                    pso_pack,
                    views,
                    next: AtomicU32::new(0),
                    stores: Mutex::default(),
                })
            }
        };
        made().inspect_err(|e| tracing::warn!(error = %e, "no FSR stand-in pass")).ok()
    })
    .as_ref()
}

impl Pass {
    fn stores(&self, device: &ID3D12Device, format: DXGI_FORMAT) -> bool {
        let mut known = self.stores.lock().unwrap_or_else(|e| e.into_inner());
        *known.entry(format.0).or_insert_with(|| {
            let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: format, ..Default::default() };
            // SAFETY: a query into our own structure of its size.
            let asked = unsafe {
                device.CheckFeatureSupport(
                    D3D12_FEATURE_FORMAT_SUPPORT,
                    (&raw mut support).cast(),
                    size_of_val(&support) as u32,
                )
            };
            asked.is_ok() && support.Support2.0 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE.0 != 0
        })
    }
}

/// Whether `input` can be stretched into `output` by [`stretch`]: 2D
/// textures of one layer, the output writable through unordered access in
/// a format that can be stored to.
pub(crate) fn fits(input: &ID3D12Resource, output: &ID3D12Resource) -> bool {
    let (Some(pass), Some(host)) = (pass(), crate::host::host()) else {
        return false;
    };
    // SAFETY: live resources.
    let (i, o) = unsafe { (input.GetDesc(), output.GetDesc()) };
    let plain = |d: &D3D12_RESOURCE_DESC| {
        d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.DepthOrArraySize == 1 && d.SampleDesc.Count == 1
    };
    plain(&i)
        && plain(&o)
        && i.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
        && o.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
        && srv_format(i.Format.0).is_some()
        && uav_format(o.Format.0).is_some_and(|f| pass.stores(&host.device, f))
}

/// Records into `list` the bilinear stretch of `input` (in a shader-read
/// state) over all of `output` (in the unordered-access state), with tone
/// mode `mode` ([`choose`]). The caller sets the title's heaps, root
/// signature and pipeline again afterwards.
pub(crate) fn stretch(list: &ID3D12GraphicsCommandList, input: &ID3D12Resource, output: &ID3D12Resource, mode: u32) -> bool {
    let (Some(pass), Some(host)) = (pass(), crate::host::host()) else {
        return false;
    };
    // SAFETY: live resources.
    let (i, o) = unsafe { (input.GetDesc(), output.GetDesc()) };
    let (Some(read), Some(write)) = (srv_format(i.Format.0), uav_format(o.Format.0)) else {
        return false;
    };
    let slot = (pass.next.fetch_add(1, Ordering::Relaxed) % (VIEWS / 2)) * 2;
    let cpu = |k: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: pass.cpu + ((slot + k) * pass.step) as usize };
    let size = [o.Width as u32, o.Height, mode, 0];
    // SAFETY: an open list; two slots of our heap; live resources; the
    // pass's objects live as long as the process.
    unsafe {
        host.device.CreateShaderResourceView(
            input,
            Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: read,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, ..Default::default() },
                },
            }),
            cpu(0),
        );
        host.device.CreateUnorderedAccessView(
            output,
            None,
            Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: write,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_UAV { MipSlice: 0, PlaneSlice: 0 },
                },
            }),
            cpu(1),
        );
        list.SetDescriptorHeaps(&[Some(pass.views.clone())]);
        list.SetComputeRootSignature(&pass.root);
        list.SetPipelineState(if mode == 2 { &pass.pso_pack } else { &pass.pso });
        list.SetComputeRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: pass.gpu + u64::from(slot * pass.step) });
        list.SetComputeRoot32BitConstants(1, 4, size.as_ptr().cast(), 0);
        list.Dispatch(size[0].div_ceil(8), size[1].div_ceil(8), 1);
        // What the stretch wrote, before whatever reads or writes it next.
        list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    // SAFETY: a borrowed pointer; the barrier never releases it.
                    pResource: std::mem::transmute_copy(output),
                }),
            },
        }]);
    }
    true
}

/// A R32_UINT texture of ours with unordered access, kept in that state,
/// and a buffer of its copyable size (common state), by size: where
/// [`stretch_shared_exponent`] packs before copying.
struct Scratch {
    texture: ID3D12Resource,
    buffer: ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Scratch {}
// SAFETY: as above.
unsafe impl Sync for Scratch {}

fn scratch(width: u32, height: u32) -> Option<std::sync::Arc<Scratch>> {
    static MADE: Mutex<Option<HashMap<(u32, u32), std::sync::Arc<Scratch>>>> = Mutex::new(None);
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    let made = made.get_or_insert_with(HashMap::new);
    if let Some(s) = made.get(&(width, height)) {
        return Some(s.clone());
    }
    let host = crate::host::host()?;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(width),
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R32_UINT,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let mut texture: Option<ID3D12Resource> = None;
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut bytes = 0u64;
    // SAFETY: plain creation and a query into our own structures.
    unsafe {
        host.device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                None,
                &mut texture,
            )
            .inspect_err(|e| tracing::warn!(error = %e, "FSR stand-in: no packing texture"))
            .ok()?;
        host.device.GetCopyableFootprints(&desc, 0, 1, 0, Some(&mut footprint), None, None, Some(&mut bytes));
    }
    let buffer = host.uav_buffer(bytes)?;
    let s = std::sync::Arc::new(Scratch { texture: texture?, buffer, footprint });
    made.insert((width, height), s.clone());
    Some(s)
}

/// Whether `input` can be stretched into R9G9B9E5 `output` by
/// [`stretch_shared_exponent`].
pub(crate) fn fits_shared_exponent(input: &ID3D12Resource, output: &ID3D12Resource) -> bool {
    // SAFETY: live resources.
    let (i, o) = unsafe { (input.GetDesc(), output.GetDesc()) };
    let plain = |d: &D3D12_RESOURCE_DESC| {
        d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.DepthOrArraySize == 1 && d.SampleDesc.Count == 1
    };
    pack_on()
        && pass().is_some()
        && o.Format.0 == SHARED_EXPONENT
        && plain(&o)
        && plain(&i)
        && i.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
        && srv_format(i.Format.0).is_some()
}

/// Records the stretch of `input` (shader-read state) into R9G9B9E5
/// `output` (copy-destination state, level 0): packed into a R32_UINT
/// texture of ours, copied into a buffer, and from it into the output as
/// R9G9B9E5 (a buffer-to-texture copy takes the bytes as they are; both
/// are 4 bytes a texel with one row pitch). The host cannot write
/// R9G9B9E5 through unordered access or as a target. Logged once per
/// output: "FSR stand-in: packed into R9G9B9E5 <output> via buffer".
pub(crate) fn stretch_shared_exponent(list: &ID3D12GraphicsCommandList, input: &ID3D12Resource, output: &ID3D12Resource) -> bool {
    use windows::core::Interface;
    // SAFETY: a live resource.
    let o = unsafe { output.GetDesc() };
    let Some(s) = scratch(o.Width as u32, o.Height) else {
        return false;
    };
    if !stretch(list, input, &s.texture, 2) {
        return false;
    }
    let mut into = s.footprint;
    into.Footprint.Format = DXGI_FORMAT(SHARED_EXPONENT);
    let at = |r: &ID3D12Resource, placed: Option<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: if placed.is_some() {
            D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT
        } else {
            D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX
        },
        Anonymous: match placed {
            Some(p) => D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: p },
            None => D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        },
    };
    let t = crate::host::transition;
    // SAFETY: an open list and live resources in the states named.
    unsafe {
        list.ResourceBarrier(&[
            t(&s.texture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            t(&s.buffer, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
        ]);
        list.CopyTextureRegion(&at(&s.buffer, Some(s.footprint)), 0, 0, 0, &at(&s.texture, None), None);
        list.ResourceBarrier(&[t(&s.buffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        list.CopyTextureRegion(&at(output, None), 0, 0, 0, &at(&s.buffer, Some(into)), None);
        list.ResourceBarrier(&[
            t(&s.buffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
            t(&s.texture, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
    }
    static TOLD: Mutex<Option<HashSet<usize>>> = Mutex::new(None);
    let mut told = TOLD.lock().unwrap_or_else(|e| e.into_inner());
    if told.get_or_insert_with(HashSet::new).insert(output.as_raw() as usize) {
        tracing::info!("FSR stand-in: packed into R9G9B9E5 {:?} via buffer ({}x{})", output.as_raw(), o.Width, o.Height);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seen(uav: bool, width: u32, height: u32, format: i32) -> Seen {
        Seen { uav, width, height, format }
    }

    #[test]
    fn picks_the_frame_and_the_upscaled_output() {
        let views = [
            seen(false, 1280, 720, 19), // depth
            seen(false, 1280, 720, 34), // motion
            seen(false, 1280, 720, 10), // colour
            seen(false, 1, 1, 10),      // exposure
            seen(false, 1920, 1080, 10), // history
            seen(true, 1920, 1080, 26),
            seen(true, 960, 540, 10),
        ];
        assert_eq!(choose(&views), Some((2, vec![5], 0)));
    }

    #[test]
    fn float_into_unorm_is_tone_mapped() {
        let views = [seen(false, 1280, 720, 10), seen(true, 1920, 1080, 24)];
        assert_eq!(choose(&views), Some((0, vec![1], 1)));
        let views = [seen(false, 1280, 720, 28), seen(true, 1920, 1080, 24)];
        assert_eq!(choose(&views), Some((0, vec![1], 0)));
    }

    #[test]
    fn nothing_without_a_smaller_colour_input_of_its_aspect() {
        assert_eq!(choose(&[seen(true, 1920, 1080, 28), seen(false, 1280, 720, 61)]), None);
        assert_eq!(choose(&[seen(true, 1920, 1080, 28), seen(false, 512, 512, 28)]), None);
        assert_eq!(choose(&[seen(false, 1280, 720, 28)]), None);
        assert_eq!(choose(&[seen(true, 1920, 1080, 67), seen(false, 1280, 720, 28)]), None);
    }

    #[test]
    fn the_stretch_compiles() {
        assert!(crate::host::compile(STRETCH, "stretch", "cs_5_1").is_ok());
        assert!(crate::host::compile(STRETCH, "pack", "cs_5_1").is_ok());
    }

    #[test]
    fn both_upscaler_pipelines_are_created_on_the_host_device() {
        if crate::host::host().is_some() {
            assert!(pass().is_some(), "create both FSR stand-in pipelines");
        }
    }

    #[test]
    fn packed_light_outputs_of_the_output_size_are_written_too() {
        let views = [
            seen(false, 1280, 720, 10),
            seen(true, 1920, 1080, 10),
            seen(true, 1920, 1080, 42),
            seen(true, 1, 1, 42),
        ];
        assert_eq!(choose(&views), Some((0, vec![1, 2], 0)));
        assert_eq!(mode_for(10, 42), 2);
        assert_eq!(mode_for(10, 67), 2);
        let views = [seen(true, 1920, 1080, 10), seen(true, 1920, 1080, 67), seen(false, 1280, 720, 10)];
        assert_eq!(choose(&views), Some((2, vec![0, 1], 0)));
        assert_eq!(mode_for(10, 28), 1);
        assert_eq!(mode_for(10, 10), 0);
    }

    #[test]
    fn one_stretch_per_output_and_frame() {
        assert!(first_this_frame(0x1000, 7));
        assert!(!first_this_frame(0x1000, 7));
        assert!(first_this_frame(0x2000, 7));
        assert!(first_this_frame(0x1000, 8));
    }
}
