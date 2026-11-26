//! Mapping PE32+ images ourselves, as the Windows loader would: at the
//! preferred base or relocated on a 64 KiB boundary, section by section,
//! with each section's protection and the image's static TLS. Imports are
//! bound separately ([`LoadedImage::bind_imports`]).

use std::collections::HashSet;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use anyhow::{Context, Result, bail, ensure};
use victory_pe::layout::{self, dir};
use victory_pe::{ImageInfo, ImportSymbol};

use super::{seh, teb, thread, tls, vm};

#[cfg(feature = "diagnostics")]
mod watchdog;
use crate::thunks::{ImportSite, ThunkBlock, UnimplementedPolicy};
use crate::{DLL_PROCESS_ATTACH, DLL_PROCESS_DETACH, ImportResolver, Provenance, Resolution};

const SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const SCN_MEM_WRITE: u32 = 0x8000_0000;
const SCN_MEM_READ: u32 = 0x4000_0000;
/// The Windows allocation granularity: every image base is a multiple.
const GRANULE: usize = 0x1_0000;
/// `PAGE_EXECUTE_WRITECOPY`: what `VirtualQuery` reports as an image's
/// allocation protection.
const PAGE_EXECUTE_WRITECOPY: u32 = 0x80;

/// A title image mapped into this process. A clone names the same mapping
/// (no reference of its own): unload one copy only.
#[derive(Debug, Clone)]
pub struct LoadedImage {
    base: usize,
    path: PathBuf,
    info: ImageInfo,
    tls_index: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingImport {
    pub dll: String,
    pub symbol: String,
}

/// Outcome of [`LoadedImage::bind_imports`].
#[derive(Debug, Default, Clone)]
pub struct BindReport {
    pub hle: usize,
    pub host: usize,
    pub title: usize,
    pub missing: Vec<MissingImport>,
}

fn page_size() -> usize {
    // SAFETY: sysconf has no preconditions.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

/// `size` bytes of fresh read-write memory at `at` exactly, or (`None`)
/// wherever the kernel likes on a 64 KiB boundary.
fn map(at: Option<usize>, size: usize) -> Option<usize> {
    let prot = libc::PROT_READ | libc::PROT_WRITE;
    let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;
    let Some(at) = at else {
        // 64 KiB more, then the ends are cut off.
        // SAFETY: a fresh anonymous mapping; no existing memory is touched.
        let p = unsafe { libc::mmap(std::ptr::null_mut(), size + GRANULE, prot, flags, -1, 0) };
        if p == libc::MAP_FAILED {
            return None;
        }
        let raw = p as usize;
        let base = raw.next_multiple_of(GRANULE);
        // SAFETY: trims our own mapping to [base, base + size).
        unsafe {
            if base != raw {
                libc::munmap(p, base - raw);
            }
            libc::munmap((base + size) as *mut c_void, raw + GRANULE - base);
        }
        return Some(base);
    };
    // SAFETY: NOREPLACE never touches an existing mapping.
    let p = unsafe {
        libc::mmap(
            at as *mut c_void,
            size,
            prot,
            flags | libc::MAP_FIXED_NOREPLACE,
            -1,
            0,
        )
    };
    if p == libc::MAP_FAILED {
        return None;
    }
    if p as usize != at {
        // A kernel before 4.17 takes the address as a hint.
        // SAFETY: our fresh mapping.
        unsafe { libc::munmap(p, size) };
        return None;
    }
    Some(at)
}

/// Maps `original` (a PE32+ file read from `path`): copies its headers and
/// sections, relocates it if its preferred base is taken, gives it static
/// TLS and its sections' protections, and makes it known to the region
/// table and the exception dispatcher. Runs no title code.
pub fn load(original: &[u8], path: &Path) -> Result<LoadedImage> {
    let info = ImageInfo::parse(original)?;
    let size = (info.size_of_image as usize).next_multiple_of(page_size());
    let opt = layout::optional_header_offset(original).context("no optional header")?;
    let headers = layout::read_u32(original, opt + 60).context("no SizeOfHeaders")? as usize;
    ensure!(
        headers <= size && headers <= original.len(),
        "headers do not fit the image"
    );
    let preferred = usize::try_from(info.image_base)?;
    let base = match map(Some(preferred), size) {
        Some(base) => base,
        None if info.has_relocations => map(None, size).context("mmap for the image failed")?,
        None => bail!("preferred base {preferred:#x} is taken and the image has no relocations"),
    };
    let unmap = || {
        // SAFETY: our mapping, not yet handed out.
        unsafe { libc::munmap(base as *mut c_void, size) };
    };
    let mapped = (|| {
        // SAFETY: the mapping is `size` bytes, read-write and ours.
        let image = unsafe { std::slice::from_raw_parts_mut(base as *mut u8, size) };
        image[..headers].copy_from_slice(&original[..headers]);
        for s in &info.sections {
            let len = match s.virtual_size {
                0 => s.file_size,
                vs => vs.min(s.file_size),
            } as usize;
            let (from, to) = (s.file_offset as usize, s.rva as usize);
            let data = original.get(from..from + len).with_context(|| {
                format!(
                    "section {} runs past the file",
                    s.name.trim_end_matches('\0')
                )
            })?;
            image
                .get_mut(to..to + len)
                .with_context(|| {
                    format!(
                        "section {} runs past the image",
                        s.name.trim_end_matches('\0')
                    )
                })?
                .copy_from_slice(data);
        }
        if base != preferred {
            relocate(image, original, base.wrapping_sub(preferred) as u64)?;
        }
        #[cfg(feature = "diagnostics")]
        if !info.is_dll() {
            watchdog::apply(image, original);
        }
        Ok(())
    })();
    if let Err(e) = mapped {
        unmap();
        return Err(e);
    }
    let tls_index = match &info.tls {
        // SAFETY: mapped and relocated just now; no title code runs yet.
        Some(t) => match unsafe { tls::add(base, t.directory_rva, !info.is_dll()) } {
            Ok(index) => Some(index),
            Err(e) => {
                unmap();
                return Err(e);
            }
        },
        None => None,
    };
    let loaded = LoadedImage {
        base,
        path: path.to_owned(),
        info,
        tls_index,
    };
    vm::adopt(base, size, PAGE_EXECUTE_WRITECOPY);
    loaded.protect(headers);
    seh::register_image(base, size);
    if let Some(file) = path.file_name() {
        seh::name_image(base, &file.to_string_lossy());
    }
    // The exe: `GetModuleHandle(NULL)` and new threads' default stack.
    if !loaded.info.is_dll() {
        teb::set_image_base(base);
        thread::set_default_stack(usize::try_from(loaded.info.stack_reserve).unwrap_or(usize::MAX));
    }
    tracing::info!(
        base = format_args!("{base:#x}"),
        preferred = format_args!("{preferred:#x}"),
        tls_index = ?loaded.tls_index,
        path = %loaded.path.display(),
        "image mapped"
    );
    thread::mapped(&loaded);
    Ok(loaded)
}

/// Reads and maps the image at `path` (see [`load`]).
pub fn load_file(path: &Path) -> Result<LoadedImage> {
    let original = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    load(&original, path).with_context(|| format!("loading {}", path.display()))
}

/// Applies the base relocations of `original` to `image`, mapped `delta`
/// bytes from its preferred base.
fn relocate(image: &mut [u8], original: &[u8], delta: u64) -> Result<()> {
    const ABSOLUTE: u16 = 0;
    const HIGHLOW: u16 = 3;
    const DIR64: u16 = 10;
    let at = layout::data_directory_offset(original, dir::BASERELOC).context("no relocations")?;
    let rva = layout::read_u32(original, at).context("no relocations")? as usize;
    let len = layout::read_u32(original, at + 4).context("no relocations")? as usize;
    let table = image
        .get(rva..rva + len)
        .context("relocation table outside the image")?
        .to_vec();
    let mut at = 0;
    while at + 8 <= table.len() {
        let page = u32::from_le_bytes(table[at..at + 4].try_into()?) as usize;
        let block = u32::from_le_bytes(table[at + 4..at + 8].try_into()?) as usize;
        ensure!(
            block >= 8 && at + block <= table.len(),
            "bad relocation block at {at:#x}"
        );
        for entry in table[at + 8..at + block].as_chunks::<2>().0 {
            let entry = u16::from_le_bytes([entry[0], entry[1]]);
            let target = page + usize::from(entry & 0xFFF);
            let (kind, width) = match entry >> 12 {
                ABSOLUTE => continue,
                DIR64 => (DIR64, 8),
                HIGHLOW => (HIGHLOW, 4),
                other => bail!("unsupported relocation type {other}"),
            };
            let field = image
                .get_mut(target..target + width)
                .context("relocation outside the image")?;
            if kind == DIR64 {
                let v = u64::from_le_bytes(field[..8].try_into()?).wrapping_add(delta);
                field.copy_from_slice(&v.to_le_bytes());
            } else {
                let v = u32::from_le_bytes(field[..4].try_into()?).wrapping_add(delta as u32);
                field.copy_from_slice(&v.to_le_bytes());
            }
        }
        at += block;
    }
    Ok(())
}

/// What `VirtualQuery` reports for a section, as on Windows: writable
/// image pages are copy-on-write until written.
fn windows_protection(characteristics: u32) -> u32 {
    match (
        characteristics & SCN_MEM_EXECUTE != 0,
        characteristics & SCN_MEM_WRITE != 0,
    ) {
        (true, true) => PAGE_EXECUTE_WRITECOPY,
        (true, false) => 0x20, // PAGE_EXECUTE_READ
        (false, true) => 0x08, // PAGE_WRITECOPY
        (false, false) => 0x02, // PAGE_READONLY
    }
}

fn native_protection(characteristics: u32) -> libc::c_int {
    libc::PROT_READ
        | if characteristics & SCN_MEM_WRITE != 0 {
            libc::PROT_WRITE
        } else {
            0
        }
        | if characteristics & SCN_MEM_EXECUTE != 0 {
            libc::PROT_EXEC
        } else {
            0
        }
}

/// Maps the DLLs the title ships in `title_dir` that `main` imports,
/// directly or through each other, in initialisation order (dependencies
/// first). Only DLL names `wanted` accepts are considered, so DLLs the host
/// or our runtime provide stay with them. File names are matched without
/// regard to case. Imports are not bound. `_cache_dir` is unused on Linux
/// (Windows writes prepared copies there).
pub fn load_title_dlls(
    main: &ImageInfo,
    title_dir: &Path,
    _cache_dir: &Path,
    wanted: &dyn Fn(&str) -> bool,
) -> Result<Vec<(String, LoadedImage)>> {
    let mut loaded = Vec::new();
    map_imports(main, title_dir, wanted, &mut HashSet::new(), &mut loaded)?;
    Ok(loaded)
}

/// Maps the DLL `name` from `title_dir` like [`load_title_dlls`] does, with
/// the title DLLs it imports (dependencies first, `name` last); empty if
/// `wanted` refuses it or the folder has no such file. For DLLs a title
/// loads itself; `wanted` must refuse those already mapped.
pub fn load_title_dll(
    name: &str,
    title_dir: &Path,
    _cache_dir: &Path,
    wanted: &dyn Fn(&str) -> bool,
) -> Result<Vec<(String, LoadedImage)>> {
    let mut loaded = Vec::new();
    map_dll(name, title_dir, wanted, &mut HashSet::new(), &mut loaded)?;
    Ok(loaded)
}

fn map_imports(
    info: &ImageInfo,
    dir: &Path,
    wanted: &dyn Fn(&str) -> bool,
    seen: &mut HashSet<String>,
    loaded: &mut Vec<(String, LoadedImage)>,
) -> Result<()> {
    for module in info.imports.iter().chain(&info.delay_imports) {
        map_dll(&module.dll, dir, wanted, seen, loaded)?;
    }
    Ok(())
}

fn map_dll(
    name: &str,
    dir: &Path,
    wanted: &dyn Fn(&str) -> bool,
    seen: &mut HashSet<String>,
    loaded: &mut Vec<(String, LoadedImage)>,
) -> Result<()> {
    let name = name.to_ascii_lowercase();
    if !seen.insert(name.clone()) || !wanted(&name) {
        return Ok(());
    }
    let Some(path) = find_file(dir, &name) else {
        return Ok(());
    };
    let image = load_file(&path)?;
    map_imports(&image.info, dir, wanted, seen, loaded)?;
    loaded.push((name, image));
    Ok(())
}

/// The file in `dir` called `name`, whatever its case, as on Windows.
pub fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    let exact = dir.join(name);
    if exact.is_file() {
        return Some(exact);
    }
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .find(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.eq_ignore_ascii_case(name))
        })
        .map(|e| e.path())
        .filter(|p| p.is_file())
}

/// The export `name` of the PE image mapped at `base`, read from its
/// export directory in memory (`GetProcAddress` on a module handle); a
/// forwarder is `None`.
///
/// # Safety
/// `base` is the base of a mapped PE32+ image.
pub unsafe fn export_at(base: usize, name: &str) -> Option<usize> {
    // SAFETY: the caller's image; every offset read is one the PE format
    // puts inside its headers or its export directory.
    unsafe {
        let at = |rva: usize| base + rva;
        let u32_at = |a: usize| (a as *const u32).read_unaligned() as usize;
        let nt = at(u32_at(at(0x3c)));
        // OptionalHeader at +0x18, DataDirectory[0] at +0x70 of it.
        let (dir, dir_size) = (u32_at(nt + 0x88), u32_at(nt + 0x8c));
        if dir == 0 {
            return None;
        }
        let names = u32_at(at(dir + 0x18));
        let (functions, name_rvas, ordinals) = (
            u32_at(at(dir + 0x1c)),
            u32_at(at(dir + 0x20)),
            u32_at(at(dir + 0x24)),
        );
        let i = (0..names).find(|&i| {
            let s = std::ffi::CStr::from_ptr(at(u32_at(at(name_rvas + 4 * i))) as *const _);
            s.to_bytes() == name.as_bytes()
        })?;
        let ordinal = (at(ordinals + 2 * i) as *const u16).read_unaligned() as usize;
        let rva = u32_at(at(functions + 4 * ordinal));
        (rva != 0 && !(dir..dir + dir_size).contains(&rva)).then(|| at(rva))
    }
}

/// Resolves through `inner`, then through the exports of the title's DLLs.
pub struct WithTitleDlls<'a> {
    pub inner: &'a dyn ImportResolver,
    pub dlls: &'a [(String, LoadedImage)],
}

impl ImportResolver for WithTitleDlls<'_> {
    fn resolve(&self, dll: &str, symbol: &ImportSymbol) -> Resolution {
        let found = self.inner.resolve(dll, symbol);
        if found != Resolution::Missing {
            return found;
        }
        self.dlls
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(dll))
            .and_then(|(_, image)| image.export_symbol(symbol))
            .map_or(Resolution::Missing, |address| Resolution::Found {
                address,
                provenance: Provenance::Title,
            })
    }
}

impl LoadedImage {
    pub fn base(&self) -> usize {
        self.base
    }

    /// Parsed from the original image (imports, entry point, TLS callbacks).
    pub fn info(&self) -> &ImageInfo {
        &self.info
    }

    /// The image's static TLS index, if it has a TLS directory.
    pub fn tls_index(&self) -> Option<u32> {
        self.tls_index
    }

    /// Address of an export of the image, if it has one by that name.
    pub fn export(&self, name: &str) -> Option<usize> {
        let e = self
            .info
            .exports
            .iter()
            .find(|e| e.name.as_deref() == Some(name))?;
        // ponytail: forwarded exports resolve to nothing; follow `e.forward` if a title DLL has one we need.
        Some(self.base + e.rva? as usize)
    }

    /// Address of an export, by name or ordinal.
    pub fn export_symbol(&self, symbol: &ImportSymbol) -> Option<usize> {
        match symbol {
            ImportSymbol::Name { name, .. } => self.export(name),
            ImportSymbol::Ordinal(o) => self
                .info
                .exports
                .iter()
                .find(|e| e.ordinal == *o)
                .and_then(|e| Some(self.base + e.rva? as usize)),
        }
    }

    /// Fills every IAT slot of the image.
    ///
    /// Each import goes to `resolver`; whatever it cannot resolve gets an
    /// unimplemented-import stub that behaves according to `policy`.
    pub fn bind_imports(
        &self,
        resolver: &dyn ImportResolver,
        policy: UnimplementedPolicy,
    ) -> Result<BindReport> {
        let mut report = BindReport::default();
        let mut slots: Vec<(u32, usize)> = Vec::with_capacity(self.info.import_count());
        let mut unresolved: Vec<(u32, &str, &ImportSymbol)> = Vec::new();

        // Delay-load slots are bound eagerly too, so the title's own
        // delay-load helper never runs.
        for module in self.info.imports.iter().chain(&self.info.delay_imports) {
            for entry in &module.entries {
                match resolver.resolve(&module.dll, &entry.symbol) {
                    Resolution::Found {
                        address,
                        provenance,
                    } => {
                        match provenance {
                            Provenance::Hle => report.hle += 1,
                            Provenance::Host => report.host += 1,
                            Provenance::Title => report.title += 1,
                        }
                        slots.push((entry.iat_slot_rva, address));
                    }
                    Resolution::Missing => {
                        unresolved.push((entry.iat_slot_rva, &module.dll, &entry.symbol));
                    }
                }
            }
        }

        if !unresolved.is_empty() {
            let mut block = ThunkBlock::new(unresolved.len())?;
            for (slot, dll, symbol) in unresolved {
                let site: &'static ImportSite = Box::leak(Box::new(ImportSite {
                    dll: dll.to_owned(),
                    symbol: symbol.to_string(),
                    policy,
                    calls: AtomicU64::new(0),
                }));
                slots.push((slot, block.emit(site)?));
                report.missing.push(MissingImport {
                    dll: site.dll.clone(),
                    symbol: site.symbol.clone(),
                });
            }
            block.seal()?;
        }

        self.write_slots(&slots)?;
        tracing::info!(
            hle = report.hle,
            host = report.host,
            missing = report.missing.len(),
            "imports bound"
        );
        Ok(report)
    }

    /// The protection of the section holding `rva` (headers: read-only).
    fn protection_at(&self, rva: usize) -> libc::c_int {
        self.info
            .sections
            .iter()
            .rev()
            .find(|s| rva >= s.rva as usize)
            .map_or(libc::PROT_READ, |s| native_protection(s.characteristics))
    }

    /// Headers read-only, then each section's own protection, also as the
    /// region table records it. A page two sections share takes the later
    /// one's.
    fn protect(&self, headers: usize) {
        let page = page_size();
        let set = |rva: usize, len: usize, characteristics: u32| {
            let prot = native_protection(characteristics);
            let start = (self.base + rva) & !(page - 1);
            let end = (self.base + rva + len).next_multiple_of(page);
            vm::adopt(start, end - start, windows_protection(characteristics));
            // SAFETY: inside our mapping of the image.
            if unsafe { libc::mprotect(start as *mut c_void, end - start, prot) } != 0 {
                tracing::warn!(
                    rva,
                    len,
                    "mprotect of an image range failed: {}",
                    std::io::Error::last_os_error()
                );
            }
        };
        set(0, headers, SCN_MEM_READ);
        let size = self.info.size_of_image as usize;
        for s in &self.info.sections {
            let rva = s.rva as usize;
            let len = match s.virtual_size {
                0 => s.file_size,
                vs => vs,
            } as usize;
            if rva < size && len != 0 {
                set(rva, len.min(size - rva), s.characteristics);
            }
        }
    }

    /// Makes each slot's page writable just long enough to write it: import
    /// tables sit in sections of differing protection.
    fn write_slots(&self, slots: &[(u32, usize)]) -> Result<()> {
        let page = page_size();
        for &(rva, address) in slots {
            ensure!(
                rva as usize + 8 <= self.info.size_of_image as usize,
                "IAT slot {rva:#x} outside the image"
            );
            let slot = (self.base + rva as usize) as *mut u64;
            let start = slot as usize & !(page - 1);
            let len = (slot as usize + 8).next_multiple_of(page) - start;
            let old = self.protection_at(start - self.base);
            // SAFETY: pages of our image mapping.
            if unsafe { libc::mprotect(start as *mut c_void, len, old | libc::PROT_WRITE) } != 0 {
                bail!(
                    "mprotect(IAT slot {rva:#x}) failed: {}",
                    std::io::Error::last_os_error()
                );
            }
            // SAFETY: inside the image, now writable; the PE format aligns slots.
            unsafe { slot.write_volatile(address as u64) };
            // SAFETY: restoring the protection we just changed.
            unsafe { libc::mprotect(start as *mut c_void, len, old) };
        }
        Ok(())
    }

    /// Calls the image's TLS callbacks with `reason` (e.g. `DLL_PROCESS_ATTACH`).
    ///
    /// # Safety
    /// Runs title code. Imports must be bound first, and the calling thread
    /// attached (a guest thread, or `attach_current`).
    pub unsafe fn run_tls_callbacks(&self, reason: u32) {
        // SAFETY: forwarded from the caller.
        unsafe { self.tls_callbacks(reason, std::ptr::null_mut()) }
    }

    /// # Safety
    /// See [`Self::run_tls_callbacks`].
    unsafe fn tls_callbacks(&self, reason: u32, reserved: *mut c_void) {
        let Some(tls) = &self.info.tls else { return };
        for &rva in &tls.callback_rvas {
            type TlsCallback = unsafe extern "win64" fn(usize, u32, *mut c_void);
            // SAFETY: the RVA came from the image's own TLS directory.
            let callback: TlsCallback = unsafe { std::mem::transmute(self.base + rva as usize) };
            // SAFETY: the caller guarantees imports are bound.
            unsafe { callback(self.base, reason, reserved) };
        }
    }

    /// Initialises a DLL as the Windows loader would: its TLS callbacks,
    /// then `DllMain(DLL_PROCESS_ATTACH)`. From then on it hears of every
    /// thread start and end on guest threads.
    ///
    /// # Safety
    /// Runs title code on the calling thread, which must be attached.
    /// Imports must be bound first, and the DLLs this one depends on
    /// attached.
    pub unsafe fn attach(&self) -> Result<()> {
        // SAFETY: forwarded from the caller.
        if unsafe { self.notify(DLL_PROCESS_ATTACH, true, std::ptr::null_mut()) } == Some(0) {
            bail!("DllMain returned FALSE");
        }
        thread::attached(self.base);
        Ok(())
    }

    /// `DLL_PROCESS_DETACH` as `FreeLibrary` sends it (TLS callbacks, then
    /// `DllMain`); the image hears of no more threads.
    ///
    /// # Safety
    /// Runs title code. The image must have been attached (or its attach
    /// failed), and no other thread may be using it.
    pub unsafe fn detach(&self) {
        thread::detached(self.base);
        // SAFETY: forwarded from the caller.
        unsafe { self.notify(DLL_PROCESS_DETACH, true, std::ptr::null_mut()) };
    }

    /// Runs the image's TLS callbacks with `reason`, then, if `dll_main` and
    /// the image is a DLL with an entry point, its `DllMain`, whose result
    /// is returned. `reserved` is the entry points' third argument.
    ///
    /// # Safety
    /// Runs title code. Imports must be bound first.
    pub(crate) unsafe fn notify(
        &self,
        reason: u32,
        dll_main: bool,
        reserved: *mut c_void,
    ) -> Option<i32> {
        // SAFETY: forwarded from the caller.
        unsafe { self.tls_callbacks(reason, reserved) };
        if !dll_main || !self.info.is_dll() || self.info.entry_rva == 0 {
            return None;
        }
        type DllMain = unsafe extern "win64" fn(usize, u32, *mut c_void) -> i32;
        // SAFETY: the entry RVA came from the original image header.
        let main: DllMain =
            unsafe { std::mem::transmute(self.base + self.info.entry_rva as usize) };
        // SAFETY: forwarded from the caller.
        Some(unsafe { main(self.base, reason, reserved) })
    }

    /// Starts the title: on a new guest thread with the stack size from the
    /// image header, attaches `dlls` in order (see [`Self::attach`]), runs
    /// the image's TLS callbacks (`DLL_PROCESS_ATTACH`), then its entry
    /// point. Returns the thread so the caller can join it; its result is
    /// the entry point's, or the code given to `ExitThread`.
    ///
    /// # Safety
    /// Runs title code. Imports of the image and of `dlls` must be bound.
    pub unsafe fn spawn_main(
        self: Arc<Self>,
        dlls: Vec<(String, LoadedImage)>,
    ) -> Result<std::thread::JoinHandle<Result<u32>>> {
        if self.info.entry_rva == 0 {
            bail!("image has no entry point");
        }
        let stack = usize::try_from(self.info.stack_reserve)
            .unwrap_or(usize::MAX)
            .max(1 << 20);
        thread::builder(stack)
            .name("title-main".into())
            .spawn(move || {
                let mut failed = None;
                let code = thread::run_guest(
                    || {},
                    || {
                        for (name, dll) in &dlls {
                            // SAFETY: forwarded from the caller of `spawn_main`.
                            if let Err(e) = unsafe { dll.attach() } {
                                failed = Some(e.context(format!("attaching {name}")));
                                return 0;
                            }
                        }
                        // SAFETY: forwarded from the caller of `spawn_main`.
                        unsafe { self.run_tls_callbacks(DLL_PROCESS_ATTACH) };
                        thread::attached(self.base);
                        type Entry = unsafe extern "win64" fn(*mut c_void) -> u32;
                        // SAFETY: the entry RVA came from the original image header.
                        let entry: Entry = unsafe {
                            std::mem::transmute(self.base + self.info.entry_rva as usize)
                        };
                        // SAFETY: forwarded from the caller of `spawn_main`.
                        unsafe { entry(std::ptr::null_mut()) }
                    },
                );
                failed.map_or(Ok(code), Err)
            })
            .context("spawning title main thread")
    }

    /// Unmaps the image.
    ///
    /// # Safety
    /// No title code may be running or run later, and no pointer into the
    /// image may be used afterwards.
    pub unsafe fn unload(self) {
        thread::unmapped(self.base);
        seh::unregister_image(self.base);
        vm::forget(self.base);
        if let Some(index) = self.tls_index {
            tls::remove(index);
        }
        let size = (self.info.size_of_image as usize).next_multiple_of(page_size());
        // SAFETY: guaranteed by the caller.
        unsafe { libc::munmap(self.base as *mut c_void, size) };
    }
}
