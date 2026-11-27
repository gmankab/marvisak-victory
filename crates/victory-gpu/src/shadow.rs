//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Offset in an imported upload heap, whose bytes already belong to the title.
    imported_offset: Option<u64>,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// The copy's bytes before the submission ran. What the GPU wrote differs.
    before: Mutex<Snapshot>,
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool (at most 8 kept, the largest).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    if pool.len() > 8 {
        pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
        pool.truncate(8);
    }
}

#[derive(Default)]
struct Snapshot {
    /// Sorted, disjoint ranges saved on first write declaration. Adjacent
    /// ranges stay separate so extending a snapshot never recopies old bytes.
    segments: Vec<(usize, Vec<u8>)>,
    reconciled: Reconciled,
}

/// Parts of this submission that no longer need comparing. Keeping the
/// gaps lets later declarations retry a readable part of a failed piece.
#[derive(Default)]
struct Reconciled(BTreeMap<usize, usize>);

impl Reconciled {
    fn next(&self, from: usize, to: usize) -> Option<(usize, usize)> {
        let from = self.0.range(..=from).next_back().map_or(from, |(_, &end)| from.max(end));
        if from >= to {
            return None;
        }
        let end = self.0.range(from..to).next().map_or(to, |(&start, _)| start);
        Some((from, end))
    }

    /// Records an uncovered range, joining any neighbors that touch it.
    fn record(&mut self, from: usize, to: usize) {
        let to = self.0.remove(&to).unwrap_or(to);
        if let Some((_, end)) = self.0.range_mut(..from).next_back()
            && *end == from
        {
            *end = to;
        } else {
            self.0.insert(from, to);
        }
    }
}

impl Snapshot {
    fn capture<'a>(&mut self, mut from: usize, to: usize, host: impl Fn(usize, usize) -> &'a [u8]) {
        let mut index = self.segments.partition_point(|(at, bytes)| at + bytes.len() <= from);
        while from < to {
            let next = self.segments.get(index).map_or(to, |(at, _)| (*at).min(to));
            if from < next {
                let mut bytes = pooled(next - from);
                bytes.extend_from_slice(host(from, next));
                self.segments.insert(index, (from, bytes));
                from = next;
            } else {
                let (at, bytes) = &self.segments[index];
                from = at + bytes.len();
            }
            index += 1;
        }
    }

    /// Reassemble a page only when its original bytes span several segments.
    /// The caller can still read and merge title memory once per page.
    fn piece<'a>(&'a self, from: usize, to: usize, scratch: &'a mut Vec<u8>) -> Option<&'a [u8]> {
        let first = self.segments.partition_point(|(at, bytes)| at + bytes.len() <= from);
        let ((at, bytes), rest) = self.segments[first..].split_first()?;
        let mut end = at + bytes.len();
        if *at > from {
            return None;
        }
        if to <= end {
            return Some(&bytes[from - at..to - at]);
        }
        scratch.clear();
        scratch.extend_from_slice(&bytes[from - at..]);
        for (at, bytes) in rest {
            if *at != end {
                return None;
            }
            scratch.extend_from_slice(&bytes[..(to - end).min(bytes.len())]);
            end += bytes.len();
            if to <= end {
                return Some(scratch);
            }
        }
        None
    }

    fn clear(&mut self) {
        self.reconciled.0.clear();
        for (_, bytes) in self.segments.drain(..) {
            give_back(bytes);
        }
    }
}

/// Applies bytes changed by the GPU to the title snapshot. Returns
/// whether any title bytes need writing.
fn merge_copy_back(title: &mut [u8], host: &[u8], before: &[u8]) -> bool {
    let (title_words, title_tail) = title.as_chunks_mut::<8>();
    let (host_words, host_tail) = host.as_chunks::<8>();
    let (before_words, before_tail) = before.as_chunks::<8>();
    let mut changed = false;
    for ((title, host), &before) in title_words.iter_mut().zip(host_words).zip(before_words) {
        let title_word = u64::from_ne_bytes(*title);
        let host_word = u64::from_ne_bytes(*host);
        let diff = host_word ^ u64::from_ne_bytes(before);
        // Adding 0x7f to the low seven bits sets the high bit if any of
        // those bits are nonzero. The sum stays within each byte.
        const HIGH: u64 = 0x8080_8080_8080_8080;
        const LOW: u64 = !HIGH;
        let nonzero = (diff | ((diff & LOW) + LOW)) & HIGH;
        // Expand each high bit to 0xff in its byte.
        let mask = (nonzero >> 7) * 255;
        let merged = (host_word & mask) | (title_word & !mask);
        changed |= merged != title_word;
        *title = merged.to_ne_bytes();
    }
    for ((title, host), &before) in title_tail.iter_mut().zip(host_tail).zip(before_tail) {
        if *host != before && *host != *title {
            *title = *host;
            changed = true;
        }
    }
    changed
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded. A submission gives each
// persistent mapping to at most one refresh worker and joins all refreshes
// before accessing the mappings again.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        self.imported_offset.unwrap_or(0) + (address - self.start)
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        if self.imported_offset.is_some() {
            list.keep(&self.buffer);
            return;
        }
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(
        buffer: ID3D12Resource,
        data: *mut u8,
        gpu: u64,
        start: u64,
        len: u64,
        imported_offset: Option<u64>,
    ) -> Shadow {
        let pages = if imported_offset.is_some() {
            0
        } else {
            (start + len).div_ceil(PAGE as u64) - start / PAGE as u64
        };
        Shadow {
            buffer,
            data,
            gpu,
            imported_offset,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (each byte is saved once per submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        if self.imported_offset.is_some() {
            return; // Upload heaps only have host GPU read views.
        }
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// only the new parts of each declared range are saved, in reused memory.
    pub(crate) fn writing(&self, address: usize, len: usize) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address + len));
        let (from, to) = self.inside(address, len);
        // SAFETY: inside the mapping.
        let host = |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        self.before.lock().unwrap_or_else(|e| e.into_inner()).capture(from, to, host);
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if from < to && !snapshot.segments.is_empty() {
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            let mut title = Vec::new();
            let mut scratch = Vec::new();
            let mut synced = Vec::new();
            let mut next = from;
            while let Some((at, end)) = snapshot.reconciled.next(next, to) {
                let n = (PIECE - (start + at) % PIECE).min(end - at);
                next = at + n;
                let Some(old) = snapshot.piece(at, next, &mut scratch) else {
                    break;
                };
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                if host != old {
                    title.resize(n, 0);
                    if !crate::host::read_into((start + at) as *const u8, &mut title) {
                        continue;
                    }
                    if merge_copy_back(&mut title, host, old) {
                        crate::watch::writing(start + at, n);
                        if !crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n) {
                            continue;
                        }
                        written += n;
                    }
                    host.copy_from_slice(&title);
                    if n == PAGE {
                        synced.push((start + at) / PAGE);
                    }
                }
                snapshot.reconciled.record(at, next);
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if first < last
                && !synced.is_empty()
                && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
                && let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true)
            {
                let base = start / PAGE;
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                    copied[page - base] = now;
                }
            }
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&r| r != (address, address + len));
        if pending.is_empty() {
            snapshot.clear();
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        if self.imported_offset.is_some() {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.copy(from, to);
        to - from
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        to - from
    }
}

#[cfg(test)]
mod tests {
    use super::{PAGE, Snapshot, merge_copy_back};

    #[test]
    fn snapshot_saves_only_new_ranges_and_reassembles_pages() {
        let mut host = vec![11; 3 * PAGE];
        let mut snapshot = Snapshot::default();
        let mut scratch = Vec::new();
        snapshot.capture(2 * PAGE, 3 * PAGE, |from, to| &host[from..to]);
        snapshot.capture(PAGE - 8, PAGE + 2, |from, to| &host[from..to]);
        assert_eq!(snapshot.segments.iter().map(|(_, bytes)| bytes.len()).sum::<usize>(), PAGE + 10);
        assert!(snapshot.piece(PAGE, 2 * PAGE, &mut scratch).is_none());

        host[PAGE..].fill(19);
        snapshot.capture(PAGE - 12, 2 * PAGE + 5, |from, to| &host[from..to]);
        snapshot.capture(PAGE - 8, 2 * PAGE + 5, |_, _| panic!("already captured"));
        assert_eq!(snapshot.piece(PAGE - 12, PAGE, &mut scratch), Some([11; 12].as_slice()));
        let before = snapshot.piece(PAGE, 2 * PAGE, &mut scratch).unwrap();
        assert_eq!(&before[..2], &[11, 11]);
        assert!(before[2..].iter().all(|&byte| byte == 19));
        assert_eq!(snapshot.piece(2 * PAGE, 2 * PAGE + 5, &mut scratch), Some([11; 5].as_slice()));

        snapshot.clear();
        snapshot.capture(PAGE, PAGE + 4, |from, to| &host[from..to]);
        assert_eq!(snapshot.piece(PAGE, PAGE + 4, &mut scratch), Some([19; 4].as_slice()));
        snapshot.clear();
    }

    #[test]
    fn redundant_writebacks_keep_later_cpu_stores() {
        let host = super::host().expect("test requires a host D3D12 device");
        let mut title = [11u8; 16];
        let address = title.as_mut_ptr() as usize;
        let (buffer, data) = host.shared_buffer(title.len() as u64).unwrap();
        let shadow = super::Shadow::new(buffer, data, 0, address as u64, title.len() as u64, None);
        // SAFETY: the live host mapping belongs to this fixture.
        unsafe { data.write_bytes(11, title.len()) };
        shadow.writing(address + 2, 4);
        shadow.writing(address + 3, 5);
        // SAFETY: the three offsets are within the host mapping.
        unsafe {
            *data.add(2) = 42;
            *data.add(5) = 43;
            *data.add(7) = 44;
        }

        title[4] = 31;
        std::hint::black_box(&mut title);
        assert_eq!(shadow.copy_back(address + 2, 4), 4);
        assert_eq!(&title[2..8], &[42, 11, 31, 43, 11, 11]);
        title[4] = 55;
        title[5] = 56;
        std::hint::black_box(&mut title);
        let second = shadow.copy_back(address + 3, 5);
        assert_eq!(&title[2..8], &[42, 11, 55, 56, 11, 44]);
        assert_eq!(second, 2);
    }

    #[cfg(not(windows))]
    #[test]
    fn smaller_writeback_retries_bytes_beyond_an_unreadable_page() {
        use victory_loader::linux::vm;

        let host = super::host().expect("test requires a host D3D12 device");
        let address = vm::alloc(0, PAGE, vm::MEM_COMMIT, vm::PAGE_READWRITE, PAGE).unwrap();
        let (buffer, data) = host.shared_buffer(PAGE as u64).unwrap();
        let shadow = super::Shadow::new(buffer, data, 0, address as u64, PAGE as u64, None);
        let tail = 8192;
        // SAFETY: both mappings were just created and belong to this test.
        unsafe {
            (address as *mut u8).write_bytes(11, PAGE);
            data.write_bytes(11, PAGE);
        }
        shadow.writing(address, PAGE);
        shadow.writing(address + tail, 4);
        // SAFETY: within the live host mapping.
        unsafe { data.add(tail).write_bytes(42, 4) };
        vm::protect(address + 4096, 4096, vm::PAGE_NOACCESS).unwrap();

        let first = shadow.copy_back(address, PAGE);
        let later = shadow.copy_back(address + tail, 4);
        // SAFETY: the tested four bytes lie outside the inaccessible page.
        let result = unsafe { ((address + tail) as *const [u8; 4]).read() };
        vm::free(address, 0, vm::MEM_RELEASE).unwrap();
        assert_eq!((first, later, result), (0, 4, [42; 4]));
    }

    #[test]
    fn word_merge_matches_byte_merge() {
        for len in (0..=40).chain([4097, 65536]) {
            let before: Vec<_> = (0..len + 1).map(|i| i as u8).collect();
            let before = &before[1..];
            for gpu_step in [1, 2, 3, 8, 64] {
                let mut after = before.to_vec();
                let mut title = before.to_vec();
                for i in (0..len).step_by(gpu_step) {
                    after[i] ^= (1 + i % 255) as u8;
                }
                for byte in title.iter_mut().skip(1).step_by(3) {
                    *byte ^= 0x80;
                }
                for i in (0..len).step_by(5) {
                    title[i] = after[i];
                }
                let expected: Vec<_> = before.iter().zip(&after).zip(&title)
                    .map(|((&old, &gpu), &cpu)| if gpu == old { cpu } else { gpu })
                    .collect();
                let changed = expected != title;
                assert_eq!(merge_copy_back(&mut title, &after, before), changed);
                assert_eq!(title, expected);
                assert!(!merge_copy_back(&mut title, &after, before));
            }
        }
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    register_shadow(resource, start, len, None);
}

/// A placed upload buffer shares the host resource of its whole guest heap.
#[cfg(not(windows))]
pub(crate) fn register_upload(
    resource: usize,
    start: u64,
    len: u64,
    buffer: &ID3D12Resource,
    offset: u64,
) {
    // SAFETY: a live buffer covering the complete upload heap.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() } + offset;
    let shadow = Arc::new(Shadow::new(buffer.clone(), std::ptr::null_mut(), gpu, start, len, Some(offset)));
    register_shadow(resource, start, len, Some(shadow));
}

fn register_shadow(resource: usize, start: u64, len: u64, shadow: Option<Arc<Shadow>>) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, data, gpu, key.0, len, None)));
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len, None));
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}
