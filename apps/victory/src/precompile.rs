//! Populate the runtime shader cache from standalone and packed containers.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use anyhow::{Context, Result, ensure};
use victory_hle::modules::d3d12x::{cache_shader, shader_cache_counts, use_shader_cache};
use victory_shader::Container;

use crate::shader::{MAX_FILE, Source, sources};

pub fn run(files: &[PathBuf], workers: usize, limit: Option<usize>) -> Result<()> {
    let cache = victory_core::Config::load()?.cache_subdir("shaders")?;
    println!("Shader cache: {}", cache.display());
    use_shader_cache(cache);
    let seen = Mutex::new(HashSet::new());
    let errors = Mutex::new(Vec::new());
    let completed = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let mut scanned = 0;

    for file in files {
        let size = std::fs::metadata(file)
            .with_context(|| format!("reading {}", file.display()))?
            .len();
        ensure!(size <= MAX_FILE, "{} is too large to scan", file.display());
        let data = std::fs::read(file).with_context(|| format!("reading {}", file.display()))?;
        let records = sources(&data);
        let count = limit.map_or(records.len(), |n| records.len().min(n - scanned));
        println!(
            "{}: processing {count} of {} records",
            file.display(),
            records.len()
        );
        scanned += count;
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..workers.min(count) {
                scope.spawn(|| {
                    loop {
                        let index = next.fetch_add(1, Relaxed);
                        let Some(source) = records[..count].get(index) else {
                            break;
                        };
                        let result = source
                            .bytes()
                            .ok_or_else(|| "invalid packed record".to_owned())
                            .and_then(|bytes| {
                                Container::parse(&bytes).map_err(|error| error.to_string())?;
                                let container = match source {
                                    // A raw record borrows the rest of the input file. The validated
                                    // DXBC length is the size of the shader passed to the runtime.
                                    Source::Raw { .. } => {
                                        let length = u32::from_le_bytes(
                                            bytes[24..28]
                                                .try_into()
                                                .expect("validated DXBC header"),
                                        );
                                        &bytes[..length as usize]
                                    }
                                    Source::Packed(_) => &bytes,
                                };
                                if seen.lock().unwrap().insert(blake3::hash(container)) {
                                    cache_shader(container)?;
                                }
                                Ok(())
                            });
                        if let Err(error) = result {
                            failed.fetch_add(1, Relaxed);
                            let mut errors = errors.lock().unwrap();
                            if errors.len() < 10 {
                                errors.push(format!(
                                    "{}{}: {error}",
                                    file.display(),
                                    source.label()
                                ));
                            }
                        }
                        let done = completed.fetch_add(1, Relaxed) + 1;
                        if done.is_multiple_of(1_000) {
                            println!("Processed {done} records");
                        }
                    }
                });
            }
        });
        if limit == Some(scanned) {
            break;
        }
    }

    let (hits, translated) = shader_cache_counts();
    let failed = failed.into_inner();
    println!("{scanned} records; {translated} translated, {hits} already cached, {failed} failed");
    for error in errors.into_inner().unwrap() {
        eprintln!("  {error}");
    }
    ensure!(
        hits + translated > 0,
        "no usable shaders were written to or read from the cache"
    );
    Ok(())
}
