//! Diagnostic compatibility hook for software-only loading measurements.

use std::ffi::OsStr;
use std::path::Path;

const BUILD: &str = "b99f876b74d4a77f";
type Patch = (usize, &'static [u8], &'static [u8]);

// These are the two sites checked by the Windows loader hook in
// https://github.com/Marvisak/VIctory/pull/454.
const PATCHES: &[Patch] = &[
    (
        0x2215c27,
        &[0x48, 0x8b, 0x05, 0x1a, 0x07, 0xfa, 0x04],
        &[0xeb, 0x18, 0x90, 0x90, 0x90, 0x90, 0x90],
    ),
    (
        0x23cf683,
        &[0xc7, 0x04, 0x25, 0, 0, 0, 0, 0x21, 0xd4, 0x8b, 0x4d],
        &[0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0, 0, 0, 0, 0],
    ),
];

fn replace(image: &mut [u8], patches: &[Patch]) -> bool {
    if !patches.iter().all(|&(at, before, after)| {
        before.len() == after.len() && image.get(at..at + before.len()) == Some(before)
    }) {
        return false;
    }
    for &(at, _, after) in patches {
        image[at..at + after.len()].copy_from_slice(after);
    }
    true
}

pub(super) fn apply(image: &mut [u8], original: &[u8]) {
    if std::env::var_os("VICTORY_NO_WATCHDOG").as_deref() != Some(OsStr::new("1")) {
        return;
    }
    if !matches!(Path::new("/dev/dri").try_exists(), Ok(false)) {
        tracing::warn!("title watchdog hook requires a software-only sandbox without /dev/dri");
        return;
    }
    if !blake3::hash(original).to_hex().starts_with(BUILD) || !replace(image, PATCHES) {
        tracing::warn!("title watchdog hook does not match this image");
        return;
    }
    tracing::info!("title watchdog hook applied to the mapped image for software diagnostics");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mismatched_site_leaves_the_other_site_untouched() {
        let patches: &[Patch] = &[(0, &[1], &[2]), (2, &[3], &[4])];
        let mut image = [1, 0, 0];
        assert!(!replace(&mut image, patches));
        assert_eq!(image, [1, 0, 0]);
        image[2] = 3;
        assert!(replace(&mut image, patches));
        assert_eq!(image, [2, 0, 4]);
    }
}
