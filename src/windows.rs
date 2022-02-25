//! Windows is unsupported currently.
//! It means, that you cannot use this allocator in native windows programs.

pub fn page_size() -> usize {
    unreachable!("windows is unsupported");
}

pub unsafe fn get_preinstalled_memory() -> (usize, usize) {
    unreachable!("windows is unsupported");
}

pub unsafe fn alloc(size: usize) -> (*mut u8, usize, u32) {
    unreachable!("windows is unsupported");
}

pub unsafe fn free(ptr: *mut u8, size: usize) -> bool {
    unreachable!("windows is unsupported");
}

pub use crate::common::get_free_borders;

#[cfg(feature = "global")]
pub fn acquire_global_lock() {
    unreachable!("windows is unsupported");
}

#[cfg(feature = "global")]
pub fn release_global_lock() {
    unreachable!("windows is unsupported");
}
