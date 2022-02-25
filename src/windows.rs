//! Windows is unsupported currently.
//! It means, that you cannot use this allocator in native windows programs.

pub fn page_size() -> usize {
    unreachable!("Windows is unsupported");
}

pub unsafe fn get_preinstalled_memory() -> (usize, usize) {
    unreachable!("Windows is unsupported");
}

pub unsafe fn alloc(size: usize) -> (*mut u8, usize, u32) {
    unreachable!("Windows is unsupported");
}

pub unsafe fn free(ptr: *mut u8, size: usize) -> bool {
    unreachable!("Windows is unsupported");
}

pub use crate::common::get_free_borders;

#[cfg(feature = "global")]
pub fn acquire_global_lock() {
    unreachable!("Windows is unsupported");
}

#[cfg(feature = "global")]
pub fn release_global_lock() {
    unreachable!("Windows is unsupported");
}
