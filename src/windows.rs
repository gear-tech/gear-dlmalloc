pub fn page_size() -> usize {
    unreachable!("unsupport windows");
}

pub unsafe fn get_preinstalled_memory() -> (usize, usize) {
    unreachable!("unsupport windows");
}

pub unsafe fn alloc(size: usize) -> (*mut u8, usize, u32) {
    unreachable!("unsupport windows");
}

pub unsafe fn free(ptr: *mut u8, size: usize) -> bool {
    unreachable!("unsupport windows");
}

pub use crate::common::get_free_borders;

#[cfg(feature = "global")]
pub fn acquire_global_lock() {
    unreachable!("unsupport windows");
}

#[cfg(feature = "global")]
pub fn release_global_lock() {
    unreachable!("unsupport windows");
}
