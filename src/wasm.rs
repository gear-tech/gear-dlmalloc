use crate::common::align_down;
use crate::dlassert;
use core::ptr;

mod gear_core {
    extern "C" {
        pub fn alloc(pages: u32) -> usize;
        pub fn free(page: u32);
    }
}

extern "C" {
    static __heap_base: i32;
}

pub fn page_size() -> usize {
    64 * 1024
}

/// Page where static data is allocated must be already in wasm linear memory.
/// A pointer where heap can be is defined by compiler in global `__heap_base`.
/// We use this addr to init remainder of page for heap allocations.
pub unsafe fn get_preinstalled_memory() -> (usize, usize) {
    // strange thing, but we must take `__heap_base` addr to get heap base address.
    let heap_base = &__heap_base as *const i32 as usize;

    let page_begin = align_down(heap_base, page_size());
    if page_begin == heap_base {
        (heap_base, 0)
    } else {
        (heap_base, page_begin + page_size() - heap_base)
    }
}

pub unsafe fn alloc(size: usize) -> (*mut u8, usize, u32) {
    crate::dlverbose!("heap base = {:?}", &__heap_base as *const i32);
    let pages = size / page_size();
    let prev = gear_core::alloc(pages as _);
    if prev == usize::max_value() {
        return (ptr::null_mut(), 0, 0);
    }
    ((prev * page_size()) as *mut u8, pages * page_size(), 0)
}

pub unsafe fn free(ptr: *mut u8, size: usize) -> bool {
    dlassert!(ptr as usize % page_size() == 0);
    dlassert!(size % page_size() == 0);

    let addr = ptr as usize;
    let first_page = addr / page_size() + (if addr % page_size() == 0 { 0 } else { 1 });
    let end_addr = addr + size;
    let last_page = end_addr / page_size() - (if end_addr % page_size() == 0 { 1 } else { 0 });

    for page in first_page..=last_page {
        gear_core::free(page as _);
    }

    true
}

pub use crate::common::get_free_borders;

#[cfg(feature = "global")]
pub fn acquire_global_lock() {}

#[cfg(feature = "global")]
pub fn release_global_lock() {}
