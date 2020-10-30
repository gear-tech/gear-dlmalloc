use core::arch::wasm32;
use core::ptr;

mod sys {
    extern "C" {
        pub fn alloc(pages: u32) -> usize;
        pub fn free(page: u32);
    }
}

pub unsafe fn alloc(size: usize) -> (*mut u8, usize, u32) {
    let pages = size / page_size();
    let prev = sys::alloc(pages as _);
    if prev == usize::max_value() {
        return (ptr::null_mut(), 0, 0);
    }
    ((prev * page_size()) as *mut u8, pages * page_size(), 0)
}

pub unsafe fn remap(_ptr: *mut u8, _oldsize: usize, _newsize: usize, _can_move: bool) -> *mut u8 {
    // TODO: I think this can be implemented near the end?
    ptr::null_mut()
}

pub unsafe fn free_part(_ptr: *mut u8, _oldsize: usize, _newsize: usize) -> bool {
    false
}

pub unsafe fn free(ptr: *mut u8, size: usize) -> bool {
    let mut pages = size / page_size();
    if size % page_size() > 0 { pages += 1; }
    let first_page = ptr as usize / page_size();

    for page in 0..pages {
        sys::free(page as _)
    }

    true
}

pub fn can_release_part(_flags: u32) -> bool {
    false
}

#[cfg(feature = "global")]
pub fn acquire_global_lock() {
    // single threaded, no need!
}

#[cfg(feature = "global")]
pub fn release_global_lock() {
    // single threaded, no need!
}

pub fn allocates_zeros() -> bool {
    true
}

pub fn page_size() -> usize {
    64 * 1024
}
