use core::ptr;

mod sys {
    extern "C" {
        pub fn alloc(pages: u32) -> usize;
        pub fn free(page: u32);
    }
}

pub fn page_size() -> usize {
    64 * 1024
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
    ptr::null_mut()
}

pub unsafe fn free_part(ptr: *mut u8, oldsize: usize, newsize: usize) -> bool {
    free(ptr.offset(newsize as _), oldsize-newsize)
}

pub unsafe fn free(ptr: *mut u8, size: usize) -> bool {
    let first_page = ptr as usize / page_size();
    let mut last_page = first_page + (size / page_size());
    if size % page_size() != 0 { last_page += 1; }

    for page in first_page..last_page {
        sys::free(page as _);
    }

    true
}

pub fn can_release_part(_flags: u32) -> bool {
    true
}

#[cfg(feature = "global")]
pub fn acquire_global_lock() {
}

#[cfg(feature = "global")]
pub fn release_global_lock() {
}

pub fn allocates_zeros() -> bool {
    true
}
