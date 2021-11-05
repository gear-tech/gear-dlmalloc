use core::ptr;

mod gear_core {
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
    let prev = gear_core::alloc(pages as _);
    if prev == usize::max_value() {
        return (ptr::null_mut(), 0, 0);
    }
    ((prev * page_size()) as *mut u8, pages * page_size(), 0)
}

pub unsafe fn remap(_ptr: *mut u8, _oldsize: usize, _newsize: usize, _can_move: bool) -> *mut u8 {
    ptr::null_mut()
}

pub unsafe fn free_part(ptr: *mut u8, oldsize: usize, newsize: usize) -> (bool, *mut u8, usize) {
    free(ptr.add(newsize), oldsize - newsize)
}

pub unsafe fn free(ptr: *mut u8, size: usize) -> (bool, *mut u8, usize) {
    let addr = ptr as usize;
    let first_page = addr / page_size() + (if addr % page_size() == 0 { 0 } else { 1 });
    let end_addr = addr + size;
    let last_page = end_addr / page_size() - (if end_addr % page_size() == 0 { 1 } else { 0 });

    for page in first_page..=last_page {
        gear_core::free(page as _);
    }

    (
        true,
        (first_page * page_size()) as *mut u8,
        (last_page - first_page + 1) * page_size(),
    )
}

pub fn can_release_part(_flags: u32) -> bool {
    true
}

#[cfg(feature = "global")]
pub fn acquire_global_lock() {}

#[cfg(feature = "global")]
pub fn release_global_lock() {}

pub fn allocates_zeros() -> bool {
    true
}
