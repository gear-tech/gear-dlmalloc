use crate::dlassert;
use crate::sys;

/// Returns min number which is >= `a` and aligned by `alignment`
pub fn align_up(a: usize, alignment: usize) -> usize {
    dlassert!(alignment.is_power_of_two());
    (a + (alignment - 1)) & !(alignment - 1)
}

/// Returns max number which is <= `a` and aligned by `alignment`
pub fn align_down(a: usize, alignemnt: usize) -> usize {
    (a / alignemnt) * alignemnt
}

/// Returns cropped memory interval in which memory can be successfully freed.
/// Actually retuerns addr and size must be aligned to page_size.
pub unsafe fn get_free_borders(ptr: *mut u8, size: usize) -> (*mut u8, usize) {
    if size < sys::page_size() {
        return (ptr, 0);
    }
    let addr = ptr as usize;

    // align addr to page size
    let aligned_addr = align_up(addr, sys::page_size());
    if addr + size <= aligned_addr {
        return (ptr, 0);
    }

    let size = addr + size - aligned_addr;
    let aligned_size = (size / sys::page_size()) * sys::page_size();
    (aligned_addr as *mut u8, aligned_size)
}
