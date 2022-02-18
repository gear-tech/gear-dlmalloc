extern crate libc;

use core::ptr;

use crate::dlassert;

pub fn page_size() -> usize {
    0x4000
}

pub unsafe fn alloc(size: usize) -> (*mut u8, usize, u32) {
    let addr = libc::mmap(
        ptr::null_mut(),
        size,
        libc::PROT_WRITE | libc::PROT_READ,
        libc::MAP_ANON | libc::MAP_PRIVATE,
        -1,
        0,
    );
    if addr == libc::MAP_FAILED {
        (ptr::null_mut(), 0, 0)
    } else {
        (addr as *mut u8, size, 0)
    }
}

fn align_up(a: usize, alignment: usize) -> usize {
    dlassert!(alignment.is_power_of_two());
    (a + (alignment - 1)) & !(alignment - 1)
}

pub unsafe fn free_borders(ptr: *mut u8, size: usize) -> (*mut u8, usize) {
    if size < page_size() {
        return (ptr, 0);
    }
    let addr = ptr as usize;

    // align addr to page size
    let aligned_addr = align_up(addr, page_size());
    if addr + size <= aligned_addr {
        return (ptr, 0);
    }

    let size = addr + size - aligned_addr;
    let aligned_size = (size / page_size()) * page_size();
    (aligned_addr as *mut u8, aligned_size)
}

pub unsafe fn free(ptr: *mut u8, size: usize) -> bool {
    libc::munmap(ptr as *mut _, size) == 0
}

#[cfg(feature = "global")]
static mut LOCK: libc::pthread_mutex_t = libc::PTHREAD_MUTEX_INITIALIZER;

#[cfg(feature = "global")]
pub fn acquire_global_lock() {
    unsafe { assert_eq!(libc::pthread_mutex_lock(&mut LOCK), 0) }
}

#[cfg(feature = "global")]
pub fn release_global_lock() {
    unsafe { assert_eq!(libc::pthread_mutex_unlock(&mut LOCK), 0) }
}
