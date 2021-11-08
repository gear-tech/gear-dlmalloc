extern crate libc;

use core::ptr;

pub unsafe fn alloc(size: usize) -> (*mut u8, usize, u32) {
    let addr = libc::mmap(
        ptr::null_mut(),
        size,
        libc::PROT_WRITE | libc::PROT_READ,
        libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
        -1,
        0,
    );
    if addr == libc::MAP_FAILED {
        (ptr::null_mut(), 0, 0)
    } else {
        (addr as *mut u8, size, 0)
    }
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
