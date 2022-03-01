extern crate windows;

use crate::dlassert;
use core::ptr;
use core::ffi::{c_void};
use once_cell::sync::Lazy;

pub fn page_size() -> usize { 4 * 1024 }

pub unsafe fn get_preinstalled_memory() -> (usize, usize) { (0, 0) }

pub unsafe fn alloc(size: usize) -> (*mut u8, usize, u32) {
    let addr = windows::Win32::System::Memory::VirtualAlloc(
        ptr::null_mut(),
        size,
        windows::Win32::System::Memory::MEM_RESERVE | windows::Win32::System::Memory::MEM_COMMIT,
        windows::Win32::System::Memory::PAGE_READWRITE,
    );

    if addr == ptr::null_mut() {
        (ptr::null_mut(), 0, 0)
    } else {
        (addr as *mut u8, size, 0)
    }
}

pub unsafe fn free(ptr: *mut u8, size: usize) -> bool {
    windows::Win32::System::Memory::VirtualFree(
        ptr as *mut c_void,
        size,
        windows::Win32::System::Memory::MEM_RELEASE).0 != 0
}

pub use crate::common::get_free_borders;

# [cfg(feature = "global")]
static MUTEX: Lazy<windows::Win32::Foundation::HANDLE> = unsafe {
    Lazy::new(|| {
        windows::Win32::System::Threading::CreateMutexA(
            ptr::null_mut(),
            false,
            windows::core::PCSTR::default (),
        )
    })
};

#[cfg(feature = "global")]
pub fn acquire_global_lock() {
    unsafe { dlassert!(windows::Win32::System::Threading::WaitForSingleObject(*MUTEX, u32::MAX) != 0); }
}

#[cfg(feature = "global")]
pub fn release_global_lock() {
    unsafe { dlassert!(windows::Win32::System::Threading::ReleaseMutex(*MUTEX).0 != 0); }
}
