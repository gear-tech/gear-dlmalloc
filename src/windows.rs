use crate::dlassert;
use crate::dlverbose;
use core::ptr;
use core::ffi::{c_void};
use once_cell::sync::Lazy;

use windows::Win32::System;

pub fn page_size() -> usize { page_size::get() }

pub unsafe fn get_preinstalled_memory() -> (usize, usize) { (0, 0) }

pub unsafe fn alloc(size: usize) -> (*mut u8, usize, u32) {
    // let process_heap = System::Memory::GetProcessHeap();
    // let addr = System::Memory::HeapAlloc(
    //     process_heap,
    //     System::Memory::HEAP_ZERO_MEMORY,
    //     size,
    // );
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
    let result = System::Memory::VirtualFree(
        ptr as *mut c_void,
        0,
        windows::Win32::System::Memory::MEM_RELEASE).0;
    // let process_heap = System::Memory::GetProcessHeap();
    // let result = System::Memory::HeapFree(process_heap, System::Memory::HEAP_NONE, ptr as *mut c_void);

    if result == 0 {
        let cause = windows::Win32::Foundation::GetLastError().0;
        dlverbose!("{}", cause);
    }

    result != 0
}

pub use crate::common::get_free_borders;

#[cfg(feature = "global")]
static LOCK: Lazy<windows::Win32::Foundation::HANDLE> = unsafe {
    Lazy::new(|| {
        windows::Win32::System::Threading::CreateMutexA(
            ptr::null_mut(),
            false,
            windows::core::PCSTR::default())
    })
};


#[cfg(feature = "global")]
pub fn acquire_global_lock() {
    let result = unsafe { windows::Win32::System::Threading::WaitForSingleObject(*LOCK, u32::MAX) };
    dlassert!(result != u32::MAX);
}

#[cfg(feature = "global")]
pub fn release_global_lock() {
    let result = unsafe { windows::Win32::System::Threading::ReleaseMutex(*LOCK).0 };
    dlassert!(result != 0);
}
