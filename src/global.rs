use core::alloc::{GlobalAlloc, Layout};
use core::ops::{Deref, DerefMut};

use Dlmalloc;

/// An instance of a "global allocator" backed by `Dlmalloc`
///
/// This API requires the `global` feature is activated, and this type
/// implements the `GlobalAlloc` trait in the standard library.
pub struct GlobalDlmalloc;

unsafe impl GlobalAlloc for GlobalDlmalloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        <Dlmalloc>::malloc(&mut get(), layout.size(), layout.align())
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        <Dlmalloc>::free(&mut get(), ptr, layout.size(), layout.align())
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        <Dlmalloc>::calloc(&mut get(), layout.size(), layout.align())
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        <Dlmalloc>::realloc(&mut get(), ptr, layout.size(), layout.align(), new_size)
    }
}

/// Returns malloced size
pub unsafe fn get_alloced_mem_size() -> usize {
    <Dlmalloc>::get_alloced_mem_size(&get())
}

static mut DLMALLOC: Dlmalloc = Dlmalloc(::dlmalloc::DLMALLOC_INIT);

struct Instance;

unsafe fn get() -> Instance {
    ::sys::acquire_global_lock();
    Instance
}

impl Deref for Instance {
    type Target = Dlmalloc;
    fn deref(&self) -> &Dlmalloc {
        unsafe { &DLMALLOC }
    }
}

impl DerefMut for Instance {
    fn deref_mut(&mut self) -> &mut Dlmalloc {
        unsafe { &mut DLMALLOC }
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        ::sys::release_global_lock()
    }
}
