//! A Rust port of the `dlmalloc` allocator.
//!
//! The `dlmalloc` allocator is described at
//! http://g.oswego.edu/dl/html/malloc.html and this Rust crate is a straight
//! port of the C code for the allocator into Rust. The implementation is
//! wrapped up in a `Dlmalloc` type and has support for Linux, OSX, and Wasm
//! currently.
//!
//! The primary purpose of this crate is that it serves as the default memory
//! allocator for the `wasm32-unknown-unknown` target in the standard library.
//! Support for other platforms is largely untested and unused, but is used when
//! testing this crate.

#![cfg_attr(target_env = "sgx", feature(llvm_asm))]
#![no_std]
#![deny(missing_docs)]
#![allow(dead_code)]
#![allow(clippy::missing_safety_doc)]

use core::cmp;
use core::ptr;

mod dlmalloc;
mod dlverbose;

#[cfg(all(feature = "global", not(test)))]
mod global;
#[cfg(all(feature = "global", not(test)))]
pub use self::global::GlobalDlmalloc;
#[cfg(all(feature = "global", not(test)))]
pub use global::get_alloced_mem_size;

/// An allocator instance
///
/// Instances of this type are used to allocate blocks of memory. For best
/// results only use one of these. Currently doesn't implement `Drop` to release
/// lingering memory back to the OS. That may happen eventually though!
pub struct Dlmalloc(dlmalloc::Dlmalloc);

/// Constant initializer for `Dlmalloc` structure.
pub const DLMALLOC_INIT: Dlmalloc = Dlmalloc(dlmalloc::DLMALLOC_INIT);

#[cfg(target_arch = "wasm32")]
#[path = "wasm.rs"]
mod sys;

#[cfg(target_os = "macos")]
#[path = "macos.rs"]
mod sys;

#[cfg(target_os = "linux")]
#[path = "linux.rs"]
mod sys;

#[cfg(target_env = "sgx")]
#[path = "sgx.rs"]
mod sys;

#[allow(clippy::new_without_default)]
impl Dlmalloc {
    /// Creates a new instance of an allocator, same as `DLMALLOC_INIT`.
    pub fn new() -> Dlmalloc {
        DLMALLOC_INIT
    }

    /// Allocates `size` bytes with `align` align.
    ///
    /// Returns a null pointer if allocation fails. Returns a valid pointer
    /// otherwise.
    ///
    /// Safety and contracts are largely governed by the `GlobalAlloc::alloc`
    /// method contracts.
    #[inline]
    pub unsafe fn malloc(&mut self, size: usize, align: usize) -> *mut u8 {
        if align <= self.0.malloc_alignment() {
            self.0.malloc(size)
        } else {
            self.0.memalign(align, size)
        }
    }

    /// Same as `malloc`, except if the allocation succeeds it's guaranteed to
    /// point to `size` bytes of zeros.
    #[inline]
    pub unsafe fn calloc(&mut self, size: usize, align: usize) -> *mut u8 {
        let ptr = self.malloc(size, align);
        if !ptr.is_null() {
            ptr::write_bytes(ptr, 0, size);
        }
        ptr
    }

    /// Deallocates a `ptr` with `size` and `align` as the previous request used
    /// to allocate it.
    ///
    /// Safety and contracts are largely governed by the `GlobalAlloc::dealloc`
    /// method contracts.
    #[inline]
    pub unsafe fn free(&mut self, ptr: *mut u8, _size: usize, _align: usize) {
        self.0.free(ptr)
    }

    /// Reallocates `ptr`, a previous allocation with `old_size` and
    /// `old_align`, to have `new_size` and the same alignment as before.
    ///
    /// Returns a null pointer if the memory couldn't be reallocated, but `ptr`
    /// is still valid. Returns a valid pointer and frees `ptr` if the request
    /// is satisfied.
    ///
    /// Safety and contracts are largely governed by the `GlobalAlloc::realloc`
    /// method contracts.
    #[inline]
    pub unsafe fn realloc(
        &mut self,
        ptr: *mut u8,
        old_size: usize,
        old_align: usize,
        new_size: usize,
    ) -> *mut u8 {
        if old_align <= self.0.malloc_alignment() {
            self.0.realloc(ptr, new_size)
        } else {
            let res = self.malloc(new_size, old_align);
            if !res.is_null() {
                let size = cmp::min(old_size, new_size);
                ptr::copy_nonoverlapping(ptr, res, size);
                self.free(ptr, old_size, old_align);
            }
            res
        }
    }

    /// Returns alloced mem size
    pub unsafe fn get_alloced_mem_size(&self) -> usize {
        self.0.get_alloced_mem_size()
    }
}
