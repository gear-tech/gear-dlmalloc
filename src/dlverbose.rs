use core::fmt::Arguments;

pub static DL_CHECKS: bool = cfg!(feature = "debug");
pub static DL_VERBOSE: bool = cfg!(feature = "verbose");
pub static VERBOSE_DEL: &str = "====================================";

#[cfg(unix)]
mod ext {
    pub fn debug(s: &str, _size: usize) {
        libc_print::libc_println!("{}", s);
    }
}

#[cfg(target_arch = "wasm32")]
mod ext {
    mod sys {
        extern "C" {
            pub fn gr_debug(msg_ptr: *const u8, msg_len: u32);
        }
    }
    pub fn debug(s: &str, size: usize) {
        unsafe { sys::gr_debug(s.as_ptr(), size as _) }
    }
}

/// Static out buffer type
type StaticStr = str_buf::StrBuf<200>;
/// Static out buffer - we use it to avoid memory allocations,
/// when something is printed inside allocator code.
static mut OUT_BUFFER: StaticStr = StaticStr::new();

/// Prints string with args.
/// What is the out stream defines in @ext module.
#[inline(never)]
pub unsafe fn dlprint_fn(args: Arguments<'_>) {
    core::fmt::write(&mut OUT_BUFFER, args).unwrap();
    ext::debug(&OUT_BUFFER, OUT_BUFFER.len());
    OUT_BUFFER.set_len(0);
}

/// Prints string with args if @DL_VERBOSE is set.
/// What is the out stream defines in @ext module.
#[macro_export]
macro_rules! dlverbose {
    ($($arg:tt)*) => {
        if crate::dlverbose::DL_VERBOSE {
            crate::dlverbose::dlprint_fn(format_args!($($arg)*))
        }
    }
}

extern crate alloc;
use self::alloc::alloc::handle_alloc_error;

/// Prints current line and throw error using @handle_alloc_error.
#[inline(never)]
pub unsafe fn dlassert_fn(line: u32) {
    dlprint_fn(format_args!("ALLOC ASSERT: {}", line));
    handle_alloc_error(self::alloc::alloc::Layout::new::<u32>());
}

/// Acts like assert using handle_alloc_error if @DL_CHECKS is set, else does nothing.
#[macro_export]
macro_rules! dlassert {
    ($check:expr) => {
        if DL_CHECKS && !($check) {
            unsafe {
                crate::dlverbose::dlassert_fn(line!());
            };
        }
    };
}
