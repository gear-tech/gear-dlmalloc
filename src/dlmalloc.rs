// This is a version of dlmalloc.c ported to Rust. You can find the original
// source at ftp://g.oswego.edu/pub/misc/malloc.c
//
// The original source was written by Doug Lea and released to the public domain
#![allow(unused)]

use core::cmp;
use core::mem;
use core::ptr;
use core::ptr::null_mut;

extern crate alloc;
use crate::dlmalloc;

use self::alloc::alloc::handle_alloc_error;

extern crate static_assertions;

use sys;

static DL_CHECKS   : bool = true; // cfg!(debug_assertions)
static DL_VERBOSE  : bool = cfg!(feature = "verbose");
static VERBOSE_DEL : &str = "====================================";

const PTR_SIZE   : usize = mem::size_of::<usize>();
const MALLIGN    : usize = 2 * PTR_SIZE;
const CHUNK_SIZE : usize = mem::size_of::<Chunk>();
static_assertions::const_assert_eq!(2 * MALLIGN, CHUNK_SIZE);
const CHUNK_MEM_OFFSET : usize = 2 * PTR_SIZE;
const SEG_SIZE   : usize = mem::size_of::<Segment>();
static_assertions::const_assert_eq!(3 * PTR_SIZE, SEG_SIZE);
const TREE_NODE_SIZE : usize = mem::size_of::<TreeChunk>();
const MIN_CHUNK_SIZE : usize = mem::size_of::<Chunk>();
// TODO: reduce to 2 * PTR_SIZE + SEG_SIZE + PTR_SIZE  - then one fencepost
const SEG_INFO_SIZE  : usize = MALLIGN + SEG_SIZE + PTR_SIZE;
static_assertions::const_assert_eq!(6 * PTR_SIZE, SEG_INFO_SIZE);
static_assertions::const_assert_eq!(DEFAULT_GRANULARITY % MALLIGN, 0);
static_assertions::const_assert!(MALLIGN > FLAG_BITS);

// Chunk state flags bits
const PINUSE:    usize = 1 << 0;
const CINUSE:    usize = 1 << 1;
const FLAG4:     usize = 1 << 2; // unused, but can be in feature
const INUSE:     usize = PINUSE | CINUSE;
const FLAG_BITS: usize = PINUSE | CINUSE | FLAG4;

#[cfg(unix)]
pub mod ext {
    pub fn debug(s: &str, size: usize) {
        libc_print::libc_println!("{}", s);
    }
}

#[cfg(target_arch = "wasm32")]
pub mod ext {
    mod sys {
        extern "C" {
            pub fn gr_debug(msg_ptr: *const u8, msg_len: u32);
        }
    }

    pub fn debug(s: &str, size: usize) {
        unsafe { sys::gr_debug(s.as_ptr(), size as _) }
    }
}

type StaticStr = str_buf::StrBuf::<200>;
static mut STATIC_BUFFER: StaticStr = StaticStr::new();
static mut MUTEX : spin::Mutex<i32> = spin::Mutex::new(0);
macro_rules! static_print {
    ($($arg:tt)*) => {{
        let lock = MUTEX.lock();
        core::fmt::write( &mut STATIC_BUFFER, format_args!($($arg)*)).unwrap();
        ext::debug( &STATIC_BUFFER, STATIC_BUFFER.len());
        STATIC_BUFFER.set_len(0);
        drop(lock);
    }}
}

macro_rules! dlverbose {
    ($($arg:tt)*) => {
        if DL_VERBOSE {
            static_print!($($arg)*);
        }
    }
}

#[inline(never)]
unsafe fn dlassert_fn( line: u32)
{
    static_print!("ALLOC ASSERT: {}", line);
    handle_alloc_error( self::alloc::alloc::Layout::new::<u32>() );
}

macro_rules! dlassert {
    ($check:expr) => {
        if !($check) { // TODO: add debug_assertions
            unsafe{ dlassert_fn(line!()); };
        }
    };
}

pub struct Dlmalloc {
    smallmap: u32,
    treemap: u32,
    smallbins: [*mut Chunk; (NSMALLBINS + 1) * 2],
    treebins: [*mut TreeChunk; NTREEBINS],
    dvsize: usize,
    topsize: usize,
    dv: *mut Chunk,
    top: *mut Chunk,
    seg: *mut Segment,
    least_addr: *mut u8,    // only for checks
}

unsafe impl Send for Dlmalloc {}

pub const DLMALLOC_INIT: Dlmalloc = Dlmalloc {
    smallmap: 0,
    treemap: 0,
    smallbins: [0 as *mut _; (NSMALLBINS + 1) * 2],
    treebins: [0 as *mut _; NTREEBINS],
    dvsize: 0,
    topsize: 0,
    dv: 0 as *mut _,
    top: 0 as *mut _,
    seg: 0 as *mut _,
    least_addr: 0 as *mut _,
};

// TODO: document this
const NSMALLBINS: usize = 32;
const NTREEBINS: usize = 32;
const SMALLBIN_SHIFT: usize = 3;
const TREEBIN_SHIFT: usize = 8;

// TODO: runtime configurable? documentation?
const DEFAULT_GRANULARITY: usize = 64 * 1024;
const DEFAULT_TRIM_THRESHOLD: usize = 1;
const MAX_RELEASE_CHECK_RATE: usize = 1;

#[repr(C)]
struct Chunk {
    prev_chunk_size: usize,
    head: usize,
    prev: *mut Chunk,
    next: *mut Chunk,
}

#[repr(C)]
struct TreeChunk {
    chunk: Chunk,
    child: [*mut TreeChunk; 2],
    parent: *mut TreeChunk,
    index: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Segment {
    base: *mut u8,
    size: usize,
    next: *mut Segment,
}

impl Segment {
    pub unsafe fn end(&self) -> *mut u8 {
        return self.base.offset( self.size as isize);
    }
    pub unsafe fn info_chunk(&self) -> *mut Chunk {
        return self.end().offset( -1 * SEG_INFO_SIZE as isize ) as *mut Chunk;
    }
    pub unsafe fn base_chunk(&self) -> *mut Chunk {
        return self.base as *mut Chunk;
    }
}

fn align_up(a: usize, alignment: usize) -> usize {
    dlassert!(alignment.is_power_of_two());
    (a + (alignment - 1)) & !(alignment - 1)
}

fn left_bits(x: u32) -> u32 {
    (x << 1) | (!(x << 1)).wrapping_add(1)
}

fn first_one_bit(x: u32) -> u32 {
    x & (!x + 1)
}

fn get_bit(mask: u32, bit_idx: u32) -> u32 {
    dlassert!( bit_idx < 32 );
    return (mask >> bit_idx) & 0b1;
}

fn leftshift_for_tree_index(x: u32) -> u32 {
    let x = x as usize;
    if x == NTREEBINS - 1 {
        0
    } else {
        (mem::size_of::<usize>() * 8 - 1 - ((x >> 1) + TREEBIN_SHIFT - 2)) as u32
    }
}

impl Dlmalloc {
    // TODO: can we get rid of this?
    pub fn malloc_alignment(&self) -> usize {
        mem::size_of::<usize>() * 2
    }

    // TODO: dox
    fn chunk_overhead(&self) -> usize {
        mem::size_of::<usize>()
    }

    fn mmap_chunk_overhead(&self) -> usize {
        2 * mem::size_of::<usize>()
    }

    // TODO: dox
    fn min_large_size(&self) -> usize {
        1 << TREEBIN_SHIFT
    }

    // TODO: dox
    fn max_small_size(&self) -> usize {
        self.min_large_size() - 1
    }

    // TODO: dox
    fn max_small_request(&self) -> usize {
        self.max_small_size() - (self.malloc_alignment() - 1) - self.chunk_overhead()
    }

    // TODO: dox
    fn min_chunk_size(&self) -> usize {
        align_up(mem::size_of::<Chunk>(), self.malloc_alignment())
    }

    // TODO: dox
    fn min_request(&self) -> usize {
        self.min_chunk_size() - self.chunk_overhead() - 1
    }

    // TODO: dox
    fn max_request(&self) -> usize {
        // min_sys_alloc_space: the largest `X` such that
        //   pad_request(X - 1)        -- minus 1, because requests of exactly
        //                                `max_request` will not be honored
        //   + self.top_foot_size()
        //   + self.malloc_alignment()
        //   + DEFAULT_GRANULARITY
        // ==
        //   usize::MAX
        let min_sys_alloc_space =
            ((!0 - (DEFAULT_GRANULARITY + SEG_INFO_SIZE + self.malloc_alignment()) + 1)
                & !self.malloc_alignment())
                - self.chunk_overhead()
                + 1;

        cmp::min((!self.min_chunk_size() + 1) << 2, min_sys_alloc_space)
    }

    fn max_chunk_size(&self) -> usize {
        return self.max_request() + PTR_SIZE;
    }

    fn pad_request(&self, amt: usize) -> usize {
        align_up(amt + self.chunk_overhead(), self.malloc_alignment())
    }

    // TODO: we do not use ptr size here - fix it
    fn small_index(&self, size: usize) -> u32 {
        (size >> SMALLBIN_SHIFT) as u32
    }

    fn small_index2size(&self, idx: u32) -> usize {
        (idx as usize) << SMALLBIN_SHIFT
    }

    fn is_small(&self, s: usize) -> bool {
        s >> SMALLBIN_SHIFT < NSMALLBINS
    }

    fn is_aligned(&self, a: usize) -> bool {
        a & (self.malloc_alignment() - 1) == 0
    }

    fn align_offset(&self, addr: *mut u8) -> usize {
        align_up(addr as usize, self.malloc_alignment()) - (addr as usize)
    }

    fn mmap_foot_pad(&self) -> usize {
        4 * mem::size_of::<usize>()
    }

    fn request2size(&self, req: usize) -> usize {
        if req < self.min_request() {
            self.min_chunk_size()
        } else {
            self.pad_request(req)
        }
    }

    pub unsafe fn calloc_must_clear(&self, ptr: *mut u8) -> bool {
        !sys::allocates_zeros() || !Chunk::mmapped(Chunk::from_mem(ptr))
    }

    unsafe fn malloc_chunk_size(&mut self, chunk_size: usize) -> *mut Chunk {
        if self.is_small(chunk_size) {
            // In the case we try to find suitable from small chunks
            let mut idx = self.small_index(chunk_size);
            let smallbits = self.smallmap >> idx;

            // Checks whether idx or idx + 1 has free chunks
            if smallbits & 0b11 != 0 {
                // If idx has no free chunk then use idx + 1
                idx += !smallbits & 1;

                let head_chunk = self.smallbin_at(idx);
                let chunk = self.unlink_first_small_chunk(head_chunk, idx);

                let smallsize = self.small_index2size(idx);
                (*chunk).head = smallsize | PINUSE | CINUSE;
                (*Chunk::next(chunk)).head |= PINUSE;

                dlverbose!( "MALLOC: use small chunk[{:?}, {:x}]", chunk, smallsize);
                return chunk;
            }

            if chunk_size > self.dvsize {
                // If we cannot use dv chunk, then tries to find first suitable chunk
                // from small bins or from tree map in other case.

                if smallbits != 0 {
                    // Has some bigger size small chunks
                    let bins_idx = (smallbits << idx).trailing_zeros();
                    let head_chunk = self.smallbin_at(bins_idx);
                    let chunk = self.unlink_first_small_chunk(head_chunk, bins_idx);

                    let smallsize = self.small_index2size(bins_idx);
                    let remainder_size = smallsize - chunk_size;

                    // TODO: mem::size_of::<usize>() != 4 why ???
                    if mem::size_of::<usize>() != 4 && remainder_size < self.min_chunk_size() {
                        // Use all size in @chunk_for_request
                        (*chunk).head = smallsize | PINUSE | CINUSE;
                        Chunk::set_pinuse_for_next(chunk, smallsize);
                    } else {
                        // In other case use lower part of @chunk_for_request
                        (*chunk).head = chunk_size | PINUSE | CINUSE;

                        // set remainder as dv
                        let remainder = Chunk::plus_offset(chunk, chunk_size);
                        (*remainder).head = remainder_size | PINUSE;
                        Chunk::set_next_chunk_prev_size( remainder, remainder_size);
                        self.replace_dv(remainder, remainder_size);
                    }

                    dlverbose!( "MALLOC: use small chunk[{:?}, {:x}]", chunk, Chunk::size(chunk));
                    return chunk;
                } else if self.treemap != 0 {
                    let mem = self.tmalloc_small(chunk_size);
                    if !mem.is_null() {
                        let chunk = Chunk::from_mem(mem);
                        dlverbose!( "MALLOC: ret small-tree chunk[{:?}, {:x}]", chunk, Chunk::size(chunk));
                        return chunk;
                    }
                }
            }
        } else if chunk_size < self.max_chunk_size() {
            if self.treemap != 0 {
                let mem = self.tmalloc_large(chunk_size);
                if !mem.is_null() {
                    let chunk = Chunk::from_mem(mem);
                    dlverbose!( "MALLOC: ret big chunk[{:?}, {:x}]", chunk, Chunk::size(chunk));
                    return chunk;
                }
            }
        } else {
            // TODO: translate this to unsupported
            return ptr::null_mut();
        }

        // Use the dv chunk if can
        if chunk_size <= self.dvsize {
            dlverbose!( "MALLOC: use dv chunk[{:?}, {:x}]", self.dv, self.dvsize);
            let chunk = self.crop_chunk(self.dv, self.dv, chunk_size);
            return chunk;
        }

        // Use the top chunk if can
        if chunk_size <= self.topsize {
            dlverbose!( "MALLOC: use top chunk[{:?}, 0x{:x}]", self.top, self.topsize);
            let chunk = self.crop_chunk(self.top, self.top, chunk_size);
            self.check_top_chunk(self.top);
            return chunk;
        }

        return ptr::null_mut();
    }

    unsafe fn malloc_internal(&mut self, size: usize) -> *mut u8 {
        let chunk_size = self.request2size(size);
        let chunk = self.malloc_chunk_size(chunk_size);
        if chunk.is_null() {
            return self.sys_alloc(chunk_size);
        }
        let mem = Chunk::to_mem(chunk);
        self.check_malloced_chunk(mem, chunk_size);
        self.check_malloc_state();
        return mem;
    }

    pub unsafe fn malloc(&mut self, size: usize) -> *mut u8 {
        dlverbose!( "{}", VERBOSE_DEL);
        dlverbose!( "MALLOC: size = 0x{:x}", size);
        self.print_segments();
        self.check_malloc_state();
        let mem = self.malloc_internal(size);
        dlverbose!( "MALLOC: result mem {:?}", mem);
        return mem;
    }

    unsafe fn sys_alloc(&mut self, size: usize) -> *mut u8 {
        dlverbose!("SYS_ALLOC: size = 0x{:x}", size);

        self.check_malloc_state();

        if size >= self.max_chunk_size() {
            return ptr::null_mut();
        }

        // keep in sync with max_request
        let aligned_size = align_up(
            size + SEG_INFO_SIZE + self.malloc_alignment(),
            DEFAULT_GRANULARITY,
        );

        let (alloced_base, alloced_size, flags) = sys::alloc(aligned_size);
        if alloced_base.is_null() {
            return alloced_base;
        }
        dlverbose!("SYS_ALLOC: new mem {:?} 0x{:x}", alloced_base, alloced_size);

        // Append alloced memory in allocator context
        if self.seg.is_null() {
            dlverbose!("SYS_ALLOC: it's newest mem");
            self.update_least_addr(alloced_base);

            self.add_segment(alloced_base, alloced_size, flags);
            self.init_small_bins();

            self.print_segments();
            self.check_top_chunk(self.top);
        } else {
            self.update_least_addr(alloced_base);

            self.add_segment(alloced_base, alloced_size, flags);

            // Checks whether there is segment which is right before alloced mem
            let mut prev_seg = ptr::null_mut();
            let mut seg = self.seg;
            while !seg.is_null() && alloced_base != (*seg).end() {
                prev_seg = seg;
                seg = (*seg).next;
            }
            if !seg.is_null() {
                // If there is then add alloced mem to the @seg
                dlverbose!("SYS_ALLOC: find seg before [{:?}, {:?}, 0x{:x}]", (*seg).base, (*seg).end(), (*seg).size);
                if prev_seg.is_null() {
                    dlassert!( (*self.seg).next == seg );
                    (*self.seg).next = (*seg).next;
                } else {
                    (*prev_seg).next = (*seg).next;
                }
                self.merge_segments(seg.as_mut().unwrap(), self.seg.as_mut().unwrap());
                self.print_segments();
            }

            // Checks whether there is segment which is right after alloced mem
            let mut prev_seg = ptr::null_mut();
            let mut seg = self.seg;
            while !seg.is_null() && (*seg).base != alloced_base.offset(alloced_size as isize) {
                prev_seg = seg;
                seg = (*seg).next;
            }
            if !seg.is_null() {
                dlverbose!("SYS_ALLOC: find seg after [{:?}, {:?}, 0x{:x}]", (*seg).base, (*seg).end(), (*seg).size);
                let next_seg = (*self.seg).next;
                self.merge_segments(self.seg.as_mut().unwrap(), seg.as_mut().unwrap());
                self.seg = next_seg;
                self.print_segments();
            }
        }

        let chunk = self.malloc_chunk_size(size);
        return if chunk.is_null() { ptr::null_mut() } else { Chunk::to_mem(chunk) };
    }

    pub unsafe fn realloc(&mut self, oldmem: *mut u8, req_size: usize) -> *mut u8 {
        self.check_malloc_state();

        if req_size >= self.max_request() {
            return ptr::null_mut();
        }

        let req_chunk_size = self.request2size(req_size);
        let old_chunk = Chunk::from_mem(oldmem);
        let old_chunk_size = Chunk::size(old_chunk);
        let old_mem_size = old_chunk_size - PTR_SIZE;

        let mut chunk = old_chunk;
        let mut chunk_size = old_chunk_size;

        dlverbose!( "{}", VERBOSE_DEL);
        dlverbose!("REALLOC: oldmem={:?} old_mem_size=0x{:x} req_size=0x{:x}", oldmem, old_mem_size, req_size);

        dlassert!( Chunk::cinuse(chunk) );
        dlassert!( chunk != self.top && chunk != self.dv );

        if req_chunk_size <= chunk_size {
            self.crop_chunk(chunk, chunk, req_chunk_size);
            return oldmem;
        } else {
            // Memory in the end of chunk memory can be corrupted in malloc or extend_free_chunk
            // because any chunk store its prev_chunk_size in prev chunk memory end:
            //
            // chunk1 beg  chunk1 end  chunk2 begin
            // |                   \  /
            // [-][-][--------------][-][-][------------]
            //       |                 |
            //       chunk1 memory     chunk1 memory end
            //
            // When chunk1 is free, then chunk2 stores chunk1 size in the chunk1 end memory.
            // When chunk1 is in use then chunk2 do not now chunk1 size and prev_chunk_size cannot be used.
            //
            // When chunk is free then supposes that chunk memory isn't used.
            // So allocator stores there additional information about free chunk.
            // 1) If chunk is small then we put it to smallbins in corresponding list,
            // so we have to store prev and next chunk in list:
            //
            // chunk beg     prev   next
            //          \      |   /
            //           [-][-][-][-]---------------...
            //                 |
            //                 chunk memory begin
            //
            // 2) If chunk is large then chunk is added to tree and we store all TreeChunk info:
            //
            // chunk beg   tree node info begin
            //          \ /
            //           {--[------------}------...
            //              |             \
            //  chunk memory begin         tree node info end

            if self.get_extended_up_chunk_size(chunk) >= req_chunk_size {
                let next_chunk = Chunk::next(chunk);
                dlassert!( !Chunk::cinuse(next_chunk) );

                let chunk_size = Chunk::size(chunk);
                let next_chunk_size = Chunk::size(next_chunk);
                let prev_in_use = if Chunk::pinuse(chunk) { PINUSE } else { 0 };

                dlverbose!("REALLOC: use after chunk[{:?}, 0x{:x}] {}", next_chunk, next_chunk_size, self.is_top_or_dv(next_chunk));

                if next_chunk != self.top && next_chunk != self.dv {
                    self.unlink_chunk(next_chunk, Chunk::size(next_chunk));
                }

                let mut remainder_size = chunk_size + next_chunk_size - req_chunk_size;
                if remainder_size < self.min_chunk_size() {
                    remainder_size = 0;
                }

                let remainder_chunk;
                if remainder_size > 0 {
                    remainder_chunk = Chunk::minus_offset(Chunk::next(next_chunk), remainder_size);
                    (*remainder_chunk).head = remainder_size | PINUSE;
                    (*Chunk::next(remainder_chunk)).prev_chunk_size = remainder_size;
                    dlassert!( !Chunk::pinuse(Chunk::next(remainder_chunk)) );
                } else {
                    remainder_chunk = ptr::null_mut();
                }

                if next_chunk == self.top {
                    self.top = remainder_chunk;
                    self.topsize = remainder_size;
                } else if next_chunk == self.dv {
                    self.dv = remainder_chunk;
                    self.dvsize = remainder_size;
                } else if remainder_size > 0 {
                    self.insert_chunk(remainder_chunk, remainder_size);
                }

                let chunk_size = chunk_size + next_chunk_size - remainder_size;
                (*chunk).head = chunk_size | CINUSE | prev_in_use;
                (*Chunk::next(chunk)).head |= PINUSE;

                self.check_malloc_state();

                return oldmem;
            }

            let new_mem = self.malloc_internal(req_size);
            if new_mem.is_null() {
                return new_mem;
            }

            let new_chunk = Chunk::from_mem(new_mem);
            let new_mem_size = Chunk::size(new_chunk) - PTR_SIZE;
            dlassert!( new_mem_size >= old_mem_size );

            dlverbose!("REALLOC: copy data from [{:?}, 0x{:x?}] to [{:?}, 0x{:x?}]", oldmem, old_mem_size, new_mem, new_mem_size);

            ptr::copy_nonoverlapping(oldmem, new_mem, old_mem_size);

            self.extend_free_chunk(chunk,true);

            self.check_malloc_state();
            return new_mem;
        }
    }

    unsafe fn crop_chunk(&mut self, mut chunk: *mut Chunk, new_chunk_pos: *mut Chunk, new_chunk_size: usize) -> *mut Chunk {
        dlassert!( self.is_aligned(new_chunk_size) );
        dlassert!( self.min_chunk_size() <= new_chunk_size );
        dlassert!( self.is_aligned(new_chunk_pos as usize) );
        dlassert!( new_chunk_pos >= chunk );

        let mut prev_in_use = if Chunk::pinuse( chunk) { PINUSE } else { 0 };

        let mut chunk_size = Chunk::size(chunk);
        dlassert!( Chunk::plus_offset(chunk, chunk_size) >= Chunk::plus_offset(new_chunk_pos, new_chunk_size) );

        dlverbose!("CROP: original chunk [{:?}, {:x?}], to new [{:?}, {:x?}]", chunk, chunk_size, new_chunk_pos, new_chunk_size);

        if new_chunk_pos != chunk {
            let remainder_size = new_chunk_pos as usize - chunk as usize;
            let remainder = chunk;
            dlassert!( remainder_size >= self.min_chunk_size() );

            chunk_size -= remainder_size;

            (*remainder).head = remainder_size | prev_in_use | CINUSE;
            (*new_chunk_pos).head = CINUSE;

            if chunk == self.top {
                self.top = new_chunk_pos;
                self.topsize = chunk_size;
            } else if chunk == self.dv {
                self.dv = new_chunk_pos;
                self.dvsize = chunk_size;
            }

            self.extend_free_chunk( remainder, true);
            dlverbose!("CROP: before rem [{:?}, {:x?}]", remainder, remainder_size);

            chunk = new_chunk_pos;
            prev_in_use = 0;
        }

        dlassert!( new_chunk_pos == chunk );
        dlassert!( chunk_size >= new_chunk_size );

        if chunk_size >= new_chunk_size + self.min_chunk_size() {
            let remainder_size = chunk_size - new_chunk_size;
            let remainder = Chunk::plus_offset(chunk, new_chunk_size);
            dlverbose!("CROP: after rem [{:?}, {:x?}]", remainder, remainder_size);

            if chunk == self.top {
                dlassert!( Chunk::cinuse( Chunk::next(chunk)) );
                self.top = remainder;
                self.topsize = remainder_size;

                (*self.top).head = self.topsize | PINUSE;
                (*self.top).head &= !CINUSE;
                (*Chunk::next(self.top)).head &= !PINUSE;
            } else if chunk == self.dv {
                dlassert!( Chunk::cinuse( Chunk::next(chunk)) );
                self.dv = remainder;
                self.dvsize = remainder_size;

                (*self.dv).head = self.dvsize | PINUSE;
                (*self.dv).head &= !CINUSE;
                (*Chunk::next(self.dv)).head &= !PINUSE;
                Chunk::set_next_chunk_prev_size(self.dv, self.dvsize);
            } else {
                (*remainder).head = remainder_size | PINUSE | CINUSE;
                self.extend_free_chunk(remainder, true);
            }

            chunk_size = new_chunk_size;
        } else {
            (*Chunk::plus_offset(chunk, chunk_size)).head |= PINUSE;
        }

        dlassert!( chunk == new_chunk_pos );
        dlassert!( chunk_size >= new_chunk_size );

        dlverbose!("CROP: cropped chunk [{:?}, {:x?}]", chunk, chunk_size);

        (*chunk).head = chunk_size | prev_in_use | CINUSE;

        if chunk == self.top {
            self.top = ptr::null_mut();
            self.topsize = 0;
        } else if chunk == self.dv {
            self.dv = ptr::null_mut();
            self.dvsize = 0;
        }

        return chunk;
    }

    // Only call this with power-of-two alignment and alignment >
    // `self.malloc_alignment()`
    pub unsafe fn memalign(&mut self, mut alignment: usize, req_size: usize) -> *mut u8 {
        dlverbose!("{}", VERBOSE_DEL);
        dlverbose!("MEMALIGN: align={:x?}, size={:x?}", alignment, req_size);

        self.check_malloc_state();

        if alignment < self.min_chunk_size() {
            alignment = self.min_chunk_size();
        }
        if req_size >= self.max_request() - alignment {
            return ptr::null_mut();
        }
        let req_chunk_size = self.request2size(req_size);
        let size_to_alloc = req_chunk_size + alignment + self.min_chunk_size() - self.chunk_overhead();
        let mem = self.malloc_internal(size_to_alloc);
        if mem.is_null() {
            return mem;
        }

        let mut chunk = Chunk::from_mem(mem);
        let mut chunk_size = Chunk::size(chunk);
        let mut prev_in_use = true;

        dlverbose!("MEMALIGN: chunk[{:?}, {:x?}]", chunk, chunk_size);

        dlassert!( Chunk::pinuse( chunk) && Chunk::cinuse( chunk) );

        let aligned_chunk;
        if mem as usize & (alignment - 1) != 0 {
            // Here we find an aligned sopt inside the chunk. Since we need to
            // give back leading space in a chunk of at least `min_chunk_size`,
            // if the first calculation places us at a spot with less than
            // `min_chunk_size` leader we can move to the next aligned spot.
            // we've allocated enough total room so that this is always possible
            let br =
                Chunk::from_mem(((mem as usize + alignment - 1) & (!alignment + 1)) as *mut u8);
            let pos = if (br as usize - chunk as usize) > self.min_chunk_size() {
                br as *mut u8
            } else {
                (br as *mut u8).offset(alignment as isize)
            };
            aligned_chunk = pos as *mut Chunk;
        } else {
            aligned_chunk = chunk;
        }

        chunk = self.crop_chunk( chunk, aligned_chunk, req_chunk_size);

        let mem_for_request = Chunk::to_mem(chunk);
        dlassert!( Chunk::size(chunk) >= req_chunk_size );
        dlassert!( align_up(mem_for_request as usize, alignment) == mem_for_request as usize );
        self.check_inuse_chunk(chunk);
        self.check_malloc_state();
        return mem_for_request;
    }

    unsafe fn init_top(&mut self, chunk: *mut Chunk, chunk_size: usize) {
        dlassert!( chunk as usize % MALLIGN == 0 );
        dlassert!( Chunk::to_mem(chunk) as usize % MALLIGN == 0 );
        self.top = chunk;
        self.topsize = chunk_size;
        (*self.top).head = chunk_size | PINUSE;
    }

    // Init next and prev ptrs to itself, other is garbage
    unsafe fn init_small_bins(&mut self) {
        for i in 0..NSMALLBINS as u32 {
            let bin = self.smallbin_at(i);
            (*bin).next = bin;
            (*bin).prev = bin;
        }
    }

    unsafe fn merge_segments(&mut self, seg1: &mut Segment, seg2: &mut Segment) {
        dlassert!( seg1.end() == seg2.base );
        dlassert!( seg1.size % DEFAULT_GRANULARITY == 0 );
        dlassert!( seg2.size % DEFAULT_GRANULARITY == 0 );
        dlassert!( seg1.base as usize % MALLIGN == 0 );
        dlassert!( seg2.base as usize % MALLIGN == 0 );

        seg2.size += seg1.size;
        seg2.base = seg1.base;

        let seg1_info_chunk = seg1.info_chunk();
        let seg2_base_chunk = seg2.base_chunk();
        let seg2_info_chunk = seg2.info_chunk();

        Chunk::change_size(seg1_info_chunk, SEG_INFO_SIZE);

        if !Chunk::pinuse(seg1_info_chunk) {
            let prev = Chunk::prev(seg1_info_chunk);
            if prev == self.top && Chunk::next(seg2_base_chunk) != seg2_info_chunk {
                self.top = ptr::null_mut();
                self.topsize = 0;
                self.insert_chunk(prev, Chunk::size(prev));
            }
            // TODO: may be we should find the biggest top free segment to be new @top
        }

        self.extend_free_chunk(seg1_info_chunk, true);
        self.check_top_chunk(self.top);
    }

    unsafe fn set_segment(&mut self, seg_base: *mut u8, seg_size: usize, prev_in_use: usize) -> *mut Segment {
        let seg_end = seg_base.offset(seg_size as isize);
        let seg_chunk = seg_end.offset(-1 * SEG_INFO_SIZE as isize) as *mut Chunk;
        let seg_info = Chunk::plus_offset(seg_chunk, 2 * PTR_SIZE) as *mut Segment;
        let fencepost_chunk = Chunk::plus_offset(seg_chunk, 4 * PTR_SIZE);

        dlassert!( seg_end   as usize % MALLIGN == 0 );
        dlassert!( seg_chunk as usize % MALLIGN == 0 );
        dlassert!( seg_info  as usize % MALLIGN == 0 );
        dlassert!( fencepost_chunk as usize % MALLIGN == 0 );
        dlassert!( fencepost_chunk as *mut u8 == seg_end.offset( -2 * PTR_SIZE as isize) );

        dlverbose!("ALLOC: add seg, info chunk {:?}", seg_chunk);

        // TODO: comments
        (*seg_chunk).head = (4 * PTR_SIZE) | prev_in_use | CINUSE;
        (*seg_info).base  = seg_base;
        (*seg_info).size  = seg_size;
        (*fencepost_chunk).head = Chunk::fencepost_head();

        return seg_info;
    }

    // add a segment to hold a new noncontiguous region
    unsafe fn add_segment(&mut self, tbase: *mut u8, tsize: usize, flags: u32) {
        dlassert!( tbase as usize % self.malloc_alignment() == 0 );
        dlassert!( tsize % DEFAULT_GRANULARITY == 0 );

        let seg = self.set_segment(tbase, tsize, 0);
        (*seg).next = self.seg;

        // insert the rest of the old top into a bin as an ordinary free chunk
        if !self.top.is_null() {
            (*Chunk::next(self.top)).prev_chunk_size = self.topsize;
            self.insert_chunk(self.top, self.topsize);
        }

        // reset the top to our new space
        let size = tsize - SEG_INFO_SIZE;
        self.init_top(tbase as *mut Chunk, size);
        (*Chunk::next(self.top)).prev_chunk_size = self.topsize;

        self.seg = seg;

        dlverbose!("SYS_ALLOC: add seg, top[{:?}, 0x{:x}]", self.top, self.topsize);

        self.check_top_chunk(self.top);
        self.check_malloc_state();
    }

    /// Finds segment which contains @ptr: @ptr is in [a, b)
    unsafe fn segment_holding(&self, ptr: *mut u8) -> *mut Segment {
        let mut sp = self.seg;
        while !sp.is_null() {
            if (*sp).base <= ptr && ptr < Segment::top(sp) {
                return sp;
            }
            sp = (*sp).next;
        }
        ptr::null_mut()
    }

    unsafe fn tmalloc_small(&mut self, size: usize) -> *mut u8 {
        let first_one_idx = self.treemap.trailing_zeros();
        let mut first_tree_chunk = *self.treebin_at(first_one_idx);

        // Iterate left and search the most suitable chunk
        let mut tree_chunk = first_tree_chunk;
        let mut free_size_left = Chunk::size(TreeChunk::chunk(tree_chunk)) - size;
        loop {
            self.check_any_chunk(TreeChunk::chunk(tree_chunk));
            tree_chunk = TreeChunk::leftmost_child(tree_chunk);
            if tree_chunk.is_null() {
                break;
            }
            let diff = Chunk::size(TreeChunk::chunk(tree_chunk)) - size;
            if diff < free_size_left {
                free_size_left = diff;
                first_tree_chunk = tree_chunk;
            }
        }

        let chunk_for_request = TreeChunk::chunk(first_tree_chunk);
        let free_tree_chunk = Chunk::plus_offset(chunk_for_request, size) as *mut TreeChunk;

        dlassert!(Chunk::size(chunk_for_request) == free_size_left + size);

        self.unlink_large_chunk(first_tree_chunk);

        if free_size_left < self.min_chunk_size() {
            // use all mem in chunk
            (*chunk_for_request).head = (free_size_left + size) | PINUSE | CINUSE;
            Chunk::set_pinuse_for_next( chunk_for_request, free_size_left + size);
        } else {
            // use only part and set free part as dv
            let free_chunk = TreeChunk::chunk(free_tree_chunk);
            (*chunk_for_request).head = size | PINUSE | CINUSE;

            (*free_chunk).head = free_size_left | PINUSE;
            Chunk::set_next_chunk_prev_size(free_chunk, free_size_left);

            self.replace_dv(free_chunk, free_size_left);
        }

        return Chunk::to_mem(chunk_for_request);
    }

    // TODO: refactoring
    unsafe fn tmalloc_large(&mut self, size: usize) -> *mut u8 {
        let mut v = ptr::null_mut();
        let mut rsize = !size + 1;
        let idx = self.compute_tree_index(size);
        let mut t = *self.treebin_at(idx);
        if !t.is_null() {
            // Traverse thre tree for this bin looking for a node with size
            // equal to the `size` above.
            let mut sizebits = size << leftshift_for_tree_index(idx);
            // Keep track of the deepest untaken right subtree
            let mut rst = ptr::null_mut();
            loop {
                let csize = Chunk::size(TreeChunk::chunk(t));
                if csize >= size && csize - size < rsize {
                    v = t;
                    rsize = csize - size;
                    if rsize == 0 {
                        break;
                    }
                }
                let rt = (*t).child[1];
                t = (*t).child[(sizebits >> (mem::size_of::<usize>() * 8 - 1)) & 1];
                if !rt.is_null() && rt != t {
                    rst = rt;
                }
                if t.is_null() {
                    // Reset `t` to the least subtree holding sizes greater than
                    // the `size` above, breaking out
                    t = rst;
                    break;
                }
                sizebits <<= 1;
            }
        }

        // Set t to the root of the next non-empty treebin
        if t.is_null() && v.is_null() {
            let leftbits = left_bits(1 << idx) & self.treemap;
            if leftbits != 0 {
                let leastbit = first_one_bit(leftbits);
                let i = leastbit.trailing_zeros();
                t = *self.treebin_at(i);
            }
        }

        // Find the smallest of this tree or subtree
        while !t.is_null() {
            let csize = Chunk::size(TreeChunk::chunk(t));
            if csize >= size && csize - size < rsize {
                rsize = csize - size;
                v = t;
            }
            t = TreeChunk::leftmost_child(t);
        }

        // If dv is a better fit, then return null so malloc will use it
        if v.is_null() || (self.dvsize >= size && !(rsize < self.dvsize - size)) {
            return ptr::null_mut();
        }

        let vc = TreeChunk::chunk(v);
        let r = Chunk::plus_offset(vc, size);
        dlassert!(Chunk::size(vc) == rsize + size);
        self.unlink_large_chunk(v);
        if rsize < self.min_chunk_size() {
            Chunk::set_inuse_and_pinuse(vc, rsize + size);
        } else {
            Chunk::set_size_and_pinuse_of_inuse_chunk(vc, size);
            Chunk::set_size_and_pinuse_of_free_chunk(r, rsize);
            self.insert_chunk(r, rsize);
        }
        Chunk::to_mem(vc)
    }

    /// In smallbins array we store Chunks instead *mut Chunks, why?
    /// Because we need only Chunk::prev and Chunk::next pointers from each chunk
    /// and chunk has size of 4 pointers then we can store prev and next for each chunk
    /// in next chunk begin:
    ///    Second chunk begin    First chunk end
    ///                      \         \||||||||||| <- prev and next for second chunk
    /// smallbins:  [----|----|----|----|----|----|----|----|----|----|--]
    ///            /          ||||||||||| <- prev and next for first chunk
    ///        First chunk begin, two first pointers never used
    ///
    /// So, size must be (bins num * 2) + 2
    unsafe fn smallbin_at(&mut self, idx: u32) -> *mut Chunk {
        let idx = (idx * 2) as usize;
        dlassert!(idx < self.smallbins.len());

        let smallbins_ptr = &self.smallbins as *const *mut Chunk;
        let idx_ptr = smallbins_ptr.offset(idx as isize ) as *mut Chunk;
        return idx_ptr;
    }

    unsafe fn treebin_at(&mut self, idx: u32) -> *mut *mut TreeChunk {
        dlassert!((idx as usize) < self.treebins.len());
        &mut *self.treebins.get_unchecked_mut(idx as usize)
    }

    fn compute_tree_index(&self, size: usize) -> u32 {
        let x = size >> TREEBIN_SHIFT;
        if x == 0 {
            0
        } else if x > 0xffff {
            NTREEBINS as u32 - 1
        } else {
            let k = mem::size_of_val(&x) * 8 - 1 - (x.leading_zeros() as usize);
            ((k << 1) + (size >> (k + TREEBIN_SHIFT - 1) & 1)) as u32
        }
    }

    unsafe fn unlink_first_small_chunk(&mut self, head: *mut Chunk, idx: u32) -> *mut Chunk {
        let chunk_to_unlink = (*head).prev;
        let new_first_chunk = (*chunk_to_unlink).prev;
        dlassert!(chunk_to_unlink != head);
        dlassert!(chunk_to_unlink != new_first_chunk);
        dlassert!(Chunk::size(chunk_to_unlink) == self.small_index2size(idx));
        if head == new_first_chunk {
            self.clear_smallmap(idx);
        } else {
            (*new_first_chunk).next = head;
            (*head).prev = new_first_chunk;
        }
        return chunk_to_unlink;
    }

    unsafe fn replace_dv(&mut self, chunk: *mut Chunk, size: usize) {
        let dv_size = self.dvsize;
        dlassert!(self.is_small(dv_size));
        if dv_size != 0 {
            self.insert_small_chunk(self.dv, dv_size);
        }
        self.dvsize = size;
        self.dv = chunk;
    }

    unsafe fn insert_chunk(&mut self, chunk: *mut Chunk, size: usize) {
        dlverbose!("ALLOC: insert [{:?}, {:?}]", chunk, Chunk::next(chunk));

        dlassert!( size == Chunk::size(chunk) );

        if self.is_small(size) {
            self.insert_small_chunk(chunk, size);
        } else {
            self.insert_large_chunk(chunk as *mut TreeChunk, size);
        }
    }

    unsafe fn insert_small_chunk(&mut self, chunk: *mut Chunk, size: usize) {
        let idx = self.small_index(size);
        let head = self.smallbin_at(idx);
        let mut f = head;
        dlassert!(size >= self.min_chunk_size());
        if !self.smallmap_is_marked(idx) {
            self.mark_smallmap(idx);
        } else {
            f = (*head).prev;
        }

        (*head).prev = chunk;
        (*f).next = chunk;
        (*chunk).prev = f;
        (*chunk).next = head;
    }

    unsafe fn insert_large_chunk(&mut self, chunk: *mut TreeChunk, size: usize) {
        let idx = self.compute_tree_index(size);
        let h = self.treebin_at(idx);
        (*chunk).index = idx;
        (*chunk).child[0] = ptr::null_mut();
        (*chunk).child[1] = ptr::null_mut();
        let chunkc = TreeChunk::chunk(chunk);
        if !self.treemap_is_marked(idx) {
            self.mark_treemap(idx);
            *h = chunk;
            (*chunk).parent = h as *mut TreeChunk; // TODO: dubious?
            (*chunkc).next = chunkc;
            (*chunkc).prev = chunkc;
        } else {
            let mut t = *h;
            let mut k = size << leftshift_for_tree_index(idx);
            loop {
                if Chunk::size(TreeChunk::chunk(t)) != size {
                    let c = &mut (*t).child[(k >> mem::size_of::<usize>() * 8 - 1) & 1];
                    k <<= 1;
                    if !c.is_null() {
                        t = *c;
                    } else {
                        *c = chunk;
                        (*chunk).parent = t;
                        (*chunkc).next = chunkc;
                        (*chunkc).prev = chunkc;
                        break;
                    }
                } else {
                    let tc = TreeChunk::chunk(t);
                    let f = (*tc).prev;
                    (*f).next = chunkc;
                    (*tc).prev = chunkc;
                    (*chunkc).prev = f;
                    (*chunkc).next = tc;
                    (*chunk).parent = ptr::null_mut();
                    break;
                }
            }
        }
    }

    unsafe fn smallmap_is_marked(&self, idx: u32) -> bool {
        self.smallmap & (1 << idx) != 0
    }

    unsafe fn mark_smallmap(&mut self, idx: u32) {
        self.smallmap |= 1 << idx;
    }

    unsafe fn clear_smallmap(&mut self, idx: u32) {
        self.smallmap &= !(1 << idx);
    }

    unsafe fn treemap_is_marked(&self, idx: u32) -> bool {
        self.treemap & (1 << idx) != 0
    }

    unsafe fn mark_treemap(&mut self, idx: u32) {
        self.treemap |= 1 << idx;
    }

    unsafe fn clear_treemap(&mut self, idx: u32) {
        self.treemap &= !(1 << idx);
    }

    unsafe fn unlink_chunk(&mut self, chunk: *mut Chunk, size: usize) {
        dlassert!( Chunk::size(chunk) == size );

        if self.is_small(size) {
            dlverbose!("ALLOC: unlink chunk[{:?}, {:?}]", chunk, Chunk::next(chunk));
            self.unlink_small_chunk(chunk, size)
        } else {
            self.unlink_large_chunk(chunk as *mut TreeChunk);
        }
    }

    unsafe fn unlink_small_chunk(&mut self, chunk: *mut Chunk, size: usize) {
        let prev_chunk = (*chunk).prev;
        let next_chunk = (*chunk).next;
        let idx = self.small_index(size);
        dlassert!(chunk != next_chunk);
        dlassert!(chunk != prev_chunk);
        dlassert!(Chunk::size(chunk) == self.small_index2size(idx));
        if next_chunk == prev_chunk {
            self.clear_smallmap(idx);
        } else {
            (*prev_chunk).next = next_chunk;
            (*next_chunk).prev = prev_chunk;
        }
    }

    unsafe fn unlink_large_chunk(&mut self, chunk: *mut TreeChunk) {
        dlverbose!("ALLOC: unlink chunk[{:?}, {:?}]", chunk, Chunk::next(TreeChunk::chunk(chunk)));
        let parent = (*chunk).parent;
        let mut r;
        if TreeChunk::next(chunk) != chunk {
            let f = TreeChunk::prev(chunk);
            r = TreeChunk::next(chunk);
            (*f).chunk.next = TreeChunk::chunk(r);
            (*r).chunk.prev = TreeChunk::chunk(f);
        } else {
            let mut rp = &mut (*chunk).child[1];
            if rp.is_null() {
                rp = &mut (*chunk).child[0];
            }
            r = *rp;
            if !rp.is_null() {
                loop {
                    let mut cp = &mut (**rp).child[1];
                    if cp.is_null() {
                        cp = &mut (**rp).child[0];
                    }
                    if cp.is_null() {
                        break;
                    }
                    rp = cp;
                }
                r = *rp;
                *rp = ptr::null_mut();
            }
        }

        if parent.is_null() {
            return;
        }

        let h = self.treebin_at((*chunk).index);
        if chunk == *h {
            *h = r;
            if r.is_null() {
                self.clear_treemap((*chunk).index);
            }
        } else {
            if (*parent).child[0] == chunk {
                (*parent).child[0] = r;
            } else {
                (*parent).child[1] = r;
            }
        }

        if !r.is_null() {
            (*r).parent = parent;
            let c0 = (*chunk).child[0];
            if !c0.is_null() {
                (*r).child[0] = c0;
                (*c0).parent = r;
            }
            let c1 = (*chunk).child[1];
            if !c1.is_null() {
                (*r).child[1] = c1;
                (*c1).parent = r;
            }
        }
    }

    unsafe fn get_extended_up_chunk_size(&mut self, chunk: *mut Chunk) -> usize {
        let next_chunk = Chunk::next(chunk);
        if !Chunk::cinuse(next_chunk) {
            return Chunk::size(chunk) + Chunk::size(next_chunk);
        } else {
            return Chunk::size(chunk);
        }
    }

    unsafe fn extend_free_chunk(&mut self, mut chunk: *mut Chunk, can_insert: bool) -> *mut Chunk {
        dlassert!( Chunk::cinuse(chunk) );
        (*chunk).head &= !CINUSE;

        // try join prev chunk
        if !Chunk::pinuse(chunk) {
            let curr_chunk_size = Chunk::size(chunk);
            let prev_chunk = Chunk::prev(chunk);
            let prev_chunk_size = Chunk::size(prev_chunk);
            dlassert!( Chunk::pinuse(prev_chunk) );

            if prev_chunk == self.top {
                self.topsize += Chunk::size(chunk);
            } else if prev_chunk == self.dv {
                self.dvsize += Chunk::size(chunk);
            } else {
                self.unlink_chunk(prev_chunk, prev_chunk_size);
            }

            dlverbose!("extend: add before chunk[{:?}, 0x{:x}] {}", prev_chunk, prev_chunk_size, self.is_top_or_dv(prev_chunk));

            chunk = prev_chunk;
            (*chunk).head = (curr_chunk_size + prev_chunk_size) | PINUSE;
        }

        // try to join next chunk
        let next_chunk = Chunk::next(chunk);
        if !Chunk::cinuse(next_chunk) {
            dlverbose!("extend: add after chunk[{:?}, 0x{:x}] {}", next_chunk, Chunk::size(next_chunk), self.is_top_or_dv(next_chunk));
            if next_chunk == self.top {
                self.top = chunk;
                self.topsize += Chunk::size(chunk);
                if chunk == self.dv { // top eats dv
                    self.dv = ptr::null_mut();
                    self.dvsize = 0;
                }
                (*chunk).head = self.topsize | PINUSE;
                (*Chunk::next(chunk)).prev_chunk_size = self.topsize;
            } else if next_chunk == self.dv {
                if chunk == self.top { // top eats dv
                    self.topsize += Chunk::size(next_chunk);
                    self.dvsize = 0;
                    self.dv = ptr::null_mut();
                    (*chunk).head = self.topsize | PINUSE;
                } else {
                    self.dvsize += Chunk::size(chunk);
                    self.dv = chunk;
                    (*chunk).head = self.dvsize | PINUSE;
                }
                (*Chunk::next(chunk)).prev_chunk_size = self.dvsize;
            } else {
                let next_chunk_size = Chunk::size(next_chunk);
                self.unlink_chunk(next_chunk, next_chunk_size);

                (*chunk).head = (Chunk::size(chunk) + next_chunk_size) | PINUSE;
                (*Chunk::next(chunk)).prev_chunk_size = Chunk::size(chunk);

                if chunk == self.dv {
                    self.dvsize = Chunk::size(chunk);
                } else if can_insert {
                    self.insert_chunk(chunk, Chunk::size(chunk));
                }
            }
        } else {
            (*next_chunk).head &= !PINUSE;
            (*next_chunk).prev_chunk_size = Chunk::size(chunk);
            if can_insert && chunk != self.top && chunk != self.dv {
                self.insert_chunk( chunk, Chunk::size(chunk));
            }
        }

        return chunk;
    }

    pub unsafe fn free(&mut self, mem: *mut u8) {
        dlverbose!( "{}", VERBOSE_DEL);
        dlverbose!( "ALLOC FREE CALL: mem={:?}", mem);

        self.check_malloc_state();

        let chunk = Chunk::from_mem(mem);
        let chunk_size = Chunk::size(chunk);
        dlverbose!( "ALLOC FREE: chunk[{:?}, 0x{:x}]", chunk, chunk_size);

        let chunk = self.extend_free_chunk(chunk, false);
        let chunk_size = Chunk::size(chunk);
        dlverbose!( "ALLOC FREE: extended chunk[{:?}, 0x{:x}] {}", chunk, chunk_size, self.is_top_or_dv(chunk));

        if chunk_size + SEG_INFO_SIZE < DEFAULT_GRANULARITY {
            Chunk::set_next_chunk_prev_size(chunk, chunk_size);
            if chunk != self.top && chunk != self.dv {
                self.insert_chunk(chunk, chunk_size);
            }
            return;
        }

        // find holding segment and prev segment in list
        let mut mem_to_free = chunk as *mut u8;
        let mut mem_to_free_end = mem_to_free.offset(chunk_size as isize);
        let mut prev_seg = ptr::null_mut() as *mut Segment;
        let mut seg = self.seg;
        while !seg.is_null() {
            if Segment::holds(seg, mem_to_free) {
                break;
            }
            prev_seg = seg;
            seg = (*seg).next;
        }
        dlassert!( !seg.is_null() );
        dlassert!( (*seg).size > chunk_size );

        let seg_begin = (*seg).base;
        let seg_end = (*seg).base.offset((*seg).size as isize);
        dlassert!( mem_to_free_end < seg_end );
        dlassert!( self.is_aligned(seg_begin as usize) );
        dlassert!( align_up((*seg).size, DEFAULT_GRANULARITY) == (*seg).size );

        dlverbose!( "ALLOC FREE: holding seg[{:?}, {:?}]", seg_begin, seg_end);
        dlverbose!( "ALLOC FREE: prev seg = {:?}", prev_seg);

        let before_remainder_size : usize;
        if mem_to_free != seg_begin {
            dlassert!( Chunk::pinuse(chunk) );
            dlassert!( mem_to_free as usize - seg_begin as usize >= self.min_chunk_size() );

            // we cannot free chunk.pred_chunk_size mem because it may be used by prev chunk mem
            mem_to_free = mem_to_free.offset(PTR_SIZE as isize);

            // additionally we need space for new segment info
            mem_to_free = mem_to_free.offset(SEG_INFO_SIZE as isize);

            // we restrict not granularity segments
            before_remainder_size = align_up(mem_to_free as usize - seg_begin as usize, DEFAULT_GRANULARITY);
            mem_to_free = seg_begin.offset( before_remainder_size as isize);
        } else {
            before_remainder_size = 0;
        }

        let mut after_remainder_size = seg_end as usize - mem_to_free_end as usize;
        if after_remainder_size > SEG_INFO_SIZE {
            dlassert!( after_remainder_size > SEG_INFO_SIZE + self.min_chunk_size() );

            // TODO: fix it
            // We need that in after remainder the most right chunk is > min_chunk_size
            after_remainder_size += self.min_chunk_size();

            after_remainder_size = align_up( after_remainder_size, DEFAULT_GRANULARITY);
            mem_to_free_end = seg_end.offset(-1 * after_remainder_size as isize);
        } else {
            dlassert!( after_remainder_size == SEG_INFO_SIZE );
            after_remainder_size = 0;
            mem_to_free_end = seg_end;
        }

        if mem_to_free as usize > mem_to_free_end as usize - DEFAULT_GRANULARITY {
            Chunk::set_next_chunk_prev_size(chunk, chunk_size);
            if chunk != self.top && chunk != self.dv {
                self.insert_chunk(chunk, chunk_size);
            }
            return;
        }

        let mem_to_free_size = mem_to_free_end as usize - mem_to_free as usize;
        dlassert!( mem_to_free_size % DEFAULT_GRANULARITY == 0 );

        dlverbose!( "ALLOC FREE: mem to free [{:?}, {:?}]", mem_to_free, mem_to_free_end);

        // We crop chunk with a reserve for before remainder segment info if there will be one
        let mut crop_chunk;
        let mut crop_chunk_size;
        if before_remainder_size != 0 {
            crop_chunk = mem_to_free.offset( -1 * SEG_INFO_SIZE as isize) as *mut Chunk;
            if (crop_chunk as usize - chunk as usize) < self.min_chunk_size() {
                // TODO: fix it
                dlverbose!( "ALLOC FREE: cannot free beacause of left remainder [{:?}, {:?}]", chunk, crop_chunk);
                Chunk::set_next_chunk_prev_size(chunk, chunk_size);
                if chunk != self.top && chunk != self.dv {
                    self.insert_chunk(chunk, chunk_size);
                }
                return;
            }
            crop_chunk_size = mem_to_free_size + SEG_INFO_SIZE;
        } else {
            crop_chunk = mem_to_free as *mut Chunk;
            crop_chunk_size = mem_to_free_size;
        }
        // If there isn't after segment remainder then we delete seg-info chunk,
        // which mustn't be cropped.
        if after_remainder_size == 0 {
            dlassert!( mem_to_free_end == seg_end );
            dlassert!( Chunk::next(chunk) as *mut u8 == seg_end.offset(-1 * SEG_INFO_SIZE as isize) );
            crop_chunk_size -= SEG_INFO_SIZE;
        }

        dlassert!( crop_chunk >= chunk );
        dlassert!( crop_chunk_size <= chunk_size );

        (*chunk).head |= CINUSE;
        let chunk = self.crop_chunk(chunk, crop_chunk, crop_chunk_size);
        dlassert!( Chunk::size(chunk) == crop_chunk_size );

        let next_seg = (*seg).next;
        let before_rem_pinuse : usize;
        if before_remainder_size > 0 {
            before_rem_pinuse = if Chunk::pinuse(chunk) { PINUSE } else { 0 };
        } else {
            before_rem_pinuse = 0;
        }
        let after_rem_pinuse : usize;
        if after_remainder_size > 0 {
            after_rem_pinuse = if Chunk::pinuse( (seg as *mut u8).offset(-2 * PTR_SIZE as isize) as *mut Chunk ) { PINUSE } else { 0 };
        } else {
            after_rem_pinuse = 0;
        }

        let (cond, free_mem, free_mem_size) = sys::free(mem_to_free, mem_to_free_size);
        dlassert!( cond );
        dlassert!( free_mem == mem_to_free );
        dlassert!( mem_to_free_size == free_mem_size );

        if before_remainder_size != 0 {
            let before_seg_info = self.set_segment(seg_begin, before_remainder_size, before_rem_pinuse);

            dlverbose!( "ALLOC FREE: before seg [{:?}, {:?}]", (*before_seg_info).base, Segment::top( before_seg_info));

            if prev_seg.is_null() {
                dlassert!( seg == self.seg );
                self.seg = before_seg_info;
            } else {
                (*prev_seg).next = before_seg_info;
            }
            prev_seg = before_seg_info;
        }

        if after_remainder_size != 0 {
            let after_seg_info = self.set_segment(mem_to_free_end, after_remainder_size, after_rem_pinuse);

            dlverbose!( "ALLOC FREE: after seg [{:?}, {:?}]", (*after_seg_info).base, Segment::top( after_seg_info));

            if prev_seg.is_null() {
                dlassert!( seg == self.seg );
                self.seg = after_seg_info;
            } else {
                (*prev_seg).next = after_seg_info;
            }
            prev_seg = after_seg_info;
        }

        if prev_seg.is_null() {
            dlassert!( seg == self.seg );
            if next_seg.is_null() {
                self.seg = ptr::null_mut();
            } else {
                self.seg = next_seg;
            }
        } else {
            (*prev_seg).next = next_seg;
        }

        self.print_segments();
        self.check_malloc_state();
    }

    unsafe fn has_segment_link(&self, ptr: *mut Segment) -> bool {
        let mut sp = self.seg;
        while !sp.is_null() {
            if Segment::holds(ptr, sp as *mut u8) {
                return true;
            }
            sp = (*sp).next;
        }
        false
    }

    // Dumps
    fn is_top_or_dv(&self, chunk: *mut Chunk) -> &'static str {
        if chunk == self.top {
            return "is top";
        } else if chunk == self.dv {
            return "is dv";
        } else {
            return "is chunk";
        }
    }

    pub unsafe fn get_alloced_mem_size(&self) -> usize {
        let mut size: usize = 0;
        let mut seg = self.seg;
        while !seg.is_null() {
            let mut chunk = (*seg).base as *mut Chunk;
            let last_chunk = Segment::top( seg).offset(-1 * SEG_INFO_SIZE as isize);
            while (chunk as *mut u8) < last_chunk {
                if Chunk::cinuse(chunk) {
                    size += Chunk::size(chunk);
                }
                chunk = Chunk::next(chunk);
            }
            dlassert!( chunk as *mut u8 == last_chunk );
            seg = (*seg).next;
        }
        return size;
    }

    unsafe fn print_segments(&mut self) {
        if !DL_VERBOSE {
            return;
        }
        let mut i = 0;
        let mut seg = self.seg;
        while !seg.is_null() && !(*seg).base.is_null() {
            i += 1;
            dlverbose!("+++++++ SEG{} {:?} [{:?}, {:?}]", i, seg, (*seg).base, Segment::top(seg));
            let mut chunk = (*seg).base as *mut Chunk;
            let last_chunk = Segment::top( seg).offset(-1 * SEG_INFO_SIZE as isize);
            while (chunk as *mut u8) < last_chunk {
                dlverbose!("SEG{} chunk [{:?}, {:?}]{}{} {}",
                    i,
                    chunk,
                    Chunk::next(chunk),
                    if Chunk::cinuse(chunk) { "c" } else { "" },
                    if Chunk::pinuse(chunk) { "p" } else { "" },
                    self.is_top_or_dv(chunk));
                chunk = Chunk::next(chunk);
            }
            dlassert!( chunk as *mut u8 == last_chunk );
            dlassert!( Chunk::size(chunk) == self.pad_request(mem::size_of::<Segment>())
                       || Chunk::size(chunk) == SEG_INFO_SIZE );

            dlverbose!("SEG{} info [{:?}, {:?}]{}{}",
                i,
                chunk,
                Chunk::next(chunk),
                if Chunk::cinuse(chunk) { "c" } else { "" },
                if Chunk::pinuse(chunk) { "p" } else { "" });

            seg = (*seg).next;
        }
    }

    // Sanity checks

    unsafe fn update_least_addr(&mut self, addr: *mut u8) {
        if !DL_CHECKS {
            return;
        }
        if self.least_addr.is_null() || addr < self.least_addr {
            self.least_addr = addr;
        }
    }

    unsafe fn check_any_chunk(&self, p: *mut Chunk) {
        if !DL_CHECKS {
            return;
        }

        // Checks whether @p intersect with some other chunk
        let mut seg = self.seg;
        while !seg.is_null() {
            let mut chunk = (*seg).base as *mut Chunk;
            let last_chunk = Segment::top( seg).offset(-1 * SEG_INFO_SIZE as isize);
            while (chunk as *mut u8) < last_chunk {
                dlassert!( !(chunk > p && chunk < Chunk::next(p)) );
                dlassert!( !(p > chunk && p < Chunk::next(chunk)) );
                chunk = Chunk::next(chunk);
            }
            seg = (*seg).next;
        }
        dlassert!(
            self.is_aligned(Chunk::to_mem(p) as usize) || (*p).head == Chunk::fencepost_head()
        );
        dlassert!(p as *mut u8 >= self.least_addr);
    }

    unsafe fn check_top_chunk(&self, p: *mut Chunk) {
        if !DL_CHECKS {
            return;
        }
        if self.top.is_null() {
            dlassert!( self.topsize == 0 );
            return;
        }
        let sp = self.segment_holding(p as *mut u8);
        let sz = (*p).head & !INUSE;
        dlassert!(!sp.is_null());
        dlassert!(
            self.is_aligned(Chunk::to_mem(p) as usize) || (*p).head == Chunk::fencepost_head()
        );
        dlassert!(p as *mut u8 >= self.least_addr);
        dlassert!(sz == self.topsize);
        dlassert!(sz > 0);
        dlassert!(sz == (*sp).base as usize + (*sp).size - p as usize - SEG_INFO_SIZE);
        dlassert!(Chunk::pinuse(p));
        dlassert!(!Chunk::pinuse(Chunk::plus_offset(p, sz)));
    }

    unsafe fn check_malloced_chunk(&self, mem: *mut u8, s: usize) {
        if !DL_CHECKS {
            return;
        }
        if mem.is_null() {
            return;
        }
        let p = Chunk::from_mem(mem);
        let sz = (*p).head & !INUSE;
        self.check_inuse_chunk(p);
        dlassert!(align_up(sz, self.malloc_alignment()) == sz);
        dlassert!(sz >= self.min_chunk_size());
        dlassert!(sz >= s);
        dlassert!(Chunk::mmapped(p) || sz < (s + self.min_chunk_size()));
    }

    unsafe fn check_inuse_chunk(&self, p: *mut Chunk) {
        self.check_any_chunk(p);
        dlassert!(Chunk::inuse(p));
        dlassert!(Chunk::pinuse(Chunk::next(p)));
        dlassert!(Chunk::mmapped(p) || Chunk::pinuse(p) || Chunk::next(Chunk::prev(p)) == p);
        if Chunk::mmapped(p) {
            self.check_mmapped_chunk(p);
        }
    }

    unsafe fn check_mmapped_chunk(&self, p: *mut Chunk) {
        if !DL_CHECKS {
            return;
        }
        let sz = Chunk::size(p);
        let len = sz + (*p).prev_chunk_size + self.mmap_foot_pad();
        dlassert!(Chunk::mmapped(p));
        dlassert!(
            self.is_aligned(Chunk::to_mem(p) as usize) || (*p).head == Chunk::fencepost_head()
        );
        dlassert!(p as *mut u8 >= self.least_addr);
        dlassert!(!self.is_small(sz));
        dlassert!(align_up(len, sys::page_size()) == len);
        dlassert!((*Chunk::plus_offset(p, sz)).head == Chunk::fencepost_head());
        dlassert!((*Chunk::plus_offset(p, sz + mem::size_of::<usize>())).head == 0);
    }

    unsafe fn check_free_chunk(&self, p: *mut Chunk) {
        if !DL_CHECKS {
            return;
        }
        let sz = Chunk::size(p);
        let next = Chunk::plus_offset(p, sz);
        self.check_any_chunk(p);
        dlassert!(!Chunk::inuse(p));
        dlassert!(!Chunk::pinuse(Chunk::next(p)));
        dlassert!(!Chunk::mmapped(p));
        if p != self.dv && p != self.top {
            if sz >= self.min_chunk_size() {
                dlassert!(align_up(sz, self.malloc_alignment()) == sz);
                dlassert!(self.is_aligned(Chunk::to_mem(p) as usize));
                dlassert!((*next).prev_chunk_size == sz);
                dlassert!(Chunk::pinuse(p));
                dlassert!(next == self.top || Chunk::inuse(next));
                dlassert!((*(*p).next).prev == p);
                dlassert!((*(*p).prev).next == p);
            } else {
                dlassert!(sz == mem::size_of::<usize>());
            }
        }
    }

    #[inline(never)]
    unsafe fn check_malloc_state(&mut self) {
        if !DL_CHECKS {
            return;
        }
        for i in 0..NSMALLBINS {
            self.check_smallbin(i as u32);
        }
        for i in 0..NTREEBINS {
            self.check_treebin(i as u32);
        }
        if self.dvsize != 0 {
            self.check_any_chunk(self.dv);
            dlassert!( self.dvsize == Chunk::size(self.dv) );
            dlassert!( self.dvsize >= self.min_chunk_size() );
            dlassert!( Chunk::pinuse(self.dv) );
            dlassert!( !Chunk::cinuse(self.dv) );
            dlassert!( !self.bin_find(self.dv) );
        }
        if !self.top.is_null() {
            self.check_top_chunk(self.top);
            dlassert!( self.topsize > 0 );
            dlassert!( !self.bin_find(self.top) );
        }

        let mut seg = self.seg;
        while !seg.is_null() && !(*seg).base.is_null() {
            let mut chunk = (*seg).base as *mut Chunk;
            let last_chunk = Segment::top( seg).offset(-1 * SEG_INFO_SIZE as isize);
            while (chunk as *mut u8) < last_chunk {
                if chunk != self.top && chunk != self.dv {
                    dlassert!( self.top < chunk || self.top >= Chunk::next(chunk) );
                    dlassert!( self.dv  < chunk || self.dv  >= Chunk::next(chunk) );
                }
                chunk = Chunk::next(chunk);
            }
            dlassert!( chunk as *mut u8 == last_chunk );
            dlassert!( Chunk::size(chunk) == self.pad_request(mem::size_of::<Segment>())
                       || Chunk::size(chunk) == SEG_INFO_SIZE );

            seg = (*seg).next;
        }
    }

    unsafe fn check_smallbin(&mut self, idx: u32) {
        if !DL_CHECKS {
            return;
        }

        let head_chunk = self.smallbin_at(idx);
        let mut bin_chunk = (*head_chunk).next;

        let idx_bin_is_empty = self.smallmap & (1 << idx) == 0;
        if bin_chunk == head_chunk {
            dlassert!(idx_bin_is_empty);
        } else if !idx_bin_is_empty {
            while bin_chunk != head_chunk {
                self.check_free_chunk(bin_chunk);

                let bin_size = Chunk::size(bin_chunk);
                dlassert!(self.small_index(bin_size) == idx);
                dlassert!((*bin_chunk).next == head_chunk
                           || Chunk::size((*bin_chunk).next) == bin_size);

                let next_mem_chunk = Chunk::next(bin_chunk);
                if (*next_mem_chunk).head != Chunk::fencepost_head() {
                    self.check_inuse_chunk(next_mem_chunk);
                }
                bin_chunk = (*bin_chunk).next;
            }
        }
    }

    unsafe fn check_treebin(&mut self, idx: u32) {
        if !DL_CHECKS {
            return;
        }
        let tb = self.treebin_at(idx);
        let t = *tb;
        let empty = self.treemap & (1 << idx) == 0;
        if t.is_null() {
            dlassert!(empty);
        }
        if !empty {
            self.check_tree(t);
        }
    }

    unsafe fn check_tree(&mut self, t: *mut TreeChunk) {
        if !DL_CHECKS {
            return;
        }
        let tc = TreeChunk::chunk(t);
        let tindex = (*t).index;
        let tsize = Chunk::size(tc);
        let idx = self.compute_tree_index(tsize);
        dlassert!(tindex == idx);
        dlassert!(tsize >= self.min_large_size());
        dlassert!(tsize >= self.min_size_for_tree_index(idx));
        dlassert!(idx == NTREEBINS as u32 - 1 || tsize < self.min_size_for_tree_index(idx + 1));

        let mut u = t;
        let mut head = ptr::null_mut::<TreeChunk>();
        loop {
            let uc = TreeChunk::chunk(u);
            self.check_any_chunk(uc);
            dlassert!((*u).index == tindex);
            dlassert!(Chunk::size(uc) == tsize);
            dlassert!(!Chunk::inuse(uc));
            dlassert!(!Chunk::pinuse(Chunk::next(uc)));
            dlassert!((*(*uc).next).prev == uc);
            dlassert!((*(*uc).prev).next == uc);
            let left = (*u).child[0];
            let right = (*u).child[1];
            if (*u).parent.is_null() {
                dlassert!(left.is_null());
                dlassert!(right.is_null());
            } else {
                dlassert!(head.is_null());
                head = u;
                dlassert!((*u).parent != u);
                dlassert!(
                    (*(*u).parent).child[0] == u
                        || (*(*u).parent).child[1] == u
                        || *((*u).parent as *mut *mut TreeChunk) == u
                );
                if !left.is_null() {
                    dlassert!((*left).parent == u);
                    dlassert!(left != u);
                    self.check_tree(left);
                }
                if !right.is_null() {
                    dlassert!((*right).parent == u);
                    dlassert!(right != u);
                    self.check_tree(right);
                }
                if !left.is_null() && !right.is_null() {
                    dlassert!(
                        Chunk::size(TreeChunk::chunk(left)) < Chunk::size(TreeChunk::chunk(right))
                    );
                }
            }

            u = TreeChunk::prev(u);
            if u == t {
                break;
            }
        }
        dlassert!(!head.is_null());
    }

    fn min_size_for_tree_index(&self, idx: u32) -> usize {
        let idx = idx as usize;
        (1 << ((idx >> 1) + TREEBIN_SHIFT)) | ((idx & 1) << ((idx >> 1) + TREEBIN_SHIFT - 1))
    }

    unsafe fn bin_find(&mut self, chunk: *mut Chunk) -> bool {
        let size = Chunk::size(chunk);
        if self.is_small(size) {
            let sidx = self.small_index(size);
            let b = self.smallbin_at(sidx);
            if !self.smallmap_is_marked(sidx) {
                return false;
            }
            let mut p = b;
            loop {
                if p == chunk {
                    return true;
                }
                p = (*p).prev;
                if p == b {
                    return false;
                }
            }
        } else {
            let tidx = self.compute_tree_index(size);
            if !self.treemap_is_marked(tidx) {
                return false;
            }
            let mut t = *self.treebin_at(tidx);
            let mut sizebits = size << leftshift_for_tree_index(tidx);
            while !t.is_null() && Chunk::size(TreeChunk::chunk(t)) != size {
                t = (*t).child[(sizebits >> (mem::size_of::<usize>() * 8 - 1)) & 1];
                sizebits <<= 1;
            }
            if t.is_null() {
                return false;
            }
            let mut u = t;
            let chunk = chunk as *mut TreeChunk;
            loop {
                if u == chunk {
                    return true;
                }
                u = TreeChunk::prev(u);
                if u == t {
                    return false;
                }
            }
        }
    }
}

impl Chunk {
    unsafe fn fencepost_head() -> usize {
        INUSE | mem::size_of::<usize>()
    }

    unsafe fn size(me: *mut Chunk) -> usize {
        (*me).head & !FLAG_BITS
    }

    unsafe fn next(me: *mut Chunk) -> *mut Chunk {
        (me as *mut u8).offset(Chunk::size(me) as isize) as *mut Chunk
    }

    unsafe fn prev(me: *mut Chunk) -> *mut Chunk {
        (me as *mut u8).offset(-((*me).prev_chunk_size as isize)) as *mut Chunk
    }

    unsafe fn cinuse(me: *mut Chunk) -> bool {
        (*me).head & CINUSE != 0
    }

    unsafe fn pinuse(me: *mut Chunk) -> bool {
        (*me).head & PINUSE != 0
    }

    unsafe fn clear_pinuse(me: *mut Chunk) {
        (*me).head &= !PINUSE;
    }

    unsafe fn inuse(me: *mut Chunk) -> bool {
        (*me).head & INUSE != PINUSE
    }

    unsafe fn mmapped(me: *mut Chunk) -> bool {
        (*me).head & INUSE == 0
    }

    unsafe fn set_inuse(me: *mut Chunk, size: usize) {
        (*me).head = ((*me).head & PINUSE) | size | CINUSE;
        let next = Chunk::plus_offset(me, size);
        (*next).head |= PINUSE;
    }

    unsafe fn set_pinuse_for_next(me: *mut Chunk, size: usize) {
        let next = Chunk::plus_offset(me, size);
        (*next).head |= PINUSE;
    }

    unsafe fn set_inuse_and_pinuse(me: *mut Chunk, size: usize) {
        (*me).head = PINUSE | size | CINUSE;
        let next = Chunk::plus_offset(me, size);
        (*next).head |= PINUSE;
    }

    unsafe fn set_size_and_pinuse_of_inuse_chunk(me: *mut Chunk, size: usize) {
        (*me).head = size | PINUSE | CINUSE;
    }

    unsafe fn set_size_and_pinuse_of_free_chunk(me: *mut Chunk, size: usize) {
        (*me).head = size | PINUSE;
        Chunk::set_next_chunk_prev_size(me, size);
    }

    unsafe fn set_free_with_pinuse(p: *mut Chunk, size: usize, n: *mut Chunk) {
        Chunk::clear_pinuse(n);
        Chunk::set_size_and_pinuse_of_free_chunk(p, size);
    }

    unsafe fn set_next_chunk_prev_size(me: *mut Chunk, size: usize) {
        let next = Chunk::plus_offset(me, size);
        (*next).prev_chunk_size = size;
    }

    unsafe fn change_size(me: *mut Chunk, size: usize) {
        (*me).head = size | ((*me).head & FLAG_BITS);
    }

    unsafe fn plus_offset(me: *mut Chunk, offset: usize) -> *mut Chunk {
        (me as *mut u8).offset(offset as isize) as *mut Chunk
    }

    unsafe fn minus_offset(me: *mut Chunk, offset: usize) -> *mut Chunk {
        (me as *mut u8).offset(-(offset as isize)) as *mut Chunk
    }

    unsafe fn to_mem(me: *mut Chunk) -> *mut u8 {
        (me as *mut u8).offset(2 * (mem::size_of::<usize>() as isize))
    }

    unsafe fn from_mem(mem: *mut u8) -> *mut Chunk {
        mem.offset(-2 * (mem::size_of::<usize>() as isize)) as *mut Chunk
    }
}

impl TreeChunk {
    unsafe fn leftmost_child(me: *mut TreeChunk) -> *mut TreeChunk {
        let left = (*me).child[0];
        if left.is_null() {
            (*me).child[1]
        } else {
            left
        }
    }

    unsafe fn chunk(me: *mut TreeChunk) -> *mut Chunk {
        &mut (*me).chunk
    }

    unsafe fn next(me: *mut TreeChunk) -> *mut TreeChunk {
        (*TreeChunk::chunk(me)).next as *mut TreeChunk
    }

    unsafe fn prev(me: *mut TreeChunk) -> *mut TreeChunk {
        (*TreeChunk::chunk(me)).prev as *mut TreeChunk
    }
}

impl Segment {
    unsafe fn holds(seg: *mut Segment, addr: *mut u8) -> bool {
        (*seg).base <= addr && addr < Segment::top(seg)
    }

    unsafe fn top(seg: *mut Segment) -> *mut u8 {
        (*seg).base.offset((*seg).size as isize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Prime the allocator with some allocations such that there will be free
    // chunks in the treemap
    unsafe fn setup_treemap(a: &mut Dlmalloc) {
        let large_request_size = NSMALLBINS * (1 << SMALLBIN_SHIFT);
        assert!(!a.is_small(large_request_size));
        let large_request1 = a.malloc(large_request_size);
        assert_ne!(large_request1, ptr::null_mut());
        let large_request2 = a.malloc(large_request_size);
        assert_ne!(large_request2, ptr::null_mut());
        a.free(large_request1);
        assert_ne!(a.treemap, 0);
    }

    #[test]
    // Test allocating, with a non-empty treemap, a specific size that used to
    // trigger an integer overflow bug
    fn treemap_alloc_overflow_minimal() {
        let mut a = DLMALLOC_INIT;
        unsafe {
            setup_treemap(&mut a);
            let min_idx31_size = (0xc000 << TREEBIN_SHIFT) - a.chunk_overhead() + 1;
            assert_ne!(a.malloc(min_idx31_size), ptr::null_mut());
        }
    }

    #[test]
    // Test allocating the maximum request size with a non-empty treemap
    fn treemap_alloc_max() {
        let mut a = DLMALLOC_INIT;
        unsafe {
            setup_treemap(&mut a);
            let max_request_size = a.max_request() - 1;
            assert_eq!(a.malloc(max_request_size), ptr::null_mut());
        }
    }
}
