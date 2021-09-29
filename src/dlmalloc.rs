// This is a version of dlmalloc.c ported to Rust. You can find the original
// source at ftp://g.oswego.edu/pub/misc/malloc.c
//
// The original source was written by Doug Lea and released to the public domain

use core::cmp;
use core::mem;
use core::ptr;

extern crate alloc;
use self::alloc::alloc::handle_alloc_error;

use sys;

#[allow(unused)]
#[cfg(target_os = "linux")]
pub mod ext {
    pub fn debug(s: &str, size: usize) {
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

pub fn myprint( s: &str, mut x: usize ) {
    let mut buf = [0u8; 20];

    let mut i = 0;
    for c in s.chars() {
        buf[i] = c as u8;
        i = i + 1;
    }
    buf[i] = ' ' as u8;
    i = i + 1;

    let mut y = x;
    while y > 0 {
        y /= 10;
        i += 1;
    }
    let size = i;
    buf[i] = '\n' as u8;

    while x > 0 {
        i = i - 1;

        buf[i] = match x % 10 {
            0 => '0',
            1 => '1',
            2 => '2',
            3 => '3',
            4 => '4',
            5 => '5',
            6 => '6',
            7 => '7',
            8 => '8',
            9 => '9',
            _ => '?',
        } as u8;

        x = x / 10;
    }

    ext::debug( core::str::from_utf8( &buf).unwrap(), size);
}

macro_rules! dlassert {
    ($check:expr) => {
        if cfg!(debug_assertions) && !($check) {
            myprint( "ALLOC ASSERT: ", line!() as usize);
            handle_alloc_error( self::alloc::alloc::Layout::new::<u32>() );
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
    footprint: usize,
    max_footprint: usize,
    seg: Segment,
    trim_check: usize,
    least_addr: *mut u8,    // only for checks, can be removed
    release_checks: usize,
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
    footprint: 0,
    max_footprint: 0,
    seg: Segment {
        base: 0 as *mut _,
        size: 0,
        next: 0 as *mut _,
        flags: 0,
    },
    trim_check: 0,
    least_addr: 0 as *mut _,
    release_checks: 0,
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
    flags: u32,
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
            ((!0 - (DEFAULT_GRANULARITY + self.top_foot_size() + self.malloc_alignment()) + 1)
                & !self.malloc_alignment())
                - self.chunk_overhead()
                + 1;

        cmp::min((!self.min_chunk_size() + 1) << 2, min_sys_alloc_space)
    }

    fn pad_request(&self, amt: usize) -> usize {
        align_up(amt + self.chunk_overhead(), self.malloc_alignment())
    }

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

    fn top_foot_size(&self) -> usize {
        self.align_offset(unsafe { Chunk::to_mem(ptr::null_mut()) })
            + self.pad_request(mem::size_of::<Segment>())
            + self.min_chunk_size()
    }

    fn mmap_foot_pad(&self) -> usize {
        4 * mem::size_of::<usize>()
    }

    fn align_as_chunk(&self, ptr: *mut u8) -> *mut Chunk {
        unsafe {
            let chunk = Chunk::to_mem(ptr as *mut Chunk);
            ptr.offset(self.align_offset(chunk) as isize) as *mut Chunk
        }
    }

    fn request2size(&self, req: usize) -> usize {
        if req < self.min_request() {
            self.min_chunk_size()
        } else {
            self.pad_request(req)
        }
    }

    unsafe fn overhead_for(&self, p: *mut Chunk) -> usize {
        if Chunk::mmapped(p) {
            self.mmap_chunk_overhead()
        } else {
            self.chunk_overhead()
        }
    }

    pub unsafe fn calloc_must_clear(&self, ptr: *mut u8) -> bool {
        !sys::allocates_zeros() || !Chunk::mmapped(Chunk::from_mem(ptr))
    }

    pub unsafe fn malloc(&mut self, size: usize) -> *mut u8 {
        self.check_malloc_state();

        let nb;
        if size <= self.max_small_request() {
            // In the case we try to find suitable from small chunks

            // Suitable small chunk size
            nb = self.request2size(size);

            let mut idx = self.small_index(nb);
            let smallbits = self.smallmap >> idx;

            // Checks whether idx or idx + 1 has free chunks
            if smallbits & 0b11 != 0 {
                // If idx has no free chunk then use idx + 1
                idx += !smallbits & 1;

                let head_chunk = self.smallbin_at(idx);
                let chunk_for_request = self.unlink_first_small_chunk(head_chunk, idx);

                let smallsize = self.small_index2size(idx);
                (*chunk_for_request).head = smallsize | PINUSE | CINUSE;
                Chunk::set_pinuse_for_next(chunk_for_request, smallsize);

                let chunk_mem = Chunk::to_mem(chunk_for_request);
                self.check_malloced_chunk(chunk_mem, nb);

                return chunk_mem;
            }

            if nb > self.dvsize {
                // If we cannot use dv chunk, then tries to find first suitable chunk
                // from small bins or from tree map in other case.

                if smallbits != 0 {
                    // Has some bigger size small chunks

                    // let leftbits = smallbits << idx; // TODO: & left_bits(1 << idx); ???
                    // let first_one_bit = first_one_bit(smallbits << idx);

                    let bins_idx = (smallbits << idx).trailing_zeros();
                    let head_chunk = self.smallbin_at(bins_idx);
                    let chunk_for_request = self.unlink_first_small_chunk(head_chunk, bins_idx);

                    let smallsize = self.small_index2size(bins_idx);
                    let free_size = smallsize - nb;
                    if mem::size_of::<usize>() != 4 && free_size < self.min_chunk_size() {
                        // Use all size in @chunk_for_request
                        Chunk::set_inuse_and_pinuse(chunk_for_request, smallsize);
                    } else {
                        Chunk::set_size_and_pinuse_of_inuse_chunk(chunk_for_request, nb);
                        let r = Chunk::plus_offset(chunk_for_request, nb);
                        Chunk::set_size_and_pinuse_of_free_chunk(r, free_size);
                        self.replace_dv(r, free_size);
                    }
                    let ret = Chunk::to_mem(chunk_for_request);
                    self.check_malloced_chunk(ret, nb);
                    return ret;
                } else if self.treemap != 0 {
                    let mem = self.tmalloc_small(nb);
                    if !mem.is_null() {
                        self.check_malloced_chunk(mem, nb);
                        self.check_malloc_state();
                        return mem;
                    }
                }
            }
        } else if size < self.max_request() {
            nb = self.pad_request(size);
            if self.treemap != 0 {
                let mem = self.tmalloc_large(nb);
                if !mem.is_null() {
                    self.check_malloced_chunk(mem, nb);
                    self.check_malloc_state();
                    return mem;
                }
            }
        } else {
            // TODO: translate this to unsupported
            return ptr::null_mut();
        }

        // use the `dv` node if we can, splitting it if necessary.
        if nb <= self.dvsize {
            let new_dv_size = self.dvsize - nb;
            let alloced_chunk = self.dv;
            if new_dv_size >= self.min_chunk_size() {
                self.dv = Chunk::plus_offset(alloced_chunk, nb);
                self.dvsize = new_dv_size;

                (*self.dv).head = new_dv_size | PINUSE;
                Chunk::set_next_chunk_prev_size(self.dv, new_dv_size);

                (*alloced_chunk).head = nb | PINUSE | CINUSE;
            } else {
                (*alloced_chunk).head = self.dvsize | PINUSE | CINUSE;
                Chunk::set_pinuse_for_next(alloced_chunk, self.dvsize);

                self.dvsize = 0;
                self.dv = ptr::null_mut();
            }

            let alloced_chunk_mem = Chunk::to_mem(alloced_chunk);
            self.check_malloced_chunk(alloced_chunk_mem, nb);
            self.check_malloc_state();

            return alloced_chunk_mem;
        }

        // Split the top node if we can
        if nb < self.topsize {
            self.topsize -= nb;
            let rsize = self.topsize;
            let p = self.top;
            self.top = Chunk::plus_offset(p, nb);
            let r = self.top;
            (*r).head = rsize | PINUSE;
            Chunk::set_size_and_pinuse_of_inuse_chunk(p, nb);
            self.check_top_chunk(self.top);
            let ret = Chunk::to_mem(p);
            self.check_malloced_chunk(ret, nb);
            self.check_malloc_state();
            return ret;
        }

        // Has no suitable memory chunk, so allocs new pages and use them
        self.sys_alloc(nb)
    }

    unsafe fn sys_alloc(&mut self, size: usize) -> *mut u8 {
        self.check_malloc_state();

        // keep in sync with max_request
        let aligned_size = align_up(
            size + self.top_foot_size() + self.malloc_alignment(),
            DEFAULT_GRANULARITY,
        );

        // libc_print::libc_print!("Try alloc {} pages\n", asize / DEFAULT_GRANULARITY);
        let (alloced_base, alloced_size, flags) = sys::alloc(aligned_size);
        if alloced_base.is_null() {
            return alloced_base;
        }
        // libc_print::libc_print!("Alocced {:?} {:X} {}\n", tbase, tsize, flags);

        self.footprint += alloced_size;
        self.max_footprint = cmp::max(self.max_footprint, self.footprint);

        // Append alloced memory in allocator context
        if self.top.is_null() {
            if self.least_addr.is_null() || alloced_base < self.least_addr {
                // update least addr when new mem is alloced
                self.least_addr = alloced_base;
            }
            self.seg.base = alloced_base;
            self.seg.size = alloced_size;
            self.seg.flags = flags;
            self.release_checks = MAX_RELEASE_CHECK_RATE;

            // TODO: ??? if top has been removed and then we make it one time more
            // then we reinit all bins ??
            self.init_small_bins();

            let new_top_size = alloced_size - self.top_foot_size();
            self.init_top(alloced_base as *mut Chunk, new_top_size);
        } else {
            // Checks whether there is segment which is right before alloced mem
            let mut seg = &mut self.seg as *mut Segment;
            while !seg.is_null() && alloced_base != Segment::top(seg) {
                seg = (*seg).next;
            }
            if !seg.is_null()
                && Segment::holds(seg, self.top as *mut u8)
            {
                // If there is and this segment contains top, then add alloced mem
                // to the segment and increase top size
                (*seg).size += alloced_size;
                self.init_top(self.top, self.topsize + alloced_size);
            } else {
                self.least_addr = cmp::min(alloced_base, self.least_addr);
                // Else checks that there is segment which is right after alloced mem
                let mut seg = &mut self.seg as *mut Segment;
                while !seg.is_null() && (*seg).base != alloced_base.offset(alloced_size as isize) {
                    seg = (*seg).next;
                }
                if !seg.is_null() {
                    // If there is one then add the alloced mem to the seg
                    let oldbase = (*seg).base;
                    (*seg).base = alloced_base;
                    (*seg).size += alloced_size;
                    // and use this seg for user request
                    return self.prepend_alloc(alloced_base, oldbase, size);
                } else {
                    // else add alloced mem as new mem segment and move top there
                    self.add_segment(alloced_base, alloced_size, flags);
                }
            }
        }

        if size < self.topsize {
            self.topsize -= size;
            let new_chunk = self.top;
            self.top = Chunk::plus_offset(new_chunk, size);
            (*self.top).head = self.topsize | PINUSE;
            (*new_chunk).head = size | PINUSE | CINUSE;

            let ret = Chunk::to_mem(new_chunk);
            self.check_top_chunk(self.top);
            self.check_malloced_chunk(ret, size);
            self.check_malloc_state();
            return ret;
        }

        return ptr::null_mut();
    }

    pub unsafe fn realloc(&mut self, oldmem: *mut u8, bytes: usize) -> *mut u8 {
        if bytes >= self.max_request() {
            return ptr::null_mut();
        }
        let nb = self.request2size(bytes);
        let oldp = Chunk::from_mem(oldmem);
        let newp = self.try_realloc_chunk(oldp, nb, true);
        if !newp.is_null() {
            self.check_inuse_chunk(newp);
            return Chunk::to_mem(newp);
        }
        let ptr = self.malloc(bytes);
        if !ptr.is_null() {
            let oc = Chunk::size(oldp) - self.overhead_for(oldp);
            ptr::copy_nonoverlapping(oldmem, ptr, cmp::min(oc, bytes));
            self.free(oldmem);
        }
        return ptr;
    }

    unsafe fn try_realloc_chunk(&mut self, p: *mut Chunk, nb: usize, can_move: bool) -> *mut Chunk {
        let oldsize = Chunk::size(p);
        let next = Chunk::plus_offset(p, oldsize);

        if Chunk::mmapped(p) {
            self.mmap_resize(p, nb, can_move)
        } else if oldsize >= nb {
            let rsize = oldsize - nb;
            if rsize >= self.min_chunk_size() {
                let r = Chunk::plus_offset(p, nb);
                Chunk::set_inuse(p, nb);
                Chunk::set_inuse(r, rsize);
                self.dispose_chunk(r, rsize);
            }
            p
        } else if next == self.top {
            // extend into top
            if oldsize + self.topsize <= nb {
                return ptr::null_mut();
            }
            let newsize = oldsize + self.topsize;
            let newtopsize = newsize - nb;
            let newtop = Chunk::plus_offset(p, nb);
            Chunk::set_inuse(p, nb);
            (*newtop).head = newtopsize | PINUSE;
            self.top = newtop;
            self.topsize = newtopsize;
            p
        } else if next == self.dv {
            // extend into dv
            let dvs = self.dvsize;
            if oldsize + dvs < nb {
                return ptr::null_mut();
            }
            let dsize = oldsize + dvs - nb;
            if dsize >= self.min_chunk_size() {
                let r = Chunk::plus_offset(p, nb);
                let n = Chunk::plus_offset(r, dsize);
                Chunk::set_inuse(p, nb);
                Chunk::set_size_and_pinuse_of_free_chunk(r, dsize);
                Chunk::clear_pinuse(n);
                self.dvsize = dsize;
                self.dv = r;
            } else {
                // exhaust dv
                let newsize = oldsize + dvs;
                Chunk::set_inuse(p, newsize);
                self.dvsize = 0;
                self.dv = ptr::null_mut();
            }
            return p;
        } else if !Chunk::cinuse(next) {
            // extend into the next free chunk
            let nextsize = Chunk::size(next);
            if oldsize + nextsize < nb {
                return ptr::null_mut();
            }
            let rsize = oldsize + nextsize - nb;
            self.unlink_chunk(next, nextsize);
            if rsize < self.min_chunk_size() {
                let newsize = oldsize + nextsize;
                Chunk::set_inuse(p, newsize);
            } else {
                let r = Chunk::plus_offset(p, nb);
                Chunk::set_inuse(p, nb);
                Chunk::set_inuse(r, rsize);
                self.dispose_chunk(r, rsize);
            }
            p
        } else {
            ptr::null_mut()
        }
    }

    unsafe fn mmap_resize(&mut self, oldp: *mut Chunk, nb: usize, can_move: bool) -> *mut Chunk {
        let oldsize = Chunk::size(oldp);
        // Can't shrink mmap regions below a small size
        if self.is_small(nb) {
            return ptr::null_mut();
        }

        // Keep the old chunk if it's big enough but not too big
        if oldsize >= nb + mem::size_of::<usize>() && (oldsize - nb) <= (DEFAULT_GRANULARITY << 1) {
            return oldp;
        }

        let offset = (*oldp).prev_chunk_size;
        let oldmmsize = oldsize + offset + self.mmap_foot_pad();
        let newmmsize =
            self.mmap_align(nb + 6 * mem::size_of::<usize>() + self.malloc_alignment() - 1);
        let ptr = sys::remap(
            (oldp as *mut u8).offset(-(offset as isize)),
            oldmmsize,
            newmmsize,
            can_move,
        );
        if ptr.is_null() {
            return ptr::null_mut();
        }
        let newp = ptr.offset(offset as isize) as *mut Chunk;
        let psize = newmmsize - offset - self.mmap_foot_pad();
        (*newp).head = psize;
        (*Chunk::plus_offset(newp, psize)).head = Chunk::fencepost_head();
        (*Chunk::plus_offset(newp, psize + mem::size_of::<usize>())).head = 0;
        self.least_addr = cmp::min(ptr, self.least_addr);
        self.footprint = self.footprint + newmmsize - oldmmsize;
        self.max_footprint = cmp::max(self.max_footprint, self.footprint);
        self.check_mmapped_chunk(newp);
        return newp;
    }

    fn mmap_align(&self, a: usize) -> usize {
        align_up(a, sys::page_size())
    }

    // Only call this with power-of-two alignment and alignment >
    // `self.malloc_alignment()`
    pub unsafe fn memalign(&mut self, mut alignment: usize, bytes: usize) -> *mut u8 {
        if alignment < self.min_chunk_size() {
            alignment = self.min_chunk_size();
        }
        if bytes >= self.max_request() - alignment {
            return ptr::null_mut();
        }
        let nb = self.request2size(bytes);
        let req = nb + alignment + self.min_chunk_size() - self.chunk_overhead();
        let mem = self.malloc(req);
        if mem.is_null() {
            return mem;
        }
        let mut p = Chunk::from_mem(mem);
        if mem as usize & (alignment - 1) != 0 {
            // Here we find an aligned sopt inside the chunk. Since we need to
            // give back leading space in a chunk of at least `min_chunk_size`,
            // if the first calculation places us at a spot with less than
            // `min_chunk_size` leader we can move to the next aligned spot.
            // we've allocated enough total room so that this is always possible
            let br =
                Chunk::from_mem(((mem as usize + alignment - 1) & (!alignment + 1)) as *mut u8);
            let pos = if (br as usize - p as usize) > self.min_chunk_size() {
                br as *mut u8
            } else {
                (br as *mut u8).offset(alignment as isize)
            };
            let newp = pos as *mut Chunk;
            let leadsize = pos as usize - p as usize;
            let newsize = Chunk::size(p) - leadsize;

            // for mmapped chunks just adjust the offset
            if Chunk::mmapped(p) {
                (*newp).prev_chunk_size = (*p).prev_chunk_size + leadsize;
                (*newp).head = newsize;
            } else {
                // give back the leader, use the rest
                Chunk::set_inuse(newp, newsize);
                Chunk::set_inuse(p, leadsize);
                self.dispose_chunk(p, leadsize);
            }
            p = newp;
        }

        // give back spare room at the end
        if !Chunk::mmapped(p) {
            let size = Chunk::size(p);
            if size > nb + self.min_chunk_size() {
                let remainder_size = size - nb;
                let remainder = Chunk::plus_offset(p, nb);
                Chunk::set_inuse(p, nb);
                Chunk::set_inuse(remainder, remainder_size);
                self.dispose_chunk(remainder, remainder_size);
            }
        }

        let mem = Chunk::to_mem(p);
        dlassert!(Chunk::size(p) >= nb);
        dlassert!(align_up(mem as usize, alignment) == mem as usize);
        self.check_inuse_chunk(p);
        return mem;
    }

    // consolidate and bin a chunk, differs from exported versions of free
    // mainly in that the chunk need not be marked as inuse
    unsafe fn dispose_chunk(&mut self, mut p: *mut Chunk, mut psize: usize) {
        let next = Chunk::plus_offset(p, psize);
        if !Chunk::pinuse(p) {
            let prevsize = (*p).prev_chunk_size;
            if Chunk::mmapped(p) {
                psize += prevsize + self.mmap_foot_pad();
                if sys::free((p as *mut u8).offset(-(prevsize as isize)), psize) {
                    self.footprint -= psize;
                }
                return;
            }
            let prev = Chunk::minus_offset(p, prevsize);
            psize += prevsize;
            p = prev;
            if p != self.dv {
                self.unlink_chunk(p, prevsize);
            } else if (*next).head & INUSE == INUSE {
                self.dvsize = psize;
                Chunk::set_free_with_pinuse(p, psize, next);
                return;
            }
        }

        if !Chunk::cinuse(next) {
            // consolidate forward
            if next == self.top {
                self.topsize += psize;
                let tsize = self.topsize;
                self.top = p;
                (*p).head = tsize | PINUSE;
                if p == self.dv {
                    self.dv = ptr::null_mut();
                    self.dvsize = 0;
                }
                return;
            } else if next == self.dv {
                self.dvsize += psize;
                let dsize = self.dvsize;
                self.dv = p;
                Chunk::set_size_and_pinuse_of_free_chunk(p, dsize);
                return;
            } else {
                let nsize = Chunk::size(next);
                psize += nsize;
                self.unlink_chunk(next, nsize);
                Chunk::set_size_and_pinuse_of_free_chunk(p, psize);
                if p == self.dv {
                    self.dvsize = psize;
                    return;
                }
            }
        } else {
            Chunk::set_free_with_pinuse(p, psize, next);
        }
        self.insert_chunk(p, psize);
    }

    unsafe fn init_top(&mut self, ptr: *mut Chunk, size: usize) {
        let offset = self.align_offset(Chunk::to_mem(ptr));
        let p = Chunk::plus_offset(ptr, offset);
        let size = size - offset;

        self.top = p;
        self.topsize = size;
        (*p).head = size | PINUSE;
        (*Chunk::plus_offset(p, size)).head = self.top_foot_size();
        self.trim_check = DEFAULT_TRIM_THRESHOLD;
    }

    // Init next and prev ptrs to itself, other is garbage
    unsafe fn init_small_bins(&mut self) {
        for i in 0..NSMALLBINS as u32 {
            let bin = self.smallbin_at(i);
            (*bin).next = bin;
            (*bin).prev = bin;
        }
    }

    unsafe fn prepend_alloc(&mut self, newbase: *mut u8, oldbase: *mut u8, alloc_chunk_size: usize) -> *mut u8 {
        let new_chunk = self.align_as_chunk(newbase);
        let old_chunk = self.align_as_chunk(oldbase);
        assert!(new_chunk as *const u8 == newbase);
        assert!(old_chunk as *const u8 == oldbase);

        // size of new segment (in normal cases, when segments is 8 aligned)
        let new_seg_size = old_chunk as usize - new_chunk as usize;

        // addr and size of free chunk
        let free_chunk = Chunk::plus_offset(new_chunk, alloc_chunk_size);
        let mut free_chunk_size = new_seg_size - alloc_chunk_size;

        (*new_chunk).head = alloc_chunk_size | PINUSE | CINUSE;

        dlassert!(old_chunk > free_chunk);
        dlassert!(Chunk::pinuse(old_chunk));
        dlassert!(free_chunk_size >= self.min_chunk_size());

        if old_chunk == self.top {
            // append free chunk to top chunk
            self.topsize += free_chunk_size;
            self.top = free_chunk;
            (*free_chunk).head = self.topsize | PINUSE;

            self.check_top_chunk(free_chunk);
        } else if old_chunk == self.dv {
            // append free chunk to dv chunk
            self.dvsize += free_chunk_size;
            self.dv = free_chunk;
            (*self.dv).head = self.dvsize | PINUSE;
            Chunk::set_next_chunk_prev_size( self.dv, self.dvsize);
        } else {
            let next_chunk;
            if !Chunk::cinuse(old_chunk) {
                // if old_chunk is free then add it to @free_chunk.
                let old_chunk_size = Chunk::size(old_chunk);
                self.unlink_chunk(old_chunk, old_chunk_size);
                next_chunk = Chunk::plus_offset(old_chunk, old_chunk_size);
                free_chunk_size += old_chunk_size;
            } else {
                // Old chunk is not removed, so have to correct its head
                next_chunk = old_chunk;
            }

            (*next_chunk).head &= !PINUSE;
            (*next_chunk).prev_chunk_size = free_chunk_size;
            (*free_chunk).head = free_chunk_size | PINUSE;

            // Chunk::set_free_with_pinuse(free_chunk, free_chunk_size, old_chunk);
            self.insert_chunk(free_chunk, free_chunk_size);

            self.check_free_chunk(free_chunk);
        }

        let new_chunk_data_ptr = Chunk::to_mem(new_chunk);
        self.check_malloced_chunk(new_chunk_data_ptr, alloc_chunk_size);
        self.check_malloc_state();

        return new_chunk_data_ptr;
    }

    // add a segment to hold a new noncontiguous region
    unsafe fn add_segment(&mut self, tbase: *mut u8, tsize: usize, flags: u32) {
        // TODO: what in the world is this function doing

        // Determine locations and sizes of segment, fenceposts, and the old top
        let old_top = self.top as *mut u8;
        let old_top_seg = self.segment_holding(old_top);
        let old_top_seg_end = Segment::top(old_top_seg);

        // 24 for wasm32
        let segment_size = self.pad_request(mem::size_of::<Segment>());

        // 47 for wasm32
        let offset = segment_size + mem::size_of::<usize>() * 4 + self.malloc_alignment() - 1;

        // "@old_top_seg_end - 47" for wasm32
        let rawsp = old_top_seg_end.offset(-(offset as isize));

        // If @old_top_seg_end is 8 aligned (I think always) then == "@old_top_seg_end - 40"
        let asp = align_up( rawsp as usize, self.malloc_alignment()) as *const u8;

        // If "@old_top_seg_end - 40 < @old_top + 16"
        let csp = if asp < old_top.offset(self.min_chunk_size() as isize) {
            old_top
        } else {
            asp
        };

        // In case @csp == @old_top => 1) ss = @old_top + 8
        //                             2) tnext = @old_top + 24 > @old_top_seg_end - 32
        // In case @csp == @asp     => 1) ss = @old_top_seg_end - 32
        //                             2) tnext = @old_top_seg_end - 16
        let sp = csp as *mut Chunk;
        let ss = Chunk::to_mem(sp) as *mut Segment;
        let tnext = Chunk::plus_offset(sp, segment_size);

        let mut p = tnext;
        let mut nfences = 0;

        // reset the top to our new space
        // size = tsize - 40
        let size = tsize - self.top_foot_size();
        self.init_top(tbase as *mut Chunk, size);

        // set up our segment record
        dlassert!(self.is_aligned(ss as usize));
        Chunk::set_size_and_pinuse_of_inuse_chunk(sp, segment_size);
        *ss = self.seg; // push our current record
        self.seg.base = tbase;
        self.seg.size = tsize;
        self.seg.flags = flags;
        self.seg.next = ss;

        // insert trailing fences
        loop {
            let nextp = Chunk::plus_offset(p, mem::size_of::<usize>());
            (*p).head = Chunk::fencepost_head();
            nfences += 1;
            if (&(*nextp).head as *const usize as *mut u8) < old_top_seg_end {
                p = nextp;
            } else {
                break;
            }
        }
        dlassert!(nfences >= 2);

        // insert the rest of the old top into a bin as an ordinary free chunk
        if csp != old_top {
            let q = old_top as *mut Chunk;
            let psize = csp as usize - old_top as usize;
            let tn = Chunk::plus_offset(q, psize);
            Chunk::set_free_with_pinuse(q, psize, tn);
            self.insert_chunk(q, psize);
        }

        self.check_top_chunk(self.top);
        self.check_malloc_state();
    }

    /// Finds segment which contains @ptr: @ptr is in [a, b)
    unsafe fn segment_holding(&self, ptr: *mut u8) -> *mut Segment {
        let mut sp = &self.seg as *const Segment as *mut Segment;
        while !sp.is_null() {
            if (*sp).base <= ptr && ptr < Segment::top(sp) {
                return sp;
            }
            sp = (*sp).next;
        }
        ptr::null_mut()
    }

    unsafe fn tmalloc_small(&mut self, size: usize) -> *mut u8 {
        let leastbit = first_one_bit(self.treemap);
        let i = leastbit.trailing_zeros();
        let mut v = *self.treebin_at(i);
        let mut t = v;
        let mut rsize = Chunk::size(TreeChunk::chunk(t)) - size;

        loop {
            t = TreeChunk::leftmost_child(t);
            if t.is_null() {
                break;
            }
            let trem = Chunk::size(TreeChunk::chunk(t)) - size;
            if trem < rsize {
                rsize = trem;
                v = t;
            }
        }

        let vc = TreeChunk::chunk(v);
        let r = Chunk::plus_offset(vc, size) as *mut TreeChunk;
        dlassert!(Chunk::size(vc) == rsize + size);
        self.unlink_large_chunk(v);
        if rsize < self.min_chunk_size() {
            Chunk::set_inuse_and_pinuse(vc, rsize + size);
        } else {
            let rc = TreeChunk::chunk(r);
            Chunk::set_size_and_pinuse_of_inuse_chunk(vc, size);
            Chunk::set_size_and_pinuse_of_free_chunk(rc, rsize);
            self.replace_dv(rc, rsize);
        }
        Chunk::to_mem(vc)
    }

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

    unsafe fn smallbin_at(&mut self, idx: u32) -> *mut Chunk {
        dlassert!(((idx * 2) as usize) < self.smallbins.len());
        &mut *self.smallbins.get_unchecked_mut((idx as usize) * 2) as *mut *mut Chunk as *mut Chunk
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
        let dvs = self.dvsize;
        dlassert!(self.is_small(dvs));
        if dvs != 0 {
            let dv = self.dv;
            self.insert_small_chunk(dv, dvs);
        }
        self.dvsize = size;
        self.dv = chunk;
    }

    unsafe fn insert_chunk(&mut self, chunk: *mut Chunk, size: usize) {
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
        if self.is_small(size) {
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

    pub unsafe fn free(&mut self, mem: *mut u8) {
        self.check_malloc_state();

        let mut p = Chunk::from_mem(mem);
        let mut psize = Chunk::size(p);
        let next = Chunk::plus_offset(p, psize);
        if !Chunk::pinuse(p) {
            let prevsize = (*p).prev_chunk_size;

            if Chunk::mmapped(p) {
                psize += prevsize + self.mmap_foot_pad();
                // libc_print::libc_print!("Free {:?} {}", (p as *mut u8).offset(-(prevsize as isize)), psize);
                if sys::free((p as *mut u8).offset(-(prevsize as isize)), psize) {
                    self.footprint -= psize;
                }
                return;
            }

            let prev = Chunk::minus_offset(p, prevsize);
            psize += prevsize;
            p = prev;
            if p != self.dv {
                self.unlink_chunk(p, prevsize);
            } else if (*next).head & INUSE == INUSE {
                self.dvsize = psize;
                Chunk::set_free_with_pinuse(p, psize, next);
                return;
            }
        }

        // Consolidate forward if we can
        if !Chunk::cinuse(next) {
            if next == self.top {
                self.topsize += psize;
                let tsize = self.topsize;
                self.top = p;
                (*p).head = tsize | PINUSE;
                if p == self.dv {
                    self.dv = ptr::null_mut();
                    self.dvsize = 0;
                }
                if self.should_trim(tsize) {
                    self.sys_trim(0);
                }
                return;
            } else if next == self.dv {
                self.dvsize += psize;
                let dsize = self.dvsize;
                self.dv = p;
                Chunk::set_size_and_pinuse_of_free_chunk(p, dsize);
                return;
            } else {
                let nsize = Chunk::size(next);
                psize += nsize;
                self.unlink_chunk(next, nsize);
                Chunk::set_size_and_pinuse_of_free_chunk(p, psize);
                if p == self.dv {
                    self.dvsize = psize;
                    return;
                }
            }
        } else {
            Chunk::set_free_with_pinuse(p, psize, next);
        }

        if self.is_small(psize) {
            self.insert_small_chunk(p, psize);
            self.check_free_chunk(p);
        } else {
            self.insert_large_chunk(p as *mut TreeChunk, psize);
            self.check_free_chunk(p);
            self.release_checks -= 1;
            if self.release_checks == 0 {
                self.release_unused_segments();
            }
        }
    }

    fn should_trim(&self, size: usize) -> bool {
        size > self.trim_check
    }

    unsafe fn sys_trim(&mut self, mut pad: usize) -> bool {
        let mut released = 0;
        if pad < self.max_request() && !self.top.is_null() {
            pad += self.top_foot_size();
            if self.topsize > pad {
                let unit = DEFAULT_GRANULARITY;
                let extra = ((self.topsize - pad + unit - 1) / unit - 1) * unit;
                let sp = self.segment_holding(self.top as *mut u8);
                dlassert!(!sp.is_null());

                    if Segment::can_release_part(sp) {
                        if (*sp).size >= extra && !self.has_segment_link(sp) {
                            let newsize = (*sp).size - extra;
                            if sys::free_part((*sp).base, (*sp).size, newsize) {
                                released = extra;
                            }
                        }
                    }

                if released != 0 {
                    (*sp).size -= released;
                    self.footprint -= released;
                    let top = self.top;
                    let topsize = self.topsize - released;
                    self.init_top(top, topsize);
                    self.check_top_chunk(self.top);
                }
            }

            released += self.release_unused_segments();

            if released == 0 && self.topsize > self.trim_check {
                self.trim_check = usize::max_value();
            }
        }

        released != 0
    }

    unsafe fn has_segment_link(&self, ptr: *mut Segment) -> bool {
        let mut sp = &self.seg as *const Segment as *mut Segment;
        while !sp.is_null() {
            if Segment::holds(ptr, sp as *mut u8) {
                return true;
            }
            sp = (*sp).next;
        }
        false
    }

    /// Unmap and unlink any mapped segments that don't contain used chunks
    unsafe fn release_unused_segments(&mut self) -> usize {
        let mut released = 0;
        let mut nsegs = 0;
        let mut pred = &mut self.seg as *mut Segment;
        let mut sp = (*pred).next;
        while !sp.is_null() {
            let base = (*sp).base;
            let size = (*sp).size;
            let next = (*sp).next;
            nsegs += 1;

            if Segment::can_release_part(sp) {
                let p = self.align_as_chunk(base);
                let psize = Chunk::size(p);
                // We can unmap if the first chunk holds the entire segment and
                // isn't pinned.
                let chunk_top = (p as *mut u8).offset(psize as isize);
                let top = base.offset((size - self.top_foot_size()) as isize);
                if !Chunk::inuse(p) && chunk_top >= top {
                    let tp = p as *mut TreeChunk;
                    dlassert!(Segment::holds(sp, sp as *mut u8));
                    if p == self.dv {
                        self.dv = ptr::null_mut();
                        self.dvsize = 0;
                    } else {
                        self.unlink_large_chunk(tp);
                    }
                    if sys::free(base, size) {
                        released += size;
                        self.footprint -= size;
                        // unlink our obsolete record
                        sp = pred;
                        (*sp).next = next;
                    } else {
                        // back out if we can't unmap
                        self.insert_large_chunk(tp, psize);
                    }
                }
            }
            pred = sp;
            sp = next;
        }
        self.release_checks = if nsegs > MAX_RELEASE_CHECK_RATE {
            nsegs
        } else {
            MAX_RELEASE_CHECK_RATE
        };
        return released;
    }

    // Sanity checks

    unsafe fn check_any_chunk(&self, p: *mut Chunk) {
        if !cfg!(zoop) {
            return;
        }
        dlassert!(
            self.is_aligned(Chunk::to_mem(p) as usize) || (*p).head == Chunk::fencepost_head()
        );
        dlassert!(p as *mut u8 >= self.least_addr);
    }

    unsafe fn check_top_chunk(&self, p: *mut Chunk) {
        if !cfg!(zoop) {
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
        dlassert!(sz == (*sp).base as usize + (*sp).size - p as usize - self.top_foot_size());
        dlassert!(Chunk::pinuse(p));
        dlassert!(!Chunk::pinuse(Chunk::plus_offset(p, sz)));
    }

    unsafe fn check_malloced_chunk(&self, mem: *mut u8, s: usize) {
        if !cfg!(zoop) {
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
        if !cfg!(zoop) {
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
        if !cfg!(zoop) {
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

    unsafe fn check_malloc_state(&mut self) {
        if !cfg!(zoop) {
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
            dlassert!(self.dvsize == Chunk::size(self.dv));
            dlassert!(self.dvsize >= self.min_chunk_size());
            let dv = self.dv;
            dlassert!(!self.bin_find(dv));
        }
        if !self.top.is_null() {
            self.check_top_chunk(self.top);
            dlassert!(self.topsize > 0);
            let top = self.top;
            dlassert!(!self.bin_find(top));
        }
        let total = self.traverse_and_check();
        dlassert!(total <= self.footprint);
        dlassert!(self.footprint <= self.max_footprint);
    }

    unsafe fn check_smallbin(&mut self, idx: u32) {
        if !cfg!(zoop) {
            return;
        }
        let b = self.smallbin_at(idx);
        let mut p = (*b).next;
        let empty = self.smallmap & (1 << idx) == 0;
        if p == b {
            dlassert!(empty)
        }
        if !empty {
            while p != b {
                let size = Chunk::size(p);
                self.check_free_chunk(p);
                dlassert!(self.small_index(size) == idx);
                dlassert!((*p).next == b || Chunk::size((*p).next) == Chunk::size(p));
                let q = Chunk::next(p);
                if (*q).head != Chunk::fencepost_head() {
                    self.check_inuse_chunk(q);
                }
                p = (*p).next;
            }
        }
    }

    unsafe fn check_treebin(&mut self, idx: u32) {
        if !cfg!(zoop) {
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
        if !cfg!(zoop) {
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

    unsafe fn traverse_and_check(&self) -> usize {
        0
    }
}

const PINUSE: usize = 1 << 0;
const CINUSE: usize = 1 << 1;
const FLAG4: usize = 1 << 2;
const INUSE: usize = PINUSE | CINUSE;
const FLAG_BITS: usize = PINUSE | CINUSE | FLAG4;

impl Chunk {
    unsafe fn fencepost_head() -> usize {
        INUSE | mem::size_of::<usize>()
    }

    unsafe fn size(me: *mut Chunk) -> usize {
        (*me).head & !FLAG_BITS
    }

    unsafe fn next(me: *mut Chunk) -> *mut Chunk {
        (me as *mut u8).offset(((*me).head & !FLAG_BITS) as isize) as *mut Chunk
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
    unsafe fn can_release_part(seg: *mut Segment) -> bool {
        sys::can_release_part((*seg).flags >> 1)
    }

    unsafe fn sys_flags(seg: *mut Segment) -> u32 {
        (*seg).flags >> 1
    }

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
