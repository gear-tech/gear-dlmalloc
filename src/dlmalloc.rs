// This is a version of dlmalloc.c ported to Rust. You can find the original
// source at ftp://g.oswego.edu/pub/misc/malloc.c
//
// The original source was written by Doug Lea and released to the public domain

#![allow(unused)]

use core::cmp;
use core::mem;
use core::ptr;
use core::ptr::null_mut;

use crate::dlassert;
use crate::dlverbose;
use dlverbose::{DL_CHECKS, DL_VERBOSE, VERBOSE_DEL};
use sys;

extern crate static_assertions;

/// Pointer size.
const PTR_SIZE: usize = mem::size_of::<usize>();
/// Malloc alignment. TODO: make it one PTR_SIZE ?
const MALIGN: usize = 2 * PTR_SIZE;
/// Chunk struct size
const CHUNK_SIZE: usize = mem::size_of::<Chunk>();
/// Segment struct size
const SEG_SIZE: usize = mem::size_of::<Segment>();
/// Chunk memory offset, see more in [Chunk]
const CHUNK_MEM_OFFSET: usize = 2 * PTR_SIZE;
/// Tree node size
const TREE_NODE_SIZE: usize = mem::size_of::<TreeChunk>();
/// Min size which memory chunk in use may have
const MIN_CHUNK_SIZE: usize = mem::size_of::<Chunk>();
/// Memory size in min chunk
const MIN_MEM_SIZE: usize = MIN_CHUNK_SIZE - PTR_SIZE;
/// Segments info size = size of seg info chunk + border chunk size, see more [Segment]
const SEG_INFO_SIZE: usize = CHUNK_MEM_OFFSET + SEG_SIZE + PTR_SIZE;
/// Default granularity is alignment for segments
const DEFAULT_GRANULARITY: usize = 64 * 1024; // 64 kBytes

static_assertions::const_assert!(2 * MALIGN == CHUNK_SIZE);
static_assertions::const_assert!(3 * PTR_SIZE == SEG_SIZE);
static_assertions::const_assert!(MIN_CHUNK_SIZE % MALIGN == 0);
static_assertions::const_assert!(6 * PTR_SIZE == SEG_INFO_SIZE);
static_assertions::const_assert!(SEG_INFO_SIZE % MALIGN == 0);
static_assertions::const_assert!(DEFAULT_GRANULARITY % MALIGN == 0);

/// Prev chunk is in use bit number
const PINUSE: usize = 1 << 0;
/// Current chunk is in use bit number
const CINUSE: usize = 1 << 1;
/// Unused
const FLAG4: usize = 1 << 2;
/// Use flag bits mask
const INUSE: usize = PINUSE | CINUSE;
/// All flag bits mask
const FLAG_BITS: usize = PINUSE | CINUSE | FLAG4;
/// Mask which is border chunk head
const BORDER_CHUNK_HEAD: usize = FLAG_BITS;

static_assertions::const_assert!(MALIGN > FLAG_BITS);

/// Number of small bins.
const NSMALLBINS: usize = 32;
/// Number of tree bins.
const NTREEBINS: usize = 32;
/// We use it to identify corresponding small bin for chunk, see [Dlmalloc::small_size]
const SMALLBIN_SHIFT: usize = 3;
/// We use it to identify corresponding tree bin for chunk, see [Dlmalloc::compute_tree_index]
const TREEBIN_SHIFT: usize = 8;

/// Dl allocator uses memory non-overlapping intervals for each request - here named Chunks.
///
/// Each chunk can be in two states: in use and free.
/// When chunk is un use, its memory can be read/written by somebody.
/// When chunk is free, we can use it for new malloc requests and make it in use.
/// Chunk info is stored in memory just before memory for request:
/// ````
/// chunk beg       head
///          \     /
///           [-][-][-----------------]
///           /      \
/// prev_chunk_size   chunk memory begin = chunk beg + CHUNK_MEM_OFFSET
/// ````
/// When chunk is free then we suppose that chunk memory isn't used.
/// So allocator stores there additional information about free chunk.
/// If chunk is small then we put it to smallbins in corresponding list,
/// so we have to store ptrs to prev and next chunk in list:
/// ````
///  chunk beg   prev      next
///           \      \    /
///            [-][-][-][-]---------------]
///                  |
///                  chunk memory begin
/// ````
/// If chunk isn't small then we store there tree node information also,
/// see [TreeChunk].
///
/// All algorithms in allocator constructed so that they do not use
/// [Chunk::prev_chunk_size], if prev chunk is in use.
/// So, we can use next chunk begin for current chunk memory:
/// ````
/// chunk1 beg            chunk1 end    chunk2 beg
///          \                      \  /
///           [-][-][----------------][-][-][-------------]
///                 |                    \
///                 chunk1 mem beg        chunk1 mem end
/// ````
/// As you can see, chunks never ovelap, but chunk memory can
/// overlap next chunk.
/// Because of this overlapping chunk memory size == chunk size - [PTR_SIZE].
/// So, in best case chunk info memory overhead for requested by malloc memory
/// is one [PTR_SIZE].
///
/// [Chunk::head] is only one field which must be correct for chunk in both states.
/// In that field we store current chunk size and chunk flag bits.
/// First seweral bits is always zero in chunk size
/// because size of chunk is always aligned to [MALIGN].
/// So, in [Chunk::head] we use left bits for size and right bits for flags, see [FLAG_BITS].
/// 1) First bit is set when prev chunk in memory is in use
///    or if there is no prev chunk (when chunk is first in segment).
/// 2) Second bit is set when current chunk is in use.
/// 3) Third flag currently used only to identify border chunk (see [Segment])
#[repr(C)]
struct Chunk {
    /// Prev in mem chunk size
    prev_chunk_size: usize,
    /// This chunk size and flag bits
    head: usize,
    /// Prev chunk in list
    prev: *mut Chunk,
    /// Next chunk in list
    next: *mut Chunk,
}

/// It's structure to store large chunks in tree.
/// This structure is stored in chunk memory, when large chunk is free:
/// ````
/// chunk beg  chunk info end        chunk end
///          \     |                /
///           [----]-------]-------]
///          /              \
/// tree node info begin     tree node info end
/// ````
/// As you can see [TreeChunk] has common [Chunk] inside and
/// also has pointers to left and right childs.
/// [Chunk] also has pointers [Chunk::next] and [Chunk::prev].
/// Why so many pointers ?
/// Because tree node is a list of same size chunks:
/// ````
///                           ||
///                         chunk1
///                        /      \
///                     chunk2 -- chunk3
///                     //           \\
///                    //           chunk6
///                 chunk4            //
///                  (  )            ...
///                 chunk5
///                 //  \\
///               ...    ...
/// ````
/// Each tree holding treenodes is a tree of unique chunk sizes.  Chunks
/// of the same size are arranged in a circularly-linked list, with only
/// the oldest chunk (the next to be used, in our FIFO ordering)
/// actually in the tree.  (Tree members are distinguished by a non-null
/// parent pointer.)  If a chunk with the same size an an existing node
/// is inserted, it is linked off the existing node using pointers that
/// work in the same way as fd/bk pointers of small chunks.
///
/// Each tree contains a power of 2 sized range of chunk sizes (the
/// smallest is 0x100 <= x < 0x180), which is is divided in half at each
/// tree level, with the chunks in the smaller half of the range (0x100
/// <= x < 0x140 for the top nose) in the left subtree and the larger
/// half (0x140 <= x < 0x180) in the right subtree.  This is, of course,
/// done by inspecting individual bits.
///
/// Using these rules, each node's left subtree contains all smaller
/// sizes than its right subtree.  However, the node at the root of each
/// subtree has no particular ordering relationship to either.  (The
/// dividing line between the subtree sizes is based on trie relation.)
/// If we remove the last chunk of a given size from the interior of the
/// tree, we need to replace it with a leaf node.  The tree ordering
/// rules permit a node to be replaced by any leaf below it.
///
/// The smallest chunk in a tree (a common operation in a best-fit
/// allocator) can be found by walking a path to the leftmost leaf in
/// the tree.  Unlike a usual binary tree, where we follow left child
/// pointers until we reach a null, here we follow the right child
/// pointer any time the left one is null, until we reach a leaf with
/// both child pointers null. The smallest chunk in the tree will be
/// somewhere along that path.
///
/// The worst case number of steps to add, find, or remove a node is
/// bounded by the number of bits differentiating chunks within
/// bins. Under current bin calculations, this ranges from 6 up to 21
/// (for 32 bit sizes) or up to 53 (for 64 bit sizes). The typical case
/// is of course much better.
#[repr(C)]
struct TreeChunk {
    /// This chunk
    chunk: Chunk,
    /// Left and right childs in tree
    child: [*mut TreeChunk; 2],
    /// Parent in tree
    parent: *mut TreeChunk,
    /// Tree index in Dlmalloc::treebins
    index: u32,
}

/// Segment is a big memory interval aligned by [DEFAULT_GRANULARITY]
/// Segment info stored inside segment memory in the end:
/// ````
///                               Border chunk begin    Border chunk end
///                                                 \       /   and seg end
///  [-----------------------------------][----(----][--)--]
///  |                                   /      \        \
///  Segment begin     Seg info chunk beg    Seg info beg \
///                                                       Seg info end
/// ````
/// So, in the end of segment we have two chunks.
/// First is segment info chunk. We store segment info in its memory.
/// This chunk is always in use.
/// Second is border chunk. This chunk has [BORDER_CHUNK_HEAD] as head
/// and it's never used, except checks.
///
/// [SEG_INFO_SIZE] is sum of border and info chunks sizes.
///
/// In allocator context segments are stored in a linked list.
/// So, segment info has field [Segment::next],
/// which points to the next segment in list.
#[repr(C)]
#[derive(Clone, Copy)]
struct Segment {
    /// Segment begin addr
    base: *mut u8,
    /// Segment size
    size: usize,
    /// Next segment in list
    next: *mut Segment,
}

/// Allocator context.
///
/// See comments for items:
/// * Data types - see [Chunk], [Segment], [TreeChunk]
/// * Malloc   - see [Dlmalloc::malloc]
/// * Memalign - see [Dlmalloc::memalign]
/// * Realloc  - see [Dlmalloc::realloc].
/// * Free     - see [Dlmalloc::free].
///
/// Some facts:
/// 1) Two neighbor chunks cannot be free in the same time.
/// If one chunk bacame to be free and there is neighbor free chunk,
/// then this chunks will be merged.
/// 2) Cannot be two neigbor segments.
/// If we allocate new segment in [Dlmalloc::sys_alloc] and there is
/// neighbor segment in allocator context, then we merge this segments.
///
pub struct Dlmalloc {
    /// Mask of available [smallbins]
    smallmap: u32,
    /// Mask of available [treebins]
    treemap: u32,
    /// First chunks in small chunks lists, one list for each small size.
    /// see more [Dlmalloc::smallbin_at]
    smallbins: [*mut Chunk; (NSMALLBINS + 1) * 2],
    /// Pointers to roots of large chunks trees.
    treebins: [*mut TreeChunk; NTREEBINS],
    /// [dv] chunk size
    dvsize: usize,
    /// [top] chunk size
    topsize: usize,
    /// [dv] is special chunk, see more in [Dlmalloc::malloc]
    dv: *mut Chunk,
    /// [top] is special chunk, see more in [Dlmalloc:malloc]
    top: *mut Chunk,
    /// Pointer to the first segment in segments list.
    /// Null if list is empty.
    seg: *mut Segment,
    /// The least allocated addr in self live (for checks only)
    least_addr: *mut u8,
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

/// Returns min number which >= a and which is aligned by `alignment`
fn align_up(a: usize, alignment: usize) -> usize {
    dlassert!(alignment.is_power_of_two());
    (a + (alignment - 1)) & !(alignment - 1)
}

/// TODO: something for binary trees search
fn leftshift_for_tree_index(idx: u32) -> u32 {
    let x = idx as usize;
    if x == NTREEBINS - 1 {
        0
    } else {
        (PTR_SIZE * 8 - 1 - ((x >> 1) + TREEBIN_SHIFT - 2)) as u32
    }
}

impl Dlmalloc {
    /// Returns align for chunk sizes and addresses
    pub fn malloc_alignment(&self) -> usize {
        MALIGN
    }

    /// Returns min size which may have large chunk, see tag_large_chunk
    fn min_large_chunk_size(&self) -> usize {
        1 << TREEBIN_SHIFT
    }

    /// Returns max mem size which can be handled as request
    fn max_request(&self) -> usize {
        // the largest `X` such that
        // req_to_chunk_size(X - 1) + SEG_INFO_SIZE + MALIGN + DEFAULT_GRANULARITY == usize::MAX
        //                       |
        //   -1 because requests of exactly max_request will not be honored
        let min_sys_alloc_space =
            ((!0 - (DEFAULT_GRANULARITY + SEG_INFO_SIZE + MALIGN) + 1) & !MALIGN) - PTR_SIZE + 1;
        cmp::min((!MIN_CHUNK_SIZE + 1) << 2, min_sys_alloc_space)
    }

    /// Returns chunk size for max request
    fn max_chunk_size(&self) -> usize {
        self.max_request() + PTR_SIZE
    }

    /// Returns index for chunk in small bins, see tag_small_bins.
    /// TODO: we do not use ptr size here - fix it
    fn small_index(&self, chunk_size: usize) -> u32 {
        (chunk_size >> SMALLBIN_SHIFT) as u32
    }

    /// Returns size of chunks in small_bin by `idx`, see tag_small_bins
    fn small_index2size(&self, idx: u32) -> usize {
        (idx as usize) << SMALLBIN_SHIFT
    }

    /// Returns whther chunk can be added to smallbins, see tag_small_bins
    fn is_chunk_small(&self, chunk_size: usize) -> bool {
        chunk_size >> SMALLBIN_SHIFT < NSMALLBINS
    }

    /// Returns size of chunk for mem request:
    /// if req size is less than min one, then min chunk is alloced.
    fn mem_to_chunk_size(&self, mem_req_size: usize) -> usize {
        if mem_req_size <= MIN_MEM_SIZE {
            MIN_CHUNK_SIZE
        } else {
            align_up(mem_req_size + PTR_SIZE, MALIGN)
        }
    }

    /// Checks whther there is a list of small chunks for given `idx`
    unsafe fn smallmap_is_marked(&self, idx: u32) -> bool {
        self.smallmap & (1 << idx) != 0
    }

    /// Marks given `idx` in smallmap
    unsafe fn mark_smallmap(&mut self, idx: u32) {
        self.smallmap |= 1 << idx;
    }

    /// Clears given `idx` in smallmap
    unsafe fn clear_smallmap(&mut self, idx: u32) {
        self.smallmap &= !(1 << idx);
    }

    /// Checks whether there is a tree for given `idx`
    unsafe fn treemap_is_marked(&self, idx: u32) -> bool {
        self.treemap & (1 << idx) != 0
    }

    /// Marks given `idx` in treemap
    unsafe fn mark_treemap(&mut self, idx: u32) {
        self.treemap |= 1 << idx;
    }

    /// Clears given `idx` in treemap
    unsafe fn clear_treemap(&mut self, idx: u32) {
        self.treemap &= !(1 << idx);
    }

    /// If there is chunk in tree/smallbins/top/dv which has size >= `chunk_size`,
    /// then returns most suitable chunk, else return null.
    unsafe fn malloc_chunk_by_size(&mut self, chunk_size: usize) -> *mut Chunk {
        if self.is_chunk_small(chunk_size) {
            // In the case we try to find suitable from small chunks
            let mut idx = self.small_index(chunk_size);
            let smallbits = self.smallmap >> idx;

            // Checks whether idx or idx + 1 has free chunks
            if smallbits & 0b11 != 0 {
                // If idx has no free chunk then use idx + 1
                idx += !smallbits & 1;

                let head_chunk = self.smallbin_at(idx);
                let chunk = self.unlink_last_small_chunk(head_chunk, idx);

                let smallsize = self.small_index2size(idx);
                (*chunk).head = smallsize | PINUSE | CINUSE;
                (*Chunk::next(chunk)).head |= PINUSE;

                dlverbose!("MALLOC: use small chunk[{:?}, {:x}]", chunk, smallsize);
                return chunk;
            }

            if chunk_size > self.dvsize {
                // If we cannot use dv chunk, then tries to find first suitable chunk
                // from small bins or from tree map in other case.

                if smallbits != 0 {
                    // Has some bigger size small chunks
                    let bins_idx = (smallbits << idx).trailing_zeros();
                    let head_chunk = self.smallbin_at(bins_idx);
                    let chunk = self.unlink_last_small_chunk(head_chunk, bins_idx);

                    let smallsize = self.small_index2size(bins_idx);
                    let remainder_size = smallsize - chunk_size;

                    // TODO: mem::size_of::<usize>() != 4 why ???
                    if mem::size_of::<usize>() != 4 && remainder_size < MIN_CHUNK_SIZE {
                        // Use all size in @chunk
                        (*chunk).head = smallsize | PINUSE | CINUSE;
                        (*Chunk::next(chunk)).head |= PINUSE;
                    } else {
                        // In other case use lower part of @chunk
                        (*chunk).head = chunk_size | PINUSE | CINUSE;

                        // set remainder as dv
                        let remainder = Chunk::plus_offset(chunk, chunk_size);
                        (*remainder).head = remainder_size | PINUSE;
                        Chunk::set_next_chunk_prev_size(remainder, remainder_size);
                        self.replace_dv(remainder, remainder_size);
                    }

                    dlverbose!(
                        "MALLOC: use small chunk[{:?}, {:x}]",
                        chunk,
                        Chunk::size(chunk)
                    );
                    return chunk;
                } else if self.treemap != 0 {
                    let mem = self.tmalloc_small(chunk_size);
                    if !mem.is_null() {
                        let chunk = Chunk::from_mem(mem);
                        dlverbose!(
                            "MALLOC: ret small-tree chunk[{:?}, {:x}]",
                            chunk,
                            Chunk::size(chunk)
                        );
                        return chunk;
                    }
                }
            }
        } else if chunk_size < self.max_chunk_size() {
            if self.treemap != 0 {
                let mem = self.tmalloc_large(chunk_size);
                if !mem.is_null() {
                    let chunk = Chunk::from_mem(mem);
                    dlverbose!(
                        "MALLOC: ret big chunk[{:?}, {:x}]",
                        chunk,
                        Chunk::size(chunk)
                    );
                    return chunk;
                }
            }
        } else {
            // TODO: translate this to unsupported
            return ptr::null_mut();
        }

        // Use the dv chunk if can
        if chunk_size <= self.dvsize {
            dlverbose!("MALLOC: use dv chunk[{:?}, {:x}]", self.dv, self.dvsize);
            let chunk = self.crop_chunk(self.dv, self.dv, chunk_size);
            return chunk;
        }

        // Use the top chunk if can
        if chunk_size <= self.topsize {
            dlverbose!(
                "MALLOC: use top chunk[{:?}, 0x{:x}]",
                self.top,
                self.topsize
            );
            let chunk = self.crop_chunk(self.top, self.top, chunk_size);
            self.check_top_chunk(self.top);
            return chunk;
        }

        ptr::null_mut()
    }

    /// Malloc func for internal usage, see more in `malloc`
    unsafe fn malloc_internal(&mut self, size: usize) -> *mut u8 {
        let chunk_size = self.mem_to_chunk_size(size);
        let chunk = self.malloc_chunk_by_size(chunk_size);
        if chunk.is_null() {
            return self.sys_alloc(chunk_size);
        }
        let mem = Chunk::to_mem(chunk);
        self.check_malloced_mem(mem, size);
        self.check_malloc_state();
        mem
    }

    /// Allocates memory interval which has size > `size` (bigger because of chunk overhead).
    /// In first memory allocation we have no available memory in allocator context.
    /// So, we request system for memory interval aligned by [DEFAULT_GRANULARITY].
    /// This memory is added as segment in segments list, head is [Dlmalloc::seg].
    /// see more in [Dlmalloc::sys_alloc]
    /// So, after that there is some available memory in allocator context.
    /// Algorithm has four ways how to allocate requested mem using available segments:
    /// 1) Use top chunk: [Dlmalloc::top] and [Dlmalloc::topsize].
    ///    Top chunk has the biggest addr between all other chunks in same segment.
    ///    It is created when new segment is alloced from system.
    ///    All memory from new segment becames to be top.
    ///    Old top (if there is one) in that case becames to be common chunk.
    /// 2) Use dv chunk: [Dlmalloc::dv] and [Dlmalloc::dvsize].
    ///    Dv chunk is created when some bigger size chunk is used to allocate
    ///    small chunk. Then remainder bacame dv (except if it is top, top stay top).
    /// 3) Use tree: [Dlmalloc::treemap] and [Dlmalloc::treebins].
    ///    All big chunks (see tag_big_chunk), which is created during allocator work,
    ///    are saved in tree. In fact it is [NTREEBINS] number of trees, where each tree
    ///    is for corresponding size range. Each tree is sorted by size binary tree.
    /// 4) Use smallbins: [Dlmalloc::smallbins] and [Dlmalloc::smallmap].
    ///    Each smallbin is list of chunks have equal small size:
    ///    zero bin is for less then 8 bytes (never used), first for 8, second for 16, 24, 32 and e.t.c.
    ///    Note: some bins is never used, it depends on [PTR_SIZE] value.
    ///    TODO: fix it, see also [Dlmalloc::small_index].
    /// All these ways have following priority:
    /// 1) if there is small bin with exatly same size as requested, then use it
    /// 2) use dv chunk if can
    /// 3) use most suitable chunk from smallbins if can
    /// 4) use chunks from tree if can
    /// 5) use top chunk if can
    /// 6) if all ways above do not works then alloc new memory from system.
    pub unsafe fn malloc(&mut self, size: usize) -> *mut u8 {
        dlverbose!("{}", VERBOSE_DEL);
        dlverbose!("MALLOC CALL: size = 0x{:x}", size);
        self.print_segments();
        self.check_malloc_state();
        let mem = self.malloc_internal(size);
        dlverbose!("MALLOC: result mem {:?}", mem);
        mem
    }

    /// Requests system to allocate memory.
    /// Requested interval is aligned to [DEFAULT_GRANULARITY].
    /// Adds new memory interval as segment in allocator context,
    /// if there is already some segments, which is neighbor, then
    /// merge old segments with new one.
    unsafe fn sys_alloc(&mut self, size: usize) -> *mut u8 {
        dlverbose!("SYS_ALLOC: size = 0x{:x}", size);

        self.check_malloc_state();

        if size >= self.max_chunk_size() {
            return ptr::null_mut();
        }

        // keep in sync with max_request
        let aligned_size = align_up(size + SEG_INFO_SIZE + MALIGN, DEFAULT_GRANULARITY);

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
                dlverbose!(
                    "SYS_ALLOC: find seg before [{:?}, {:?}, 0x{:x}]",
                    (*seg).base,
                    (*seg).end(),
                    (*seg).size
                );
                if prev_seg.is_null() {
                    dlassert!((*self.seg).next == seg);
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
            while !seg.is_null() && (*seg).base != alloced_base.add(alloced_size) {
                prev_seg = seg;
                seg = (*seg).next;
            }
            if !seg.is_null() {
                dlverbose!(
                    "SYS_ALLOC: find seg after [{:?}, {:?}, 0x{:x}]",
                    (*seg).base,
                    (*seg).end(),
                    (*seg).size
                );
                let next_seg = (*self.seg).next;
                self.merge_segments(self.seg.as_mut().unwrap(), seg.as_mut().unwrap());
                self.seg = next_seg;
                self.print_segments();
            }
        }

        let chunk = self.malloc_chunk_by_size(size);
        if chunk.is_null() {
            ptr::null_mut()
        } else {
            Chunk::to_mem(chunk)
        }
    }

    /// If new requested size is less then old size,
    /// then just crops existen chunk and returns its memory.
    /// In other case checks whether existen chunk can be extended up, so that new
    /// chunk size will be bigger then requested.
    /// If so, then extends it and crops to requested size.
    /// In other case we need other chunk, which have suit size, so malloc is used.
    /// All data from old chunk copied to new.
    pub unsafe fn realloc(&mut self, oldmem: *mut u8, req_size: usize) -> *mut u8 {
        self.check_malloc_state();

        if req_size >= self.max_request() {
            return ptr::null_mut();
        }

        let req_chunk_size = self.mem_to_chunk_size(req_size);
        let old_chunk = Chunk::from_mem(oldmem);
        let old_chunk_size = Chunk::size(old_chunk);
        let old_mem_size = old_chunk_size - PTR_SIZE;

        let mut chunk = old_chunk;
        let mut chunk_size = old_chunk_size;

        dlverbose!("{}", VERBOSE_DEL);
        dlverbose!(
            "REALLOC: oldmem={:?} old_mem_size=0x{:x} req_size=0x{:x}",
            oldmem,
            old_mem_size,
            req_size
        );

        dlassert!(Chunk::cinuse(chunk));
        dlassert!(chunk != self.top && chunk != self.dv);

        if req_chunk_size <= chunk_size {
            self.crop_chunk(chunk, chunk, req_chunk_size);
            oldmem
        } else {
            if self.get_extended_up_chunk_size(chunk) >= req_chunk_size {
                let next_chunk = Chunk::next(chunk);
                dlassert!(!Chunk::cinuse(next_chunk));

                let chunk_size = Chunk::size(chunk);
                let next_chunk_size = Chunk::size(next_chunk);
                let prev_in_use = if Chunk::pinuse(chunk) { PINUSE } else { 0 };

                dlverbose!(
                    "REALLOC: use after chunk[{:?}, 0x{:x}] {}",
                    next_chunk,
                    next_chunk_size,
                    self.is_top_or_dv(next_chunk)
                );

                if next_chunk != self.top && next_chunk != self.dv {
                    self.unlink_chunk(next_chunk, Chunk::size(next_chunk));
                }

                let mut remainder_size = chunk_size + next_chunk_size - req_chunk_size;
                if remainder_size < MIN_CHUNK_SIZE {
                    remainder_size = 0;
                }

                let remainder_chunk;
                if remainder_size > 0 {
                    remainder_chunk = Chunk::minus_offset(Chunk::next(next_chunk), remainder_size);
                    (*remainder_chunk).head = remainder_size | PINUSE;
                    (*Chunk::next(remainder_chunk)).prev_chunk_size = remainder_size;
                    dlassert!(!Chunk::pinuse(Chunk::next(remainder_chunk)));
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
            dlassert!(new_mem_size >= old_mem_size);

            dlverbose!(
                "REALLOC: copy data from [{:?}, 0x{:x?}] to [{:?}, 0x{:x?}]",
                oldmem,
                old_mem_size,
                new_mem,
                new_mem_size
            );

            ptr::copy_nonoverlapping(oldmem, new_mem, old_mem_size);

            self.extend_free_chunk(chunk, true);

            self.check_malloc_state();
            new_mem
        }
    }

    /// Crops `chunk` so that it will have addr `new_chunk_pos`
    /// and size which `new_chunk_size` <= size <= `new_chunk_size` + [MIN_CHUNK_SIZE]
    /// If there is remainders then extend-free them (see [Dlmalloc::extend_free_chunk])
    /// ````
    ///         new_chunk_pos            new_chunk_pos + new_chunk_size
    ///         |                            |
    /// [-------(----------------------------)--------]
    /// |
    /// chunk
    /// ````
    /// Will be transformed to:
    /// ````
    ///  Before remainder        After remainder
    /// /                                    \
    /// [------][----------------------------][-------]
    ///         |
    ///         chunk
    /// ````
    /// Before and after remainders will be extend-free.
    /// Chunk will be set as [CINUSE] and returned.
    unsafe fn crop_chunk(
        &mut self,
        mut chunk: *mut Chunk,
        new_chunk_pos: *mut Chunk,
        new_chunk_size: usize,
    ) -> *mut Chunk {
        dlassert!(new_chunk_size % MALIGN == 0);
        dlassert!(MIN_CHUNK_SIZE <= new_chunk_size);
        dlassert!(new_chunk_pos as usize % MALIGN == 0);
        dlassert!(new_chunk_pos >= chunk);

        let mut prev_in_use = if Chunk::pinuse(chunk) { PINUSE } else { 0 };

        let mut chunk_size = Chunk::size(chunk);
        dlassert!(
            Chunk::plus_offset(chunk, chunk_size)
                >= Chunk::plus_offset(new_chunk_pos, new_chunk_size)
        );

        dlverbose!(
            "CROP: original chunk [{:?}, {:x?}], to new [{:?}, {:x?}]",
            chunk,
            chunk_size,
            new_chunk_pos,
            new_chunk_size
        );

        if new_chunk_pos != chunk {
            let remainder_size = new_chunk_pos as usize - chunk as usize;
            let remainder = chunk;
            dlassert!(remainder_size >= MIN_CHUNK_SIZE);

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

            self.extend_free_chunk(remainder, true);
            dlverbose!("CROP: before rem [{:?}, {:x?}]", remainder, remainder_size);

            chunk = new_chunk_pos;
            prev_in_use = 0;
        }

        dlassert!(new_chunk_pos == chunk);
        dlassert!(chunk_size >= new_chunk_size);

        if chunk_size >= new_chunk_size + MIN_CHUNK_SIZE {
            let remainder_size = chunk_size - new_chunk_size;
            let remainder = Chunk::plus_offset(chunk, new_chunk_size);
            dlverbose!("CROP: after rem [{:?}, {:x?}]", remainder, remainder_size);

            if chunk == self.top {
                dlassert!(Chunk::cinuse(Chunk::next(chunk)));
                self.top = remainder;
                self.topsize = remainder_size;

                (*self.top).head = self.topsize | PINUSE;
                (*self.top).head &= !CINUSE;
                (*Chunk::next(self.top)).head &= !PINUSE;
            } else if chunk == self.dv {
                dlassert!(Chunk::cinuse(Chunk::next(chunk)));
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

        dlassert!(chunk == new_chunk_pos);
        dlassert!(chunk_size >= new_chunk_size);

        dlverbose!("CROP: cropped chunk [{:?}, {:x?}]", chunk, chunk_size);

        (*chunk).head = chunk_size | prev_in_use | CINUSE;

        if chunk == self.top {
            self.top = ptr::null_mut();
            self.topsize = 0;
        } else if chunk == self.dv {
            self.dv = ptr::null_mut();
            self.dvsize = 0;
        }

        chunk
    }

    /// When user want alignment, which is bigger then [MALIGN],
    /// then we just use [Dlmalloc::malloc_internal] for bigger than requested size.
    /// After that we crop malloced chunk, so that returned memory is aligned as need.
    /// Remainder is stored in smallbins or tree.
    pub unsafe fn memalign(&mut self, mut alignment: usize, req_size: usize) -> *mut u8 {
        dlverbose!("{}", VERBOSE_DEL);
        dlverbose!("MEMALIGN: align={:x?}, size={:x?}", alignment, req_size);

        self.check_malloc_state();

        if alignment < MIN_CHUNK_SIZE {
            alignment = MIN_CHUNK_SIZE;
        }
        if req_size >= self.max_request() - alignment {
            return ptr::null_mut();
        }
        let req_chunk_size = self.mem_to_chunk_size(req_size);
        let size_to_alloc = req_chunk_size + alignment + MIN_CHUNK_SIZE - PTR_SIZE;
        let mem = self.malloc_internal(size_to_alloc);
        if mem.is_null() {
            return mem;
        }

        let mut chunk = Chunk::from_mem(mem);
        let mut chunk_size = Chunk::size(chunk);
        let mut prev_in_use = true;

        dlverbose!("MEMALIGN: chunk[{:?}, {:x?}]", chunk, chunk_size);

        dlassert!(Chunk::pinuse(chunk) && Chunk::cinuse(chunk));

        let aligned_chunk;
        if mem as usize & (alignment - 1) != 0 {
            // Here we find an aligned sopt inside the chunk. Since we need to
            // give back leading space in a chunk of at least `min_chunk_size`,
            // if the first calculation places us at a spot with less than
            // `min_chunk_size` leader we can move to the next aligned spot.
            // we've allocated enough total room so that this is always possible
            let br =
                Chunk::from_mem(((mem as usize + alignment - 1) & (!alignment + 1)) as *mut u8);
            let pos = if (br as usize - chunk as usize) > MIN_CHUNK_SIZE {
                br as *mut u8
            } else {
                (br as *mut u8).add(alignment)
            };
            aligned_chunk = pos as *mut Chunk;
        } else {
            aligned_chunk = chunk;
        }

        chunk = self.crop_chunk(chunk, aligned_chunk, req_chunk_size);

        let mem_for_request = Chunk::to_mem(chunk);
        dlassert!(Chunk::size(chunk) >= req_chunk_size);
        dlassert!(align_up(mem_for_request as usize, alignment) == mem_for_request as usize);
        self.check_cinuse_chunk(chunk);
        self.check_malloc_state();
        mem_for_request
    }

    /// Init top chunk
    unsafe fn init_top(&mut self, chunk: *mut Chunk, chunk_size: usize) {
        dlassert!(chunk as usize % MALIGN == 0);
        dlassert!(Chunk::to_mem(chunk) as usize % MALIGN == 0);
        self.top = chunk;
        self.topsize = chunk_size;
        (*self.top).head = chunk_size | PINUSE;
    }

    /// Init next and prev ptrs to itself, other is garbage
    unsafe fn init_small_bins(&mut self) {
        for i in 0..NSMALLBINS as u32 {
            let bin = self.smallbin_at(i);
            (*bin).next = bin;
            (*bin).prev = bin;
        }
    }

    /// Merge two neighbor segments. `seg1` will be deleted, `seg2` is result segment.
    /// If `seg1` has top chunk, then remove top and insert its chunk.
    /// `seg1` info chunk and border chunks will be free-extended.
    unsafe fn merge_segments(&mut self, seg1: &mut Segment, seg2: &mut Segment) {
        dlassert!(seg1.end() == seg2.base);
        dlassert!(seg1.size % DEFAULT_GRANULARITY == 0);
        dlassert!(seg2.size % DEFAULT_GRANULARITY == 0);
        dlassert!(seg1.base as usize % MALIGN == 0);
        dlassert!(seg2.base as usize % MALIGN == 0);

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
            // TODO: may be we should find the biggest top free segment to be new top
        }

        self.extend_free_chunk(seg1_info_chunk, true);
        self.check_top_chunk(self.top);
    }

    /// Set seg info chunk and border chunk.
    unsafe fn set_segment_info(
        &mut self,
        seg_base: *mut u8,
        seg_size: usize,
        prev_in_use: usize,
    ) -> *mut Segment {
        let seg_end = seg_base.add(seg_size);
        let seg_chunk = seg_end.sub(SEG_INFO_SIZE) as *mut Chunk;
        let seg_info = Chunk::plus_offset(seg_chunk, 2 * PTR_SIZE) as *mut Segment;
        let border_chunk = Chunk::plus_offset(seg_chunk, 4 * PTR_SIZE);

        dlassert!(seg_end as usize % MALIGN == 0);
        dlassert!(seg_chunk as usize % MALIGN == 0);
        dlassert!(seg_info as usize % MALIGN == 0);
        dlassert!(border_chunk as usize % MALIGN == 0);
        dlassert!(border_chunk as *mut u8 == seg_end.sub(2 * PTR_SIZE));

        dlverbose!("ALLOC: add seg, info chunk {:?}", seg_chunk);

        // see [Segment]
        (*seg_chunk).head = (4 * PTR_SIZE) | prev_in_use | CINUSE;
        (*seg_info).base = seg_base;
        (*seg_info).size = seg_size;
        (*border_chunk).head = BORDER_CHUNK_HEAD;

        seg_info
    }

    /// Add new memory as segment in allocator context
    unsafe fn add_segment(&mut self, tbase: *mut u8, tsize: usize, flags: u32) {
        dlassert!(tbase as usize % MALIGN == 0);
        dlassert!(tsize % DEFAULT_GRANULARITY == 0);

        let seg = self.set_segment_info(tbase, tsize, 0);
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

        dlverbose!(
            "SYS_ALLOC: add seg, top[{:?}, 0x{:x}]",
            self.top,
            self.topsize
        );

        self.check_top_chunk(self.top);
        self.check_malloc_state();
    }

    /// Finds segment which contains `ptr`: means `ptr` is in [a, b)
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

    /// Find the smallest chunk in trees, in order to allocate memory for
    /// small chunk.
    unsafe fn tmalloc_small(&mut self, size: usize) -> *mut u8 {
        let first_one_idx = self.treemap.trailing_zeros();
        let first_tree_chunk = *self.treebin_at(first_one_idx);

        // Iterate left and search the most suitable chunk
        let mut tree_chunk = first_tree_chunk;
        let mut best_tree_chunk = first_tree_chunk;
        let mut remainder_size = Chunk::size(TreeChunk::chunk(tree_chunk)) - size;
        loop {
            self.check_any_chunk(TreeChunk::chunk(tree_chunk));
            tree_chunk = TreeChunk::leftmost_child(tree_chunk);
            if tree_chunk.is_null() {
                break;
            }
            let diff = Chunk::size(TreeChunk::chunk(tree_chunk)) - size;
            if diff < remainder_size {
                remainder_size = diff;
                best_tree_chunk = tree_chunk;
            }
        }

        let chunk = TreeChunk::chunk(best_tree_chunk);
        dlassert!(Chunk::size(chunk) == remainder_size + size);

        self.unlink_large_chunk(best_tree_chunk);

        if remainder_size < MIN_CHUNK_SIZE {
            // use all mem in chunk
            (*chunk).head = (remainder_size + size) | PINUSE | CINUSE;
            (*Chunk::next(chunk)).head |= PINUSE;
        } else {
            // use part and set remainder as dv
            (*chunk).head = size | PINUSE | CINUSE;
            let remainder = Chunk::next(chunk);
            (*remainder).head = remainder_size | PINUSE;
            Chunk::set_next_chunk_prev_size(remainder, remainder_size);
            self.replace_dv(remainder, remainder_size);
        }

        Chunk::to_mem(chunk)
    }

    /// Find most suitable chunk in trees for `size`
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
                t = (*t).child[(sizebits >> (PTR_SIZE * 8 - 1)) & 1];
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
            let leftbits = self.treemap & (u32::MAX << idx) << 1;
            if leftbits != 0 {
                let idx = leftbits.trailing_zeros();
                t = *self.treebin_at(idx);
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

        // If dv is a better fit, then returns null so malloc will use it
        if v.is_null() || (self.dvsize >= size && rsize >= (self.dvsize - size)) {
            return ptr::null_mut();
        }

        let vc = TreeChunk::chunk(v);
        let r = Chunk::plus_offset(vc, size);
        dlassert!(Chunk::size(vc) == rsize + size);
        self.unlink_large_chunk(v);
        if rsize < MIN_CHUNK_SIZE {
            (*vc).head = (rsize + size) | CINUSE | PINUSE;
            (*Chunk::next(vc)).head |= PINUSE;
        } else {
            (*vc).head = size | CINUSE | PINUSE;
            (*r).head = rsize | PINUSE;
            Chunk::set_next_chunk_prev_size(r, rsize);
            self.insert_chunk(r, rsize);
        }
        Chunk::to_mem(vc)
    }

    /// Returns smallbin head for `idx`.
    ///
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
        smallbins_ptr.add(idx) as *mut Chunk
    }

    /// Returns `idx` tree root chunk
    unsafe fn treebin_at(&mut self, idx: u32) -> *mut *mut TreeChunk {
        dlassert!((idx as usize) < self.treebins.len());
        &mut *self.treebins.get_unchecked_mut(idx as usize)
    }

    /// Returns index of tree which can contain chunks for `size`
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

    /// Unlinks and returns last chunk from small chunks list
    unsafe fn unlink_last_small_chunk(&mut self, head: *mut Chunk, idx: u32) -> *mut Chunk {
        let chunk = (*head).prev;
        let new_first_chunk = (*chunk).prev;
        dlassert!(chunk != head);
        dlassert!(chunk != new_first_chunk);
        dlassert!(Chunk::size(chunk) == self.small_index2size(idx));
        if head == new_first_chunk {
            self.clear_smallmap(idx);
        } else {
            (*new_first_chunk).next = head;
            (*head).prev = new_first_chunk;
        }
        chunk
    }

    /// Replaces [Dlmalloc::dv] to given `chunk`.
    /// Inserts old dv as common chunk.
    unsafe fn replace_dv(&mut self, chunk: *mut Chunk, size: usize) {
        let dv_size = self.dvsize;
        dlassert!(self.is_chunk_small(dv_size));
        if dv_size != 0 {
            self.insert_chunk(self.dv, dv_size);
        }
        self.dvsize = size;
        self.dv = chunk;
    }

    /// Inserts free chunk to allocator context
    unsafe fn insert_chunk(&mut self, chunk: *mut Chunk, size: usize) {
        dlverbose!("ALLOC: insert [{:?}, {:?}]", chunk, Chunk::next(chunk));

        dlassert!(size == Chunk::size(chunk));

        if self.is_chunk_small(size) {
            self.insert_small_chunk(chunk, size);
        } else {
            self.insert_large_chunk(chunk as *mut TreeChunk, size);
        }
    }

    /// Inserts small free chunk to allocator context
    unsafe fn insert_small_chunk(&mut self, chunk: *mut Chunk, size: usize) {
        let idx = self.small_index(size);
        let head = self.smallbin_at(idx);
        let mut f = head;
        dlassert!(size >= MIN_CHUNK_SIZE);
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

    /// Inserts large free chunk to allocator context
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
                    let c = &mut (*t).child[(k >> (PTR_SIZE * 8 - 1)) & 1];
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

    /// Unlinks free chunk from list or tree
    unsafe fn unlink_chunk(&mut self, chunk: *mut Chunk, size: usize) {
        dlassert!(Chunk::size(chunk) == size);

        if self.is_chunk_small(size) {
            dlverbose!("ALLOC: unlink chunk[{:?}, {:?}]", chunk, Chunk::next(chunk));
            self.unlink_small_chunk(chunk, size)
        } else {
            self.unlink_large_chunk(chunk as *mut TreeChunk);
        }
    }

    /// Unlinks small free chunk from small chunks list
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

    /// Unlinks large free chunk from tree
    unsafe fn unlink_large_chunk(&mut self, chunk: *mut TreeChunk) {
        dlverbose!(
            "ALLOC: unlink chunk[{:?}, {:?}]",
            chunk,
            Chunk::next(TreeChunk::chunk(chunk))
        );
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
        } else if (*parent).child[0] == chunk {
            (*parent).child[0] = r;
        } else {
            (*parent).child[1] = r;
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

    /// Returns size of chunk if we will extend it up
    unsafe fn get_extended_up_chunk_size(&mut self, chunk: *mut Chunk) -> usize {
        let next_chunk = Chunk::next(chunk);
        if !Chunk::cinuse(next_chunk) {
            Chunk::size(chunk) + Chunk::size(next_chunk)
        } else {
            Chunk::size(chunk)
        }
    }

    /// Frees and extends chunk.
    /// Takes [CINUSE] chunk and marks it as free.
    /// Because two neighbor chunks cannot be both free,
    /// we must merge our chunk with all free chunks around.
    /// `can_insert` arg controls whether we have to insert
    /// the result free chunk into list/tree (if it isn't top or dv).
    unsafe fn extend_free_chunk(&mut self, mut chunk: *mut Chunk, can_insert: bool) -> *mut Chunk {
        dlassert!(Chunk::cinuse(chunk));
        (*chunk).head &= !CINUSE;

        // try join prev chunk
        if !Chunk::pinuse(chunk) {
            let curr_chunk_size = Chunk::size(chunk);
            let prev_chunk = Chunk::prev(chunk);
            let prev_chunk_size = Chunk::size(prev_chunk);
            dlassert!(Chunk::pinuse(prev_chunk));

            if prev_chunk == self.top {
                self.topsize += Chunk::size(chunk);
            } else if prev_chunk == self.dv {
                self.dvsize += Chunk::size(chunk);
            } else {
                self.unlink_chunk(prev_chunk, prev_chunk_size);
            }

            dlverbose!(
                "extend: add before chunk[{:?}, 0x{:x}] {}",
                prev_chunk,
                prev_chunk_size,
                self.is_top_or_dv(prev_chunk)
            );

            chunk = prev_chunk;
            (*chunk).head = (curr_chunk_size + prev_chunk_size) | PINUSE;
        }

        // try to join next chunk
        let next_chunk = Chunk::next(chunk);
        if !Chunk::cinuse(next_chunk) {
            dlverbose!(
                "extend: add after chunk[{:?}, 0x{:x}] {}",
                next_chunk,
                Chunk::size(next_chunk),
                self.is_top_or_dv(next_chunk)
            );
            if next_chunk == self.top {
                self.top = chunk;
                self.topsize += Chunk::size(chunk);
                if chunk == self.dv {
                    // top eats dv
                    self.dv = ptr::null_mut();
                    self.dvsize = 0;
                }
                (*chunk).head = self.topsize | PINUSE;
                (*Chunk::next(chunk)).prev_chunk_size = self.topsize;
            } else if next_chunk == self.dv {
                if chunk == self.top {
                    // top eats dv
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
                self.insert_chunk(chunk, Chunk::size(chunk));
            }
        }

        chunk
    }

    /// When user call free mem, in our context it means - free one chunk.
    /// There can be already free neighbor chunks, so we extend our chunk
    /// to all free chunks around. Then if chunk is big enought we can return some memory to system.
    /// To understand what memory can be returned to system, you should now one simple rule:
    /// segments has size aligned by [DEFAULT_GRANULARITY]. So, when we return memory to the system,
    /// we may change segments size or delete some segments or create new segments,
    /// and in all cases we always must satisfy this rule.
    /// Let's see example:
    /// ````
    ///  Segment begin    Default granuality                       Segment end
    ///  |                |               \                                  |
    ///  [================|========(=======|================|====)===========]
    ///                            |                             |
    ///                            Chunk begin                   Chunk end
    /// ````
    /// Here chunk is free, and we want to return some part of chunk's memory to the system.
    /// To avoid unaligned segments, we call system free for only one granuality part:
    /// ````
    ///  Segment1                                           Segment
    ///  |                                                  |
    ///  [================|========(=======]                [====)===========]
    ///                            |       |                |    |
    ///                       Chunk1    Chunk1 end     Chunk2    Chunk2 end
    /// ````
    /// We create new Segment1 and crop old Segment.
    /// Memory between segments isn't in allocator context now.
    /// Chunk1 and Chunk2 are free remainders, which is added to tree/smallbins/top,
    /// depends on size and context.
    ///
    /// If chunk is not big enought to sys-free something, then it is marked as free
    /// and added to allocator context just like remainders above.
    /// TODO: we also must call free in sys_alloc when one page is excess
    /// TODO: we also must call free in realloc when we free chunk
    pub unsafe fn free(&mut self, mem: *mut u8) {
        dlverbose!("{}", VERBOSE_DEL);
        dlverbose!("ALLOC FREE CALL: mem={:?}", mem);

        self.check_malloc_state();

        let chunk = Chunk::from_mem(mem);
        let chunk_size = Chunk::size(chunk);
        dlverbose!("ALLOC FREE: chunk[{:?}, 0x{:x}]", chunk, chunk_size);

        let chunk = self.extend_free_chunk(chunk, false);
        let chunk_size = Chunk::size(chunk);
        dlverbose!(
            "ALLOC FREE: extended chunk[{:?}, 0x{:x}] {}",
            chunk,
            chunk_size,
            self.is_top_or_dv(chunk)
        );

        if chunk_size + SEG_INFO_SIZE < DEFAULT_GRANULARITY {
            Chunk::set_next_chunk_prev_size(chunk, chunk_size);
            if chunk != self.top && chunk != self.dv {
                self.insert_chunk(chunk, chunk_size);
            }
            return;
        }

        let mut mem_to_free = chunk as *mut u8;
        let mut mem_to_free_end = mem_to_free.add(chunk_size);

        // find holding segment and prev segment in list
        let mut prev_seg = ptr::null_mut() as *mut Segment;
        let mut seg = self.seg;
        while !seg.is_null() {
            if Segment::holds(seg, mem_to_free) {
                break;
            }
            prev_seg = seg;
            seg = (*seg).next;
        }
        dlassert!(!seg.is_null());
        dlassert!((*seg).size > chunk_size);

        let seg_begin = (*seg).base;
        let seg_end = (*seg).base.add((*seg).size);
        dlassert!(mem_to_free_end < seg_end);
        dlassert!(seg_begin as usize % MALIGN == 0);
        dlassert!((*seg).size % DEFAULT_GRANULARITY == 0);

        dlverbose!("ALLOC FREE: holding seg[{:?}, {:?}]", seg_begin, seg_end);
        dlverbose!("ALLOC FREE: prev seg = {:?}", prev_seg);

        let before_remainder_size: usize;
        if mem_to_free != seg_begin {
            dlassert!(Chunk::pinuse(chunk));
            dlassert!(mem_to_free as usize - seg_begin as usize >= MIN_CHUNK_SIZE);

            // we cannot free chunk.pred_chunk_size mem because it may be used by prev chunk mem
            mem_to_free = mem_to_free.add(PTR_SIZE);

            // additionally we need space for new segment info
            mem_to_free = mem_to_free.add(SEG_INFO_SIZE);

            // we restrict not granularity segments
            before_remainder_size = align_up(
                mem_to_free as usize - seg_begin as usize,
                DEFAULT_GRANULARITY,
            );
            mem_to_free = seg_begin.add(before_remainder_size);
        } else {
            before_remainder_size = 0;
        }

        let mut after_remainder_size = seg_end as usize - mem_to_free_end as usize;
        if after_remainder_size > SEG_INFO_SIZE {
            // If there is chunk(s) between, then it must be at least min chunk size
            dlassert!(after_remainder_size >= SEG_INFO_SIZE + MIN_CHUNK_SIZE);

            // TODO: fix it
            // We need that in after remainder the most right chunk is > min_chunk_size
            after_remainder_size += MIN_CHUNK_SIZE;

            after_remainder_size = align_up(after_remainder_size, DEFAULT_GRANULARITY);
            mem_to_free_end = seg_end.sub(after_remainder_size);
        } else {
            dlassert!(after_remainder_size == SEG_INFO_SIZE);
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
        dlassert!(mem_to_free_size % DEFAULT_GRANULARITY == 0);

        dlverbose!(
            "ALLOC FREE: mem to free [{:?}, {:?}]",
            mem_to_free,
            mem_to_free_end
        );

        // We crop chunk with a reserve for before remainder segment info if there will be one
        let mut crop_chunk;
        let mut crop_chunk_size;
        if before_remainder_size != 0 {
            crop_chunk = mem_to_free.sub(SEG_INFO_SIZE) as *mut Chunk;
            if (crop_chunk as usize - chunk as usize) < MIN_CHUNK_SIZE {
                // TODO: fix it
                dlverbose!(
                    "ALLOC FREE: cannot free beacause of left remainder [{:?}, {:?}]",
                    chunk,
                    crop_chunk
                );
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
            dlassert!(mem_to_free_end == seg_end);
            dlassert!(Chunk::next(chunk) as *mut u8 == seg_end.sub(SEG_INFO_SIZE));
            crop_chunk_size -= SEG_INFO_SIZE;
        }

        dlassert!(crop_chunk >= chunk);
        dlassert!(crop_chunk_size <= chunk_size);

        (*chunk).head |= CINUSE;
        let chunk = self.crop_chunk(chunk, crop_chunk, crop_chunk_size);
        dlassert!(Chunk::size(chunk) == crop_chunk_size);

        let next_seg = (*seg).next;
        let before_rem_pinuse: usize;
        if before_remainder_size > 0 {
            before_rem_pinuse = if Chunk::pinuse(chunk) { PINUSE } else { 0 };
        } else {
            before_rem_pinuse = 0;
        }
        let after_rem_pinuse: usize;
        if after_remainder_size > 0 {
            after_rem_pinuse = if Chunk::pinuse((seg as *mut u8).sub(2 * PTR_SIZE) as *mut Chunk) {
                PINUSE
            } else {
                0
            };
        } else {
            after_rem_pinuse = 0;
        }

        let (cond, free_mem, free_mem_size) = sys::free(mem_to_free, mem_to_free_size);
        dlassert!(cond);
        dlassert!(free_mem == mem_to_free);
        dlassert!(mem_to_free_size == free_mem_size);

        if before_remainder_size != 0 {
            let before_seg_info =
                self.set_segment_info(seg_begin, before_remainder_size, before_rem_pinuse);

            dlverbose!(
                "ALLOC FREE: before seg [{:?}, {:?}]",
                (*before_seg_info).base,
                Segment::top(before_seg_info)
            );

            if prev_seg.is_null() {
                dlassert!(seg == self.seg);
                self.seg = before_seg_info;
            } else {
                (*prev_seg).next = before_seg_info;
            }
            prev_seg = before_seg_info;
        }

        if after_remainder_size != 0 {
            let after_seg_info =
                self.set_segment_info(mem_to_free_end, after_remainder_size, after_rem_pinuse);

            dlverbose!(
                "ALLOC FREE: after seg [{:?}, {:?}]",
                (*after_seg_info).base,
                Segment::top(after_seg_info)
            );

            if prev_seg.is_null() {
                dlassert!(seg == self.seg);
                self.seg = after_seg_info;
            } else {
                (*prev_seg).next = after_seg_info;
            }
            prev_seg = after_seg_info;
        }

        if prev_seg.is_null() {
            dlassert!(seg == self.seg);
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

    /// Returns static string about chunk status in context
    fn is_top_or_dv(&self, chunk: *mut Chunk) -> &'static str {
        if chunk == self.top {
            "is top"
        } else if chunk == self.dv {
            "is dv"
        } else {
            "is chunk"
        }
    }

    /// Bypasses all segments and counts sum of in use chunks sizes.
    pub unsafe fn get_alloced_mem_size(&self) -> usize {
        let mut size: usize = 0;
        let mut seg = self.seg;
        while !seg.is_null() {
            let mut chunk = (*seg).base as *mut Chunk;
            let last_chunk = Segment::top(seg).sub(SEG_INFO_SIZE);
            while (chunk as *mut u8) < last_chunk {
                if Chunk::cinuse(chunk) {
                    size += Chunk::size(chunk);
                }
                chunk = Chunk::next(chunk);
            }
            dlassert!(chunk as *mut u8 == last_chunk);
            seg = (*seg).next;
        }
        size
    }

    /// Prints all segments and their chunks
    unsafe fn print_segments(&mut self) {
        if !DL_VERBOSE {
            return;
        }
        let mut i = 0;
        let mut seg = self.seg;
        while !seg.is_null() {
            i += 1;
            dlverbose!(
                "+++++++ SEG{} {:?} [{:?}, {:?}]",
                i,
                seg,
                (*seg).base,
                Segment::top(seg)
            );
            let mut chunk = (*seg).base as *mut Chunk;
            let last_chunk = Segment::top(seg).sub(SEG_INFO_SIZE);
            while (chunk as *mut u8) < last_chunk {
                dlverbose!(
                    "SEG{} chunk [{:?}, {:?}]{}{} {}",
                    i,
                    chunk,
                    Chunk::next(chunk),
                    if Chunk::cinuse(chunk) { "c" } else { "" },
                    if Chunk::pinuse(chunk) { "p" } else { "" },
                    self.is_top_or_dv(chunk)
                );
                chunk = Chunk::next(chunk);
            }

            dlverbose!(
                "SEG{} info [{:?}, {:?}]{}{}",
                i,
                chunk,
                Chunk::next(chunk),
                if Chunk::cinuse(chunk) { "c" } else { "" },
                if Chunk::pinuse(chunk) { "p" } else { "" }
            );

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

        dlassert!(!p.is_null());
        dlassert!(p as usize % MALIGN == 0);
        dlassert!(Chunk::size(p) % MALIGN == 0);
        dlassert!(Chunk::to_mem(p) as usize % MALIGN == 0);
        dlassert!(p as *mut u8 >= self.least_addr);

        // Checks that @p doesn't intersect some other chunk
        let mut seg = self.seg;
        while !seg.is_null() {
            let mut chunk = (*seg).base as *mut Chunk;
            let last_chunk = Segment::top(seg).sub(SEG_INFO_SIZE);
            while (chunk as *mut u8) < last_chunk {
                dlassert!(!(chunk > p && chunk < Chunk::next(p)));
                dlassert!(!(p > chunk && p < Chunk::next(chunk)));
                chunk = Chunk::next(chunk);
            }
            seg = (*seg).next;
        }
    }

    unsafe fn check_top_chunk(&self, p: *mut Chunk) {
        if !DL_CHECKS {
            return;
        }
        if self.top.is_null() {
            dlassert!(self.topsize == 0);
            return;
        }
        self.check_any_chunk(p);

        let sp = self.segment_holding(p as *mut u8);
        let sz = Chunk::size(p);
        dlassert!(!sp.is_null());
        dlassert!(sz == self.topsize);
        dlassert!(sz != 0);
        dlassert!(sz == (*sp).base as usize + (*sp).size - p as usize - SEG_INFO_SIZE);
        dlassert!(Chunk::pinuse(p));
        dlassert!(!Chunk::pinuse(Chunk::plus_offset(p, sz)));
    }

    unsafe fn check_malloced_mem(&self, mem: *mut u8, req_size: usize) {
        if !DL_CHECKS {
            return;
        }
        if mem.is_null() {
            return;
        }
        let p = Chunk::from_mem(mem);
        let sz = Chunk::size(p);
        self.check_any_chunk(p);
        self.check_cinuse_chunk(p);
        dlassert!(sz >= MIN_CHUNK_SIZE);
        dlassert!(sz >= req_size + PTR_SIZE);
    }

    unsafe fn check_cinuse_chunk(&self, p: *mut Chunk) {
        self.check_any_chunk(p);
        dlassert!(Chunk::cinuse(p));
        dlassert!(Chunk::pinuse(Chunk::next(p)));
        dlassert!(Chunk::pinuse(p) || Chunk::next(Chunk::prev(p)) == p);
    }

    unsafe fn check_free_chunk(&self, p: *mut Chunk) {
        if !DL_CHECKS {
            return;
        }
        self.check_any_chunk(p);
        let sz = Chunk::size(p);
        let next = Chunk::plus_offset(p, sz);
        dlassert!(!Chunk::cinuse(p) && Chunk::pinuse(p));
        dlassert!(!Chunk::pinuse(Chunk::next(p)));
        dlassert!((*next).prev_chunk_size == sz);
        dlassert!(Chunk::cinuse(next));

        if p != self.dv && p != self.top {
            // TODO: change when add half-chunk
            if sz >= MIN_CHUNK_SIZE {
                dlassert!((*(*p).next).prev == p);
                dlassert!((*(*p).prev).next == p);
            }
        }
    }

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
            dlassert!(self.dvsize == Chunk::size(self.dv));
            dlassert!(self.dvsize >= MIN_CHUNK_SIZE);
            dlassert!(Chunk::pinuse(self.dv));
            dlassert!(!Chunk::cinuse(self.dv));
            dlassert!(!self.bin_find(self.dv));
        }
        if !self.top.is_null() {
            self.check_top_chunk(self.top);
            dlassert!(self.topsize > 0);
            dlassert!(!self.bin_find(self.top));
        }

        // Bypasses all segments
        let mut seg = self.seg;
        while !seg.is_null() && !(*seg).base.is_null() {
            let mut chunk = (*seg).base as *mut Chunk;
            let last_chunk = Segment::top(seg).sub(SEG_INFO_SIZE);
            while (chunk as *mut u8) < last_chunk {
                if chunk != self.top && chunk != self.dv {
                    dlassert!(self.top < chunk || self.top >= Chunk::next(chunk));
                    dlassert!(self.dv < chunk || self.dv >= Chunk::next(chunk));
                }
                chunk = Chunk::next(chunk);
            }
            dlassert!(chunk as *mut u8 == last_chunk);
            dlassert!(Chunk::size(chunk) == SEG_INFO_SIZE - 2 * PTR_SIZE);

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
                dlassert!(
                    (*bin_chunk).next == head_chunk || Chunk::size((*bin_chunk).next) == bin_size
                );

                let next_mem_chunk = Chunk::next(bin_chunk);
                if !(*next_mem_chunk).is_border_chunk() {
                    self.check_cinuse_chunk(next_mem_chunk);
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
        dlassert!(tsize >= self.min_large_chunk_size());
        dlassert!(tsize >= self.min_size_for_tree_index(idx));
        dlassert!(idx == NTREEBINS as u32 - 1 || tsize < self.min_size_for_tree_index(idx + 1));

        let mut u = t;
        let mut head = ptr::null_mut::<TreeChunk>();
        loop {
            let uc = TreeChunk::chunk(u);
            self.check_any_chunk(uc);
            dlassert!((*u).index == tindex);
            dlassert!(Chunk::size(uc) == tsize);
            dlassert!(!Chunk::cinuse(uc) && Chunk::pinuse(uc));
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
        if self.is_chunk_small(size) {
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
                t = (*t).child[(sizebits >> (PTR_SIZE * 8 - 1)) & 1];
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
    fn is_border_chunk(&self) -> bool {
        self.head == BORDER_CHUNK_HEAD
    }
    unsafe fn size(me: *mut Chunk) -> usize {
        (*me).head & !FLAG_BITS
    }
    unsafe fn next(me: *mut Chunk) -> *mut Chunk {
        Chunk::plus_offset(me, Chunk::size(me))
    }
    unsafe fn prev(me: *mut Chunk) -> *mut Chunk {
        dlassert!(!Chunk::pinuse(me));
        Chunk::minus_offset(me, (*me).prev_chunk_size)
    }
    unsafe fn cinuse(me: *mut Chunk) -> bool {
        (*me).head & CINUSE != 0
    }
    unsafe fn pinuse(me: *mut Chunk) -> bool {
        (*me).head & PINUSE != 0
    }
    unsafe fn set_next_chunk_prev_size(me: *mut Chunk, size: usize) {
        (*Chunk::next(me)).prev_chunk_size = size;
    }
    unsafe fn change_size(me: *mut Chunk, size: usize) {
        (*me).head = size | ((*me).head & FLAG_BITS);
    }
    unsafe fn plus_offset(me: *mut Chunk, offset: usize) -> *mut Chunk {
        (me as *mut u8).add(offset) as *mut Chunk
    }
    unsafe fn minus_offset(me: *mut Chunk, offset: usize) -> *mut Chunk {
        (me as *mut u8).sub(offset) as *mut Chunk
    }
    unsafe fn to_mem(me: *mut Chunk) -> *mut u8 {
        (me as *mut u8).add(CHUNK_MEM_OFFSET)
    }
    unsafe fn from_mem(mem: *mut u8) -> *mut Chunk {
        mem.sub(CHUNK_MEM_OFFSET) as *mut Chunk
    }
}

impl TreeChunk {
    unsafe fn leftmost_child(me: *mut TreeChunk) -> *mut TreeChunk {
        if (*me).child[0].is_null() {
            (*me).child[1]
        } else {
            (*me).child[0]
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
        (*seg).base.add((*seg).size)
    }
    pub unsafe fn end(&self) -> *mut u8 {
        self.base.add(self.size)
    }
    pub unsafe fn info_chunk(&self) -> *mut Chunk {
        self.end().sub(SEG_INFO_SIZE) as *mut Chunk
    }
    pub unsafe fn base_chunk(&self) -> *mut Chunk {
        self.base as *mut Chunk
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Prime the allocator with some allocations such that there will be free
    // chunks in the treemap
    unsafe fn setup_treemap(a: &mut Dlmalloc) {
        let large_request_size = NSMALLBINS * (1 << SMALLBIN_SHIFT);
        assert!(!a.is_chunk_small(large_request_size));
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
            let min_idx31_size = (0xc000 << TREEBIN_SHIFT) - PTR_SIZE + 1;
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
