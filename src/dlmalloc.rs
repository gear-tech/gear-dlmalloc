// This is a version of dlmalloc.c ported to Rust. You can find the original
// source at ftp://g.oswego.edu/pub/misc/malloc.c
//
// The original source was written by Doug Lea and released to the public domain

#![allow(unused)]

use core::cmp;
use core::isize::MIN;
use core::mem;
use core::ptr;
use core::ptr::null_mut;

use crate::common::{align_down, align_up};
use crate::dlassert;
use crate::dlverbose;
use crate::dlverbose_no_flush;
use dlverbose::{DL_CHECKS, DL_DEBUG, DL_VERBOSE, VERBOSE_DEL};
use sys;

extern crate static_assertions;

/// Pointer size.
const PTR_SIZE: usize = mem::size_of::<usize>();
/// Malloc alignment.
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
/// Default granularity is min size for system memory allocation and free.
const DEFAULT_GRANULARITY: usize = 64 * 1024; // 64 kBytes

static_assertions::const_assert!(2 * MALIGN == CHUNK_SIZE);
static_assertions::const_assert!(3 * PTR_SIZE == SEG_SIZE);
static_assertions::const_assert!(MIN_CHUNK_SIZE % MALIGN == 0);
static_assertions::const_assert!(6 * PTR_SIZE == SEG_INFO_SIZE);
static_assertions::const_assert!(SEG_INFO_SIZE % MALIGN == 0);
static_assertions::const_assert!(DEFAULT_GRANULARITY % MALIGN == 0);
static_assertions::const_assert!(DEFAULT_GRANULARITY >= MIN_CHUNK_SIZE + SEG_INFO_SIZE);

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

/// Sizes for each [Dlmalloc::sbuff] cell.
/// Each hex number is size of one cell, for example 0x4321,
/// means that first cell will have size 0x1 * [MALIGN], second 0x2 * [MALIGN],
/// third 0x3 * [MALIGN] and forth 0x4 * [MALIGN].
const SBUFF_IDX_SIZES: usize = 0x86611;

/// Max index which cell can have plus one.
/// Actually is number of cells in [Dlmalloc::sbuff]
const SBUFF_IDX_MAX: usize = {
    let mut sizes = SBUFF_IDX_SIZES;
    let mut idx = 0;
    while sizes != 0 {
        idx += 1;
        sizes >>= 4;
    }
    idx
};
static_assertions::const_assert!(SBUFF_IDX_MAX < 8);

/// Max size which cell can have
const SBUFF_MAX: usize = { (SBUFF_IDX_SIZES >> (4 * (SBUFF_IDX_MAX - 1))) * MALIGN };

/// Sum of all cell sizes in [Dlmalloc::sbuff]
const SBUFF_SIZE: usize = {
    let mut sizes = SBUFF_IDX_SIZES;
    let mut res = 0;
    while sizes != 0 {
        res += sizes & 0xF;
        sizes >>= 4;
    }
    res * MALIGN
};

/// Hex number which identify offsets for each sbuff cell.
const SBUFF_IDX_OFFSETS: usize = {
    let mut sizes = SBUFF_IDX_SIZES;
    let mut offset = 0;
    let mut res = 0;
    let mut idx = 0;
    while sizes != 0 {
        res += offset << idx;
        offset += sizes & 0xF;
        idx += 4;
        sizes >>= 4;
    }
    res
};

/// Max offset - offset which last cell in sbuff has.
/// We check that this offset is less then 0x10,
/// so that each offset can be stored as one hex number in [SBUFF_IDX_OFFSETS].
const SBUFF_MAX_OFFSET: usize = {
    let mut offset = 0;
    let mut idx = 0;
    while idx < SBUFF_IDX_MAX - 1 {
        offset += (SBUFF_IDX_SIZES >> (4 * idx)) & 0xF;
        idx += 1;
    }
    offset
};
static_assertions::const_assert!(SBUFF_MAX_OFFSET < 0x10);

/// Dl allocator uses memory non-overlapping intervals for each request - here named Chunks.
///
/// Each chunk can be in two states: in use and free.
/// When chunk is in use, its memory can be read/written by somebody.
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
///
/// Half chunks.
/// Some times there is situations, when [MALIGN]-size chunks are created.
/// For example, when we malloc chunk for mem-request
/// and remainder is < [MIN_CHUNK_SIZE], but >= [MALIGN].
/// Then we cannot insert this remainder into smallbins, but can mark it as free.
/// Such chunk won't be used in malloc, but if some neighbor chunk
/// become free, this two will be merged.
///````
/// chunk for allocation                  very small remainder
/// |                                                   \
/// [--(------------------------------------------------)-]
///    |                                                |
///    requested mem begin                      requested mem end
///
///
/// chunk for allocation                    chunk end    free half chunk
/// |                                                \  /
/// [--(----------------------------------------------][-)-]
///    |                                                 |
///    requested mem begin                      requested mem end
///````
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

/// Segment is continuous memory section, which we occupied for allocations.
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
/// If one chunk became free and there is neighbor free chunk,
/// then this chunks will be merged.
/// 2) Cannot be two neighbor segments.
/// If we allocate new segment in [Dlmalloc::sys_alloc] and there is
/// neighbor segment in allocator context, then we merge this segments.
/// 3) [MIN_CHUNK_SIZE] is min size, which allocated chunk may have.
/// Free chunks also may have size == [MALIGN]. These chunks are not
/// stored into tree or smallbins and cannot be used in malloc.
/// But this chunks can be merged with other free neighbor chunk.
/// see more in [Chunk].
/// 4) If no heap memory is allocated yet, then dlmalloc use static
/// buffer for small requests allocations, in order to increase
/// allocation performance. See more in [Dlmalloc::malloc].
/// 5) Some memory can be preinstalled by system,
/// this memory is added in context in [Dlmalloc::sys_alloc] first call.
///
#[repr(align(16))]
#[repr(C)]
pub struct Dlmalloc {
    /// Static memory buffer, it uses to alloc small memory
    /// requests, if there is no allocated memory yet.
    /// TODO: move it to dynamic initialization
    sbuff: [u8; SBUFF_SIZE],
    /// Mark sbuff cells which is in use.
    sbuff_mask: u8,
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
    /// Whether preinstalled memory initialization has been done?
    preinstallation_is_done: bool,
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
    sbuff: [0; SBUFF_SIZE],
    sbuff_mask: 0,
    preinstallation_is_done: false,
    least_addr: 0 as *mut _,
};

fn to_pinuse(prev_in_use: bool) -> usize {
    if prev_in_use {
        PINUSE
    } else {
        0
    }
}

fn to_cinuse(curr_in_use: bool) -> usize {
    if curr_in_use {
        CINUSE
    } else {
        0
    }
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

    /// Set top or dv to `new_val` or `new_size` if `chunk` is top or dv
    /// Returns true if chunk is top or dv
    /// TODO: apply this fn
    unsafe fn set_top_or_dv(
        &mut self,
        chunk: *mut Chunk,
        new_val: *mut Chunk,
        new_size: usize,
    ) -> bool {
        if chunk == self.top {
            self.top = new_val;
            self.topsize = new_size;
        } else if chunk == self.dv {
            self.dv = new_val;
            self.dvsize = new_size;
        } else {
            return false;
        }
        true
    }

    /// Returns size of [Dlmalloc::sbuff] cell for `idx` ``
    unsafe fn sbuff_idx_to_size(idx: usize) -> usize {
        dlassert!(idx < SBUFF_IDX_MAX);
        ((SBUFF_IDX_SIZES >> (4 * idx)) & 0xF) * MALIGN
    }

    /// Returns offset of [Dlmalloc::sbuff] cell for `idx`
    unsafe fn sbuff_idx_to_offset(idx: usize) -> usize {
        dlassert!(idx < SBUFF_IDX_MAX);
        ((SBUFF_IDX_OFFSETS >> (4 * idx)) & 0xF) * MALIGN
    }

    /// Returns idx of [Dlmalloc::sbuff] cell which has `offset`.
    /// If cannot find such cell, then returns [SBUFF_IDX_MAX]
    unsafe fn sbuff_offset_to_idx(offset: usize) -> usize {
        for idx in 0..SBUFF_IDX_MAX {
            if offset == Dlmalloc::sbuff_idx_to_offset(idx) {
                return idx;
            }
        }
        SBUFF_IDX_MAX
    }

    /// Returns index of free [Dlmalloc::sbuff] cell, which has
    /// size >= then `size`.
    /// If cannot find such cell then returns [SBUFF_IDX_MAX]
    unsafe fn sbuff_size_to_idx(&self, size: usize) -> usize {
        if size > SBUFF_MAX {
            return SBUFF_IDX_MAX;
        }

        for idx in 0..SBUFF_IDX_MAX {
            if self.sbuff_mask & (1 << idx) != 0 {
                continue;
            }
            if size <= Dlmalloc::sbuff_idx_to_size(idx) {
                return idx;
            }
        }
        SBUFF_IDX_MAX
    }

    /// Currently used only for wasm32 target.
    /// If there is memory, which is already alloced by system for current program,
    /// then here we add this memory to allocator context.
    pub unsafe fn init_preinstalled_memory(&mut self, mem_begin: usize, mem_end: usize) {
        dlassert!(!self.preinstallation_is_done);
        self.preinstallation_is_done = true;

        dlverbose!(
            "DL INIT PREINSALLED MEMORY: mem_begin = {:#x}, mem_end = {:#x}",
            mem_begin,
            mem_end,
        );

        dlassert!(self.seg.is_null());
        dlassert!(mem_end % DEFAULT_GRANULARITY == 0);

        let mem_begin = align_up(mem_begin, MALIGN);
        let size = mem_end - mem_begin;
        if size == 0 {
            return;
        }

        let req_chunk = self.append_mem_in_alloc_ctx(mem_begin as *mut u8, size, 0u32);
        dlassert!(req_chunk as usize == mem_begin);
        dlassert!(Chunk::size(req_chunk) == size - SEG_INFO_SIZE);
        dlassert!(self.top == req_chunk);
    }

    /// If there is chunk in tree/smallbins/top/dv which has size >= `chunk_size`,
    /// then returns most suitable chunk, else return null.
    unsafe fn malloc_chunk_by_size(&mut self, chunk_size: usize) -> *mut Chunk {
        if self.is_chunk_small(chunk_size) {
            // In the case we try to find suitable from small chunks
            let mut idx = self.small_index(chunk_size);
            let smallbits = self.smallmap >> idx;

            // Checks whether idx or idx + 1 has free chunks
            // TODO: remove usage of idx + 1
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

                    if remainder_size >= MALIGN {
                        // set remainder as dv
                        (*chunk).head = chunk_size | PINUSE | CINUSE;
                        let remainder = Chunk::plus_offset(chunk, chunk_size);
                        (*remainder).head = remainder_size | PINUSE;
                        Chunk::set_next_chunk_prev_size(remainder, remainder_size);
                        if remainder_size >= MIN_CHUNK_SIZE {
                            self.replace_dv(remainder, remainder_size);
                        }
                    } else {
                        dlassert!(remainder_size == 0);
                        (*chunk).head = smallsize | PINUSE | CINUSE;
                        (*Chunk::next(chunk)).head |= PINUSE;
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
            return ptr::null_mut();
        }

        // Use the dv chunk if can
        if chunk_size <= self.dvsize {
            dlverbose!("MALLOC: use dv chunk[{:?}, {:x}]", self.dv, self.dvsize);
            let chunk = self.dv;
            self.crop_chunk(self.dv, self.dv, chunk_size, true);
            return chunk;
        }

        // Use the top chunk if can
        if chunk_size <= self.topsize {
            dlverbose!(
                "MALLOC: use top chunk[{:?}, 0x{:x}]",
                self.top,
                self.topsize
            );
            let chunk = self.top;
            self.crop_chunk(self.top, self.top, chunk_size, true);
            self.check_top_chunk(self.top);
            return chunk;
        }

        ptr::null_mut()
    }

    /// Zeroes chunk unused memory tail (for debug reasons only)
    unsafe fn debug_zero_tail(ptr: *mut u8, req_size: usize, size: usize) {
        for i in req_size..size {
            *(ptr.add(i)) = 0;
        }
    }

    /// Calculates memory interval: just sum all bytes values (for debug reasons only)
    unsafe fn debug_mem_sum(ptr: *mut u8, size: usize) -> u64 {
        let mut x: u64 = 0;
        for i in 0..size {
            x += *(ptr.add(i)) as u64;
        }
        x
    }

    /// Prints memory value from first to the last byte (for debug/log reasons only)
    unsafe fn debug_print_mem(ptr: *mut u8, size: usize) {
        for i in 0..size {
            let x = *(ptr.add(i));
            dlverbose_no_flush!("{:02X}", x);
        }
        dlverbose!("");
    }

    /// Malloc func for internal usage, see more in [Dlmalloc::malloc]
    unsafe fn malloc_internal(&mut self, size: usize, can_use_sbuff: bool) -> *mut u8 {
        // Tries to use memory from statcic buffer first if can.
        if can_use_sbuff && self.seg.is_null() {
            let sbuff = &mut self.sbuff as *mut u8;
            dlassert!(sbuff as usize % MALIGN == 0);
            let idx = self.sbuff_size_to_idx(size);
            if idx < SBUFF_IDX_MAX {
                dlverbose!("DL MALLOC: use sbuff cell {}", idx);
                self.sbuff_mask |= (1 << idx);
                return sbuff.add(Dlmalloc::sbuff_idx_to_offset(idx));
            }
        }

        let chunk_size = self.mem_to_chunk_size(size);
        let chunk = self.malloc_chunk_by_size(chunk_size);
        if chunk.is_null() {
            return self.sys_alloc(chunk_size);
        }
        let mem = Chunk::to_mem(chunk);
        self.check_malloced_mem(mem, size);
        mem
    }

    /// Allocates memory interval which has size > `size` (bigger because of chunk overhead).
    /// In first memory allocation we have no available memory in allocator context.
    ///
    /// If requested size is small enought then we can use static buffer [Dlmalloc::sbuff]
    /// and allocate requested size their. This buffer begin addr is aligned by 16 bytes,
    /// so this must be enought for archs which has pointer size <= 8 bytes, and
    /// all cells in sbuff is aligned by [MALIGN].
    /// `Cell` in sbuff is a static memory interval which has const length.
    /// All cells is numerated by their index. We store offsets and sizes for each
    /// cell in [SBUFF_IDX_OFFSETS] and [SBUFF_IDX_SIZES].
    ///
    /// If there is no free cells in static buffer or requested size is not small enought,
    /// then we request system for memory interval bigger then [DEFAULT_GRANULARITY].
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
        let mem = self.malloc_internal(size, true);
        dlverbose!("MALLOC: result mem {:?}", mem);
        mem
    }

    /// Requests system to allocate memory.
    /// Adds new memory interval as segment in allocator context,
    /// if there is already some segments, which is neighbor, then
    /// merge old segments with a new one.
    /// Addtionally, if there is preinstalled memory,
    /// then we tries to init and use it.
    unsafe fn sys_alloc(&mut self, size: usize) -> *mut u8 {
        dlverbose!("DL SYS ALLOC: size = 0x{:x}", size);

        if size >= self.max_chunk_size() {
            return ptr::null_mut();
        }

        let mut req_chunk = if !self.preinstallation_is_done {
            // First call of sys_alloc tries to use preinstalled memory,
            // but before we must initialize it if there is one.
            dlassert!(self.seg.is_null());
            let (mem_addr, mem_size) = sys::get_preinstalled_memory();
            self.init_preinstalled_memory(mem_addr, mem_addr + mem_size);
            if self.topsize >= size {
                self.top
            } else {
                ptr::null_mut()
            }
        } else {
            ptr::null_mut()
        };

        if req_chunk.is_null() {
            let aligned_size = align_up(size + SEG_INFO_SIZE + MALIGN, DEFAULT_GRANULARITY);
            let (alloced_base, alloced_size, flags) = sys::alloc(aligned_size);
            dlverbose!(
                "DL SYS ALLOC: new mem {:?} 0x{:x}",
                alloced_base,
                alloced_size
            );

            if alloced_base.is_null() {
                return alloced_base;
            }

            req_chunk = self.append_mem_in_alloc_ctx(alloced_base, alloced_size, flags);
        }

        if self.top.is_null() {
            // Looking for a new top
            dlassert!(self.topsize == 0);
            let mut seg = self.seg;
            while !seg.is_null() {
                let chunk = (*seg).info_chunk();
                seg = (*seg).next;
                if Chunk::pinuse(chunk) {
                    continue;
                }
                let chunk = Chunk::prev(chunk);
                if chunk == self.dv {
                    continue;
                }
                let size = Chunk::size(chunk);
                if size < MIN_CHUNK_SIZE || size <= self.topsize {
                    continue;
                }
                self.top = chunk;
                self.topsize = size;
            }
            if !self.top.is_null() {
                self.unlink_chunk(self.top);
            }
            self.check_top_chunk(self.top);
        }

        dlassert!(Chunk::size(req_chunk) >= size);

        if self.crop_chunk(req_chunk, req_chunk, size, false) {
            // Free the remainder
            let next_chunk = Chunk::next(req_chunk);
            self.free_chunk(next_chunk);
        }

        Chunk::to_mem(req_chunk)
    }

    /// Appends alloced memory in allocator context
    unsafe fn append_mem_in_alloc_ctx(
        &mut self,
        alloced_base: *mut u8,
        alloced_size: usize,
        flags: u32,
    ) -> *mut Chunk {
        let mut req_chunk;
        if self.seg.is_null() {
            dlverbose!("DL APPEND: it's a newest mem");
            self.update_least_addr(alloced_base);
            self.add_segment(alloced_base, alloced_size, flags);
            // TODO: make it in constructor
            self.init_small_bins();
            self.check_top_chunk(self.top);
            req_chunk = self.top;
        } else {
            self.update_least_addr(alloced_base);

            self.add_segment(alloced_base, alloced_size, flags);

            req_chunk = alloced_base as *mut Chunk;

            // Checks whether there is segment, which is right before alloced mem
            let mut prev_seg = ptr::null_mut();
            let mut seg = self.seg;
            while !seg.is_null() && alloced_base != (*seg).end() {
                prev_seg = seg;
                seg = (*seg).next;
            }
            if !seg.is_null() {
                // If there is then add alloced mem to the @seg
                dlverbose!(
                    "DL APPEND: find seg before [{:?}, {:?}, 0x{:x}]",
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
                req_chunk = self.merge_segments(seg.as_mut().unwrap(), self.seg.as_mut().unwrap());
            }

            // Checks whether there is segment, which is right after alloced mem
            let mut prev_seg = ptr::null_mut();
            let mut seg = self.seg;
            while !seg.is_null() && (*seg).base != alloced_base.add(alloced_size) {
                prev_seg = seg;
                seg = (*seg).next;
            }
            if !seg.is_null() {
                dlverbose!(
                    "DL APPEND: find seg after [{:?}, {:?}, 0x{:x}]",
                    (*seg).base,
                    (*seg).end(),
                    (*seg).size
                );
                let next_seg = (*self.seg).next;
                req_chunk = self.merge_segments(self.seg.as_mut().unwrap(), seg.as_mut().unwrap());
                self.seg = next_seg;
            }
        }
        req_chunk
    }

    /// If new requested size is less then old size,
    /// then just crops existen chunk and returns its memory.
    /// In other case checks whether existen chunk can be extended up, so that new
    /// chunk size will be bigger then requested.
    /// If so, then extends it and crops to requested size.
    /// In other case we need other chunk, which have suit size, so malloc is used.
    /// All data from old chunk copied to new.
    pub unsafe fn realloc(&mut self, old_mem: *mut u8, req_size: usize) -> *mut u8 {
        dlverbose!("{}", VERBOSE_DEL);
        dlverbose!(
            "DL REALLOC CALL: old_mem={:?} req_size=0x{:x}",
            old_mem,
            req_size
        );

        self.check_malloc_state();

        if req_size >= self.max_request() {
            return ptr::null_mut();
        }

        // Separate handling for memory which was allocated in
        // static buffer [Dlmalloc::sbuff].
        let sbuff = &mut self.sbuff as *mut u8;
        if old_mem >= sbuff && old_mem <= sbuff.add(SBUFF_SIZE) {
            let offset = old_mem as usize - sbuff as usize;
            let idx = Dlmalloc::sbuff_offset_to_idx(offset);
            let size = Dlmalloc::sbuff_idx_to_size(idx);
            dlassert!(idx < SBUFF_IDX_MAX);
            dlverbose!("DL REALLOC: in sbuff idx={} size=0x{:x}", idx, size);

            if size >= req_size {
                dlverbose!("DL REALLOC: use old sbuff cell");
                return old_mem;
            }

            self.sbuff_mask &= !(1 << idx);
            let new_mem = self.malloc_internal(req_size, true);
            dlverbose!("DL REALLOC: new mem {:?}", new_mem);
            ptr::copy_nonoverlapping(old_mem, new_mem, size);
            return new_mem;
        }

        let req_chunk_size = self.mem_to_chunk_size(req_size);
        let old_chunk = Chunk::from_mem(old_mem);
        let old_chunk_size = Chunk::size(old_chunk);
        let old_mem_size = old_chunk_size - PTR_SIZE;

        let mut chunk = old_chunk;
        let mut chunk_size = old_chunk_size;

        dlverbose!(
            "REALLOC: oldmem={:?} old_mem_size=0x{:x} req_size=0x{:x}",
            old_mem,
            old_mem_size,
            req_size
        );

        dlassert!(Chunk::cinuse(chunk));
        dlassert!(chunk != self.top && chunk != self.dv);

        if req_chunk_size == chunk_size {
            old_mem
        } else if req_chunk_size < chunk_size {
            // requested mem < then old - so crop chunk and free the remainder
            self.crop_chunk(chunk, chunk, req_chunk_size, false);
            let next_chunk = Chunk::next(chunk);
            dlassert!(!Chunk::cinuse(next_chunk));
            self.extend_free_chunk(next_chunk, false);
            self.free_chunk(next_chunk);
            old_mem
        } else if self.get_extended_up_chunk_size(chunk) >= req_chunk_size {
            // We can use next free chunk
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
                self.unlink_chunk(next_chunk);
            }

            let mut remainder_size = chunk_size + next_chunk_size - req_chunk_size;
            dlassert!(remainder_size % MALIGN == 0);

            let mut remainder_chunk;
            if remainder_size > 0 {
                remainder_chunk = Chunk::minus_offset(Chunk::next(next_chunk), remainder_size);
                (*remainder_chunk).head = remainder_size | PINUSE;
                (*Chunk::next(remainder_chunk)).prev_chunk_size = remainder_size;
                dlassert!(!Chunk::pinuse(Chunk::next(remainder_chunk)));
            } else {
                remainder_chunk = ptr::null_mut();
            }

            let chunk_size = chunk_size + next_chunk_size - remainder_size;

            if remainder_size < MIN_CHUNK_SIZE {
                // In this case we set top or dv as null
                remainder_chunk = ptr::null_mut();
                remainder_size = 0;
            }

            if next_chunk == self.top {
                self.top = remainder_chunk;
                self.topsize = remainder_size;
            } else if next_chunk == self.dv {
                self.dv = remainder_chunk;
                self.dvsize = remainder_size;
            } else if remainder_size >= MIN_CHUNK_SIZE {
                self.insert_chunk(remainder_chunk, remainder_size);
            }

            (*chunk).head = chunk_size | CINUSE | prev_in_use;
            (*Chunk::next(chunk)).head |= PINUSE;

            self.check_malloc_state();
            old_mem
        } else {
            // Alloc new mem and copy all data to it
            let new_mem = self.malloc_internal(req_size, false);
            if new_mem.is_null() {
                return new_mem;
            }

            let new_chunk = Chunk::from_mem(new_mem);
            let new_mem_size = Chunk::size(new_chunk) - PTR_SIZE;
            dlassert!(new_mem_size >= old_mem_size);

            dlverbose!(
                "REALLOC: copy data from [{:?}, 0x{:x?}] to [{:?}, 0x{:x?}]",
                old_mem,
                old_mem_size,
                new_mem,
                new_mem_size
            );

            ptr::copy_nonoverlapping(old_mem, new_mem, old_mem_size);

            chunk = self.extend_free_chunk(chunk, false);
            self.free_chunk(chunk);

            self.check_malloc_state();
            new_mem
        }
    }

    /// Crops `chunk` so that it will have addr `new_chunk_pos`
    /// and size which `new_chunk_size` <= size <= `new_chunk_size` + [MIN_CHUNK_SIZE]
    /// If there is remainders, then makes them as separate free chunks.
    /// ````
    ///         new_chunk_pos            new_chunk_pos + new_chunk_size
    ///         |                            |
    /// [-------(----------------------------)--------]
    /// |
    /// chunk
    /// ````
    /// Will be transformed to:
    /// ````
    ///   Before remainder     After remainder
    ///  /                                   \
    /// [------][----------------------------][-------]
    ///         |
    ///         chunk
    /// ````
    /// If `can_insert` is true, then inserts the remainders,
    /// but in that case you must be shure that remainders cannot be extended:
    /// i.e there is no neighbor free chunks.
    /// Cropped chunk is marked as [CINUSE].
    /// Returns whether there is after remainder.
    unsafe fn crop_chunk(
        &mut self,
        mut chunk: *mut Chunk,
        new_chunk_pos: *mut Chunk,
        new_chunk_size: usize,
        can_insert: bool,
    ) -> bool {
        let mut chunk_size = Chunk::size(chunk);
        dlverbose!(
            "DL CROP: original chunk [{:?}, {:x?}], to new [{:?}, {:x?}]",
            chunk,
            chunk_size,
            new_chunk_pos,
            new_chunk_size
        );

        dlassert!(new_chunk_size % MALIGN == 0);
        dlassert!(MIN_CHUNK_SIZE <= new_chunk_size);
        dlassert!(new_chunk_pos as usize % MALIGN == 0);
        dlassert!(new_chunk_pos >= chunk);
        dlassert!(
            Chunk::plus_offset(chunk, chunk_size)
                >= Chunk::plus_offset(new_chunk_pos, new_chunk_size)
        );

        let mut prev_in_use = if Chunk::pinuse(chunk) { PINUSE } else { 0 };
        if new_chunk_pos != chunk {
            let remainder_size = new_chunk_pos as usize - chunk as usize;
            let remainder = chunk;
            dlassert!(remainder_size >= MALIGN);

            chunk_size -= remainder_size;
            (*remainder).head = remainder_size | prev_in_use;
            self.set_top_or_dv(chunk, new_chunk_pos, chunk_size);
            if can_insert && remainder_size >= MIN_CHUNK_SIZE {
                dlassert!(Chunk::pinuse(remainder));
                self.insert_chunk(remainder, remainder_size);
            }

            dlverbose!("CROP: before rem [{:?}, {:x?}]", remainder, remainder_size);

            chunk = new_chunk_pos;
            (*chunk).prev_chunk_size = remainder_size;
            prev_in_use = 0;
        }

        dlassert!(new_chunk_pos == chunk);
        dlassert!(chunk_size >= new_chunk_size);

        let has_after_rem;
        let next_chunk = Chunk::plus_offset(chunk, chunk_size);
        if chunk_size >= new_chunk_size + MALIGN {
            let mut remainder_size = chunk_size - new_chunk_size;
            let mut remainder = Chunk::plus_offset(chunk, new_chunk_size);
            dlverbose!("CROP: after rem [{:?}, {:x?}]", remainder, remainder_size);

            (*remainder).head = remainder_size | PINUSE;
            (*Chunk::next(remainder)).head &= !PINUSE;
            (*Chunk::next(remainder)).prev_chunk_size = remainder_size;

            if remainder_size < MIN_CHUNK_SIZE {
                dlassert!(remainder_size == MALIGN);
                // If remainder is half chunk, then if `chunk` is top or dv,
                // they must be set as null.
                remainder_size = 0;
                remainder = ptr::null_mut();
            }

            if self.set_top_or_dv(chunk, remainder, remainder_size) {
                dlassert!(Chunk::cinuse(next_chunk));
            } else if can_insert && remainder_size >= MIN_CHUNK_SIZE {
                dlassert!(Chunk::cinuse(next_chunk));
                self.insert_chunk(remainder, remainder_size);
            }

            chunk_size = new_chunk_size;
            has_after_rem = true;
        } else {
            dlassert!(chunk_size == new_chunk_size);
            (*next_chunk).head |= PINUSE;
            (*next_chunk).prev_chunk_size = chunk_size;
            has_after_rem = false;
        }

        dlassert!(chunk == new_chunk_pos);
        dlassert!(chunk_size == new_chunk_size);

        (*chunk).head = chunk_size | prev_in_use | CINUSE;
        self.set_top_or_dv(chunk, ptr::null_mut(), 0);

        dlverbose!("CROP: cropped chunk [{:?}, {:x?}]", chunk, chunk_size);
        has_after_rem
    }

    /// When user want alignment, which is bigger then [MALIGN],
    /// then we just use [Dlmalloc::malloc_internal] for bigger than requested size.
    /// After that we crop malloced chunk, so that returned memory is aligned as need.
    /// Remainder is stored in smallbins or tree.
    pub unsafe fn memalign(&mut self, mut alignment: usize, req_size: usize) -> *mut u8 {
        dlverbose!("{}", VERBOSE_DEL);
        dlverbose!("MEMALIGN: align={:x?}, size={:x?}", alignment, req_size);

        self.check_malloc_state();
        dlassert!(alignment >= MIN_CHUNK_SIZE);

        if req_size >= self.max_request() - alignment {
            return ptr::null_mut();
        }

        let req_chunk_size = self.mem_to_chunk_size(req_size);
        let size_to_alloc = req_chunk_size + alignment;
        let mut mem = self.malloc_internal(size_to_alloc, false);
        if mem.is_null() {
            return mem;
        }

        let mut chunk = Chunk::from_mem(mem);
        dlverbose!("MEMALIGN: chunk[{:?}, {:x?}]", chunk, Chunk::size(chunk));

        dlassert!(Chunk::pinuse(chunk) && Chunk::cinuse(chunk));

        mem = align_up(mem as usize, alignment) as *mut u8;
        let aligned_chunk = Chunk::from_mem(mem);

        if self.crop_chunk(chunk, aligned_chunk, req_chunk_size, false) {
            self.extend_free_chunk(Chunk::next(aligned_chunk), true);
        }
        if chunk != aligned_chunk && Chunk::size(chunk) >= MIN_CHUNK_SIZE {
            self.insert_chunk(chunk, Chunk::size(chunk));
        }

        self.check_cinuse_chunk(aligned_chunk);
        self.check_malloc_state();
        mem
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
    /// `seg1` info chunk and border chunks will be free-extended
    /// and returns this extanded free chunk.
    unsafe fn merge_segments(&mut self, seg1: &mut Segment, seg2: &mut Segment) -> *mut Chunk {
        dlassert!(seg1.end() == seg2.base);
        dlassert!(seg1.size % MALIGN == 0);
        dlassert!(seg2.size % MALIGN == 0);
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
        }

        let chunk = self.extend_free_chunk(seg1_info_chunk, false);
        self.check_top_chunk(self.top);
        chunk
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
        dlassert!(tsize % MALIGN == 0);

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

        if remainder_size < MALIGN {
            dlassert!(remainder_size == 0);
            (*chunk).head = size | PINUSE | CINUSE;
            (*Chunk::next(chunk)).head |= PINUSE;
        } else {
            // use part and set remainder as dv if can
            (*chunk).head = size | PINUSE | CINUSE;
            let remainder = Chunk::next(chunk);
            (*remainder).head = remainder_size | PINUSE;
            Chunk::set_next_chunk_prev_size(remainder, remainder_size);
            if remainder_size >= MIN_CHUNK_SIZE {
                self.replace_dv(remainder, remainder_size);
            }
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
        if rsize < MALIGN {
            dlassert!(rsize == 0);
            (*vc).head = size | CINUSE | PINUSE;
            (*Chunk::next(vc)).head |= PINUSE;
        } else {
            (*vc).head = size | CINUSE | PINUSE;
            (*r).head = rsize | PINUSE;
            Chunk::set_next_chunk_prev_size(r, rsize);
            if rsize >= MIN_CHUNK_SIZE {
                self.insert_chunk(r, rsize);
            }
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
        dlassert!(size >= MIN_CHUNK_SIZE);
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
        dlassert!(size >= MIN_CHUNK_SIZE);
        dlassert!(!Chunk::cinuse(chunk));

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

    /// Inserts large free chunk in tree
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
    unsafe fn unlink_chunk(&mut self, chunk: *mut Chunk) {
        let size = Chunk::size(chunk);
        if size < MIN_CHUNK_SIZE {
            dlassert!(size == MALIGN);
        } else if self.is_chunk_small(size) {
            self.unlink_small_chunk(chunk)
        } else {
            self.unlink_large_chunk(chunk as *mut TreeChunk);
        }
    }

    /// Unlinks small free chunk from small chunks list
    unsafe fn unlink_small_chunk(&mut self, chunk: *mut Chunk) {
        let size = Chunk::size(chunk);
        dlverbose!(
            "ALLOC: unlink small chunk[{:?}, {:?}, 0x{:x}]",
            chunk,
            Chunk::next(chunk),
            size
        );
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
        let size = Chunk::size(TreeChunk::chunk(chunk));
        dlverbose!(
            "ALLOC: unlink large chunk[{:?}, {:?}, 0x{:x}]",
            chunk,
            TreeChunk::next(chunk),
            size
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
    /// Because two neighbor chunks cannot be both free,
    /// we must merge `chunk` with all free neighbor chunks.
    /// `can_insert` arg controls whether we have to insert
    /// the result free chunk into list/tree (if it isn't top or dv).
    unsafe fn extend_free_chunk(&mut self, mut chunk: *mut Chunk, can_insert: bool) -> *mut Chunk {
        dlverbose!("DL EXTEND: chunk[{:?}, {:#x}]", chunk, Chunk::size(chunk));

        dlassert!(Chunk::size(chunk) >= MALIGN);

        // try join prev chunk
        if !Chunk::pinuse(chunk) {
            let curr_chunk_size = Chunk::size(chunk);
            let prev_chunk = Chunk::prev(chunk);
            let prev_chunk_size = Chunk::size(prev_chunk);
            dlassert!(Chunk::pinuse(prev_chunk));

            dlverbose!(
                "extend: add before chunk[{:?}, 0x{:x}] {}",
                prev_chunk,
                prev_chunk_size,
                self.is_top_or_dv(prev_chunk)
            );

            if prev_chunk == self.top {
                self.topsize += Chunk::size(chunk);
            } else if prev_chunk == self.dv {
                self.dvsize += Chunk::size(chunk);
            } else {
                self.unlink_chunk(prev_chunk);
            }

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
            dlassert!(chunk != self.top);

            if next_chunk == self.top {
                self.top = chunk;
                self.topsize += Chunk::size(chunk);
                if chunk == self.dv {
                    // top eats dv
                    self.dv = ptr::null_mut();
                    self.dvsize = 0;
                }
                (*chunk).head = self.topsize | PINUSE;
            } else if next_chunk == self.dv {
                self.dv = chunk;
                self.dvsize += Chunk::size(chunk);
                (*chunk).head = self.dvsize | PINUSE;
            } else {
                self.unlink_chunk(next_chunk);
                (*chunk).head = (Chunk::size(chunk) + Chunk::size(next_chunk)) | PINUSE;
                if chunk == self.dv {
                    self.dvsize = Chunk::size(chunk);
                } else if can_insert {
                    self.insert_chunk(chunk, Chunk::size(chunk));
                }
            }
            (*Chunk::next(chunk)).prev_chunk_size = Chunk::size(chunk);
        } else {
            (*chunk).head &= !CINUSE;
            (*next_chunk).head &= !PINUSE;
            (*next_chunk).prev_chunk_size = Chunk::size(chunk);
            if can_insert
                && chunk != self.top
                && chunk != self.dv
                && Chunk::size(chunk) >= MIN_CHUNK_SIZE
            {
                self.insert_chunk(chunk, Chunk::size(chunk));
            }
        }

        chunk
    }

    /// Tries to free `chunk`, see more in [Dlmalloc::free]
    /// If `chunk` has no intervals which suit to be freed by system,
    /// then just insert `chunk` if need.
    unsafe fn free_chunk(&mut self, chunk: *mut Chunk) {
        dlassert!(Chunk::pinuse(chunk));
        dlassert!(Chunk::cinuse(Chunk::next(chunk)));

        dlverbose!("DL FREE: chunk[{:?}, {:?}]", chunk, Chunk::next(chunk));

        let chunk_size = Chunk::size(chunk);
        dlassert!(chunk_size >= MALIGN);

        if chunk_size + SEG_INFO_SIZE < DEFAULT_GRANULARITY {
            if chunk != self.top && chunk != self.dv && chunk_size >= MIN_CHUNK_SIZE {
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

        let seg_base = (*seg).base;
        let seg_end = (*seg).base.add((*seg).size);
        dlassert!(mem_to_free_end < seg_end);
        dlassert!(seg_base as usize % MALIGN == 0);

        dlverbose!("DL FREE: holding seg[{:?}, {:?}]", seg_base, seg_end);
        dlverbose!("DL FREE: prev seg = {:?}", prev_seg);

        let next_chunk = Chunk::next(chunk);

        if mem_to_free != seg_base {
            dlassert!(Chunk::pinuse(chunk));

            // If there is chunk(s) between @mem_to_free and @seg_begin,
            // then interval between must be at least @MIN_CHUNK_SIZE,
            // because at least one chunk must be in use.
            dlassert!(mem_to_free as usize - seg_base as usize >= MIN_CHUNK_SIZE);

            // we cannot free chunk.pred_chunk_size mem because it may be used by prev chunk mem
            mem_to_free = mem_to_free.add(PTR_SIZE);

            // additionally we need space for new segment info
            mem_to_free = mem_to_free.add(SEG_INFO_SIZE);
        }

        let mut after_seg_size = seg_end as usize - mem_to_free_end as usize;
        if after_seg_size > SEG_INFO_SIZE {
            // If there is chunk(s) between, then it must be at least [MIN_CHUNK_SIZE]
            // because at least one chunk must be in use.
            dlassert!(after_seg_size >= SEG_INFO_SIZE + MIN_CHUNK_SIZE);
        } else {
            // Next chunk is seg info chunk - so we can free it also.
            dlassert!(after_seg_size == SEG_INFO_SIZE);
            mem_to_free_end = seg_end;
        }

        dlverbose!(
            "DL FREE: mem can be freed [{:?}, {:?}]",
            mem_to_free,
            mem_to_free_end
        );

        let mem_to_free_size = mem_to_free_end as usize - mem_to_free as usize;
        let (mem_to_free, mem_to_free_size) = sys::get_free_borders(mem_to_free, mem_to_free_size);
        let mem_to_free_end = mem_to_free.add(mem_to_free_size);
        dlverbose!(
            "DL FREE: mem can be freed by system [{:?}, {:?}]",
            mem_to_free,
            mem_to_free_end,
        );

        if mem_to_free_size == 0 {
            if chunk != self.top && chunk != self.dv {
                self.insert_chunk(chunk, chunk_size);
            }
            return;
        }

        let before_seg_size = mem_to_free as usize - seg_base as usize;
        let after_seg_size = seg_end as usize - mem_to_free_end as usize;

        let mut crop_chunk;
        let mut crop_chunk_size;
        if before_seg_size != 0 {
            // We crop chunk with a reserve for before remainder segment info
            crop_chunk = mem_to_free.sub(SEG_INFO_SIZE) as *mut Chunk;
            dlassert!((crop_chunk as usize - chunk as usize) >= MALIGN);
            crop_chunk_size = mem_to_free_size + SEG_INFO_SIZE;
        } else {
            crop_chunk = mem_to_free as *mut Chunk;
            crop_chunk_size = mem_to_free_size;
        }
        // If there isn't after segment remainder then we delete seg-info chunk,
        // which mustn't be cropped.
        if after_seg_size == 0 {
            dlassert!(mem_to_free_end == seg_end);
            dlassert!(Chunk::next(chunk) as *mut u8 == seg_end.sub(SEG_INFO_SIZE));
            crop_chunk_size -= SEG_INFO_SIZE;
        }

        self.crop_chunk(chunk, crop_chunk, crop_chunk_size, true);

        let next_seg = (*seg).next;
        let after_rem_pinuse = to_pinuse(after_seg_size > 0 && Chunk::pinuse((*seg).info_chunk()));

        let success = sys::free(mem_to_free, mem_to_free_size);
        dlassert!(success);

        if before_seg_size != 0 {
            let before_seg_info = self.set_segment_info(seg_base, before_seg_size, 0);

            dlverbose!(
                "DL FREE: before seg [{:?}, {:?}]",
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

        if after_seg_size != 0 {
            let after_seg_info =
                self.set_segment_info(mem_to_free_end, after_seg_size, after_rem_pinuse);

            dlverbose!(
                "DL FREE: after seg [{:?}, {:?}]",
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
    }

    /// When user call free mem, in our context it means - free one chunk.
    /// There can be already free neighbor chunks, so we extend our chunk
    /// to all free chunks around. Then if chunk is big enought we can return some memory to the system.
    /// The size of interval that can be free cannot be less then [DEFAULT_GRANULARITY]
    /// and also must satisfy system restrictions.
    /// Let's see an example:
    /// ````
    ///  Segment begin                   Can be free by system            Segment end
    ///  |                                  /              \                 |
    ///  [=========================(=======|================|====)===========]
    ///                            |                             |
    ///                            Chunk begin                   Chunk end
    /// ````
    /// Here chunk is free, and we want to return some part of chunk's memory to the system.
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
    pub unsafe fn free(&mut self, mem: *mut u8) {
        dlverbose!("{}", VERBOSE_DEL);
        dlverbose!("DL FREE CALL: mem={:?}", mem);
        self.print_segments();
        self.check_malloc_state();

        // Separate handling for memory which was allocated in
        // static buffer [Dlmalloc::sbuff].
        let sbuff = &mut self.sbuff as *mut u8;
        if mem >= sbuff && mem <= sbuff.add(SBUFF_SIZE) {
            let offset = mem as usize - sbuff as usize;
            let idx = Dlmalloc::sbuff_offset_to_idx(offset);
            dlassert!(idx < SBUFF_IDX_MAX);
            self.sbuff_mask &= !(1 << idx);
            dlverbose!(
                "DL FREE: is in sbuff cell {}, sbuff_mask={:x}",
                idx,
                self.sbuff_mask
            );
            return;
        }

        let chunk = Chunk::from_mem(mem);
        let chunk_size = Chunk::size(chunk);
        dlassert!(chunk_size >= MIN_CHUNK_SIZE);

        let chunk = self.extend_free_chunk(chunk, false);
        dlverbose!(
            "ALLOC FREE: extended chunk[{:?}, 0x{:x}] {}",
            chunk,
            Chunk::size(chunk),
            self.is_top_or_dv(chunk)
        );

        self.free_chunk(chunk);

        self.print_segments();
        self.check_malloc_state();
    }

    /// Returns static string about chunk status in context
    unsafe fn is_top_or_dv(&self, chunk: *mut Chunk) -> &'static str {
        if chunk == self.top {
            "is top"
        } else if chunk == self.dv {
            "is dv"
        } else if Chunk::size(chunk) == MALIGN {
            "is half chunk"
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

        // Prints all cells info from self.sbuff
        for i in 0..SBUFF_IDX_MAX {
            let size = Dlmalloc::sbuff_idx_to_size(i);
            if self.sbuff_mask & (1 << i) == 0 {
                dlverbose!("[{}, -]", size);
            } else {
                dlverbose!(
                    "[{}, {}]",
                    size,
                    Dlmalloc::debug_mem_sum(
                        self.sbuff
                            .as_mut_ptr()
                            .add(Dlmalloc::sbuff_idx_to_offset(i)),
                        size
                    )
                );
            }
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
                    "chunk [{:?}, {:?}]{}{} {}, sum = {}",
                    chunk,
                    Chunk::next(chunk),
                    if Chunk::cinuse(chunk) { "c" } else { "" },
                    if Chunk::pinuse(chunk) { "p" } else { "" },
                    self.is_top_or_dv(chunk),
                    if Chunk::cinuse(chunk) {
                        Dlmalloc::debug_mem_sum(Chunk::to_mem(chunk), Chunk::size(chunk) - PTR_SIZE)
                    } else {
                        0
                    }
                );
                chunk = Chunk::next(chunk);
            }

            dlverbose!(
                "info [{:?}, {:?}]{}{}",
                chunk,
                Chunk::next(chunk),
                if Chunk::cinuse(chunk) { "c" } else { "" },
                if Chunk::pinuse(chunk) { "p" } else { "" }
            );

            seg = (*seg).next;
        }

        dlverbose!(r"\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/");
    }

    // Sanity checks

    unsafe fn update_least_addr(&mut self, addr: *mut u8) {
        if !DL_DEBUG {
            return;
        }
        if self.least_addr.is_null() || addr < self.least_addr {
            self.least_addr = addr;
        }
    }

    unsafe fn check_any_chunk(&self, p: *mut Chunk) {
        if !DL_DEBUG {
            return;
        }

        dlassert!(!p.is_null());
        dlassert!(p as usize % MALIGN == 0);
        dlassert!(Chunk::size(p) % MALIGN == 0);
        dlassert!(Chunk::to_mem(p) as usize % MALIGN == 0);
        dlassert!(p as *mut u8 >= self.least_addr);

        if !DL_CHECKS {
            return;
        }

        // Checks that `p` doesn't intersect some other chunk
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
        if !DL_DEBUG {
            return;
        }
        if self.top.is_null() {
            dlassert!(self.topsize == 0);
            return;
        }
        self.check_any_chunk(p);

        if !DL_CHECKS {
            return;
        }

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
        if !DL_DEBUG {
            return;
        }
        if mem.is_null() {
            return;
        }
        let p = Chunk::from_mem(mem);
        let sz = Chunk::size(p);
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
        if !DL_DEBUG {
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
            if sz >= MIN_CHUNK_SIZE {
                dlassert!((*(*p).next).prev == p);
                dlassert!((*(*p).prev).next == p);
            } else {
                dlassert!(sz == MALIGN);
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

        // Bypasses all segments and chunks
        let mut seg = self.seg;
        while !seg.is_null() {
            let mut chunk = (*seg).base as *mut Chunk;
            let last_chunk = Segment::top(seg).sub(SEG_INFO_SIZE);
            while (chunk as *mut u8) < last_chunk {
                if chunk != self.top && chunk != self.dv {
                    dlassert!(self.top < chunk || self.top >= Chunk::next(chunk));
                    dlassert!(self.dv < chunk || self.dv >= Chunk::next(chunk));
                    dlassert!(
                        Chunk::cinuse(chunk)
                            || Chunk::size(chunk) == MALIGN
                            || self.bin_find(chunk)
                    );
                }
                dlassert!(
                    Chunk::cinuse(chunk)
                        || Chunk::size(chunk) < 2 * DEFAULT_GRANULARITY + SEG_INFO_SIZE
                );
                dlassert!(Chunk::pinuse(chunk) || !Chunk::cinuse(Chunk::prev(chunk)));
                let next = Chunk::next(chunk);
                dlassert!(Chunk::pinuse(next) == Chunk::cinuse(chunk));
                dlassert!(Chunk::cinuse(chunk) || Chunk::cinuse(next));
                dlassert!(
                    Chunk::pinuse(chunk)
                        || Chunk::size(Chunk::prev(chunk)) == (*chunk).prev_chunk_size
                );

                chunk = next;
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
