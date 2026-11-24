use {
    crate::execution_budget::{
        MAX_CALL_DEPTH, MAX_HEAP_FRAME_BYTES, MAX_INSTRUCTION_STACK_DEPTH_SIMD_0268,
        MIN_HEAP_FRAME_BYTES,
    },
    solana_sbpf::{aligned_memory::AlignedMemory, ebpf::HOST_ALIGN, vm::CallFrame},
    std::{
        array,
        ops::{Deref, DerefMut},
    },
};

trait Reset {
    fn reset(&mut self, len: usize);
}

struct Pool<T: Reset, const SIZE: usize> {
    items: [Option<T>; SIZE],
    next_empty: usize,
}

impl<T: Reset, const SIZE: usize> Pool<T, SIZE> {
    fn new(items: [T; SIZE]) -> Self {
        Self {
            items: items.map(|i| Some(i)),
            next_empty: SIZE,
        }
    }

    fn len(&self) -> usize {
        SIZE
    }

    fn get(&mut self) -> Option<T> {
        if self.next_empty == 0 {
            return None;
        }
        self.next_empty = self.next_empty.saturating_sub(1);
        self.items
            .get_mut(self.next_empty)
            .and_then(|item| item.take())
    }

    fn put(&mut self, mut value: T, len: usize) -> bool {
        self.items
            .get_mut(self.next_empty)
            .map(|item| {
                value.reset(len);
                item.replace(value);
                self.next_empty = self.next_empty.saturating_add(1);
                true
            })
            .unwrap_or(false)
    }
}

impl Reset for AlignedMemory<{ HOST_ALIGN }> {
    fn reset(&mut self, len: usize) {
        let slice = self.as_slice_mut();
        let len = len.min(slice.len());
        if let Some(head) = slice.get_mut(..len) {
            // Most chunks are still zero from the previous reset. On x86_64,
            // memset of these buffers is limited by the stores, so only the
            // nonzero chunks are cleared. On aarch64, glibc's memset clears
            // whole cache lines with DC ZVA, which is faster than reading the
            // buffer to find them.
            if cfg!(target_arch = "x86_64") {
                clear_nonzero_chunks(head);
            } else {
                head.fill(0);
            }
        }
    }
}

/// Size of the chunks that [`clear_nonzero_chunks`] checks and clears.
const RESET_CHUNK_SIZE: usize = 4096;

/// Sets all bytes of `buf` to zero, writing only the 4 KiB chunks that contain
/// a nonzero byte.
///
/// A program usually writes to a few chunks of its stack and heap, and the
/// other chunks are still zero from the previous reset.
fn clear_nonzero_chunks(buf: &mut [u8]) {
    for chunk in buf.chunks_mut(RESET_CHUNK_SIZE) {
        if !is_zeroed(chunk) {
            chunk.fill(0);
        }
    }
}

/// Returns whether all bytes of `bytes` are zero.
///
/// The bytes are read as u64 words, 64 words (512 bytes) at a time, and the
/// check stops at the first block that has a nonzero word.
fn is_zeroed(bytes: &[u8]) -> bool {
    let (blocks, rest) = bytes.as_chunks::<512>();
    blocks.iter().all(|block| {
        let (words, _) = block.as_chunks::<8>();
        words
            .iter()
            .fold(0, |acc, word| acc | u64::from_ne_bytes(*word))
            == 0
    }) && rest.iter().all(|byte| *byte == 0)
}

pub struct CallFrameBuffer(Box<[CallFrame; MAX_CALL_DEPTH]>);

impl Default for CallFrameBuffer {
    fn default() -> Self {
        let mut mem = Box::<[CallFrame; MAX_CALL_DEPTH]>::new_uninit();
        let ptr = mem.as_mut_ptr().cast::<CallFrame>();
        for i in 0..MAX_CALL_DEPTH {
            unsafe { ptr.add(i).write(CallFrame::default()) }
        }
        Self(unsafe { mem.assume_init() })
    }
}

impl Reset for CallFrameBuffer {
    fn reset(&mut self, _len: usize) {
        self.fill(CallFrame::default())
    }
}

impl Deref for CallFrameBuffer {
    type Target = [CallFrame];

    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

impl DerefMut for CallFrameBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut_slice()
    }
}

pub struct VmMemoryPool {
    stack: Pool<AlignedMemory<{ HOST_ALIGN }>, MAX_INSTRUCTION_STACK_DEPTH_SIMD_0268>,
    heap: Pool<AlignedMemory<{ HOST_ALIGN }>, MAX_INSTRUCTION_STACK_DEPTH_SIMD_0268>,
    call_frame: Pool<CallFrameBuffer, MAX_INSTRUCTION_STACK_DEPTH_SIMD_0268>,
}

/// Measurement only. A zero-filled allocation starts with pages that the
/// process has not written yet, and reads from such a page go to the kernel's
/// shared zero page. In a long-running validator, programs with deep call
/// stacks or large heaps write to the pages of the pooled buffers sooner or
/// later. Write a nonzero byte every 4 KiB and then clear the buffer, so that
/// every page has its own memory from the start.
fn write_every_page(mut memory: AlignedMemory<{ HOST_ALIGN }>) -> AlignedMemory<{ HOST_ALIGN }> {
    let bytes = memory.as_slice_mut();
    for byte in bytes.iter_mut().step_by(4096) {
        *byte = 1;
    }
    // The allocation is known to be zero, so keep the compiler from removing
    // the writes above and the clear below.
    let bytes = std::hint::black_box(bytes);
    bytes.fill(0);
    memory
}

impl VmMemoryPool {
    pub fn new() -> Self {
        Self {
            stack: Pool::new(array::from_fn(|_| {
                #[allow(clippy::arithmetic_side_effects)]
                write_every_page(AlignedMemory::zero_filled(
                    solana_sbpf::vm::get_stack_frame_size() * MAX_CALL_DEPTH,
                ))
            })),
            heap: Pool::new(array::from_fn(|_| {
                write_every_page(AlignedMemory::zero_filled(MAX_HEAP_FRAME_BYTES as usize))
            })),
            call_frame: Pool::new(array::from_fn(|_| CallFrameBuffer::default())),
        }
    }

    pub fn stack_len(&self) -> usize {
        self.stack.len()
    }

    pub fn heap_len(&self) -> usize {
        self.heap.len()
    }

    #[allow(clippy::arithmetic_side_effects)]
    pub fn get_stack(&mut self, size: usize) -> AlignedMemory<{ HOST_ALIGN }> {
        debug_assert!(size == solana_sbpf::vm::get_stack_frame_size() * MAX_CALL_DEPTH);
        self.stack
            .get()
            .unwrap_or_else(|| AlignedMemory::zero_filled(size))
    }

    pub fn put_stack(&mut self, stack: AlignedMemory<{ HOST_ALIGN }>) -> bool {
        let len = stack.len();
        self.stack.put(stack, len)
    }

    pub fn get_heap(&mut self, heap_size: u32) -> AlignedMemory<{ HOST_ALIGN }> {
        debug_assert!((MIN_HEAP_FRAME_BYTES..=MAX_HEAP_FRAME_BYTES).contains(&heap_size));
        self.heap
            .get()
            .unwrap_or_else(|| AlignedMemory::zero_filled(MAX_HEAP_FRAME_BYTES as usize))
    }

    pub fn put_heap(&mut self, heap: AlignedMemory<{ HOST_ALIGN }>, mapped_len: usize) -> bool {
        let heap_size = heap.len();
        debug_assert!(
            heap_size >= MIN_HEAP_FRAME_BYTES as usize
                && heap_size <= MAX_HEAP_FRAME_BYTES as usize
        );
        debug_assert!(mapped_len <= heap_size);
        self.heap.put(heap, mapped_len.min(heap_size))
    }

    pub fn get_call_frames(&mut self) -> CallFrameBuffer {
        self.call_frame.get().unwrap_or_default()
    }

    pub fn put_call_frames(&mut self, call_frame: CallFrameBuffer) -> bool {
        self.call_frame.put(call_frame, 0)
    }
}

#[cfg(test)]
mod test {
    use {
        super::*,
        rand::{Rng, SeedableRng, rngs::StdRng},
    };

    #[derive(Debug, Eq, PartialEq)]
    struct Item(u8, u8);
    impl Reset for Item {
        fn reset(&mut self, _len: usize) {
            self.1 = 0;
        }
    }

    #[test]
    fn test_heap_shrink_then_grow_stays_zeroed() {
        let mut pool = VmMemoryPool::new();
        let big = MAX_HEAP_FRAME_BYTES;
        let small = MIN_HEAP_FRAME_BYTES;

        let mut heap = pool.get_heap(big);
        heap.as_slice_mut().fill(0xaa);
        assert!(pool.put_heap(heap, big as usize));

        let mut heap = pool.get_heap(small);
        heap.as_slice_mut()
            .get_mut(..small as usize)
            .unwrap()
            .fill(0xbb);
        assert!(pool.put_heap(heap, small as usize));

        let heap = pool.get_heap(big);
        assert!(heap.as_slice().iter().all(|byte| *byte == 0));
    }

    /// Clears the first `len` bytes of copies of `bytes` with
    /// `clear_nonzero_chunks` and with `Reset`, and checks both against
    /// `fill(0)`.
    fn assert_reset_matches_fill(bytes: &[u8], len: usize) {
        let head = len.min(bytes.len());
        let mut expected = bytes.to_vec();
        expected.get_mut(..head).unwrap().fill(0);

        let mut cleared = bytes.to_vec();
        clear_nonzero_chunks(cleared.get_mut(..head).unwrap());
        assert_eq!(cleared, expected);

        let mut memory = AlignedMemory::<{ HOST_ALIGN }>::from_slice(bytes);
        memory.reset(len);
        assert_eq!(memory.as_slice(), expected.as_slice());
    }

    #[test]
    #[allow(clippy::arithmetic_side_effects)]
    fn test_reset_single_nonzero_byte() {
        // A full chunk, then a partial chunk of one 512-byte block and a tail
        // that does not end on a word.
        let size = RESET_CHUNK_SIZE + 512 + 77;
        for position in 0..size {
            let mut bytes = vec![0; size];
            *bytes.get_mut(position).unwrap() = 1;
            assert_reset_matches_fill(&bytes, size);
            // The byte is past `len` and must stay.
            assert_reset_matches_fill(&bytes, position);
        }
    }

    #[test]
    #[allow(clippy::arithmetic_side_effects)]
    fn test_reset_matches_fill() {
        let mut rng = StdRng::seed_from_u64(0);
        let stack_size = solana_sbpf::vm::get_stack_frame_size() * MAX_CALL_DEPTH;
        let heap_size = MAX_HEAP_FRAME_BYTES as usize;
        for _ in 0..500 {
            let size = match rng.random_range(0..4) {
                0 => stack_size,
                1 => heap_size,
                _ => rng.random_range(0..4 * RESET_CHUNK_SIZE),
            };
            let mut bytes = vec![0; size];
            // Single bytes and runs of random bytes, like the stack frames and
            // heap allocations that a program writes.
            if size > 0 {
                for _ in 0..rng.random_range(0..8) {
                    let start = rng.random_range(0..size);
                    if rng.random_bool(0.5) {
                        *bytes.get_mut(start).unwrap() = rng.random_range(1..=u8::MAX);
                    } else {
                        let end = rng.random_range(start..=size.min(start + 2 * RESET_CHUNK_SIZE));
                        rng.fill(bytes.get_mut(start..end).unwrap());
                    }
                }
            }
            // `reset` clamps a `len` past the end of the memory.
            let len = rng.random_range(0..=size + 8);
            assert_reset_matches_fill(&bytes, len);
        }
    }

    #[test]
    fn test_pool() {
        let mut pool = Pool::<Item, 2>::new([Item(0, 1), Item(1, 1)]);
        assert_eq!(pool.get(), Some(Item(1, 1)));
        assert_eq!(pool.get(), Some(Item(0, 1)));
        assert_eq!(pool.get(), None);
        pool.put(Item(1, 1), 0);
        assert_eq!(pool.get(), Some(Item(1, 0)));
        pool.put(Item(2, 2), 0);
        pool.put(Item(3, 3), 0);
        assert!(!pool.put(Item(4, 4), 0));
        assert_eq!(pool.get(), Some(Item(3, 0)));
        assert_eq!(pool.get(), Some(Item(2, 0)));
        assert_eq!(pool.get(), None);
    }
}
