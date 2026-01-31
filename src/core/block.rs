//! Block abstractions for PagedAttention.
//!
//! PagedAttention divides KV cache into fixed-size blocks, similar to
//! how operating systems manage virtual memory with pages.
//!
//! See [`docs/paged_attention.md`] for detailed documentation.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::error::{Error, Result};

/// Default block size (tokens per block).
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// Compute a cumulative hash for a block of tokens including its prefix chain.
///
/// Used for prefix caching to identify shared prefixes. The hash includes the
/// parent block's hash to create a chain, ensuring that blocks at the same
/// position with different prefixes have different hashes.
///
/// # Arguments
///
/// * `token_ids` - Slice of token IDs for this block
/// * `parent_hash` - Hash of the previous block in the chain (None for first block)
///
/// # Returns
///
/// Cumulative hash value that uniquely identifies this block AND all previous blocks
///
/// # Example
///
/// ```
/// use nano_vllm::core::block::hash_token_block;
///
/// let tokens = [1u32, 2, 3, 4];
/// let hash1 = hash_token_block(&tokens, None);
/// let hash2 = hash_token_block(&tokens, Some(hash1));
///
/// // Same tokens but different prefix chain -> different hash
/// assert_ne!(hash1, hash2);
/// ```
pub fn hash_token_block(token_ids: &[u32], parent_hash: Option<u64>) -> u64 {
    let mut hasher = DefaultHasher::new();

    // Include parent hash if present
    if let Some(ph) = parent_hash {
        ph.hash(&mut hasher);
    }

    // Hash all token IDs
    for &token in token_ids {
        token.hash(&mut hasher);
    }

    hasher.finish()
}

/// A fixed-size chunk of KV cache memory.
///
/// Each block stores KV states for `block_size` tokens.
/// Blocks are the unit of allocation in the [`BlockManager`](super::block_manager::BlockManager).
///
/// # Fields
///
/// * `block_id` - Unique identifier for this physical block
/// * `block_size` - Number of tokens this block can hold
/// * `ref_count` - Reference count for prefix sharing
/// * `prefix_hash` - Hash of tokens stored in this block (for prefix caching)
/// * `is_full` - Whether the block is completely filled
#[derive(Debug, Clone)]
pub struct Block {
    /// Unique identifier for this physical block.
    block_id: usize,
    /// Number of tokens this block can hold.
    block_size: usize,
    /// Reference count for prefix sharing.
    ref_count: usize,
    /// Hash of tokens stored in this block (for prefix caching).
    prefix_hash: Option<u64>,
    /// Whether the block is completely filled.
    is_full: bool,
}

impl Block {
    /// Create a new block with the given ID.
    ///
    /// # Arguments
    ///
    /// * `block_id` - Unique identifier for this block
    /// * `block_size` - Number of tokens this block can hold
    pub fn new(block_id: usize, block_size: usize) -> Self {
        Self {
            block_id,
            block_size,
            ref_count: 1,
            prefix_hash: None,
            is_full: false,
        }
    }

    /// Create a new block with default block size.
    pub fn with_default_size(block_id: usize) -> Self {
        Self::new(block_id, DEFAULT_BLOCK_SIZE)
    }

    /// Get the block ID.
    pub fn block_id(&self) -> usize {
        self.block_id
    }

    /// Get the block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get the current reference count.
    pub fn ref_count(&self) -> usize {
        self.ref_count
    }

    /// Get the prefix hash.
    pub fn prefix_hash(&self) -> Option<u64> {
        self.prefix_hash
    }

    /// Check if the block is full.
    pub fn is_full(&self) -> bool {
        self.is_full
    }

    /// Set the prefix hash.
    pub fn set_prefix_hash(&mut self, hash: u64) {
        self.prefix_hash = Some(hash);
    }

    /// Mark the block as full.
    pub fn set_full(&mut self, is_full: bool) {
        self.is_full = is_full;
    }

    /// Increment reference count (when sharing with another sequence).
    pub fn increment_ref(&mut self) {
        self.ref_count += 1;
    }

    /// Decrement reference count.
    ///
    /// # Returns
    ///
    /// The new reference count after decrementing.
    pub fn decrement_ref(&mut self) -> usize {
        self.ref_count = self.ref_count.saturating_sub(1);
        self.ref_count
    }
}

/// Maps a sequence's logical positions to physical block IDs.
///
/// Think of this like a page table in virtual memory:
/// - Logical block index: Position in the sequence (0, 1, 2, ...)
/// - Physical block ID: Actual block in the global cache pool
///
/// Token at position `p` is stored in:
/// - Logical block: `p / block_size`
/// - Slot within block: `p % block_size`
/// - Physical block: `block_ids[p / block_size]`
///
/// # Example
///
/// ```
/// use nano_vllm::core::block::BlockTable;
///
/// let mut table = BlockTable::new(16);
/// table.append_block(5);   // Tokens 0-15
/// table.append_block(12);  // Tokens 16-31
/// table.append_block(3);   // Tokens 32-47
///
/// // Token 20 -> logical block 1 -> physical block 12
/// assert_eq!(table.get_block_id(1).unwrap(), 12);
/// ```
#[derive(Debug, Clone, Default)]
pub struct BlockTable {
    /// Physical block IDs in logical order.
    block_ids: Vec<usize>,
    /// Number of tokens per block.
    block_size: usize,
}

impl BlockTable {
    /// Create a new empty block table.
    pub fn new(block_size: usize) -> Self {
        Self {
            block_ids: Vec::new(),
            block_size,
        }
    }

    /// Create a new block table with default block size.
    pub fn with_default_size() -> Self {
        Self::new(DEFAULT_BLOCK_SIZE)
    }

    /// Get the block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get physical block ID for a logical block index.
    ///
    /// # Arguments
    ///
    /// * `logical_block_idx` - Which block in the sequence (0-indexed)
    ///
    /// # Returns
    ///
    /// Physical block ID in the global cache, or error if out of bounds.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BlockIndexOutOfBounds`] if the logical block index
    /// is not allocated.
    pub fn get_block_id(&self, logical_block_idx: usize) -> Result<usize> {
        self.block_ids
            .get(logical_block_idx)
            .copied()
            .ok_or_else(|| Error::BlockIndexOutOfBounds {
                logical_idx: logical_block_idx,
                num_blocks: self.block_ids.len(),
            })
    }

    /// Add a new physical block to the table.
    ///
    /// Called when the sequence grows and needs more blocks.
    pub fn append_block(&mut self, block_id: usize) {
        self.block_ids.push(block_id);
    }

    /// Number of blocks allocated to this sequence.
    pub fn num_blocks(&self) -> usize {
        self.block_ids.len()
    }

    /// Check if the table is empty.
    pub fn is_empty(&self) -> bool {
        self.block_ids.is_empty()
    }

    /// Get all physical block IDs for this sequence.
    pub fn get_physical_block_ids(&self) -> &[usize] {
        &self.block_ids
    }

    /// Get physical slot indices for all tokens in the sequence.
    ///
    /// Returns a list where `slot_mapping[i]` is the global slot index
    /// for token `i`. Used for writing KV to the cache.
    ///
    /// Global slot = `block_id * block_size + slot_within_block`
    ///
    /// # Arguments
    ///
    /// * `seq_len` - Number of tokens in the sequence
    ///
    /// # Returns
    ///
    /// Vector of global slot indices for each token position.
    pub fn get_slot_mapping(&self, seq_len: usize) -> Vec<usize> {
        let mut slots = Vec::with_capacity(seq_len);

        for pos in 0..seq_len {
            let logical_block = pos / self.block_size;
            let slot_in_block = pos % self.block_size;

            if let Some(&physical_block) = self.block_ids.get(logical_block) {
                let global_slot = physical_block * self.block_size + slot_in_block;
                slots.push(global_slot);
            }
        }

        slots
    }

    /// Clear all blocks from the table.
    pub fn clear(&mut self) {
        self.block_ids.clear();
    }
}

/// Compute number of blocks needed for a sequence of given length.
///
/// # Arguments
///
/// * `seq_len` - Number of tokens in the sequence
/// * `block_size` - Number of tokens per block
///
/// # Returns
///
/// Number of blocks needed to store all tokens.
///
/// # Example
///
/// ```
/// use nano_vllm::core::block::compute_num_blocks;
///
/// assert_eq!(compute_num_blocks(35, 16), 3);  // 35 tokens -> 3 blocks
/// assert_eq!(compute_num_blocks(32, 16), 2);  // 32 tokens -> 2 blocks exactly
/// assert_eq!(compute_num_blocks(0, 16), 0);   // 0 tokens -> 0 blocks
/// ```
pub fn compute_num_blocks(seq_len: usize, block_size: usize) -> usize {
    seq_len.div_ceil(block_size)
}
