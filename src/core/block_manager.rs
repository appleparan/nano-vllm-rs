//! Block manager for PagedAttention.
//!
//! The BlockManager handles allocation and deallocation of KV cache blocks,
//! similar to how an operating system manages physical memory pages.
//!
//! ## Features
//!
//! - **Free list allocation**: O(1) block allocation and deallocation
//! - **Reference counting**: Enables block sharing for prefix caching
//! - **Prefix caching**: Hash-based lookup for shared prefixes
//!
//! ## Example
//!
//! ```
//! use nano_vllm::core::block_manager::BlockManager;
//!
//! let mut manager = BlockManager::new(1024, 16);
//!
//! // Allocate a block
//! let block_id = manager.allocate().unwrap();
//!
//! // Use the block...
//!
//! // Free when done
//! manager.free(block_id);
//! ```

use std::collections::{HashMap, VecDeque};

use crate::core::block::{Block, DEFAULT_BLOCK_SIZE};
use crate::error::{Error, Result};

/// Manages allocation and deallocation of KV cache blocks.
///
/// The BlockManager maintains:
/// - A free list for O(1) allocation/deallocation
/// - Reference counting for shared blocks
/// - A prefix cache for hash-based block reuse
#[derive(Debug)]
pub struct BlockManager {
    /// All blocks indexed by block_id.
    blocks: HashMap<usize, Block>,
    /// Free block IDs (LIFO for cache locality).
    free_list: VecDeque<usize>,
    /// Prefix hash -> block_id mapping for prefix caching.
    prefix_cache: HashMap<u64, usize>,
    /// Number of tokens per block.
    block_size: usize,
    /// Total number of blocks.
    num_blocks: usize,
}

impl BlockManager {
    /// Create a new block manager with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `num_blocks` - Total number of blocks to manage
    /// * `block_size` - Number of tokens per block
    ///
    /// # Example
    ///
    /// ```
    /// use nano_vllm::core::block_manager::BlockManager;
    ///
    /// let manager = BlockManager::new(1024, 16);
    /// assert_eq!(manager.num_free_blocks(), 1024);
    /// ```
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        // Initialize all blocks in the free list
        let free_list: VecDeque<usize> = (0..num_blocks).collect();

        Self {
            blocks: HashMap::with_capacity(num_blocks),
            free_list,
            prefix_cache: HashMap::new(),
            block_size,
            num_blocks,
        }
    }

    /// Create a new block manager with default block size.
    pub fn with_default_block_size(num_blocks: usize) -> Self {
        Self::new(num_blocks, DEFAULT_BLOCK_SIZE)
    }

    /// Get the block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get the total number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Get the number of free blocks.
    pub fn num_free_blocks(&self) -> usize {
        self.free_list.len()
    }

    /// Get the number of used blocks.
    pub fn num_used_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Check if there are free blocks available.
    pub fn has_free_blocks(&self) -> bool {
        !self.free_list.is_empty()
    }

    /// Check if a specific number of blocks can be allocated.
    pub fn can_allocate(&self, num_blocks: usize) -> bool {
        self.free_list.len() >= num_blocks
    }

    /// Allocate a single block.
    ///
    /// # Returns
    ///
    /// The block ID of the allocated block, or an error if no blocks are available.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfBlocks`] if no free blocks are available.
    ///
    /// # Example
    ///
    /// ```
    /// use nano_vllm::core::block_manager::BlockManager;
    ///
    /// let mut manager = BlockManager::new(2, 16);
    ///
    /// let b1 = manager.allocate().unwrap();
    /// let b2 = manager.allocate().unwrap();
    ///
    /// // Third allocation fails
    /// assert!(manager.allocate().is_err());
    /// ```
    pub fn allocate(&mut self) -> Result<usize> {
        let block_id = self.free_list.pop_front().ok_or(Error::OutOfBlocks)?;

        let block = Block::new(block_id, self.block_size);
        self.blocks.insert(block_id, block);

        Ok(block_id)
    }

    /// Allocate multiple blocks at once.
    ///
    /// # Arguments
    ///
    /// * `num_blocks` - Number of blocks to allocate
    ///
    /// # Returns
    ///
    /// Vector of allocated block IDs, or an error if not enough blocks.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfBlocks`] if not enough free blocks are available.
    pub fn allocate_many(&mut self, num_blocks: usize) -> Result<Vec<usize>> {
        if !self.can_allocate(num_blocks) {
            return Err(Error::OutOfBlocks);
        }

        let mut block_ids = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            block_ids.push(self.allocate()?);
        }

        Ok(block_ids)
    }

    /// Free a block and return it to the free list.
    ///
    /// The block is only freed if its reference count reaches zero.
    /// If the block has a prefix hash, it is removed from the prefix cache.
    ///
    /// # Arguments
    ///
    /// * `block_id` - The block to free
    ///
    /// # Returns
    ///
    /// `true` if the block was freed, `false` if it still has references.
    pub fn free(&mut self, block_id: usize) -> bool {
        if let Some(block) = self.blocks.get_mut(&block_id) {
            let new_ref_count = block.decrement_ref();

            if new_ref_count == 0 {
                // Remove from prefix cache if present
                if let Some(hash) = block.prefix_hash() {
                    self.prefix_cache.remove(&hash);
                }

                // Remove block and return to free list
                self.blocks.remove(&block_id);
                self.free_list.push_back(block_id);
                return true;
            }
        }
        false
    }

    /// Free multiple blocks.
    ///
    /// # Arguments
    ///
    /// * `block_ids` - The blocks to free
    ///
    /// # Returns
    ///
    /// Number of blocks actually freed (those whose ref count reached zero).
    pub fn free_many(&mut self, block_ids: &[usize]) -> usize {
        block_ids.iter().filter(|&&id| self.free(id)).count()
    }

    /// Get a reference to a block.
    pub fn get_block(&self, block_id: usize) -> Option<&Block> {
        self.blocks.get(&block_id)
    }

    /// Get a mutable reference to a block.
    pub fn get_block_mut(&mut self, block_id: usize) -> Option<&mut Block> {
        self.blocks.get_mut(&block_id)
    }

    /// Increment the reference count of a block.
    ///
    /// Used when sharing a block with another sequence (e.g., prefix caching).
    ///
    /// # Returns
    ///
    /// The new reference count, or `None` if the block doesn't exist.
    pub fn increment_ref(&mut self, block_id: usize) -> Option<usize> {
        self.blocks.get_mut(&block_id).map(|block| {
            block.increment_ref();
            block.ref_count()
        })
    }

    /// Decrement the reference count of a block without freeing.
    ///
    /// Use [`free`](Self::free) if you want to return the block to the free list
    /// when the reference count reaches zero.
    ///
    /// # Returns
    ///
    /// The new reference count, or `None` if the block doesn't exist.
    pub fn decrement_ref(&mut self, block_id: usize) -> Option<usize> {
        self.blocks
            .get_mut(&block_id)
            .map(|block| block.decrement_ref())
    }

    // ========== Prefix Caching ==========

    /// Try to get a cached block by prefix hash.
    ///
    /// If found, the block's reference count is incremented.
    ///
    /// # Arguments
    ///
    /// * `prefix_hash` - Hash of the token prefix
    ///
    /// # Returns
    ///
    /// The block ID if found in cache, or `None` if not cached.
    ///
    /// # Example
    ///
    /// ```
    /// use nano_vllm::core::block_manager::BlockManager;
    /// use nano_vllm::core::block::hash_token_block;
    ///
    /// let mut manager = BlockManager::new(10, 16);
    /// let tokens = [1u32, 2, 3, 4];
    /// let hash = hash_token_block(&tokens, None);
    ///
    /// // First allocation - not cached
    /// assert!(manager.get_cached_block(hash).is_none());
    ///
    /// // Allocate and cache
    /// let block_id = manager.allocate().unwrap();
    /// manager.cache_block(block_id, hash);
    ///
    /// // Now it's cached
    /// let cached = manager.get_cached_block(hash);
    /// assert_eq!(cached, Some(block_id));
    /// ```
    pub fn get_cached_block(&mut self, prefix_hash: u64) -> Option<usize> {
        if let Some(&block_id) = self.prefix_cache.get(&prefix_hash) {
            // Verify the block still exists and increment ref count
            if self.blocks.contains_key(&block_id) {
                self.increment_ref(block_id);
                return Some(block_id);
            } else {
                // Block was freed but cache entry remained - clean it up
                self.prefix_cache.remove(&prefix_hash);
            }
        }
        None
    }

    /// Add a block to the prefix cache.
    ///
    /// # Arguments
    ///
    /// * `block_id` - The block to cache
    /// * `prefix_hash` - Hash of the token prefix stored in this block
    pub fn cache_block(&mut self, block_id: usize, prefix_hash: u64) {
        if let Some(block) = self.blocks.get_mut(&block_id) {
            block.set_prefix_hash(prefix_hash);
            self.prefix_cache.insert(prefix_hash, block_id);
        }
    }

    /// Allocate a block, checking the prefix cache first.
    ///
    /// If a block with the given prefix hash is already cached, it will be
    /// reused (with incremented reference count). Otherwise, a new block
    /// is allocated.
    ///
    /// # Arguments
    ///
    /// * `prefix_hash` - Hash of the token prefix
    ///
    /// # Returns
    ///
    /// A tuple of (block_id, was_cached) where `was_cached` indicates whether
    /// the block was reused from the cache.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfBlocks`] if not cached and no free blocks available.
    pub fn allocate_with_prefix(&mut self, prefix_hash: u64) -> Result<(usize, bool)> {
        // Check cache first
        if let Some(block_id) = self.get_cached_block(prefix_hash) {
            return Ok((block_id, true));
        }

        // Allocate new block
        let block_id = self.allocate()?;
        self.cache_block(block_id, prefix_hash);
        Ok((block_id, false))
    }

    /// Check if a prefix hash is in the cache.
    pub fn is_prefix_cached(&self, prefix_hash: u64) -> bool {
        self.prefix_cache.contains_key(&prefix_hash)
    }

    /// Get the number of cached prefixes.
    pub fn num_cached_prefixes(&self) -> usize {
        self.prefix_cache.len()
    }

    /// Clear the prefix cache.
    ///
    /// This does not free the blocks, only removes them from the cache.
    pub fn clear_prefix_cache(&mut self) {
        for (&hash, &block_id) in &self.prefix_cache {
            if let Some(block) = self.blocks.get_mut(&block_id) {
                // Clear the hash from the block, but keep it as Some
                // to indicate it was previously cached
                let _ = hash; // Just to avoid unused warning
                block.set_prefix_hash(0); // Clear hash
            }
        }
        self.prefix_cache.clear();
    }

    /// Reset the block manager to initial state.
    ///
    /// All blocks are freed and returned to the free list.
    pub fn reset(&mut self) {
        self.blocks.clear();
        self.prefix_cache.clear();
        self.free_list.clear();
        self.free_list.extend(0..self.num_blocks);
    }
}
