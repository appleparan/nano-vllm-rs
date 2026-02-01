//! Integration tests for BlockManager.

use nano_vllm::Error;
use nano_vllm::core::block::hash_token_block;
use nano_vllm::core::block_manager::BlockManager;

#[test]
fn test_block_manager_creation() {
    let manager = BlockManager::new(100, 16);
    assert_eq!(manager.num_blocks(), 100);
    assert_eq!(manager.block_size(), 16);
    assert_eq!(manager.num_free_blocks(), 100);
    assert_eq!(manager.num_used_blocks(), 0);
}

#[test]
fn test_allocate_and_free() {
    let mut manager = BlockManager::new(10, 16);

    // Allocate
    let block_id = manager.allocate().unwrap();
    assert_eq!(manager.num_free_blocks(), 9);
    assert_eq!(manager.num_used_blocks(), 1);

    // Block should exist
    assert!(manager.get_block(block_id).is_some());
    assert_eq!(manager.get_block(block_id).unwrap().ref_count(), 1);

    // Free
    assert!(manager.free(block_id));
    assert_eq!(manager.num_free_blocks(), 10);
    assert_eq!(manager.num_used_blocks(), 0);
    assert!(manager.get_block(block_id).is_none());
}

#[test]
fn test_allocate_many() {
    let mut manager = BlockManager::new(10, 16);

    let blocks = manager.allocate_many(5).unwrap();
    assert_eq!(blocks.len(), 5);
    assert_eq!(manager.num_free_blocks(), 5);
    assert_eq!(manager.num_used_blocks(), 5);

    // Allocating more than available should fail
    assert!(manager.allocate_many(6).is_err());
}

#[test]
fn test_out_of_blocks() {
    let mut manager = BlockManager::new(2, 16);

    manager.allocate().unwrap();
    manager.allocate().unwrap();

    // Should fail
    assert!(matches!(manager.allocate(), Err(Error::OutOfBlocks)));
}

#[test]
fn test_reference_counting() {
    let mut manager = BlockManager::new(10, 16);

    let block_id = manager.allocate().unwrap();

    // Increment ref count
    assert_eq!(manager.increment_ref(block_id), Some(2));
    assert_eq!(manager.increment_ref(block_id), Some(3));

    // First two frees should not return block to free list
    assert!(!manager.free(block_id)); // ref_count: 3 -> 2
    assert!(!manager.free(block_id)); // ref_count: 2 -> 1
    assert_eq!(manager.num_used_blocks(), 1);

    // Third free should return to free list
    assert!(manager.free(block_id)); // ref_count: 1 -> 0
    assert_eq!(manager.num_used_blocks(), 0);
    assert_eq!(manager.num_free_blocks(), 10);
}

#[test]
fn test_prefix_caching_basic() {
    let mut manager = BlockManager::new(10, 16);

    let tokens = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let hash = hash_token_block(&tokens, None);

    // Not cached initially
    assert!(!manager.is_prefix_cached(hash));
    assert!(manager.get_cached_block(hash).is_none());

    // Allocate and cache
    let block_id = manager.allocate().unwrap();
    manager.cache_block(block_id, hash);

    // Now cached
    assert!(manager.is_prefix_cached(hash));
    assert_eq!(manager.num_cached_prefixes(), 1);
}

#[test]
fn test_prefix_cache_reuse() {
    let mut manager = BlockManager::new(10, 16);

    let tokens = [1u32, 2, 3, 4];
    let hash = hash_token_block(&tokens, None);

    // First allocation
    let (block_id1, cached1) = manager.allocate_with_prefix(hash).unwrap();
    assert!(!cached1);
    assert_eq!(manager.get_block(block_id1).unwrap().ref_count(), 1);

    // Second allocation with same hash - should reuse
    let (block_id2, cached2) = manager.allocate_with_prefix(hash).unwrap();
    assert!(cached2);
    assert_eq!(block_id1, block_id2);
    assert_eq!(manager.get_block(block_id1).unwrap().ref_count(), 2);

    // Only 1 block used
    assert_eq!(manager.num_used_blocks(), 1);
}

#[test]
fn test_prefix_cache_cleared_on_free() {
    let mut manager = BlockManager::new(10, 16);

    let tokens = [1u32, 2, 3];
    let hash = hash_token_block(&tokens, None);

    let block_id = manager.allocate().unwrap();
    manager.cache_block(block_id, hash);
    assert!(manager.is_prefix_cached(hash));

    // Free the block
    manager.free(block_id);

    // Cache entry should be removed
    assert!(!manager.is_prefix_cached(hash));
}

#[test]
fn test_free_many() {
    let mut manager = BlockManager::new(10, 16);

    let blocks = manager.allocate_many(5).unwrap();
    assert_eq!(manager.num_used_blocks(), 5);

    // Increment ref on first block
    manager.increment_ref(blocks[0]);

    // Free all - only 4 should actually free (first has ref_count 2)
    let freed = manager.free_many(&blocks);
    assert_eq!(freed, 4);
    assert_eq!(manager.num_used_blocks(), 1);

    // Free the remaining one
    assert!(manager.free(blocks[0]));
    assert_eq!(manager.num_used_blocks(), 0);
}

#[test]
fn test_can_allocate() {
    let mut manager = BlockManager::new(5, 16);

    assert!(manager.can_allocate(5));
    assert!(!manager.can_allocate(6));

    manager.allocate_many(3).unwrap();
    assert!(manager.can_allocate(2));
    assert!(!manager.can_allocate(3));
}

#[test]
fn test_reset() {
    let mut manager = BlockManager::new(10, 16);

    // Allocate some blocks
    let blocks = manager.allocate_many(5).unwrap();

    // Cache one
    let hash = hash_token_block(&[1u32, 2, 3], None);
    manager.cache_block(blocks[0], hash);

    // Reset
    manager.reset();

    // Everything should be back to initial state
    assert_eq!(manager.num_free_blocks(), 10);
    assert_eq!(manager.num_used_blocks(), 0);
    assert_eq!(manager.num_cached_prefixes(), 0);
}
