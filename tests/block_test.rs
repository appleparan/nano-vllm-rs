//! Integration tests for Block and BlockTable.

use nano_vllm::core::block::{compute_num_blocks, hash_token_block, Block, BlockTable};

#[test]
fn test_block_creation() {
    let block = Block::new(42, 16);
    assert_eq!(block.block_id(), 42);
    assert_eq!(block.block_size(), 16);
    assert_eq!(block.ref_count(), 1);
    assert!(block.prefix_hash().is_none());
    assert!(!block.is_full());
}

#[test]
fn test_block_ref_counting() {
    let mut block = Block::new(0, 16);
    assert_eq!(block.ref_count(), 1);

    block.increment_ref();
    assert_eq!(block.ref_count(), 2);

    block.increment_ref();
    assert_eq!(block.ref_count(), 3);

    assert_eq!(block.decrement_ref(), 2);
    assert_eq!(block.decrement_ref(), 1);
    assert_eq!(block.decrement_ref(), 0);

    // Should not go below 0
    assert_eq!(block.decrement_ref(), 0);
}

#[test]
fn test_block_prefix_hash() {
    let mut block = Block::new(0, 16);
    assert!(block.prefix_hash().is_none());

    block.set_prefix_hash(12345);
    assert_eq!(block.prefix_hash(), Some(12345));
}

#[test]
fn test_block_table_basic() {
    let mut table = BlockTable::new(16);
    assert!(table.is_empty());
    assert_eq!(table.num_blocks(), 0);

    table.append_block(5);
    table.append_block(12);
    table.append_block(3);

    assert!(!table.is_empty());
    assert_eq!(table.num_blocks(), 3);
    assert_eq!(table.get_physical_block_ids(), &[5, 12, 3]);
}

#[test]
fn test_block_table_get_block_id() {
    let mut table = BlockTable::new(16);
    table.append_block(5);
    table.append_block(12);
    table.append_block(3);

    assert_eq!(table.get_block_id(0).unwrap(), 5);
    assert_eq!(table.get_block_id(1).unwrap(), 12);
    assert_eq!(table.get_block_id(2).unwrap(), 3);

    // Out of bounds
    assert!(table.get_block_id(3).is_err());
    assert!(table.get_block_id(100).is_err());
}

#[test]
fn test_block_table_slot_mapping() {
    let mut table = BlockTable::new(16);
    table.append_block(5);
    table.append_block(12);

    let slots = table.get_slot_mapping(20);
    assert_eq!(slots.len(), 20);

    // First 16 tokens in block 5 (slots 80-95)
    assert_eq!(slots[0], 5 * 16); // 80
    assert_eq!(slots[15], 5 * 16 + 15); // 95

    // Next 4 tokens in block 12 (slots 192-195)
    assert_eq!(slots[16], 12 * 16); // 192
    assert_eq!(slots[19], 12 * 16 + 3); // 195
}

#[test]
fn test_hash_token_block() {
    let tokens = [1u32, 2, 3, 4, 5];

    // Hash without parent
    let hash1 = hash_token_block(&tokens, None);

    // Same tokens with different parent should produce different hash
    let hash2 = hash_token_block(&tokens, Some(999));
    assert_ne!(hash1, hash2);

    // Same tokens with same parent should produce same hash
    let hash3 = hash_token_block(&tokens, Some(999));
    assert_eq!(hash2, hash3);

    // Different tokens should produce different hash
    let different_tokens = [1u32, 2, 3, 4, 6];
    let hash4 = hash_token_block(&different_tokens, None);
    assert_ne!(hash1, hash4);
}

#[test]
fn test_compute_num_blocks() {
    assert_eq!(compute_num_blocks(0, 16), 0);
    assert_eq!(compute_num_blocks(1, 16), 1);
    assert_eq!(compute_num_blocks(15, 16), 1);
    assert_eq!(compute_num_blocks(16, 16), 1);
    assert_eq!(compute_num_blocks(17, 16), 2);
    assert_eq!(compute_num_blocks(32, 16), 2);
    assert_eq!(compute_num_blocks(35, 16), 3);
    assert_eq!(compute_num_blocks(100, 16), 7);
}
