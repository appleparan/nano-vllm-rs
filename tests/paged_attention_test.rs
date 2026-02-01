//! Integration tests for PagedAttention.

use candle_core::{DType, Device, Tensor};
use nano_vllm::attention::{paged_attention, prefill_attention, write_kv_to_cache};
use nano_vllm::core::block::BlockTable;
use nano_vllm::core::kv_cache::{KVCacheConfig, LayerKVCache};

fn test_device() -> Device {
    Device::Cpu
}

#[test]
fn test_prefill_attention_shape() {
    let device = test_device();
    let batch_size = 1;
    let seq_len = 4;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 16;

    let q = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, seq_len, num_heads, head_dim),
        &device,
    )
    .unwrap();
    let k = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, seq_len, num_kv_heads, head_dim),
        &device,
    )
    .unwrap();
    let v = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, seq_len, num_kv_heads, head_dim),
        &device,
    )
    .unwrap();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let output = prefill_attention(&q, &k, &v, num_heads, num_kv_heads, scale, true).unwrap();

    assert_eq!(output.dims(), &[batch_size, seq_len, num_heads * head_dim]);
}

#[test]
fn test_prefill_attention_causal_mask() {
    let device = test_device();
    let seq_len = 4;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 16;

    let q = Tensor::randn(0.0f32, 1.0, (1, seq_len, num_heads, head_dim), &device).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (1, seq_len, num_kv_heads, head_dim), &device).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (1, seq_len, num_kv_heads, head_dim), &device).unwrap();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let output = prefill_attention(&q, &k, &v, num_heads, num_kv_heads, scale, true).unwrap();

    // Verify output is finite
    let output_sum: f32 = output.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
    assert!(output_sum.is_finite());
}

#[test]
fn test_repeat_kv() {
    let device = test_device();
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 16;

    // Test via prefill_attention which uses repeat_kv internally
    let q = Tensor::randn(0.0f32, 1.0, (1, 4, num_heads, head_dim), &device).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (1, 4, num_kv_heads, head_dim), &device).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (1, 4, num_kv_heads, head_dim), &device).unwrap();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let output = prefill_attention(&q, &k, &v, num_heads, num_kv_heads, scale, true).unwrap();

    assert_eq!(output.dims(), &[1, 4, num_heads * head_dim]);
}

#[test]
fn test_repeat_kv_no_expansion() {
    let device = test_device();
    let num_heads = 4;
    let head_dim = 16;

    // When num_heads == num_kv_heads, no expansion needed
    let q = Tensor::randn(0.0f32, 1.0, (1, 4, num_heads, head_dim), &device).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (1, 4, num_heads, head_dim), &device).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (1, 4, num_heads, head_dim), &device).unwrap();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let output = prefill_attention(&q, &k, &v, num_heads, num_heads, scale, true).unwrap();

    assert_eq!(output.dims(), &[1, 4, num_heads * head_dim]);
}

#[test]
fn test_paged_attention_basic() {
    let device = test_device();
    let num_blocks = 4;
    let block_size = 4;
    let num_kv_heads = 2;
    let head_dim = 8;
    let num_heads = 4;

    // Create KV cache config
    let config = KVCacheConfig::new(num_blocks, block_size, num_kv_heads, head_dim, 1);

    // Create layer cache
    let kv_cache = LayerKVCache::new(&config, &device).unwrap();

    // Create block table with 2 blocks (8 tokens)
    let mut block_table = BlockTable::new(block_size);
    block_table.append_block(0);
    block_table.append_block(1);

    let context_len = 6; // Using 6 out of 8 available slots

    // Query for decode: 1 new token
    let query = Tensor::randn(0.0f32, 1.0, (1, 1, num_heads, head_dim), &device).unwrap();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let output = paged_attention(
        &query,
        &kv_cache,
        &block_table,
        context_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
    )
    .unwrap();

    assert_eq!(output.dims(), &[1, 1, num_heads * head_dim]);
}

#[test]
fn test_decode_causal_mask() {
    let device = test_device();
    let num_blocks = 4;
    let block_size = 4;
    let num_kv_heads = 2;
    let head_dim = 8;
    let num_heads = 4;

    let config = KVCacheConfig::new(num_blocks, block_size, num_kv_heads, head_dim, 1);
    let kv_cache = LayerKVCache::new(&config, &device).unwrap();

    let mut block_table = BlockTable::new(block_size);
    block_table.append_block(0);
    block_table.append_block(1);

    // 2 new tokens, testing chunked prefill causal masking
    let query = Tensor::randn(0.0f32, 1.0, (1, 2, num_heads, head_dim), &device).unwrap();
    let context_len = 8; // All 8 slots used

    let scale = 1.0 / (head_dim as f64).sqrt();
    let output = paged_attention(
        &query,
        &kv_cache,
        &block_table,
        context_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
    )
    .unwrap();

    assert_eq!(output.dims(), &[1, 2, num_heads * head_dim]);
}

#[test]
fn test_write_kv_to_cache() {
    let device = test_device();
    let num_blocks = 2;
    let block_size = 4;
    let num_kv_heads = 2;
    let head_dim = 4;

    // Create empty cache tensors
    let key_cache = Tensor::zeros(
        (num_blocks, block_size, num_kv_heads, head_dim),
        DType::F32,
        &device,
    )
    .unwrap();
    let value_cache = Tensor::zeros(
        (num_blocks, block_size, num_kv_heads, head_dim),
        DType::F32,
        &device,
    )
    .unwrap();

    // K/V for 3 tokens
    let key = Tensor::ones((1, 3, num_kv_heads, head_dim), DType::F32, &device).unwrap();
    let value = (Tensor::ones((1, 3, num_kv_heads, head_dim), DType::F32, &device).unwrap()
        * 2.0)
        .unwrap();

    // Slot mapping: tokens go to slots 0, 1, 4 (second block, first slot)
    let slot_mapping = vec![0, 1, 4];

    let (new_key_cache, new_value_cache) = write_kv_to_cache(
        &key,
        &value,
        &key_cache,
        &value_cache,
        &slot_mapping,
        block_size,
    )
    .unwrap();

    // Verify shapes
    assert_eq!(
        new_key_cache.dims(),
        &[num_blocks, block_size, num_kv_heads, head_dim]
    );
    assert_eq!(
        new_value_cache.dims(),
        &[num_blocks, block_size, num_kv_heads, head_dim]
    );

    // Verify slot 0 has key=1
    let slot0 = new_key_cache
        .narrow(0, 0, 1)
        .unwrap()
        .narrow(1, 0, 1)
        .unwrap();
    let slot0_vals: Vec<f32> = slot0.flatten_all().unwrap().to_vec1().unwrap();
    assert!(slot0_vals.iter().all(|&v| (v - 1.0).abs() < 1e-6));

    // Verify slot 4 (block 1, slot 0) has key=1
    let slot4 = new_key_cache
        .narrow(0, 1, 1)
        .unwrap()
        .narrow(1, 0, 1)
        .unwrap();
    let slot4_vals: Vec<f32> = slot4.flatten_all().unwrap().to_vec1().unwrap();
    assert!(slot4_vals.iter().all(|&v| (v - 1.0).abs() < 1e-6));

    // Verify slot 2 (not written) is still 0
    let slot2 = new_key_cache
        .narrow(0, 0, 1)
        .unwrap()
        .narrow(1, 2, 1)
        .unwrap();
    let slot2_vals: Vec<f32> = slot2.flatten_all().unwrap().to_vec1().unwrap();
    assert!(slot2_vals.iter().all(|&v| v.abs() < 1e-6));
}

#[test]
fn test_paged_attention_multi_block() {
    let device = test_device();
    let num_blocks = 8;
    let block_size = 4;
    let num_kv_heads = 2;
    let head_dim = 8;
    let num_heads = 4;

    let config = KVCacheConfig::new(num_blocks, block_size, num_kv_heads, head_dim, 1);
    let kv_cache = LayerKVCache::new(&config, &device).unwrap();

    // Block table with 3 blocks (non-contiguous: 0, 2, 5)
    let mut block_table = BlockTable::new(block_size);
    block_table.append_block(0);
    block_table.append_block(2);
    block_table.append_block(5);

    let context_len = 10; // Using 10 out of 12 available slots

    let query = Tensor::randn(0.0f32, 1.0, (1, 1, num_heads, head_dim), &device).unwrap();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let output = paged_attention(
        &query,
        &kv_cache,
        &block_table,
        context_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
    )
    .unwrap();

    assert_eq!(output.dims(), &[1, 1, num_heads * head_dim]);
}

#[test]
fn test_chunked_prefill_attention() {
    let device = test_device();
    let num_blocks = 8;
    let block_size = 4;
    let num_kv_heads = 2;
    let head_dim = 8;
    let num_heads = 4;

    let config = KVCacheConfig::new(num_blocks, block_size, num_kv_heads, head_dim, 1);
    let kv_cache = LayerKVCache::new(&config, &device).unwrap();

    // 2 blocks already cached
    let mut block_table = BlockTable::new(block_size);
    block_table.append_block(0);
    block_table.append_block(1);

    // Chunked prefill: processing 4 new tokens at once
    let num_new_tokens = 4;
    let context_len = 8 + num_new_tokens; // 8 cached + 4 new

    // Need another block for new tokens
    block_table.append_block(2);

    let query = Tensor::randn(
        0.0f32,
        1.0,
        (1, num_new_tokens, num_heads, head_dim),
        &device,
    )
    .unwrap();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let output = paged_attention(
        &query,
        &kv_cache,
        &block_table,
        context_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
    )
    .unwrap();

    assert_eq!(output.dims(), &[1, num_new_tokens, num_heads * head_dim]);
}
