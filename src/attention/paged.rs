//! PagedAttention implementation.
//!
//! PagedAttention enables efficient KV cache management by splitting the cache
//! into fixed-size blocks, similar to OS virtual memory. This allows:
//! - Non-contiguous memory allocation
//! - Memory sharing via copy-on-write
//! - Prefix caching for shared prompts
//!
//! ## Two Phases
//!
//! 1. **Prefill**: Process all prompt tokens at once with standard SDPA
//!    - Efficient for batched matrix operations
//!    - K/V are written to cache after computation
//!
//! 2. **Decode**: Process one token at a time with paged K/V access
//!    - Gather K/V from non-contiguous blocks via BlockTable
//!    - Compute attention with the full sequence context
//!
//! ## Memory Layout
//!
//! ```text
//! Standard KV Cache:   [batch, seq_len, num_kv_heads, head_dim]
//! Paged KV Cache:      [num_blocks, block_size, num_kv_heads, head_dim]
//!
//! Mapping via BlockTable:
//!   Token at position p -> Block (p / block_size), Slot (p % block_size)
//!   Physical location = block_table[logical_block] * block_size + slot
//! ```

use candle_core::{Device, Tensor, D};

use crate::core::block::BlockTable;
use crate::core::kv_cache::LayerKVCache;
use crate::error::Result;

/// Prefill attention using standard Scaled Dot-Product Attention.
///
/// For the prefill phase, we use contiguous K/V tensors for efficient
/// batched matrix operations. The K/V states are written to the paged
/// cache after computation.
///
/// # Arguments
///
/// * `query` - Query tensor [batch, seq_len, num_heads, head_dim]
/// * `key` - Key tensor [batch, seq_len, num_kv_heads, head_dim]
/// * `value` - Value tensor [batch, seq_len, num_kv_heads, head_dim]
/// * `num_heads` - Number of query heads
/// * `num_kv_heads` - Number of KV heads (for GQA)
/// * `scale` - Attention score scaling factor (1/sqrt(head_dim))
/// * `causal` - Whether to apply causal masking
///
/// # Returns
///
/// Attention output [batch, seq_len, num_heads, head_dim]
pub fn prefill_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f64,
    causal: bool,
) -> Result<Tensor> {
    // Expand K, V for GQA
    let key = repeat_kv(key, num_heads, num_kv_heads)?;
    let value = repeat_kv(value, num_heads, num_kv_heads)?;

    // Transpose for attention: [batch, num_heads, seq_len, head_dim]
    let q = query.transpose(1, 2)?.contiguous()?;
    let k = key.transpose(1, 2)?.contiguous()?;
    let v = value.transpose(1, 2)?.contiguous()?;

    let (batch_size, _, seq_len, head_dim) = q.dims4()?;

    // Compute attention scores: Q @ K^T / sqrt(d)
    let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;

    // Apply causal mask if needed
    let attn_weights = if causal && seq_len > 1 {
        let mask = create_causal_mask(seq_len, seq_len, q.device())?;
        attn_weights.broadcast_add(&mask)?
    } else {
        attn_weights
    };

    // Softmax
    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

    // Attention @ V
    let output = attn_weights.matmul(&v)?;

    // Transpose back: [batch, seq_len, num_heads, head_dim]
    let output = output.transpose(1, 2)?.contiguous()?;
    Ok(output.reshape((batch_size, seq_len, num_heads * head_dim))?)
}

/// Paged attention for decode phase.
///
/// Gathers K/V from block-based cache using the BlockTable mapping,
/// then computes attention for the new query token(s).
///
/// # Arguments
///
/// * `query` - Query tensor [batch, num_new_tokens, num_heads, head_dim]
/// * `kv_cache` - Block-based KV cache for this layer
/// * `block_table` - Logical to physical block mapping
/// * `context_len` - Total context length (cached tokens + new tokens)
/// * `num_heads` - Number of query heads
/// * `num_kv_heads` - Number of KV heads (for GQA)
/// * `head_dim` - Dimension per head
/// * `scale` - Attention score scaling factor
///
/// # Returns
///
/// Attention output [batch, num_new_tokens, num_heads * head_dim]
#[allow(clippy::too_many_arguments)]
pub fn paged_attention(
    query: &Tensor,
    kv_cache: &LayerKVCache,
    block_table: &BlockTable,
    context_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
) -> Result<Tensor> {
    let (batch_size, num_new_tokens, _, _) = query.dims4()?;
    let block_size = block_table.block_size();

    // Get physical block IDs for this sequence
    let block_ids = block_table.get_physical_block_ids();

    if block_ids.is_empty() {
        return Err(crate::error::Error::Config(
            "BlockTable is empty - no blocks allocated".into(),
        ));
    }

    // Gather K, V from blocks
    // Shape: [num_blocks, block_size, num_kv_heads, head_dim]
    let gathered_k = kv_cache.gather_keys(block_ids)?;
    let gathered_v = kv_cache.gather_values(block_ids)?;

    // Reshape to continuous sequence
    // [num_blocks * block_size, num_kv_heads, head_dim] -> trim to context_len
    let num_blocks = block_ids.len();
    let gathered_k = gathered_k.reshape((num_blocks * block_size, num_kv_heads, head_dim))?;
    let gathered_v = gathered_v.reshape((num_blocks * block_size, num_kv_heads, head_dim))?;

    // Narrow to actual context length (blocks may have extra slots)
    let k = gathered_k.narrow(0, 0, context_len)?;
    let v = gathered_v.narrow(0, 0, context_len)?;

    // Add batch dimension: [1, context_len, num_kv_heads, head_dim]
    let k = k.unsqueeze(0)?;
    let v = v.unsqueeze(0)?;

    // Expand K, V for GQA
    let k = repeat_kv(&k, num_heads, num_kv_heads)?;
    let v = repeat_kv(&v, num_heads, num_kv_heads)?;

    // Transpose for attention
    let q = query.transpose(1, 2)?.contiguous()?; // [batch, num_heads, num_new_tokens, head_dim]
    let k = k.transpose(1, 2)?.contiguous()?; // [batch, num_heads, context_len, head_dim]
    let v = v.transpose(1, 2)?.contiguous()?;

    // Compute attention scores
    let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;

    // For decode, new tokens can attend to all context (no future masking needed
    // because we're generating one token at a time)
    // However, if num_new_tokens > 1 (chunked prefill), we need partial causal mask

    let attn_weights = if num_new_tokens > 1 {
        // Chunked prefill: create causal mask for the new tokens portion
        let mask = create_decode_causal_mask(num_new_tokens, context_len, query.device())?;
        attn_weights.broadcast_add(&mask)?
    } else {
        // Single token decode: can attend to everything
        attn_weights
    };

    // Softmax
    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

    // Attention @ V
    let output = attn_weights.matmul(&v)?;

    // Transpose back and reshape
    let output = output.transpose(1, 2)?.contiguous()?;
    Ok(output.reshape((batch_size, num_new_tokens, num_heads * head_dim))?)
}

/// Write K/V states to the block-based cache.
///
/// Uses slot mapping to write each token's K/V to the correct block and slot.
///
/// # Arguments
///
/// * `key` - Key tensor [batch=1, seq_len, num_kv_heads, head_dim]
/// * `value` - Value tensor [batch=1, seq_len, num_kv_heads, head_dim]
/// * `kv_cache` - Mutable reference to layer KV cache
/// * `slot_mapping` - Global slot indices for each token position
///
/// # Returns
///
/// Updated KV cache tensors (key_cache, value_cache)
pub fn write_kv_to_cache(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &[usize],
    block_size: usize,
) -> Result<(Tensor, Tensor)> {
    let (_, seq_len, num_kv_heads, head_dim) = key.dims4()?;

    if slot_mapping.len() != seq_len {
        return Err(crate::error::Error::Config(format!(
            "slot_mapping length {} doesn't match seq_len {}",
            slot_mapping.len(),
            seq_len
        )));
    }

    // For each position, write to the corresponding slot in cache
    // This is a simple loop implementation - could be optimized with scatter
    let mut new_key_cache = key_cache.clone();
    let mut new_value_cache = value_cache.clone();

    for (pos, &global_slot) in slot_mapping.iter().enumerate() {
        let block_id = global_slot / block_size;
        let slot_in_block = global_slot % block_size;

        // Extract K/V for this position: [num_kv_heads, head_dim]
        let k_pos = key.narrow(1, pos, 1)?.squeeze(1)?.squeeze(0)?;
        let v_pos = value.narrow(1, pos, 1)?.squeeze(1)?.squeeze(0)?;

        // Expand to match cache shape for scatter
        // [1, 1, num_kv_heads, head_dim]
        let k_expanded = k_pos.unsqueeze(0)?.unsqueeze(0)?;
        let v_expanded = v_pos.unsqueeze(0)?.unsqueeze(0)?;

        // Use index to create a view at the right position
        // Note: This is inefficient but correct. Real implementations use custom kernels.
        new_key_cache = scatter_to_cache(
            &new_key_cache,
            &k_expanded,
            block_id,
            slot_in_block,
            num_kv_heads,
            head_dim,
        )?;
        new_value_cache = scatter_to_cache(
            &new_value_cache,
            &v_expanded,
            block_id,
            slot_in_block,
            num_kv_heads,
            head_dim,
        )?;
    }

    Ok((new_key_cache, new_value_cache))
}

/// Helper function to scatter a single KV entry to the cache.
fn scatter_to_cache(
    cache: &Tensor,
    entry: &Tensor,
    block_id: usize,
    slot: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let (num_blocks, block_size, _, _) = cache.dims4()?;

    // Get the entry data
    let entry_data = entry.squeeze(0)?.squeeze(0)?; // [num_kv_heads, head_dim]

    // Create a mask tensor and use it to update
    // For simplicity, we'll reconstruct the affected block
    let device = cache.device();
    let dtype = cache.dtype();

    // Extract the block
    let block = cache.narrow(0, block_id, 1)?; // [1, block_size, num_kv_heads, head_dim]

    // Create updated block with new entry at slot
    let mut block_data: Vec<f32> = block.flatten_all()?.to_vec1()?;

    // Get entry data
    let entry_flat: Vec<f32> = entry_data.flatten_all()?.to_vec1()?;

    // Calculate offset and update
    let slot_offset = slot * num_kv_heads * head_dim;
    for (i, &val) in entry_flat.iter().enumerate() {
        block_data[slot_offset + i] = val;
    }

    // Reconstruct the block
    let updated_block =
        Tensor::from_vec(block_data, (1, block_size, num_kv_heads, head_dim), device)?
            .to_dtype(dtype)?;

    // Reconstruct full cache with updated block
    if block_id == 0 && num_blocks == 1 {
        Ok(updated_block)
    } else if block_id == 0 {
        let rest = cache.narrow(0, 1, num_blocks - 1)?;
        Ok(Tensor::cat(&[updated_block, rest], 0)?)
    } else if block_id == num_blocks - 1 {
        let first = cache.narrow(0, 0, block_id)?;
        Ok(Tensor::cat(&[first, updated_block], 0)?)
    } else {
        let first = cache.narrow(0, 0, block_id)?;
        let rest = cache.narrow(0, block_id + 1, num_blocks - block_id - 1)?;
        Ok(Tensor::cat(&[first, updated_block, rest], 0)?)
    }
}

/// Repeats KV heads to match the number of query heads (for GQA).
///
/// Input: [batch, seq_len, num_kv_heads, head_dim]
/// Output: [batch, seq_len, num_heads, head_dim]
fn repeat_kv(x: &Tensor, num_heads: usize, num_kv_heads: usize) -> Result<Tensor> {
    let n_rep = num_heads / num_kv_heads;
    if n_rep == 1 {
        return Ok(x.clone());
    }

    let (batch, seq_len, _, head_dim) = x.dims4()?;

    // Expand and repeat
    let x = x.unsqueeze(3)?;
    let x = x.expand((batch, seq_len, num_kv_heads, n_rep, head_dim))?;
    Ok(x.reshape((batch, seq_len, num_heads, head_dim))?)
}

/// Creates a causal attention mask for prefill.
///
/// Returns a mask where future positions are -inf.
fn create_causal_mask(seq_len: usize, kv_seq_len: usize, device: &Device) -> Result<Tensor> {
    let neg_inf = f32::NEG_INFINITY;

    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..kv_seq_len).map(move |j| if j > i { neg_inf } else { 0.0f32 })
        })
        .collect();

    Ok(Tensor::from_vec(mask, (1, 1, seq_len, kv_seq_len), device)?)
}

/// Creates a causal mask for decode with partial prefill.
///
/// For chunked prefill where new_tokens are processed together,
/// each new token can attend to all previous context plus earlier new tokens.
fn create_decode_causal_mask(
    num_new_tokens: usize,
    context_len: usize,
    device: &Device,
) -> Result<Tensor> {
    let neg_inf = f32::NEG_INFINITY;
    let cached_len = context_len - num_new_tokens;

    let mask: Vec<f32> = (0..num_new_tokens)
        .flat_map(|i| {
            let query_pos = cached_len + i;
            (0..context_len).map(move |key_pos| {
                if key_pos > query_pos {
                    neg_inf
                } else {
                    0.0f32
                }
            })
        })
        .collect();

    Ok(Tensor::from_vec(mask, (1, 1, num_new_tokens, context_len), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use crate::core::kv_cache::KVCacheConfig;

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

        let q = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, num_heads, head_dim), &device)
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

        assert_eq!(
            output.dims(),
            &[batch_size, seq_len, num_heads * head_dim]
        );
    }

    #[test]
    fn test_prefill_attention_causal_mask() {
        let device = test_device();
        let seq_len = 4;

        let mask = create_causal_mask(seq_len, seq_len, &device).unwrap();
        let mask_vals: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

        // Row 0: [0, -inf, -inf, -inf]
        assert_eq!(mask_vals[0], 0.0);
        assert!(mask_vals[1].is_infinite());
        // Row 1: [0, 0, -inf, -inf]
        assert_eq!(mask_vals[4], 0.0);
        assert_eq!(mask_vals[5], 0.0);
        assert!(mask_vals[6].is_infinite());
        // Row 3: [0, 0, 0, 0]
        assert_eq!(mask_vals[12], 0.0);
        assert_eq!(mask_vals[15], 0.0);
    }

    #[test]
    fn test_repeat_kv() {
        let device = test_device();
        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 2, 16), &device).unwrap();

        let expanded = repeat_kv(&x, 4, 2).unwrap();

        assert_eq!(expanded.dims(), &[1, 4, 4, 16]);
    }

    #[test]
    fn test_repeat_kv_no_expansion() {
        let device = test_device();
        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 4, 16), &device).unwrap();

        let expanded = repeat_kv(&x, 4, 4).unwrap();

        assert_eq!(expanded.dims(), &[1, 4, 4, 16]);
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

        // Create layer cache and fill with random values
        let kv_cache = crate::core::kv_cache::LayerKVCache::new(&config, &device).unwrap();

        // Create block table with 2 blocks (8 tokens)
        let mut block_table = BlockTable::new(block_size);
        block_table.append_block(0);
        block_table.append_block(1);

        let context_len = 6; // Using 6 out of 8 available slots

        // Query for decode: 1 new token
        let query =
            Tensor::randn(0.0f32, 1.0, (1, 1, num_heads, head_dim), &device).unwrap();

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

        // 2 new tokens, total context of 5 (3 cached + 2 new)
        let mask = create_decode_causal_mask(2, 5, &device).unwrap();
        let mask_vals: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

        // mask shape: [1, 1, 2, 5]
        // Token at pos 3 (first new): can attend to 0,1,2,3, not 4
        // [0, 0, 0, 0, -inf]
        assert_eq!(mask_vals[0], 0.0);
        assert_eq!(mask_vals[3], 0.0);
        assert!(mask_vals[4].is_infinite());

        // Token at pos 4 (second new): can attend to all 0,1,2,3,4
        // [0, 0, 0, 0, 0]
        assert_eq!(mask_vals[5], 0.0);
        assert_eq!(mask_vals[9], 0.0);
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
        let value =
            (Tensor::ones((1, 3, num_kv_heads, head_dim), DType::F32, &device).unwrap() * 2.0)
                .unwrap();

        // Slot mapping: tokens go to slots 0, 1, 4 (second block, first slot)
        let slot_mapping = vec![0, 1, 4];

        let (new_key_cache, new_value_cache) =
            write_kv_to_cache(&key, &value, &key_cache, &value_cache, &slot_mapping, block_size)
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
        let kv_cache = crate::core::kv_cache::LayerKVCache::new(&config, &device).unwrap();

        // Block table with 3 blocks (non-contiguous: 0, 2, 5)
        let mut block_table = BlockTable::new(block_size);
        block_table.append_block(0);
        block_table.append_block(2);
        block_table.append_block(5);

        let context_len = 10; // Using 10 out of 12 available slots

        let query =
            Tensor::randn(0.0f32, 1.0, (1, 1, num_heads, head_dim), &device).unwrap();

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
        let kv_cache = crate::core::kv_cache::LayerKVCache::new(&config, &device).unwrap();

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
}
