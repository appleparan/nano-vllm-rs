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

use candle_core::{D, Device, Tensor};

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
        .flat_map(|i| (0..kv_seq_len).map(move |j| if j > i { neg_inf } else { 0.0f32 }))
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
            (0..context_len).map(
                move |key_pos| {
                    if key_pos > query_pos { neg_inf } else { 0.0f32 }
                },
            )
        })
        .collect();

    Ok(Tensor::from_vec(
        mask,
        (1, 1, num_new_tokens, context_len),
        device,
    )?)
}
