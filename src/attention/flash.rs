//! Flash Attention implementation.
//!
//! Flash Attention is a memory-efficient attention algorithm that reduces memory
//! usage from O(n²) to O(n) by computing attention in tiles and using online
//! softmax computation.
//!
//! ## Algorithm Overview
//!
//! Instead of materializing the full attention matrix, Flash Attention:
//! 1. Splits Q, K, V into blocks
//! 2. Computes attention scores block by block
//! 3. Uses online softmax to maintain running statistics (max, sum)
//! 4. Accumulates output incrementally with proper rescaling
//!
//! ## References
//!
//! - FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
//!   <https://arxiv.org/abs/2205.14135>
//! - FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
//!   <https://arxiv.org/abs/2307.08691>

use candle_core::{D, Device, Result, Tensor};

/// Configuration for Flash Attention.
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Block size for query dimension.
    pub block_size_q: usize,
    /// Block size for key/value dimension.
    pub block_size_kv: usize,
    /// Whether to apply causal masking.
    pub causal: bool,
    /// Softmax scale factor (typically 1/sqrt(head_dim)).
    pub softmax_scale: f32,
}

impl FlashAttentionConfig {
    /// Create a new FlashAttentionConfig.
    pub fn new(head_dim: usize, causal: bool) -> Self {
        Self {
            block_size_q: 64,
            block_size_kv: 64,
            causal,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Create config with custom block sizes.
    pub fn with_block_sizes(mut self, block_size_q: usize, block_size_kv: usize) -> Self {
        self.block_size_q = block_size_q;
        self.block_size_kv = block_size_kv;
        self
    }

    /// Create config with custom softmax scale.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.softmax_scale = scale;
        self
    }
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_size_q: 64,
            block_size_kv: 64,
            causal: true,
            softmax_scale: 1.0 / 8.0, // Assume head_dim=64
        }
    }
}

/// Flash Attention dispatcher - selects CPU or CUDA implementation.
///
/// # Arguments
///
/// * `query` - Query tensor [batch, seq_len, num_heads, head_dim]
/// * `key` - Key tensor [batch, seq_len, num_kv_heads, head_dim]
/// * `value` - Value tensor [batch, seq_len, num_kv_heads, head_dim]
/// * `config` - Flash Attention configuration
///
/// # Returns
///
/// Attention output [batch, seq_len, num_heads, head_dim]
pub fn flash_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    config: &FlashAttentionConfig,
) -> Result<Tensor> {
    match query.device() {
        Device::Cpu => flash_attention_cpu(query, key, value, config),
        // TODO: Add CUDA dispatch when FFI bindings are ready
        // The custom kernel is compiled but not yet linked
        Device::Cuda(_) => {
            // For now, move to CPU for computation
            // This is temporary until FFI bindings are implemented
            let q_cpu = query.to_device(&Device::Cpu)?;
            let k_cpu = key.to_device(&Device::Cpu)?;
            let v_cpu = value.to_device(&Device::Cpu)?;
            let output = flash_attention_cpu(&q_cpu, &k_cpu, &v_cpu, config)?;
            output.to_device(query.device())
        }
        _ => flash_attention_cpu(query, key, value, config),
    }
}

/// CPU reference implementation of Flash Attention.
///
/// This is a block-by-block implementation with online softmax for educational
/// purposes. It demonstrates the Flash Attention algorithm without GPU-specific
/// optimizations.
///
/// # Algorithm
///
/// For each query block Q_i:
///   1. Initialize: O_i = 0, l_i = 0, m_i = -inf
///   2. For each key/value block K_j, V_j:
///      a. Compute S_ij = Q_i @ K_j^T * scale
///      b. Apply causal mask if needed
///      c. Update running max: m_new = max(m_i, rowmax(S_ij))
///      d. Compute P_ij = exp(S_ij - m_new)
///      e. Update running sum: l_new = l_i * exp(m_i - m_new) + rowsum(P_ij)
///      f. Update output: O_i = O_i * (l_i/l_new * exp(m_i - m_new)) + P_ij @ V_j / l_new
///      g. Update: m_i = m_new, l_i = l_new
///
/// # Arguments
///
/// * `query` - Query tensor [batch, seq_len_q, num_heads, head_dim]
/// * `key` - Key tensor [batch, seq_len_kv, num_kv_heads, head_dim]
/// * `value` - Value tensor [batch, seq_len_kv, num_kv_heads, head_dim]
/// * `config` - Flash Attention configuration
///
/// # Returns
///
/// Attention output [batch, seq_len_q, num_heads, head_dim]
pub fn flash_attention_cpu(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    config: &FlashAttentionConfig,
) -> Result<Tensor> {
    let (batch_size, seq_len_q, num_heads, head_dim) = query.dims4()?;
    let (_, seq_len_kv, num_kv_heads, _) = key.dims4()?;

    // Handle GQA: expand K, V to match num_heads
    let (key, value) = if num_kv_heads != num_heads {
        let n_rep = num_heads / num_kv_heads;
        let key = repeat_kv(key, n_rep)?;
        let value = repeat_kv(value, n_rep)?;
        (key, value)
    } else {
        (key.clone(), value.clone())
    };

    // Transpose to [batch, num_heads, seq_len, head_dim] for easier block processing
    let q = query.transpose(1, 2)?.contiguous()?;
    let k = key.transpose(1, 2)?.contiguous()?;
    let v = value.transpose(1, 2)?.contiguous()?;

    let block_size_q = config.block_size_q.min(seq_len_q);
    let block_size_kv = config.block_size_kv.min(seq_len_kv);
    let num_blocks_q = seq_len_q.div_ceil(block_size_q);
    let num_blocks_kv = seq_len_kv.div_ceil(block_size_kv);

    // Initialize output tensor
    let mut output = Tensor::zeros(
        (batch_size, num_heads, seq_len_q, head_dim),
        q.dtype(),
        q.device(),
    )?;

    // Process each query block
    for q_block_idx in 0..num_blocks_q {
        let q_start = q_block_idx * block_size_q;
        let q_end = (q_start + block_size_q).min(seq_len_q);
        let q_len = q_end - q_start;

        // Extract query block: [batch, num_heads, q_len, head_dim]
        let q_block = q.narrow(2, q_start, q_len)?;

        // Initialize running statistics for this query block
        // m: running max per row, l: running sum per row
        // Shape: [batch, num_heads, q_len]
        let neg_inf = f32::NEG_INFINITY;
        let mut m_i = Tensor::full(neg_inf, (batch_size, num_heads, q_len), q.device())?
            .to_dtype(q.dtype())?;
        let mut l_i = Tensor::zeros((batch_size, num_heads, q_len), q.dtype(), q.device())?;
        let mut o_i = Tensor::zeros(
            (batch_size, num_heads, q_len, head_dim),
            q.dtype(),
            q.device(),
        )?;

        // Process each key/value block
        for kv_block_idx in 0..num_blocks_kv {
            let kv_start = kv_block_idx * block_size_kv;
            let kv_end = (kv_start + block_size_kv).min(seq_len_kv);
            let kv_len = kv_end - kv_start;

            // Skip blocks that are entirely masked (causal)
            if config.causal && kv_start > q_end - 1 {
                continue;
            }

            // Extract key/value blocks: [batch, num_heads, kv_len, head_dim]
            let k_block = k.narrow(2, kv_start, kv_len)?;
            let v_block = v.narrow(2, kv_start, kv_len)?;

            // Compute attention scores: Q @ K^T
            // [batch, num_heads, q_len, head_dim] @ [batch, num_heads, head_dim, kv_len]
            // = [batch, num_heads, q_len, kv_len]
            let k_t = k_block.transpose(D::Minus2, D::Minus1)?;
            let s_ij = q_block.matmul(&k_t)?;
            let s_ij = (s_ij * config.softmax_scale as f64)?;

            // Apply causal mask if needed
            let s_ij = if config.causal {
                apply_causal_mask_block(&s_ij, q_start, kv_start)?
            } else {
                s_ij
            };

            // Online softmax update
            // Step 1: Compute row-wise max of current block
            let m_ij = s_ij.max(D::Minus1)?; // [batch, num_heads, q_len]

            // Step 2: New running max
            let m_new = m_i.maximum(&m_ij)?;

            // Step 3: Compute exp(S_ij - m_new) for numerical stability
            let m_new_expanded = m_new.unsqueeze(D::Minus1)?; // [batch, num_heads, q_len, 1]
            let p_ij = (s_ij.broadcast_sub(&m_new_expanded))?.exp()?;

            // Step 4: Row sum of P_ij
            let l_ij = p_ij.sum(D::Minus1)?; // [batch, num_heads, q_len]

            // Step 5: Rescale factor for previous accumulator
            let m_diff = (&m_i - &m_new)?;
            let alpha = m_diff.exp()?; // exp(m_old - m_new)

            // Step 6: New running sum
            let l_new = ((&l_i * &alpha)? + &l_ij)?;

            // Step 7: Update output
            // O_new = O_old * (l_old * alpha / l_new) + P_ij @ V / l_new
            let alpha_expanded = alpha.unsqueeze(D::Minus1)?;
            let l_i_expanded = l_i.unsqueeze(D::Minus1)?;
            let l_new_expanded = l_new.unsqueeze(D::Minus1)?;

            // Rescale old output
            let scale_old =
                (l_i_expanded.broadcast_mul(&alpha_expanded))?.broadcast_div(&l_new_expanded)?;
            let o_scaled = o_i.broadcast_mul(&scale_old)?;

            // Compute P_ij @ V_j
            let pv = p_ij.matmul(&v_block)?; // [batch, num_heads, q_len, head_dim]

            // Add new contribution
            let pv_scaled = pv.broadcast_div(&l_new_expanded)?;
            o_i = (o_scaled + pv_scaled)?;

            // Update running statistics
            m_i = m_new;
            l_i = l_new;
        }

        // Write output block back
        // We need to use slice_assign or cat to update the output tensor
        output = write_block(&output, &o_i, q_start)?;
    }

    // Transpose back to [batch, seq_len, num_heads, head_dim]
    output.transpose(1, 2)?.contiguous()
}

/// Repeat KV heads for GQA.
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (batch, seq_len, num_kv_heads, head_dim) = x.dims4()?;
    let x = x.unsqueeze(3)?; // [batch, seq_len, num_kv_heads, 1, head_dim]
    let x = x.expand((batch, seq_len, num_kv_heads, n_rep, head_dim))?;
    x.reshape((batch, seq_len, num_kv_heads * n_rep, head_dim))
}

/// Apply causal mask to a block of attention scores.
fn apply_causal_mask_block(scores: &Tensor, q_start: usize, kv_start: usize) -> Result<Tensor> {
    let (_batch, _num_heads, q_len, kv_len) = scores.dims4()?;
    let device = scores.device();
    let dtype = scores.dtype();

    let neg_inf = f32::NEG_INFINITY;

    // Create mask: position i can attend to position j if j <= i (in global coords)
    // Global query position: q_start + i
    // Global key position: kv_start + j
    // Mask condition: kv_start + j > q_start + i means mask out
    let mask_data: Vec<f32> = (0..q_len)
        .flat_map(|i| {
            let global_q = q_start + i;
            (0..kv_len)
                .map(move |j| {
                    let global_k = kv_start + j;
                    if global_k > global_q { neg_inf } else { 0.0 }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mask = Tensor::from_vec(mask_data, (1, 1, q_len, kv_len), device)?.to_dtype(dtype)?;
    scores.broadcast_add(&mask)
}

/// Write a block back to the output tensor.
fn write_block(output: &Tensor, block: &Tensor, start: usize) -> Result<Tensor> {
    let (_batch, _num_heads, seq_len, _head_dim) = output.dims4()?;
    let block_len = block.dim(2)?;

    if start == 0 && block_len == seq_len {
        // Block covers entire sequence
        return Ok(block.clone());
    }

    // Build output by concatenating parts
    let mut parts = Vec::new();

    if start > 0 {
        let before = output.narrow(2, 0, start)?;
        parts.push(before);
    }

    parts.push(block.clone());

    let end = start + block_len;
    if end < seq_len {
        let after = output.narrow(2, end, seq_len - end)?;
        parts.push(after);
    }

    if parts.len() == 1 {
        Ok(parts.into_iter().next().unwrap())
    } else {
        Tensor::cat(&parts.iter().collect::<Vec<_>>(), 2)
    }
}

// TODO: Add CUDA FFI bindings when ready
// The custom kernel is in kernels/flash_attn_fwd.cu
