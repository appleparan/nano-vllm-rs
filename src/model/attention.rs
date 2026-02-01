//! Qwen3 Attention implementation.
//!
//! This module implements Grouped Query Attention (GQA) with:
//! - Per-head RMSNorm on Q and K (Qwen3 specific)
//! - Rotary Position Embeddings (RoPE)
//! - KV cache support for incremental decoding

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use super::norm::RmsNorm;
use super::rope::RotaryEmbedding;

/// Qwen3 Attention with Grouped Query Attention (GQA).
///
/// Key features:
/// - GQA: Multiple query heads share each KV head pair
/// - Per-head normalization: RMSNorm applied to Q and K per head
/// - RoPE: Position encoding via rotation
#[derive(Debug, Clone)]
pub struct Qwen3Attention {
    /// Query projection [hidden_size] -> [num_heads * head_dim].
    q_proj: Linear,
    /// Key projection [hidden_size] -> [num_kv_heads * head_dim].
    k_proj: Linear,
    /// Value projection [hidden_size] -> [num_kv_heads * head_dim].
    v_proj: Linear,
    /// Output projection [num_heads * head_dim] -> [hidden_size].
    o_proj: Linear,
    /// Per-head Q normalization.
    q_norm: RmsNorm,
    /// Per-head K normalization.
    k_norm: RmsNorm,
    /// Rotary position embeddings.
    rotary_emb: RotaryEmbedding,
    /// Number of query heads.
    num_heads: usize,
    /// Number of key-value heads.
    num_kv_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Hidden dimension.
    #[allow(dead_code)]
    hidden_size: usize,
    /// Scaling factor for attention scores.
    scale: f64,
}

impl Qwen3Attention {
    /// Creates a new Qwen3Attention from a VarBuilder.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - Model hidden dimension
    /// * `num_heads` - Number of query heads
    /// * `num_kv_heads` - Number of key-value heads (for GQA)
    /// * `head_dim` - Dimension per head
    /// * `max_seq_len` - Maximum sequence length for RoPE
    /// * `rope_theta` - RoPE frequency base
    /// * `rms_norm_eps` - Epsilon for RMSNorm
    /// * `vb` - VarBuilder for loading weights
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        rms_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let q_proj = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        // Per-head normalization weights
        let q_norm_weight = vb.get((head_dim,), "q_norm.weight")?;
        let k_norm_weight = vb.get((head_dim,), "k_norm.weight")?;
        let q_norm = RmsNorm::new(q_norm_weight, rms_norm_eps);
        let k_norm = RmsNorm::new(k_norm_weight, rms_norm_eps);

        let rotary_emb =
            RotaryEmbedding::new(head_dim, max_seq_len, rope_theta, vb.dtype(), vb.device())?;

        let scale = 1.0 / (head_dim as f64).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
            scale,
        })
    }

    /// Creates a new Qwen3Attention with random weights for testing.
    #[allow(clippy::too_many_arguments)]
    pub fn new_random(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        rms_norm_eps: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let scale_init = 0.02;

        // Q, K, V, O projections
        let q_weight = Tensor::randn(
            0.0f32,
            scale_init,
            (num_heads * head_dim, hidden_size),
            device,
        )?
        .to_dtype(dtype)?;
        let k_weight = Tensor::randn(
            0.0f32,
            scale_init,
            (num_kv_heads * head_dim, hidden_size),
            device,
        )?
        .to_dtype(dtype)?;
        let v_weight = Tensor::randn(
            0.0f32,
            scale_init,
            (num_kv_heads * head_dim, hidden_size),
            device,
        )?
        .to_dtype(dtype)?;
        let o_weight = Tensor::randn(
            0.0f32,
            scale_init,
            (hidden_size, num_heads * head_dim),
            device,
        )?
        .to_dtype(dtype)?;

        let q_proj = Linear::new(q_weight, None);
        let k_proj = Linear::new(k_weight, None);
        let v_proj = Linear::new(v_weight, None);
        let o_proj = Linear::new(o_weight, None);

        // Per-head norm weights (initialized to 1)
        let q_norm = RmsNorm::new_ones(head_dim, rms_norm_eps, dtype, device)?;
        let k_norm = RmsNorm::new_ones(head_dim, rms_norm_eps, dtype, device)?;

        let rotary_emb = RotaryEmbedding::new(head_dim, max_seq_len, rope_theta, dtype, device)?;

        let scale = 1.0 / (head_dim as f64).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
            scale,
        })
    }

    /// Returns the number of query heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Returns the number of KV heads.
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Returns the head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Forward pass through the attention layer.
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - Input tensor [batch, seq_len, hidden_size]
    /// * `start_pos` - Starting position for RoPE (0 for prefill, >0 for decode)
    /// * `kv_cache` - Optional KV cache (k_cache, v_cache) for incremental decoding
    /// * `attention_mask` - Optional attention mask
    ///
    /// # Returns
    ///
    /// Output tensor [batch, seq_len, hidden_size]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        start_pos: usize,
        kv_cache: Option<&mut (Tensor, Tensor)>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // 1. Project Q, K, V
        let q = self.q_proj.forward(hidden_states)?; // [batch, seq_len, num_heads * head_dim]
        let k = self.k_proj.forward(hidden_states)?; // [batch, seq_len, num_kv_heads * head_dim]
        let v = self.v_proj.forward(hidden_states)?; // [batch, seq_len, num_kv_heads * head_dim]

        // 2. Reshape to [batch, seq_len, num_heads, head_dim]
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;

        // 3. Per-head RMSNorm on Q and K (Qwen3 specific)
        let q = self.apply_head_norm(&q, &self.q_norm)?;
        let k = self.apply_head_norm(&k, &self.k_norm)?;

        // 4. Apply RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, start_pos)?;

        // 5. Handle KV cache
        let (k, v) = match kv_cache {
            Some((k_cache, v_cache)) => {
                // Append new K, V to cache
                let k = Tensor::cat(&[k_cache.clone(), k], 1)?;
                let v = Tensor::cat(&[v_cache.clone(), v], 1)?;
                // Update cache
                *k_cache = k.clone();
                *v_cache = v.clone();
                (k, v)
            }
            None => (k, v),
        };

        let kv_seq_len = k.dims()[1];

        // 6. Expand K, V for GQA
        let k = self.repeat_kv(&k)?; // [batch, kv_seq_len, num_heads, head_dim]
        let v = self.repeat_kv(&v)?;

        // 7. Transpose for attention: [batch, num_heads, seq_len, head_dim]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // 8. Compute attention scores: Q @ K^T / sqrt(d)
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * self.scale)?;
        // attn_weights: [batch, num_heads, seq_len, kv_seq_len]

        // 9. Apply causal mask
        let attn_weights = match attention_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => {
                // Create causal mask if not provided
                let mask = self.create_causal_mask(seq_len, kv_seq_len, start_pos, q.device())?;
                attn_weights.broadcast_add(&mask)?
            }
        };

        // 10. Softmax
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // 11. Attention @ V
        let attn_output = attn_weights.matmul(&v)?;
        // attn_output: [batch, num_heads, seq_len, head_dim]

        // 12. Transpose back and reshape
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?;
        // [batch, seq_len, num_heads, head_dim]
        let attn_output =
            attn_output.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        // 13. Output projection
        self.o_proj.forward(&attn_output)
    }

    /// Applies RMSNorm per head.
    ///
    /// Input: [batch, seq_len, num_heads, head_dim]
    /// Norm is applied on the last dimension (head_dim) for each head.
    fn apply_head_norm(&self, x: &Tensor, norm: &RmsNorm) -> Result<Tensor> {
        // The norm expects [..., hidden_size] and normalizes the last dim
        // Our input is [batch, seq_len, num_heads, head_dim]
        // We want to normalize head_dim for each head
        norm.forward(x)
    }

    /// Repeats KV heads to match the number of query heads (for GQA).
    ///
    /// Input: [batch, seq_len, num_kv_heads, head_dim]
    /// Output: [batch, seq_len, num_heads, head_dim]
    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            return Ok(x.clone());
        }

        let (batch, seq_len, num_kv_heads, head_dim) = x.dims4()?;

        // Expand and repeat
        // [batch, seq_len, num_kv_heads, head_dim]
        // -> [batch, seq_len, num_kv_heads, 1, head_dim]
        // -> [batch, seq_len, num_kv_heads, n_rep, head_dim]
        // -> [batch, seq_len, num_heads, head_dim]
        let x = x.unsqueeze(3)?;
        let x = x.expand((batch, seq_len, num_kv_heads, n_rep, head_dim))?;
        x.reshape((batch, seq_len, self.num_heads, head_dim))
    }

    /// Creates a causal attention mask.
    ///
    /// For prefill (start_pos=0): Standard causal mask
    /// For decode (start_pos>0): Allow attending to all cached positions
    fn create_causal_mask(
        &self,
        seq_len: usize,
        kv_seq_len: usize,
        start_pos: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let neg_inf = f32::NEG_INFINITY;

        if seq_len == 1 {
            // Decode: single query can attend to all kv positions
            Tensor::zeros((1, 1, 1, kv_seq_len), DType::F32, device)
        } else {
            // Prefill: causal mask
            let mask: Vec<f32> = (0..seq_len)
                .flat_map(|i| {
                    let query_pos = start_pos + i;
                    (0..kv_seq_len).map(
                        move |key_pos| {
                            if key_pos > query_pos {
                                neg_inf
                            } else {
                                0.0f32
                            }
                        },
                    )
                })
                .collect();
            Tensor::from_vec(mask, (1, 1, seq_len, kv_seq_len), device)
        }
    }
}
