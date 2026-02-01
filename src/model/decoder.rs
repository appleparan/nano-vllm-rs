//! Qwen3 Decoder Layer implementation.
//!
//! A decoder layer combines self-attention and MLP with residual connections
//! and pre-norm architecture.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::attention::Qwen3Attention;
use super::mlp::Qwen3Mlp;
use super::norm::RmsNorm;

/// Qwen3 Decoder Layer.
///
/// Each layer consists of:
/// 1. Input layer norm -> Self-attention -> Residual add
/// 2. Post-attention layer norm -> MLP -> Residual add
///
/// This is the "pre-norm" architecture where normalization happens
/// before each sub-layer rather than after.
#[derive(Debug, Clone)]
pub struct Qwen3DecoderLayer {
    /// Input layer normalization (before attention).
    input_layernorm: RmsNorm,
    /// Self-attention module.
    self_attn: Qwen3Attention,
    /// Post-attention layer normalization (before MLP).
    post_attention_layernorm: RmsNorm,
    /// Feed-forward MLP.
    mlp: Qwen3Mlp,
}

impl Qwen3DecoderLayer {
    /// Creates a new Qwen3DecoderLayer from a VarBuilder.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - Model hidden dimension
    /// * `intermediate_size` - MLP intermediate dimension
    /// * `num_heads` - Number of query heads
    /// * `num_kv_heads` - Number of key-value heads
    /// * `head_dim` - Dimension per head
    /// * `max_seq_len` - Maximum sequence length
    /// * `rope_theta` - RoPE frequency base
    /// * `rms_norm_eps` - Epsilon for RMSNorm
    /// * `use_flash_attention` - Whether to use Flash Attention
    /// * `vb` - VarBuilder for loading weights
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        rms_norm_eps: f64,
        use_flash_attention: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_layernorm_weight = vb.get((hidden_size,), "input_layernorm.weight")?;
        let input_layernorm = RmsNorm::new(input_layernorm_weight, rms_norm_eps);

        let self_attn = Qwen3Attention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len,
            rope_theta,
            rms_norm_eps,
            use_flash_attention,
            vb.pp("self_attn"),
        )?;

        let post_attention_layernorm_weight =
            vb.get((hidden_size,), "post_attention_layernorm.weight")?;
        let post_attention_layernorm = RmsNorm::new(post_attention_layernorm_weight, rms_norm_eps);

        let mlp = Qwen3Mlp::new(hidden_size, intermediate_size, vb.pp("mlp"))?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    /// Creates a new Qwen3DecoderLayer with random weights for testing.
    #[allow(clippy::too_many_arguments)]
    pub fn new_random(
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        rms_norm_eps: f64,
        use_flash_attention: bool,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let input_layernorm = RmsNorm::new_ones(hidden_size, rms_norm_eps, dtype, device)?;

        let self_attn = Qwen3Attention::new_random(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len,
            rope_theta,
            rms_norm_eps,
            use_flash_attention,
            dtype,
            device,
        )?;

        let post_attention_layernorm = RmsNorm::new_ones(hidden_size, rms_norm_eps, dtype, device)?;

        let mlp = Qwen3Mlp::new_random(hidden_size, intermediate_size, dtype, device)?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    /// Returns a reference to the attention module.
    pub fn self_attn(&self) -> &Qwen3Attention {
        &self.self_attn
    }

    /// Returns a reference to the MLP module.
    pub fn mlp(&self) -> &Qwen3Mlp {
        &self.mlp
    }

    /// Forward pass through the decoder layer.
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - Input tensor [batch, seq_len, hidden_size]
    /// * `start_pos` - Starting position for RoPE
    /// * `kv_cache` - Optional KV cache for incremental decoding
    /// * `attention_mask` - Optional attention mask
    ///
    /// # Returns
    ///
    /// Output tensor [batch, seq_len, hidden_size]
    ///
    /// # Architecture
    ///
    /// ```text
    /// Input (hidden_states)
    ///    │
    ///    ├───────────────────────────┐ (residual)
    ///    │                           │
    ///    ▼                           │
    /// input_layernorm                │
    ///    │                           │
    ///    ▼                           │
    /// self_attention                 │
    ///    │                           │
    ///    ▼                           │
    ///    + ◄─────────────────────────┘
    ///    │
    ///    ├───────────────────────────┐ (residual)
    ///    │                           │
    ///    ▼                           │
    /// post_attention_layernorm       │
    ///    │                           │
    ///    ▼                           │
    /// mlp                            │
    ///    │                           │
    ///    ▼                           │
    ///    + ◄─────────────────────────┘
    ///    │
    ///    ▼
    /// Output
    /// ```
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        start_pos: usize,
        kv_cache: Option<&mut (Tensor, Tensor)>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // First residual block: Attention
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states =
            self.self_attn
                .forward(&hidden_states, start_pos, kv_cache, attention_mask)?;
        let hidden_states = (residual + hidden_states)?;

        // Second residual block: MLP
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        residual + hidden_states
    }
}
