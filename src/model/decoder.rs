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
        let hidden_states = self
            .self_attn
            .forward(&hidden_states, start_pos, kv_cache, attention_mask)?;
        let hidden_states = (residual + hidden_states)?;

        // Second residual block: MLP
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        residual + hidden_states
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> Device {
        Device::Cpu
    }

    #[test]
    fn test_decoder_layer_creation() {
        let device = test_device();
        let layer = Qwen3DecoderLayer::new_random(
            64,     // hidden_size
            256,    // intermediate_size
            4,      // num_heads
            2,      // num_kv_heads
            16,     // head_dim
            1024,   // max_seq_len
            10000.0, // rope_theta
            1e-6,   // rms_norm_eps
            DType::F32,
            &device,
        )
        .unwrap();

        assert_eq!(layer.self_attn().num_heads(), 4);
        assert_eq!(layer.self_attn().num_kv_heads(), 2);
        assert_eq!(layer.mlp().hidden_size(), 64);
        assert_eq!(layer.mlp().intermediate_size(), 256);
    }

    #[test]
    fn test_decoder_layer_forward_shape() {
        let device = test_device();
        let hidden_size = 64;
        let batch_size = 2;
        let seq_len = 8;

        let layer = Qwen3DecoderLayer::new_random(
            hidden_size,
            256,
            4,
            2,
            16,
            1024,
            10000.0,
            1e-6,
            DType::F32,
            &device,
        )
        .unwrap();

        let x = Tensor::randn(0.0f32, 0.1, (batch_size, seq_len, hidden_size), &device).unwrap();

        let output = layer.forward(&x, 0, None, None).unwrap();

        // Output should have the same shape as input
        assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_decoder_layer_with_kv_cache() {
        let device = test_device();
        let hidden_size = 64;
        let num_kv_heads = 2;
        let head_dim = 16;

        let layer = Qwen3DecoderLayer::new_random(
            hidden_size,
            256,
            4,
            num_kv_heads,
            head_dim,
            1024,
            10000.0,
            1e-6,
            DType::F32,
            &device,
        )
        .unwrap();

        // Prefill
        let x_prefill = Tensor::randn(0.0f32, 0.1, (1, 4, hidden_size), &device).unwrap();
        let mut kv_cache = (
            Tensor::zeros((1, 0, num_kv_heads, head_dim), DType::F32, &device).unwrap(),
            Tensor::zeros((1, 0, num_kv_heads, head_dim), DType::F32, &device).unwrap(),
        );

        let output1 = layer.forward(&x_prefill, 0, Some(&mut kv_cache), None).unwrap();
        assert_eq!(output1.dims(), &[1, 4, hidden_size]);
        assert_eq!(kv_cache.0.dims(), &[1, 4, num_kv_heads, head_dim]);

        // Decode
        let x_decode = Tensor::randn(0.0f32, 0.1, (1, 1, hidden_size), &device).unwrap();
        let output2 = layer.forward(&x_decode, 4, Some(&mut kv_cache), None).unwrap();
        assert_eq!(output2.dims(), &[1, 1, hidden_size]);
        assert_eq!(kv_cache.0.dims(), &[1, 5, num_kv_heads, head_dim]);
    }

    #[test]
    fn test_decoder_layer_residual() {
        let device = test_device();
        let hidden_size = 32;

        let layer = Qwen3DecoderLayer::new_random(
            hidden_size,
            128,
            4,
            2,
            8,
            1024,
            10000.0,
            1e-6,
            DType::F32,
            &device,
        )
        .unwrap();

        // With residual connections, output should have similar magnitude to input
        // (unlike without residual where it could explode or vanish)
        let x = Tensor::randn(0.0f32, 1.0, (1, 4, hidden_size), &device).unwrap();
        let output = layer.forward(&x, 0, None, None).unwrap();

        // Check output is finite and reasonable
        let output_sum: f32 = output.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert!(output_sum.is_finite());
        assert!(output_sum > 0.0);
    }

    #[test]
    fn test_decoder_layer_with_qwen3_config() {
        // Qwen3-0.6B scaled down for testing
        let device = test_device();
        let hidden_size = 256;
        let intermediate_size = 768;
        let num_heads = 8;
        let num_kv_heads = 4;
        let head_dim = 32;

        let layer = Qwen3DecoderLayer::new_random(
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
            4096,
            1000000.0, // Qwen3 uses 1M
            1e-6,
            DType::F32,
            &device,
        )
        .unwrap();

        let x = Tensor::randn(0.0f32, 0.1, (1, 4, hidden_size), &device).unwrap();
        let output = layer.forward(&x, 0, None, None).unwrap();

        assert_eq!(output.dims(), &[1, 4, hidden_size]);
    }
}
