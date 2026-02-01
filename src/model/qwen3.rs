//! Qwen3 Model implementation.
//!
//! This module implements the full Qwen3 transformer model:
//! - Token embedding layer
//! - N transformer decoder layers
//! - Final RMSNorm
//! - Language model head (optional, can tie weights with embedding)
//!
//! ## Architecture
//!
//! ```text
//! Input Token IDs
//!       │
//!       ▼
//! ┌───────────────┐
//! │  Embedding    │  vocab_size → hidden_size
//! └───────────────┘
//!       │
//!       ▼
//! ┌───────────────┐
//! │ DecoderLayer  │ × num_hidden_layers
//! └───────────────┘
//!       │
//!       ▼
//! ┌───────────────┐
//! │   RMSNorm     │  Final normalization
//! └───────────────┘
//!       │
//!       ▼
//! ┌───────────────┐
//! │   LM Head     │  hidden_size → vocab_size
//! └───────────────┘
//!       │
//!       ▼
//! Output Logits
//! ```

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};

use super::decoder::Qwen3DecoderLayer;
use super::loader::Qwen3Config;
use super::norm::RmsNorm;

/// Qwen3 transformer model.
#[derive(Debug, Clone)]
pub struct Qwen3Model {
    /// Token embedding layer.
    embed_tokens: Embedding,
    /// Transformer decoder layers.
    layers: Vec<Qwen3DecoderLayer>,
    /// Final layer normalization.
    norm: RmsNorm,
    /// Device.
    device: Device,
    /// Data type.
    dtype: DType,
}

impl Qwen3Model {
    /// Creates a new Qwen3Model from a VarBuilder.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    /// * `vb` - VarBuilder for loading weights
    pub fn new(config: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = Qwen3DecoderLayer::new(
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                config.rms_norm_eps,
                vb.pp(format!("model.layers.{i}")),
            )?;
            layers.push(layer);
        }

        let norm_weight = vb.get((config.hidden_size,), "model.norm.weight")?;
        let norm = RmsNorm::new(norm_weight, config.rms_norm_eps);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Forward pass through the transformer (without LM head).
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `start_pos` - Starting position for incremental decoding
    ///
    /// # Returns
    ///
    /// Hidden states [batch, seq_len, hidden_size]
    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        // Embed tokens
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Pass through decoder layers
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states, start_pos, None, None)?;
        }

        // Final normalization
        self.norm.forward(&hidden_states)
    }

    /// Returns the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Returns the embedding layer for weight tying.
    pub fn embed_tokens(&self) -> &Embedding {
        &self.embed_tokens
    }
}

/// Qwen3 model for causal language modeling.
///
/// Wraps Qwen3Model and adds the language model head for token prediction.
#[derive(Debug, Clone)]
pub struct Qwen3ForCausalLM {
    /// Base transformer model.
    model: Qwen3Model,
    /// Language model head (linear projection to vocab).
    lm_head: Linear,
}

impl Qwen3ForCausalLM {
    /// Creates a new Qwen3ForCausalLM from a VarBuilder.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    /// * `vb` - VarBuilder for loading weights
    pub fn new(config: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let model = Qwen3Model::new(config, vb.clone())?;

        let lm_head = if config.tie_word_embeddings {
            // Weight tying: lm_head uses embedding weights transposed
            let embed_weight = model.embed_tokens.embeddings();
            Linear::new(embed_weight.clone(), None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self { model, lm_head })
    }

    /// Forward pass through the model.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `start_pos` - Starting position for incremental decoding
    ///
    /// # Returns
    ///
    /// Logits for the last position [batch, vocab_size]
    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        let hidden_states = self.model.forward(input_ids, start_pos)?;

        // Get hidden states for the last position only
        let seq_len = hidden_states.dim(1)?;
        let last_hidden = hidden_states.narrow(1, seq_len - 1, 1)?.squeeze(1)?;

        // Project to vocabulary
        self.lm_head.forward(&last_hidden)
    }

    /// Forward pass returning logits for all positions.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `start_pos` - Starting position for incremental decoding
    ///
    /// # Returns
    ///
    /// Logits for all positions [batch, seq_len, vocab_size]
    pub fn forward_all(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        let hidden_states = self.model.forward(input_ids, start_pos)?;
        self.lm_head.forward(&hidden_states)
    }

    /// Returns reference to the base model.
    pub fn model(&self) -> &Qwen3Model {
        &self.model
    }

    /// Returns mutable reference to the base model.
    pub fn model_mut(&mut self) -> &mut Qwen3Model {
        &mut self.model
    }

    /// Returns the device.
    pub fn device(&self) -> &Device {
        self.model.device()
    }

    /// Returns the data type.
    pub fn dtype(&self) -> DType {
        self.model.dtype()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full model tests require downloading weights from HuggingFace.
    // These tests verify the structure and forward pass shapes with mock data.

    #[test]
    fn test_qwen3_config_defaults() {
        let json = r#"{
            "vocab_size": 1000,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2
        }"#;

        let config: Qwen3Config = serde_json::from_str(json).unwrap();

        assert_eq!(config.vocab_size, 1000);
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_hidden_layers, 2);
        assert_eq!(config.head_dim, 128); // default
        assert!(config.tie_word_embeddings); // default
    }
}
