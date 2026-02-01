//! Speculative decoding engine.
//!
//! Orchestrates draft model, target model, and rejection sampling
//! for faster token generation.

use candle_core::{DType, Device, Result, Tensor, D};

use super::config::SpeculativeConfig;
use super::sampler::RejectionSampler;
use crate::model::Qwen3ForCausalLM;

/// Speculative decoding engine.
///
/// Coordinates between a small draft model and a large target model
/// to accelerate autoregressive text generation.
///
/// ## Workflow
///
/// 1. **Draft**: Generate K tokens with the fast draft model
/// 2. **Verify**: Run target model on all tokens in one forward pass
/// 3. **Accept/Reject**: Use rejection sampling to determine final tokens
///
/// ## Example
///
/// ```text
/// Input:  [The, quick, brown]
/// Draft:  [fox, jumps, over, the]     <- 4 speculative tokens
/// Target: Verify all 5 positions      <- 1 forward pass
/// Result: [fox, jumps, over]          <- 3 accepted + 1 resampled
/// ```
pub struct SpeculativeEngine {
    /// Large target model (ground truth).
    target_model: Qwen3ForCausalLM,
    /// Small draft model (fast approximation).
    draft_model: Qwen3ForCausalLM,
    /// Configuration.
    config: SpeculativeConfig,
    /// Rejection sampler.
    rejection_sampler: RejectionSampler,
    /// Device (CPU/GPU).
    device: Device,
    /// Data type.
    dtype: DType,
}

impl SpeculativeEngine {
    /// Create a new speculative engine.
    ///
    /// # Arguments
    ///
    /// * `target_model` - Large model for verification (e.g., Qwen3-4B)
    /// * `draft_model` - Small model for drafting (e.g., Qwen3-0.6B)
    /// * `config` - Speculative decoding configuration
    pub fn new(
        target_model: Qwen3ForCausalLM,
        draft_model: Qwen3ForCausalLM,
        config: SpeculativeConfig,
    ) -> Self {
        let device = target_model.device().clone();
        let dtype = target_model.dtype();

        Self {
            target_model,
            draft_model,
            config,
            rejection_sampler: RejectionSampler::new(),
            device,
            dtype,
        }
    }

    /// Create with a seeded RNG for reproducibility.
    pub fn with_seed(
        target_model: Qwen3ForCausalLM,
        draft_model: Qwen3ForCausalLM,
        config: SpeculativeConfig,
        seed: u64,
    ) -> Self {
        let device = target_model.device().clone();
        let dtype = target_model.dtype();

        Self {
            target_model,
            draft_model,
            config,
            rejection_sampler: RejectionSampler::with_seed(seed),
            device,
            dtype,
        }
    }

    /// Generate K draft tokens using the draft model.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Current token sequence [1, seq_len]
    /// * `start_pos` - Starting position for attention
    ///
    /// # Returns
    ///
    /// `(draft_tokens, draft_logits)`:
    /// - `draft_tokens`: K generated token IDs
    /// - `draft_logits`: Logits for each draft position [K, vocab_size]
    fn draft(
        &mut self,
        input_ids: &Tensor,
        start_pos: usize,
    ) -> Result<(Vec<u32>, Tensor)> {
        let k = self.config.num_speculative_tokens;
        let mut tokens = Vec::with_capacity(k);
        let mut all_logits = Vec::with_capacity(k);
        let mut current_ids = input_ids.clone();

        for i in 0..k {
            // Forward pass with draft model
            let logits = self.draft_model.forward(&current_ids, start_pos + i)?;

            // Store logits for rejection sampling
            all_logits.push(logits.clone());

            // Greedy decoding for draft (most common approach)
            let token = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
            tokens.push(token);

            // Append token for next iteration
            current_ids = self.append_token(&current_ids, token)?;
        }

        // Stack logits: [K, vocab_size]
        let logits_tensor = Tensor::stack(&all_logits, 0)?;

        Ok((tokens, logits_tensor))
    }

    /// Verify draft tokens with the target model.
    ///
    /// Runs a single forward pass with all draft tokens appended,
    /// returning logits for the last K+1 positions.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Original input sequence [1, seq_len]
    /// * `draft_tokens` - Draft tokens to verify
    /// * `start_pos` - Starting position for attention
    ///
    /// # Returns
    ///
    /// Target logits for positions [seq_len-1, seq_len, ..., seq_len+K-1]
    /// Shape: [K+1, vocab_size]
    fn verify(
        &mut self,
        input_ids: &Tensor,
        draft_tokens: &[u32],
        start_pos: usize,
    ) -> Result<Tensor> {
        let k = draft_tokens.len();

        // Append all draft tokens to input
        let extended_ids = self.append_tokens(input_ids, draft_tokens)?;

        // Forward pass with target model - get all position logits
        let all_logits = self.target_model.forward_all(&extended_ids, start_pos)?;

        // Extract last K+1 positions
        // all_logits shape: [1, seq_len + K, vocab_size]
        let seq_len = all_logits.dim(1)?;
        let start = seq_len - k - 1;

        // Narrow to [1, K+1, vocab_size] then squeeze to [K+1, vocab_size]
        let target_logits = all_logits.narrow(1, start, k + 1)?.squeeze(0)?;

        Ok(target_logits)
    }

    /// Perform one speculative decoding step.
    ///
    /// Generates K draft tokens, verifies with target model,
    /// and returns accepted tokens using rejection sampling.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Current token sequence [1, seq_len]
    /// * `start_pos` - Starting position for attention
    /// * `temperature` - Sampling temperature
    ///
    /// # Returns
    ///
    /// Vector of accepted tokens (1 to K+1 tokens)
    pub fn speculative_step(
        &mut self,
        input_ids: &Tensor,
        start_pos: usize,
        temperature: f32,
    ) -> Result<Vec<u32>> {
        // Step 1: Draft K tokens
        let (draft_tokens, draft_logits) = self.draft(input_ids, start_pos)?;

        // Step 2: Verify with target model
        let target_logits = self.verify(input_ids, &draft_tokens, start_pos)?;

        // Step 3: Rejection sampling
        let (accepted, final_token, _num_accepted) = self.rejection_sampler.verify(
            &draft_tokens,
            &draft_logits,
            &target_logits,
            temperature,
        )?;

        // Combine accepted tokens with final token
        let mut result = accepted;
        result.push(final_token);

        Ok(result)
    }

    /// Append a single token to input_ids.
    fn append_token(&self, input_ids: &Tensor, token: u32) -> Result<Tensor> {
        let token_tensor = Tensor::new(&[token], &self.device)?
            .to_dtype(input_ids.dtype())?
            .unsqueeze(0)?;

        Tensor::cat(&[input_ids, &token_tensor], 1)
    }

    /// Append multiple tokens to input_ids.
    fn append_tokens(&self, input_ids: &Tensor, tokens: &[u32]) -> Result<Tensor> {
        if tokens.is_empty() {
            return Ok(input_ids.clone());
        }

        let tokens_tensor = Tensor::new(tokens, &self.device)?
            .to_dtype(input_ids.dtype())?
            .unsqueeze(0)?;

        Tensor::cat(&[input_ids, &tokens_tensor], 1)
    }

    /// Get the speculative configuration.
    pub fn config(&self) -> &SpeculativeConfig {
        &self.config
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get reference to the target model.
    pub fn target_model(&self) -> &Qwen3ForCausalLM {
        &self.target_model
    }

    /// Get mutable reference to the target model.
    pub fn target_model_mut(&mut self) -> &mut Qwen3ForCausalLM {
        &mut self.target_model
    }

    /// Get reference to the draft model.
    pub fn draft_model(&self) -> &Qwen3ForCausalLM {
        &self.draft_model
    }

    /// Get mutable reference to the draft model.
    pub fn draft_model_mut(&mut self) -> &mut Qwen3ForCausalLM {
        &mut self.draft_model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full integration tests require loading actual models.
    // These are basic structural tests.

    #[test]
    fn test_config_accessor() {
        // This test just verifies the config accessor compiles correctly.
        // Full tests are in speculative_inference_test.rs
        let config = SpeculativeConfig::new("Qwen/Qwen3-0.6B")
            .num_tokens(4);

        assert_eq!(config.num_speculative_tokens, 4);
        assert_eq!(config.draft_model_id, "Qwen/Qwen3-0.6B");
    }
}
