//! LLM Inference Engine.
//!
//! The LLMEngine orchestrates all components for text generation:
//! - Model for forward passes
//! - Scheduler for request management
//! - Sampler for token selection
//! - Tokenizer for text encoding/decoding
//!
//! ## Engine Flow
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      LLMEngine                              │
//! └─────────────────────────────────────────────────────────────┘
//!                            │
//!           add_request()    │    step() / generate()
//!                ▼           │           ▼
//!         ┌──────────┐       │    ┌──────────────┐
//!         │ Tokenize │       │    │  Scheduler   │
//!         │  prompt  │       │    │   outputs    │
//!         └──────────┘       │    └──────────────┘
//!                │           │           │
//!                ▼           │           ▼
//!         ┌──────────┐       │    ┌──────────────┐
//!         │ Scheduler│       │    │    Model     │
//!         │   add    │       │    │   forward    │
//!         └──────────┘       │    └──────────────┘
//!                            │           │
//!                            │           ▼
//!                            │    ┌──────────────┐
//!                            │    │   Sampler    │
//!                            │    │   sample     │
//!                            │    └──────────────┘
//!                            │           │
//!                            │           ▼
//!                            │    ┌──────────────┐
//!                            │    │   Decode     │
//!                            │    │   tokens     │
//!                            │    └──────────────┘
//! ```

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use tokenizers::Tokenizer;

use super::sampler::Sampler;
use crate::config::{EngineConfig, SamplingConfig};
use crate::core::sequence::{FinishReason, Sequence, SequenceId};
use crate::error::{Error, Result};
use crate::model::{Qwen3Config, Qwen3ForCausalLM};
use crate::scheduler::Scheduler;
use crate::speculative::{SpeculativeConfig, SpeculativeEngine};
use crate::SchedulerConfig;

/// Output from a generation request.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    /// Unique request ID.
    pub request_id: SequenceId,
    /// Input prompt text.
    pub prompt: String,
    /// Generated output text.
    pub output_text: String,
    /// Generated token IDs.
    pub output_tokens: Vec<u32>,
    /// Reason for finishing.
    pub finish_reason: Option<FinishReason>,
    /// Total number of tokens (prompt + output).
    pub total_tokens: usize,
}

/// Request for text generation.
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    /// Unique request ID (auto-assigned if None).
    pub request_id: Option<SequenceId>,
    /// Input prompt text.
    pub prompt: String,
    /// Sampling configuration for this request.
    pub sampling_config: SamplingConfig,
    /// Priority (higher = more important).
    pub priority: i32,
}

impl GenerationRequest {
    /// Create a new generation request with default settings.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            request_id: None,
            prompt: prompt.into(),
            sampling_config: SamplingConfig::default(),
            priority: 0,
        }
    }

    /// Set the maximum tokens to generate.
    pub fn max_tokens(mut self, max_tokens: usize) -> Self {
        self.sampling_config.max_tokens = max_tokens;
        self
    }

    /// Set the temperature for sampling.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.sampling_config.temperature = temperature;
        self
    }

    /// Set top-k sampling parameter.
    pub fn top_k(mut self, top_k: usize) -> Self {
        self.sampling_config.top_k = top_k;
        self
    }

    /// Set top-p (nucleus) sampling parameter.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.sampling_config.top_p = top_p;
        self
    }

    /// Set request priority.
    pub fn priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

/// LLM Inference Engine.
///
/// Orchestrates model, scheduler, and sampler for text generation.
/// Optionally supports speculative decoding for faster generation.
///
/// When speculative decoding is enabled, the engine uses a small draft model
/// to generate tokens speculatively, then verifies with a larger target model.
pub struct LLMEngine {
    /// The language model (None when using speculative decoding).
    model: Option<Qwen3ForCausalLM>,
    /// Model configuration.
    model_config: Qwen3Config,
    /// Request scheduler.
    scheduler: Scheduler,
    /// Tokenizer for encoding/decoding text.
    tokenizer: Tokenizer,
    /// Sampling configs per request.
    sampling_configs: HashMap<SequenceId, SamplingConfig>,
    /// Sampler instances per request.
    samplers: HashMap<SequenceId, Sampler>,
    /// Original prompts per request.
    prompts: HashMap<SequenceId, String>,
    /// Counter for request IDs.
    next_request_id: SequenceId,
    /// Device (CPU/GPU).
    device: Device,
    /// Data type.
    dtype: DType,
    /// End-of-sequence token ID.
    eos_token_id: u32,
    /// Optional speculative decoding engine.
    /// When Some, the engine uses speculative decoding for generation.
    speculative_engine: Option<SpeculativeEngine>,
}

impl LLMEngine {
    /// Create a new LLMEngine.
    ///
    /// # Arguments
    ///
    /// * `model` - The loaded Qwen3 model
    /// * `model_config` - Model configuration
    /// * `tokenizer` - Tokenizer for text encoding/decoding
    /// * `engine_config` - Engine configuration
    pub fn new(
        model: Qwen3ForCausalLM,
        model_config: Qwen3Config,
        tokenizer: Tokenizer,
        engine_config: EngineConfig,
    ) -> Result<Self> {
        let device = model.device().clone();
        let dtype = model.dtype();

        // Create scheduler config from engine config
        let scheduler_config = SchedulerConfig {
            max_num_seqs: engine_config.max_num_seqs,
            max_prefill_tokens: engine_config.max_prefill_tokens,
            enable_chunked_prefill: true,
            chunk_size: 512,
            enable_priority: true,
            enable_preemption: engine_config.enable_preemption,
        };

        let scheduler = Scheduler::new(
            scheduler_config,
            engine_config.block_size,
            engine_config.num_blocks,
        );

        // Get EOS token ID from tokenizer
        let eos_token_id = tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.token_to_id("</s>"))
            .or_else(|| tokenizer.token_to_id("<|im_end|>"))
            .unwrap_or(151643); // Qwen3 default EOS

        Ok(Self {
            model: Some(model),
            model_config,
            scheduler,
            tokenizer,
            sampling_configs: HashMap::new(),
            samplers: HashMap::new(),
            prompts: HashMap::new(),
            next_request_id: 1,
            device,
            dtype,
            eos_token_id,
            speculative_engine: None,
        })
    }

    /// Create a new LLMEngine with speculative decoding enabled.
    ///
    /// Uses separate target and draft models for speculative decoding.
    /// The target model (larger) is used for verification, while the
    /// draft model (smaller) generates speculative tokens.
    ///
    /// # Arguments
    ///
    /// * `target_model` - Large target model for verification (e.g., Qwen3-4B)
    /// * `draft_model` - Small draft model for speculation (e.g., Qwen3-0.6B)
    /// * `model_config` - Target model configuration
    /// * `tokenizer` - Tokenizer for text encoding/decoding
    /// * `engine_config` - Engine configuration
    /// * `speculative_config` - Speculative decoding configuration
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Load target model (larger)
    /// let target_model = Qwen3ForCausalLM::new(&target_config, false, target_vb)?;
    /// // Load draft model (smaller)
    /// let draft_model = Qwen3ForCausalLM::new(&draft_config, false, draft_vb)?;
    ///
    /// let engine = LLMEngine::new_with_speculative(
    ///     target_model,
    ///     draft_model,
    ///     target_config,
    ///     tokenizer,
    ///     engine_config,
    ///     SpeculativeConfig::default(),
    /// )?;
    /// ```
    pub fn new_with_speculative(
        target_model: Qwen3ForCausalLM,
        draft_model: Qwen3ForCausalLM,
        model_config: Qwen3Config,
        tokenizer: Tokenizer,
        engine_config: EngineConfig,
        speculative_config: SpeculativeConfig,
    ) -> Result<Self> {
        let device = target_model.device().clone();
        let dtype = target_model.dtype();

        // Create scheduler config from engine config
        let scheduler_config = SchedulerConfig {
            max_num_seqs: engine_config.max_num_seqs,
            max_prefill_tokens: engine_config.max_prefill_tokens,
            enable_chunked_prefill: true,
            chunk_size: 512,
            enable_priority: true,
            enable_preemption: engine_config.enable_preemption,
        };

        let scheduler = Scheduler::new(
            scheduler_config,
            engine_config.block_size,
            engine_config.num_blocks,
        );

        // Get EOS token ID from tokenizer
        let eos_token_id = tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.token_to_id("</s>"))
            .or_else(|| tokenizer.token_to_id("<|im_end|>"))
            .unwrap_or(151643); // Qwen3 default EOS

        // Create speculative engine with both models
        let speculative_engine = SpeculativeEngine::new(
            target_model,
            draft_model,
            speculative_config,
        );

        Ok(Self {
            // In speculative mode, model is None
            // All model access goes through speculative_engine
            model: None,
            model_config,
            scheduler,
            tokenizer,
            sampling_configs: HashMap::new(),
            samplers: HashMap::new(),
            prompts: HashMap::new(),
            next_request_id: 1,
            device,
            dtype,
            eos_token_id,
            speculative_engine: Some(speculative_engine),
        })
    }

    /// Add a generation request to the engine.
    ///
    /// Returns the assigned request ID.
    pub fn add_request(&mut self, request: GenerationRequest) -> Result<SequenceId> {
        // Assign request ID
        let request_id = request.request_id.unwrap_or_else(|| {
            let id = self.next_request_id;
            self.next_request_id += 1;
            id
        });

        // Tokenize the prompt
        let encoding = self
            .tokenizer
            .encode(request.prompt.as_str(), false)
            .map_err(|e| Error::Tokenization(e.to_string()))?;

        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

        if prompt_tokens.is_empty() {
            return Err(Error::Tokenization("Empty prompt".to_string()));
        }

        // Create sequence
        let mut sequence = Sequence::new(request_id, prompt_tokens);
        sequence.set_priority(request.priority);

        // Add to scheduler
        self.scheduler.add_sequence(sequence);

        // Store configs and create sampler
        self.sampling_configs
            .insert(request_id, request.sampling_config.clone());
        self.samplers
            .insert(request_id, Sampler::new(&request.sampling_config));
        self.prompts.insert(request_id, request.prompt);

        Ok(request_id)
    }

    /// Run a single step of inference.
    ///
    /// Returns the outputs for any completed sequences.
    pub fn step(&mut self) -> Result<Vec<GenerationOutput>> {
        // Get scheduling decision
        let scheduler_outputs = self.scheduler.schedule();

        if scheduler_outputs.is_empty() {
            return Ok(vec![]);
        }

        let mut completed = Vec::new();

        // Process prefill sequences
        for &seq_id in &scheduler_outputs.prefill_sequences {
            self.process_prefill(seq_id, &scheduler_outputs)?;
        }

        // Process decode sequences
        for &seq_id in &scheduler_outputs.decode_sequences {
            if let Some(output) = self.process_decode(seq_id)? {
                completed.push(output);
            }
        }

        Ok(completed)
    }

    /// Process prefill phase for a sequence.
    fn process_prefill(
        &mut self,
        seq_id: SequenceId,
        scheduler_outputs: &crate::scheduler::SchedulerOutputs,
    ) -> Result<()> {
        // Extract all needed data from sequence first to avoid borrow issues
        let (start, end, tokens, prompt_len) = {
            let sequence = self
                .scheduler
                .get_sequence(seq_id)
                .ok_or(Error::SequenceNotFound(seq_id))?;

            let num_prefilled = sequence.num_prefilled_tokens();
            let chunk_size = scheduler_outputs
                .prefill_chunk_sizes
                .get(&seq_id)
                .copied()
                .unwrap_or(sequence.prompt_len() - num_prefilled);

            let start = num_prefilled;
            let end = (start + chunk_size).min(sequence.prompt_len());
            let tokens: Vec<u32> = sequence.all_token_ids()[start..end].to_vec();
            let prompt_len = sequence.prompt_len();

            (start, end, tokens, prompt_len)
        };

        // Forward pass - use model or speculative_engine's target model
        let input_ids = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.forward_prefill(&input_ids, start)?;

        // If prefill is complete, sample first token
        if end == prompt_len {
            // Get sampler for this sequence
            let sampler = self
                .samplers
                .get_mut(&seq_id)
                .ok_or_else(|| Error::Config(format!("No sampler for sequence {seq_id}")))?;

            let new_tokens = sampler.sample(&logits)?;
            let new_token = new_tokens[0];

            // Update sequence via scheduler
            self.scheduler.append_token(seq_id, new_token)?;
            self.scheduler.mark_prefilled(seq_id, end)?;

            // Check for completion
            self.check_completion(seq_id, new_token)?;
        } else {
            // Update prefill progress
            self.scheduler.mark_prefilled(seq_id, end)?;
        }

        Ok(())
    }

    /// Forward pass for prefill phase.
    /// Uses the appropriate model depending on whether speculative decoding is enabled.
    fn forward_prefill(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        if let Some(ref mut spec_engine) = self.speculative_engine {
            // Use speculative engine's target model for prefill
            Ok(spec_engine.target_model_mut().forward(input_ids, start_pos)?)
        } else if let Some(ref mut model) = self.model {
            Ok(model.forward(input_ids, start_pos)?)
        } else {
            Err(Error::Config("No model available".to_string()))
        }
    }

    /// Process decode phase for a sequence.
    fn process_decode(&mut self, seq_id: SequenceId) -> Result<Option<GenerationOutput>> {
        // Route to speculative or standard decode
        if self.speculative_engine.is_some() {
            self.process_decode_speculative(seq_id)
        } else {
            self.process_decode_standard(seq_id)
        }
    }

    /// Standard decode: generate one token at a time.
    fn process_decode_standard(&mut self, seq_id: SequenceId) -> Result<Option<GenerationOutput>> {
        let sequence = self
            .scheduler
            .get_sequence(seq_id)
            .ok_or(Error::SequenceNotFound(seq_id))?;

        // Get all tokens for forward pass
        // Note: Without KV cache integration, we need to pass all tokens
        // to maintain context. This is inefficient but correct.
        // Future optimization: integrate KV cache for O(1) decode.
        let all_tokens = sequence.all_token_ids().to_vec();
        if all_tokens.is_empty() {
            return Err(Error::Config("Sequence has no tokens".to_string()));
        }

        // Forward pass with all tokens, returning logits for last position
        let input_ids = Tensor::new(all_tokens.as_slice(), &self.device)?.unsqueeze(0)?;

        let logits = if let Some(ref mut model) = self.model {
            model.forward(&input_ids, 0)?
        } else {
            return Err(Error::Config("No model available for standard decode".to_string()));
        };

        // Sample next token
        let sampler = self
            .samplers
            .get_mut(&seq_id)
            .ok_or_else(|| Error::Config(format!("No sampler for sequence {seq_id}")))?;

        let new_tokens = sampler.sample(&logits)?;
        let new_token = new_tokens[0];

        // Update sequence
        self.scheduler.append_token(seq_id, new_token)?;

        // Check for completion
        let output = self.check_completion(seq_id, new_token)?;

        Ok(output)
    }

    /// Speculative decode: generate multiple tokens using draft-verify.
    fn process_decode_speculative(&mut self, seq_id: SequenceId) -> Result<Option<GenerationOutput>> {
        let sequence = self
            .scheduler
            .get_sequence(seq_id)
            .ok_or(Error::SequenceNotFound(seq_id))?;

        // Get all tokens
        let all_tokens = sequence.all_token_ids().to_vec();
        if all_tokens.is_empty() {
            return Err(Error::Config("Sequence has no tokens".to_string()));
        }

        // Get sampling config for temperature
        let sampling_config = self
            .sampling_configs
            .get(&seq_id)
            .ok_or_else(|| Error::Config(format!("No sampling config for sequence {seq_id}")))?
            .clone();

        // Create input tensor
        let input_ids = Tensor::new(all_tokens.as_slice(), &self.device)?.unsqueeze(0)?;

        // Run speculative step
        let spec_engine = self.speculative_engine.as_mut()
            .ok_or_else(|| Error::Config("Speculative engine not available".to_string()))?;

        let new_tokens = spec_engine.speculative_step(
            &input_ids,
            0,
            sampling_config.temperature,
        )?;

        // Append all generated tokens and check for completion
        for &token in &new_tokens {
            self.scheduler.append_token(seq_id, token)?;

            // Check for completion after each token
            if let Some(output) = self.check_completion(seq_id, token)? {
                return Ok(Some(output));
            }
        }

        Ok(None)
    }

    /// Check if a sequence should be finished.
    fn check_completion(
        &mut self,
        seq_id: SequenceId,
        new_token: u32,
    ) -> Result<Option<GenerationOutput>> {
        let sampling_config = self
            .sampling_configs
            .get(&seq_id)
            .ok_or_else(|| Error::Config(format!("No sampling config for sequence {seq_id}")))?;

        let sequence = self
            .scheduler
            .get_sequence(seq_id)
            .ok_or(Error::SequenceNotFound(seq_id))?;

        let output_len = sequence.output_len();
        let finish_reason;

        // Check EOS token
        if new_token == self.eos_token_id {
            finish_reason = Some(FinishReason::EndOfSequence);
        }
        // Check max tokens
        else if output_len >= sampling_config.max_tokens {
            finish_reason = Some(FinishReason::MaxTokens);
        }
        // Check stop sequences (simplified - would need proper token matching)
        else {
            finish_reason = None;
        }

        if let Some(reason) = finish_reason {
            // Mark sequence as finished
            self.scheduler.finish_sequence(seq_id, reason);

            // Generate output
            let output = self.create_output(seq_id, reason)?;
            return Ok(Some(output));
        }

        Ok(None)
    }

    /// Create generation output for a finished sequence.
    fn create_output(
        &self,
        seq_id: SequenceId,
        finish_reason: FinishReason,
    ) -> Result<GenerationOutput> {
        let sequence = self
            .scheduler
            .get_sequence(seq_id)
            .ok_or(Error::SequenceNotFound(seq_id))?;

        let output_tokens = sequence.output_token_ids().to_vec();
        let prompt = self.prompts.get(&seq_id).cloned().unwrap_or_default();

        // Decode output tokens
        let output_text = self
            .tokenizer
            .decode(&output_tokens, true)
            .map_err(|e| Error::Tokenization(e.to_string()))?;

        Ok(GenerationOutput {
            request_id: seq_id,
            prompt,
            output_text,
            output_tokens,
            finish_reason: Some(finish_reason),
            total_tokens: sequence.total_len(),
        })
    }

    /// Run generation until all requests are complete.
    ///
    /// Returns outputs for all completed sequences.
    pub fn generate(&mut self) -> Result<Vec<GenerationOutput>> {
        let mut all_outputs = Vec::new();

        loop {
            let outputs = self.step()?;
            all_outputs.extend(outputs);

            // Check if all sequences are done
            if !self.has_pending_requests() {
                break;
            }
        }

        Ok(all_outputs)
    }

    /// Generate text from a single prompt (convenience method).
    ///
    /// # Arguments
    ///
    /// * `prompt` - Input text
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature
    ///
    /// # Returns
    ///
    /// Generated text
    pub fn generate_text(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<String> {
        let request = GenerationRequest::new(prompt)
            .max_tokens(max_tokens)
            .temperature(temperature);

        self.add_request(request)?;
        let outputs = self.generate()?;

        outputs
            .into_iter()
            .next()
            .map(|o| o.output_text)
            .ok_or_else(|| Error::Config("No output generated".to_string()))
    }

    /// Check if there are pending (waiting or running) requests.
    pub fn has_pending_requests(&self) -> bool {
        self.scheduler.has_pending_requests()
    }

    /// Get the number of pending requests.
    pub fn num_pending_requests(&self) -> usize {
        self.scheduler.num_waiting() + self.scheduler.num_running()
    }

    /// Get the model's device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the model's dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get reference to the tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get reference to the model configuration.
    pub fn model_config(&self) -> &Qwen3Config {
        &self.model_config
    }
}
