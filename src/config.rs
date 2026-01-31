//! Configuration types for nano-vllm.

use serde::{Deserialize, Serialize};

/// Engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Maximum number of sequences to process concurrently.
    pub max_num_seqs: usize,
    /// Maximum number of tokens to prefill per iteration.
    pub max_prefill_tokens: usize,
    /// Block size for PagedAttention (tokens per block).
    pub block_size: usize,
    /// Total number of blocks for KV cache.
    pub num_blocks: usize,
    /// Enable PagedAttention.
    pub use_paged_attention: bool,
    /// Enable prefix caching.
    pub enable_prefix_caching: bool,
    /// Enable priority-based preemption.
    pub enable_preemption: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_prefill_tokens: 4096,
            block_size: 16,
            num_blocks: 1024,
            use_paged_attention: true,
            enable_prefix_caching: true,
            enable_preemption: false,
        }
    }
}

/// Scheduler configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Maximum number of sequences to schedule per iteration.
    pub max_num_seqs: usize,
    /// Maximum tokens to prefill per iteration.
    pub max_prefill_tokens: usize,
    /// Enable chunked prefill for long prompts.
    pub enable_chunked_prefill: bool,
    /// Chunk size for chunked prefill.
    pub chunk_size: usize,
    /// Enable priority-based scheduling.
    pub enable_priority: bool,
    /// Enable preemption of low-priority sequences.
    pub enable_preemption: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_prefill_tokens: 4096,
            enable_chunked_prefill: true,
            chunk_size: 512,
            enable_priority: true,
            enable_preemption: false,
        }
    }
}

/// Sampling configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Temperature for sampling (1.0 = no change).
    pub temperature: f32,
    /// Top-k sampling (0 = disabled).
    pub top_k: usize,
    /// Top-p (nucleus) sampling (1.0 = disabled).
    pub top_p: f32,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Stop sequences.
    pub stop_sequences: Vec<String>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            max_tokens: 256,
            stop_sequences: Vec::new(),
        }
    }
}

/// Model configuration (Llama-style).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Intermediate dimension (MLP).
    pub intermediate_size: usize,
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA).
    pub num_key_value_heads: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// RoPE theta.
    pub rope_theta: f64,
    /// Maximum sequence length.
    pub max_position_embeddings: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        // TinyLlama-1.1B defaults
        Self {
            vocab_size: 32000,
            hidden_size: 2048,
            intermediate_size: 5632,
            num_hidden_layers: 22,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 2048,
        }
    }
}

impl ModelConfig {
    /// Head dimension (hidden_size / num_attention_heads).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Number of query heads per KV head group (for GQA).
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}
