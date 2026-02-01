//! Speculative decoding configuration.

use serde::{Deserialize, Serialize};

/// Configuration for speculative decoding.
///
/// Speculative decoding uses a small draft model to generate K tokens,
/// then verifies them with the larger target model in a single forward pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    /// Number of tokens to speculate per iteration (K).
    /// Higher values can improve throughput if acceptance rate is high,
    /// but may waste computation if many tokens are rejected.
    pub num_speculative_tokens: usize,

    /// Draft model HuggingFace ID.
    /// Should be a smaller, faster model with the same vocabulary.
    /// Example: "Qwen/Qwen3-0.6B" as draft for "Qwen/Qwen3-4B" target.
    pub draft_model_id: String,

    /// Draft model revision (branch, tag, or commit hash).
    pub draft_revision: String,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            num_speculative_tokens: 4,
            draft_model_id: "Qwen/Qwen3-0.6B".to_string(),
            draft_revision: "main".to_string(),
        }
    }
}

impl SpeculativeConfig {
    /// Create a new speculative config with the given draft model.
    pub fn new(draft_model_id: impl Into<String>) -> Self {
        Self {
            draft_model_id: draft_model_id.into(),
            ..Default::default()
        }
    }

    /// Set the number of speculative tokens.
    pub fn num_tokens(mut self, k: usize) -> Self {
        self.num_speculative_tokens = k;
        self
    }

    /// Set the draft model revision.
    pub fn revision(mut self, revision: impl Into<String>) -> Self {
        self.draft_revision = revision.into();
        self
    }
}
