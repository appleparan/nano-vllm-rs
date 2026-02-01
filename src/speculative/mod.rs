//! Speculative decoding.
//!
//! This module implements draft-verify speculative decoding
//! for faster token generation.
//!
//! ## How it works
//!
//! 1. **Draft Phase**: A small, fast draft model (e.g., Qwen3-0.6B) generates
//!    K tokens speculatively.
//!
//! 2. **Verify Phase**: The larger target model (e.g., Qwen3-4B) processes all
//!    K+1 positions in a single forward pass.
//!
//! 3. **Rejection Sampling**: Each draft token is accepted or rejected based on
//!    the probability ratio between target and draft distributions.
//!
//! ## Example
//!
//! ```text
//! Draft (K=4):     [prompt] -> t1 -> t2 -> t3 -> t4
//! Target verify:   [prompt, t1, t2, t3, t4] -> logits for all 5 positions
//! Rejection:       Accept t1, t2, reject t3, resample -> final: t1, t2, t3'
//! ```
//!
//! ## Configuration
//!
//! - Target model: Qwen/Qwen3-4B (larger, slower, more accurate)
//! - Draft model: Qwen/Qwen3-0.6B (smaller, faster, approximate)
//! - K (num_speculative_tokens): 4 (typical range: 2-8)

pub mod config;
pub mod sampler;

pub use config::SpeculativeConfig;
pub use sampler::RejectionSampler;
