//! Model implementations.
//!
//! This module contains:
//! - Model loading from HuggingFace
//! - Qwen3 architecture (RMSNorm, RoPE, GQA, SwiGLU)

pub mod attention;
pub mod decoder;
pub mod mlp;
pub mod norm;
pub mod rope;

pub use attention::Qwen3Attention;
pub use decoder::Qwen3DecoderLayer;
pub use mlp::Qwen3Mlp;
pub use norm::RmsNorm;
pub use rope::RotaryEmbedding;

// TODO: Stage 7 - Model Loader
// pub mod loader;
// pub mod config;
