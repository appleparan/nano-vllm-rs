//! Model implementations.
//!
//! This module contains:
//! - Model loading from HuggingFace
//! - Qwen3 architecture (RMSNorm, RoPE, GQA, SwiGLU)

pub mod attention;
pub mod decoder;
pub mod loader;
pub mod mlp;
pub mod norm;
pub mod qwen3;
pub mod rope;

pub use attention::Qwen3Attention;
pub use decoder::Qwen3DecoderLayer;
pub use loader::{download_model, load_config, load_safetensors, ModelFiles, Qwen3Config};
pub use mlp::Qwen3Mlp;
pub use norm::RmsNorm;
pub use qwen3::{Qwen3ForCausalLM, Qwen3Model};
pub use rope::RotaryEmbedding;
