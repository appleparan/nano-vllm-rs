//! nano-vllm: A minimalistic LLM inference engine in Rust.
//!
//! This crate implements core vLLM optimizations for educational purposes:
//! - PagedAttention for efficient KV cache management
//! - Continuous batching for high throughput
//! - Prefix caching for shared prompts
//! - Speculative decoding for faster generation

pub mod config;
pub mod error;

pub mod attention;
pub mod core;
pub mod engine;
pub mod model;
pub mod scheduler;
pub mod speculative;

pub use attention::{flash_attention, flash_attention_cpu, FlashAttentionConfig};
pub use config::{EngineConfig, ModelConfig, SamplingConfig, SchedulerConfig};
pub use engine::{GenerationOutput, GenerationRequest, LLMEngine, Sampler};
pub use error::{Error, Result};
pub use model::{
    download_model, load_config, load_safetensors, ModelFiles, Qwen3Config, Qwen3ForCausalLM,
};
pub use scheduler::{Scheduler, SchedulerOutputs};
