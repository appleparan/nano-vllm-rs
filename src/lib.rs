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

pub use config::{EngineConfig, ModelConfig, SamplingConfig, SchedulerConfig};
pub use error::{Error, Result};
pub use scheduler::{Scheduler, SchedulerOutputs};
