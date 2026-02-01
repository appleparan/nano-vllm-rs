//! nano-vllm: A minimalistic LLM inference engine in Rust.
//!
//! This crate implements core vLLM optimizations for educational purposes:
//! - PagedAttention for efficient KV cache management
//! - Continuous batching for high throughput
//! - Prefix caching for shared prompts
//! - Speculative decoding for faster generation
//!
//! ## Educational Modes
//!
//! The crate includes educational modes for learning LLM inference:
//! - **Narrator Mode**: Real-time plain-English commentary
//! - **X-Ray Mode**: Mathematical/tensor visualizations
//! - **Dashboard Mode**: Rich terminal UI with live updates
//! - **Tutorial Mode**: Interactive step-by-step learning

pub mod config;
pub mod educational;
pub mod error;

pub mod attention;
pub mod core;
pub mod engine;
pub mod model;
pub mod scheduler;
pub mod speculative;

pub use attention::{FlashAttentionConfig, flash_attention, flash_attention_cpu};
pub use config::{EngineConfig, ModelConfig, SamplingConfig, SchedulerConfig};
pub use engine::{GenerationOutput, GenerationRequest, LLMEngine, Sampler};
pub use error::{Error, Result};
pub use model::{
    ModelFiles, Qwen3Config, Qwen3ForCausalLM, download_model, load_config, load_safetensors,
};
pub use scheduler::{Scheduler, SchedulerOutputs};
pub use speculative::{RejectionSampler, SpeculativeConfig, SpeculativeEngine};

// Educational mode re-exports
pub use educational::{
    EducationalConfig, InferenceNarrator, InteractiveTutorial, NarratorConfig, XRayConfig,
    XRayVisualizer,
};
