//! Attention implementations.
//!
//! This module contains:
//! - PagedAttention for block-based KV cache access
//! - FlashAttention for memory-efficient attention
//!
//! # CUDA Kernel
//!
//! The custom Flash Attention CUDA kernel is in `kernels/flash_attn_fwd.cu`.
//! It is compiled via `build.rs` when the `cuda` feature is enabled.
//! FFI bindings can be added later for integration.

pub mod flash;
pub mod paged;

pub use flash::{FlashAttentionConfig, flash_attention, flash_attention_cpu};
pub use paged::{paged_attention, prefill_attention, write_kv_to_cache};
