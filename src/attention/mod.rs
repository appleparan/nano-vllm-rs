//! Attention implementations.
//!
//! This module contains:
//! - PagedAttention for block-based KV cache access
//! - FlashAttention for memory-efficient attention

pub mod flash;
pub mod paged;

pub use flash::{flash_attention, flash_attention_cpu, FlashAttentionConfig};
pub use paged::{paged_attention, prefill_attention, write_kv_to_cache};
