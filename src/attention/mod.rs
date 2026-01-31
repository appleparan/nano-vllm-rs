//! Attention implementations.
//!
//! This module contains:
//! - PagedAttention for block-based KV cache access
//! - FlashAttention integration (optional)

pub mod paged;

pub use paged::{paged_attention, prefill_attention, write_kv_to_cache};

// TODO: Stage 10 - FlashAttention integration
// pub mod flash;
