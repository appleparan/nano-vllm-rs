//! Core infrastructure for nano-vllm.
//!
//! This module contains the fundamental building blocks:
//! - Block and BlockTable for PagedAttention
//! - BlockManager for memory allocation
//! - Sequence for request tracking
//! - KVCache for key-value storage

pub mod block;
pub mod block_manager;
pub mod kv_cache;
pub mod sequence;
