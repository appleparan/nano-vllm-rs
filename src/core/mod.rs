//! Core infrastructure for nano-vllm.
//!
//! This module contains the fundamental building blocks:
//! - Block and BlockTable for PagedAttention
//! - BlockManager for memory allocation
//! - Sequence for request tracking
//! - KVCache for key-value storage

pub mod block;

// TODO: Stage 2 - BlockManager
// pub mod block_manager;

// TODO: Stage 3 - Sequence & KV Cache
// pub mod sequence;
// pub mod kv_cache;
