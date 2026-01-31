//! Error types for nano-vllm.

use thiserror::Error;

/// Result type alias for nano-vllm operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for nano-vllm.
#[derive(Error, Debug)]
pub enum Error {
    /// Block allocation failed - no free blocks available.
    #[error("out of KV cache blocks")]
    OutOfBlocks,

    /// Block index out of bounds.
    #[error("logical block {logical_idx} not allocated (table has {num_blocks} blocks)")]
    BlockIndexOutOfBounds {
        /// The requested logical block index.
        logical_idx: usize,
        /// Number of blocks in the table.
        num_blocks: usize,
    },

    /// Sequence not found in scheduler.
    #[error("sequence {0} not found")]
    SequenceNotFound(u64),

    /// Invalid sequence state transition.
    #[error("invalid state transition: {from:?} -> {to:?}")]
    InvalidStateTransition {
        /// The current state.
        from: &'static str,
        /// The target state.
        to: &'static str,
    },

    /// Model loading failed.
    #[error("failed to load model: {0}")]
    ModelLoad(String),

    /// Tokenization error.
    #[error("tokenization error: {0}")]
    Tokenization(String),

    /// Tensor operation error.
    #[error("tensor error: {0}")]
    Tensor(#[from] candle_core::Error),

    /// Configuration error.
    #[error("configuration error: {0}")]
    Config(String),

    /// IO error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parsing error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}
