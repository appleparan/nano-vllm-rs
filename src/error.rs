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

    /// Sequence not found in scheduler.
    #[error("sequence {0} not found")]
    SequenceNotFound(u64),

    /// Invalid sequence state transition.
    #[error("invalid state transition: {from:?} -> {to:?}")]
    InvalidStateTransition {
        from: &'static str,
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
