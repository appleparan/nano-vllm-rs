//! Model loading utilities.
//!
//! This module provides functions for:
//! - Downloading models from HuggingFace Hub
//! - Loading SafeTensors weights
//! - Creating VarBuilder for model initialization

use std::path::PathBuf;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{Repo, RepoType, api::sync::Api};

use crate::error::{Error, Result};

/// Downloads model files from HuggingFace Hub.
///
/// # Arguments
///
/// * `model_id` - HuggingFace model ID (e.g., "Qwen/Qwen3-0.6B")
/// * `revision` - Git revision (branch, tag, or commit hash). Use "main" for latest.
///
/// # Returns
///
/// Paths to downloaded files: (config.json, model weights, tokenizer files)
pub fn download_model(model_id: &str, revision: &str) -> Result<ModelFiles> {
    let api = Api::new().map_err(|e| Error::ModelLoad(format!("Failed to create HF API: {e}")))?;

    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    // Download config.json
    let config_path = repo
        .get("config.json")
        .map_err(|e| Error::ModelLoad(format!("Failed to download config.json: {e}")))?;

    // Download model weights (try safetensors first, then pytorch)
    let weights_paths = download_weights(&repo)?;

    // Download tokenizer files
    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| Error::ModelLoad(format!("Failed to download tokenizer.json: {e}")))?;

    Ok(ModelFiles {
        config: config_path,
        weights: weights_paths,
        tokenizer: tokenizer_path,
    })
}

/// Downloads model weight files.
fn download_weights(repo: &hf_hub::api::sync::ApiRepo) -> Result<Vec<PathBuf>> {
    // Try to get model.safetensors first (single file)
    if let Ok(path) = repo.get("model.safetensors") {
        return Ok(vec![path]);
    }

    // Try to get model.safetensors.index.json for sharded models
    if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        let index_content = std::fs::read_to_string(&index_path)
            .map_err(|e| Error::ModelLoad(format!("Failed to read safetensors index: {e}")))?;

        let index: serde_json::Value = serde_json::from_str(&index_content)
            .map_err(|e| Error::ModelLoad(format!("Failed to parse safetensors index: {e}")))?;

        // Extract unique shard filenames from weight_map
        let weight_map = index["weight_map"].as_object().ok_or_else(|| {
            Error::ModelLoad("Invalid safetensors index: missing weight_map".into())
        })?;

        let mut shard_files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .collect();
        shard_files.sort();
        shard_files.dedup();

        let mut paths = Vec::new();
        for filename in shard_files {
            let path = repo
                .get(&filename)
                .map_err(|e| Error::ModelLoad(format!("Failed to download {filename}: {e}")))?;
            paths.push(path);
        }
        return Ok(paths);
    }

    Err(Error::ModelLoad(
        "No SafeTensors weights found. This implementation only supports SafeTensors format."
            .into(),
    ))
}

/// Paths to downloaded model files.
#[derive(Debug, Clone)]
pub struct ModelFiles {
    /// Path to config.json.
    pub config: PathBuf,
    /// Paths to weight files (SafeTensors).
    pub weights: Vec<PathBuf>,
    /// Path to tokenizer.json.
    pub tokenizer: PathBuf,
}

/// Creates a VarBuilder from SafeTensors files.
///
/// # Arguments
///
/// * `paths` - Paths to SafeTensors files
/// * `dtype` - Data type for tensors
/// * `device` - Device to load tensors to
///
/// # Returns
///
/// VarBuilder for loading model weights
///
/// # Safety
///
/// Uses memory-mapped file access for efficient loading of large model weights.
/// This is safe as long as the files are not modified while being read.
#[allow(unsafe_code)]
pub fn load_safetensors(
    paths: &[PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, device)? };
    Ok(vb)
}

/// Loads Qwen3 configuration from config.json.
///
/// # Arguments
///
/// * `path` - Path to config.json
///
/// # Returns
///
/// Parsed Qwen3Config
pub fn load_config(path: &PathBuf) -> Result<Qwen3Config> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| Error::ModelLoad(format!("Failed to read config.json: {e}")))?;

    let config: Qwen3Config = serde_json::from_str(&content)
        .map_err(|e| Error::ModelLoad(format!("Failed to parse config.json: {e}")))?;

    Ok(config)
}

/// Qwen3 model configuration from HuggingFace config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Intermediate dimension (MLP).
    pub intermediate_size: usize,
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA).
    pub num_key_value_heads: usize,
    /// Dimension per attention head.
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    /// RMSNorm epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    /// RoPE theta.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// Maximum sequence length.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    /// Whether to tie word embeddings with lm_head.
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
}

fn default_head_dim() -> usize {
    128
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_rope_theta() -> f64 {
    1000000.0
}

fn default_max_position_embeddings() -> usize {
    40960
}

fn default_tie_word_embeddings() -> bool {
    true
}

use serde::Deserialize;

impl Qwen3Config {
    /// Convert to our internal ModelConfig.
    pub fn to_model_config(&self) -> crate::config::ModelConfig {
        crate::config::ModelConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
        }
    }
}
