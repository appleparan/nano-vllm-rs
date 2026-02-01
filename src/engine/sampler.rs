//! Token sampling strategies.
//!
//! This module provides sampling methods for selecting the next token
//! from a probability distribution:
//!
//! - **Temperature**: Controls randomness (lower = more deterministic)
//! - **Top-k**: Limits choices to k most likely tokens
//! - **Top-p (nucleus)**: Limits choices to tokens covering p probability mass
//!
//! ## Sampling Pipeline
//!
//! ```text
//! Logits [vocab_size]
//!     │
//!     ▼ Temperature scaling
//! Logits / temperature
//!     │
//!     ▼ Top-k filtering (optional)
//! Keep top k tokens
//!     │
//!     ▼ Softmax
//! Probabilities
//!     │
//!     ▼ Top-p filtering (optional)
//! Cumulative prob ≤ p
//!     │
//!     ▼ Renormalize + Sample
//! Selected token
//! ```

use candle_core::{D, IndexOp, Result, Tensor};
use rand::SeedableRng;
use rand::distributions::Distribution;

use crate::config::SamplingConfig;

/// Token sampler with configurable sampling strategies.
#[derive(Debug, Clone)]
pub struct Sampler {
    /// Temperature for scaling logits.
    temperature: f32,
    /// Top-k value (0 = disabled).
    top_k: usize,
    /// Top-p value (1.0 = disabled).
    top_p: f32,
    /// Random number generator.
    rng: rand::rngs::StdRng,
}

impl Sampler {
    /// Creates a new sampler with the given configuration.
    pub fn new(config: &SamplingConfig) -> Self {
        Self {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            rng: rand::rngs::StdRng::from_entropy(),
        }
    }

    /// Creates a new sampler with a specific seed for reproducibility.
    pub fn with_seed(config: &SamplingConfig, seed: u64) -> Self {
        Self {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    /// Sample a token from logits.
    ///
    /// # Arguments
    ///
    /// * `logits` - Raw logits from the model [vocab_size] or [batch, vocab_size]
    ///
    /// # Returns
    ///
    /// Sampled token ID(s)
    pub fn sample(&mut self, logits: &Tensor) -> Result<Vec<u32>> {
        let dims = logits.dims();

        match dims.len() {
            1 => {
                // Single sequence: [vocab_size]
                let token = self.sample_single(logits)?;
                Ok(vec![token])
            }
            2 => {
                // Batch: [batch, vocab_size]
                let batch_size = dims[0];
                let mut tokens = Vec::with_capacity(batch_size);
                for i in 0..batch_size {
                    let seq_logits = logits.i(i)?;
                    let token = self.sample_single(&seq_logits)?;
                    tokens.push(token);
                }
                Ok(tokens)
            }
            _ => Err(candle_core::Error::Msg(format!(
                "Expected 1D or 2D logits, got {}D",
                dims.len()
            ))),
        }
    }

    /// Sample a single token from 1D logits.
    fn sample_single(&mut self, logits: &Tensor) -> Result<u32> {
        let device = logits.device();
        let vocab_size = logits.dim(0)?;

        // Apply temperature
        let logits = if self.temperature != 1.0 && self.temperature > 0.0 {
            (logits / self.temperature as f64)?
        } else {
            logits.clone()
        };

        // For temperature = 0, use greedy decoding (argmax)
        if self.temperature == 0.0 {
            return self.argmax(&logits);
        }

        // Apply top-k filtering
        let (logits, indices) = if self.top_k > 0 && self.top_k < vocab_size {
            self.apply_top_k(&logits, self.top_k)?
        } else {
            let indices: Vec<u32> = (0..vocab_size as u32).collect();
            let indices_tensor = Tensor::from_vec(indices.clone(), vocab_size, device)?;
            (logits, indices_tensor)
        };

        // Compute probabilities with softmax
        let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;

        // Apply top-p filtering
        let (probs, indices) = if self.top_p < 1.0 && self.top_p > 0.0 {
            self.apply_top_p(&probs, &indices, self.top_p)?
        } else {
            (probs, indices)
        };

        // Sample from the distribution
        self.sample_from_probs(&probs, &indices)
    }

    /// Greedy decoding: select the token with highest logit.
    fn argmax(&self, logits: &Tensor) -> Result<u32> {
        let max_idx = logits.argmax(D::Minus1)?;
        let token = max_idx.to_scalar::<u32>()?;
        Ok(token)
    }

    /// Apply top-k filtering: keep only the k tokens with highest logits.
    fn apply_top_k(&self, logits: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
        let device = logits.device();
        let vocab_size = logits.dim(0)?;
        let k = k.min(vocab_size);

        // Get logits as vec and sort
        let logits_vec: Vec<f32> = logits.to_vec1()?;
        let mut indexed: Vec<(usize, f32)> = logits_vec
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        let top_k: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();
        let indices: Vec<u32> = top_k.iter().map(|(i, _)| *i as u32).collect();
        let values: Vec<f32> = top_k.iter().map(|(_, v)| *v).collect();

        let indices_tensor = Tensor::from_vec(indices, k, device)?;
        let logits_tensor = Tensor::from_vec(values, k, device)?;

        Ok((logits_tensor, indices_tensor))
    }

    /// Apply top-p (nucleus) filtering: keep tokens until cumulative probability exceeds p.
    fn apply_top_p(&self, probs: &Tensor, indices: &Tensor, p: f32) -> Result<(Tensor, Tensor)> {
        let device = probs.device();
        let n = probs.dim(0)?;

        // Get probs and indices as vecs
        let probs_vec: Vec<f32> = probs.to_vec1()?;
        let indices_vec: Vec<u32> = indices.to_vec1()?;

        // Sort by probability (descending)
        let mut indexed: Vec<(usize, f32)> =
            probs_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find cutoff where cumulative prob exceeds p
        let mut cumulative = 0.0f32;
        let mut cutoff_idx = n;
        for (i, (_, prob)) in indexed.iter().enumerate() {
            cumulative += prob;
            if cumulative > p {
                cutoff_idx = i + 1; // Include the token that pushed us over
                break;
            }
        }

        // Take tokens up to cutoff
        let selected: Vec<(usize, f32)> = indexed.into_iter().take(cutoff_idx).collect();
        let new_probs: Vec<f32> = selected.iter().map(|(_, v)| *v).collect();
        let new_indices: Vec<u32> = selected.iter().map(|(i, _)| indices_vec[*i]).collect();

        let probs_tensor = Tensor::from_vec(new_probs, cutoff_idx, device)?;
        let indices_tensor = Tensor::from_vec(new_indices, cutoff_idx, device)?;

        Ok((probs_tensor, indices_tensor))
    }

    /// Sample from probability distribution using the stored RNG.
    fn sample_from_probs(&mut self, probs: &Tensor, indices: &Tensor) -> Result<u32> {
        let probs_vec: Vec<f32> = probs.to_vec1()?;
        let indices_vec: Vec<u32> = indices.to_vec1()?;

        // Renormalize probabilities
        let sum: f32 = probs_vec.iter().sum();
        let normalized: Vec<f64> = probs_vec.iter().map(|&p| (p / sum) as f64).collect();

        // Sample using weighted distribution
        let dist = rand::distributions::WeightedIndex::new(&normalized)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create distribution: {e}")))?;

        let sampled_idx = dist.sample(&mut self.rng);
        Ok(indices_vec[sampled_idx])
    }

    /// Set temperature.
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature;
    }

    /// Set top-k.
    pub fn set_top_k(&mut self, top_k: usize) {
        self.top_k = top_k;
    }

    /// Set top-p.
    pub fn set_top_p(&mut self, top_p: f32) {
        self.top_p = top_p;
    }
}
