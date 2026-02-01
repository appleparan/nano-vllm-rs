//! RMS Normalization implementation.
//!
//! Root Mean Square Layer Normalization is a simplified version of LayerNorm
//! that only rescales inputs by their RMS value, without centering (no mean subtraction).
//!
//! Reference: <https://arxiv.org/abs/1910.07467>

use candle_core::{DType, Result, Tensor};

/// RMS Normalization layer.
///
/// Unlike LayerNorm, RMSNorm only normalizes by the root mean square,
/// without subtracting the mean. This is computationally simpler and
/// has been shown to work well in transformer models.
///
/// Formula: `output = (x / rms(x)) * weight`
/// where `rms(x) = sqrt(mean(x^2) + eps)`
#[derive(Debug, Clone)]
pub struct RmsNorm {
    /// Learnable scale parameter [hidden_size].
    weight: Tensor,
    /// Small constant for numerical stability.
    eps: f64,
}

impl RmsNorm {
    /// Creates a new RmsNorm layer.
    ///
    /// # Arguments
    ///
    /// * `weight` - Learnable scale tensor of shape [hidden_size]
    /// * `eps` - Small constant for numerical stability (typically 1e-6)
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Creates a new RmsNorm layer with ones as weights.
    ///
    /// Useful for testing or initialization before loading weights.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - Size of the hidden dimension
    /// * `eps` - Small constant for numerical stability
    /// * `dtype` - Data type for the weight tensor
    /// * `device` - Device to create the tensor on
    pub fn new_ones(
        hidden_size: usize,
        eps: f64,
        dtype: DType,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let weight = Tensor::ones(hidden_size, dtype, device)?;
        Ok(Self { weight, eps })
    }

    /// Returns a reference to the weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Returns the epsilon value.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Applies RMS normalization to the input tensor.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [..., hidden_size]
    ///
    /// # Returns
    ///
    /// Normalized tensor of the same shape as input.
    ///
    /// # Formula
    ///
    /// ```text
    /// rms = sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
    /// output = (x / rms) * weight
    /// ```
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [..., hidden_size]
        // Compute x^2
        let x_squared = x.sqr()?;

        // Compute mean along the last dimension, keeping dims for broadcasting
        // mean(x^2) -> [..., 1]
        let variance = x_squared.mean_keepdim(candle_core::D::Minus1)?;

        // rms = sqrt(variance + eps)
        let rms = (variance + self.eps)?.sqrt()?;

        // Normalize: x / rms
        let normalized = x.broadcast_div(&rms)?;

        // Scale by weight
        normalized.broadcast_mul(&self.weight)
    }
}
