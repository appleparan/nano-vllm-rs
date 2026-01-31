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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_rmsnorm_creation() {
        let device = Device::Cpu;
        let weight = Tensor::ones(64, DType::F32, &device).unwrap();
        let norm = RmsNorm::new(weight, 1e-6);

        assert_eq!(norm.eps(), 1e-6);
        assert_eq!(norm.weight().dims(), &[64]);
    }

    #[test]
    fn test_rmsnorm_new_ones() {
        let device = Device::Cpu;
        let norm = RmsNorm::new_ones(128, 1e-6, DType::F32, &device).unwrap();

        assert_eq!(norm.weight().dims(), &[128]);
    }

    #[test]
    fn test_rmsnorm_forward_shape() {
        let device = Device::Cpu;
        let hidden_size = 64;
        let batch_size = 2;
        let seq_len = 4;

        let norm = RmsNorm::new_ones(hidden_size, 1e-6, DType::F32, &device).unwrap();

        // Input: [batch, seq_len, hidden_size]
        let x = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device).unwrap();

        let output = norm.forward(&x).unwrap();

        // Output should have the same shape
        assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_rmsnorm_forward_values() {
        let device = Device::Cpu;
        let hidden_size = 4;

        // Create norm with weight = [1, 1, 1, 1]
        let norm = RmsNorm::new_ones(hidden_size, 1e-6, DType::F32, &device).unwrap();

        // Simple input: [1, 1, 4] with values [1, 2, 3, 4]
        let x = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)
            .unwrap()
            .reshape((1, 1, 4))
            .unwrap();

        let output = norm.forward(&x).unwrap();

        // Expected RMS = sqrt((1 + 4 + 9 + 16) / 4 + eps) = sqrt(7.5 + eps) ≈ 2.739
        // Expected output ≈ [1/2.739, 2/2.739, 3/2.739, 4/2.739] ≈ [0.365, 0.730, 1.095, 1.461]
        let output_vec: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();

        // Check approximate values
        let rms = (7.5f32 + 1e-6).sqrt();
        assert!((output_vec[0] - 1.0 / rms).abs() < 1e-5);
        assert!((output_vec[1] - 2.0 / rms).abs() < 1e-5);
        assert!((output_vec[2] - 3.0 / rms).abs() < 1e-5);
        assert!((output_vec[3] - 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn test_rmsnorm_with_custom_weight() {
        let device = Device::Cpu;

        // Create norm with weight = [2, 2, 2, 2]
        let weight = Tensor::new(&[2.0f32, 2.0, 2.0, 2.0], &device).unwrap();
        let norm = RmsNorm::new(weight, 1e-6);

        let x = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)
            .unwrap()
            .reshape((1, 1, 4))
            .unwrap();

        let output = norm.forward(&x).unwrap();
        let output_vec: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();

        // With weight=2, output should be 2x the unit weight case
        let rms = (7.5f32 + 1e-6).sqrt();
        assert!((output_vec[0] - 2.0 * 1.0 / rms).abs() < 1e-5);
        assert!((output_vec[1] - 2.0 * 2.0 / rms).abs() < 1e-5);
    }
}
