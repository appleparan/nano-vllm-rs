//! Qwen3 MLP (Multi-Layer Perceptron) implementation.
//!
//! This module implements the SwiGLU feed-forward network used in Qwen3.
//! SwiGLU uses a gating mechanism with SiLU activation for better performance.
//!
//! Reference: <https://arxiv.org/abs/2002.05202>

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

/// SiLU (Sigmoid Linear Unit) activation function.
///
/// Also known as Swish: `silu(x) = x * sigmoid(x)`
fn silu(x: &Tensor) -> Result<Tensor> {
    x.mul(&candle_nn::ops::sigmoid(x)?)
}

/// Qwen3 MLP with SwiGLU activation.
///
/// The MLP consists of three linear projections:
/// - `gate_proj`: Projects input to intermediate size and applies SiLU
/// - `up_proj`: Projects input to intermediate size
/// - `down_proj`: Projects gated output back to hidden size
///
/// Formula: `output = down_proj(silu(gate_proj(x)) * up_proj(x))`
#[derive(Debug, Clone)]
pub struct Qwen3Mlp {
    /// Gate projection with SiLU activation.
    gate_proj: Linear,
    /// Up projection.
    up_proj: Linear,
    /// Down projection.
    down_proj: Linear,
    /// Hidden dimension.
    hidden_size: usize,
    /// Intermediate dimension.
    intermediate_size: usize,
}

impl Qwen3Mlp {
    /// Creates a new Qwen3Mlp from a VarBuilder.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - Input/output dimension
    /// * `intermediate_size` - Intermediate (expanded) dimension
    /// * `vb` - VarBuilder for loading weights
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            hidden_size,
            intermediate_size,
        })
    }

    /// Creates a new Qwen3Mlp with random weights for testing.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - Input/output dimension
    /// * `intermediate_size` - Intermediate dimension
    /// * `dtype` - Data type for weights
    /// * `device` - Device to create tensors on
    pub fn new_random(
        hidden_size: usize,
        intermediate_size: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let _map = candle_nn::VarMap::new();

        // Initialize with random weights (small scale for stability)
        let scale = 0.02;
        let gate_weight = Tensor::randn(0.0f32, scale, (intermediate_size, hidden_size), device)?
            .to_dtype(dtype)?;
        let up_weight = Tensor::randn(0.0f32, scale, (intermediate_size, hidden_size), device)?
            .to_dtype(dtype)?;
        let down_weight = Tensor::randn(0.0f32, scale, (hidden_size, intermediate_size), device)?
            .to_dtype(dtype)?;

        let gate_proj = Linear::new(gate_weight, None);
        let up_proj = Linear::new(up_weight, None);
        let down_proj = Linear::new(down_weight, None);

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            hidden_size,
            intermediate_size,
        })
    }

    /// Returns the hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Returns the intermediate size.
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    /// Forward pass through the MLP.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [..., hidden_size]
    ///
    /// # Returns
    ///
    /// Output tensor of shape [..., hidden_size]
    ///
    /// # Formula
    ///
    /// ```text
    /// gate = silu(gate_proj(x))  # [..., intermediate_size]
    /// up = up_proj(x)            # [..., intermediate_size]
    /// output = down_proj(gate * up)  # [..., hidden_size]
    /// ```
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // gate_proj + silu
        let gate = silu(&self.gate_proj.forward(x)?)?;

        // up_proj
        let up = self.up_proj.forward(x)?;

        // Element-wise multiplication
        let gated = gate.mul(&up)?;

        // down_proj
        self.down_proj.forward(&gated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> Device {
        Device::Cpu
    }

    #[test]
    fn test_silu() {
        let device = test_device();
        let x = Tensor::new(&[-1.0f32, 0.0, 1.0, 2.0], &device).unwrap();
        let y = silu(&x).unwrap();
        let values: Vec<f32> = y.to_vec1().unwrap();

        // silu(x) = x * sigmoid(x)
        // silu(-1) = -1 * 0.2689 ≈ -0.2689
        // silu(0) = 0
        // silu(1) = 1 * 0.7311 ≈ 0.7311
        // silu(2) = 2 * 0.8808 ≈ 1.7616
        assert!((values[0] - (-0.2689)).abs() < 0.01);
        assert!(values[1].abs() < 0.001);
        assert!((values[2] - 0.7311).abs() < 0.01);
        assert!((values[3] - 1.7616).abs() < 0.01);
    }

    #[test]
    fn test_mlp_creation() {
        let device = test_device();
        let mlp = Qwen3Mlp::new_random(64, 256, DType::F32, &device).unwrap();

        assert_eq!(mlp.hidden_size(), 64);
        assert_eq!(mlp.intermediate_size(), 256);
    }

    #[test]
    fn test_mlp_forward_shape() {
        let device = test_device();
        let hidden_size = 64;
        let intermediate_size = 256;
        let batch_size = 2;
        let seq_len = 4;

        let mlp = Qwen3Mlp::new_random(hidden_size, intermediate_size, DType::F32, &device).unwrap();

        // Input: [batch, seq_len, hidden_size]
        let x =
            Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device).unwrap();

        let output = mlp.forward(&x).unwrap();

        // Output should be [batch, seq_len, hidden_size]
        assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_mlp_forward_2d() {
        let device = test_device();
        let hidden_size = 32;
        let intermediate_size = 128;
        let seq_len = 8;

        let mlp = Qwen3Mlp::new_random(hidden_size, intermediate_size, DType::F32, &device).unwrap();

        // 2D input: [seq_len, hidden_size]
        let x = Tensor::randn(0.0f32, 1.0, (seq_len, hidden_size), &device).unwrap();

        let output = mlp.forward(&x).unwrap();

        // Output should be [seq_len, hidden_size]
        assert_eq!(output.dims(), &[seq_len, hidden_size]);
    }

    #[test]
    fn test_mlp_with_qwen3_config() {
        // Qwen3-0.6B: hidden_size=1024, intermediate_size=3072
        let device = test_device();
        let mlp = Qwen3Mlp::new_random(1024, 3072, DType::F32, &device).unwrap();

        let x = Tensor::randn(0.0f32, 0.1, (1, 4, 1024), &device).unwrap();
        let output = mlp.forward(&x).unwrap();

        assert_eq!(output.dims(), &[1, 4, 1024]);
    }
}
