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
