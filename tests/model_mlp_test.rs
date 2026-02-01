//! Integration tests for Qwen3Mlp.

use candle_core::{DType, Device, Tensor};
use nano_vllm::model::Qwen3Mlp;

fn test_device() -> Device {
    Device::Cpu
}

fn silu(x: &Tensor) -> candle_core::Result<Tensor> {
    x.mul(&candle_nn::ops::sigmoid(x)?)
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
    let x = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device).unwrap();

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
