//! Integration tests for RmsNorm.

use candle_core::{DType, Device, Tensor};
use nano_vllm::model::RmsNorm;

fn test_device() -> Device {
    Device::Cpu
}

#[test]
fn test_rmsnorm_creation() {
    let device = test_device();
    let weight = Tensor::ones(64, DType::F32, &device).unwrap();
    let norm = RmsNorm::new(weight, 1e-6);

    assert_eq!(norm.eps(), 1e-6);
    assert_eq!(norm.weight().dims(), &[64]);
}

#[test]
fn test_rmsnorm_new_ones() {
    let device = test_device();
    let norm = RmsNorm::new_ones(128, 1e-6, DType::F32, &device).unwrap();

    assert_eq!(norm.weight().dims(), &[128]);
}

#[test]
fn test_rmsnorm_forward_shape() {
    let device = test_device();
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
    let device = test_device();
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
    let device = test_device();

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
