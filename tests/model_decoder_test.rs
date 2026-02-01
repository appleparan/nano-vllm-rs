//! Integration tests for Qwen3DecoderLayer.

use candle_core::{DType, Device, Tensor};
use nano_vllm::model::Qwen3DecoderLayer;

fn test_device() -> Device {
    Device::Cpu
}

#[test]
fn test_decoder_layer_creation() {
    let device = test_device();
    let layer = Qwen3DecoderLayer::new_random(
        64,      // hidden_size
        256,     // intermediate_size
        4,       // num_heads
        2,       // num_kv_heads
        16,      // head_dim
        1024,    // max_seq_len
        10000.0, // rope_theta
        1e-6,    // rms_norm_eps
        false,   // use_flash_attention
        DType::F32,
        &device,
    )
    .unwrap();

    assert_eq!(layer.self_attn().num_heads(), 4);
    assert_eq!(layer.self_attn().num_kv_heads(), 2);
    assert_eq!(layer.mlp().hidden_size(), 64);
    assert_eq!(layer.mlp().intermediate_size(), 256);
}

#[test]
fn test_decoder_layer_forward_shape() {
    let device = test_device();
    let hidden_size = 64;
    let batch_size = 2;
    let seq_len = 8;

    let layer = Qwen3DecoderLayer::new_random(
        hidden_size,
        256,
        4,
        2,
        16,
        1024,
        10000.0,
        1e-6,
        false, // use_flash_attention
        DType::F32,
        &device,
    )
    .unwrap();

    let x = Tensor::randn(0.0f32, 0.1, (batch_size, seq_len, hidden_size), &device).unwrap();

    let output = layer.forward(&x, 0, None, None).unwrap();

    // Output should have the same shape as input
    assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
}

#[test]
fn test_decoder_layer_with_kv_cache() {
    let device = test_device();
    let hidden_size = 64;
    let num_kv_heads = 2;
    let head_dim = 16;

    let layer = Qwen3DecoderLayer::new_random(
        hidden_size,
        256,
        4,
        num_kv_heads,
        head_dim,
        1024,
        10000.0,
        1e-6,
        false, // use_flash_attention
        DType::F32,
        &device,
    )
    .unwrap();

    // Prefill
    let x_prefill = Tensor::randn(0.0f32, 0.1, (1, 4, hidden_size), &device).unwrap();
    let mut kv_cache = (
        Tensor::zeros((1, 0, num_kv_heads, head_dim), DType::F32, &device).unwrap(),
        Tensor::zeros((1, 0, num_kv_heads, head_dim), DType::F32, &device).unwrap(),
    );

    let output1 = layer
        .forward(&x_prefill, 0, Some(&mut kv_cache), None)
        .unwrap();
    assert_eq!(output1.dims(), &[1, 4, hidden_size]);
    assert_eq!(kv_cache.0.dims(), &[1, 4, num_kv_heads, head_dim]);

    // Decode
    let x_decode = Tensor::randn(0.0f32, 0.1, (1, 1, hidden_size), &device).unwrap();
    let output2 = layer
        .forward(&x_decode, 4, Some(&mut kv_cache), None)
        .unwrap();
    assert_eq!(output2.dims(), &[1, 1, hidden_size]);
    assert_eq!(kv_cache.0.dims(), &[1, 5, num_kv_heads, head_dim]);
}

#[test]
fn test_decoder_layer_residual() {
    let device = test_device();
    let hidden_size = 32;

    let layer = Qwen3DecoderLayer::new_random(
        hidden_size,
        128,
        4,
        2,
        8,
        1024,
        10000.0,
        1e-6,
        false, // use_flash_attention
        DType::F32,
        &device,
    )
    .unwrap();

    // With residual connections, output should have similar magnitude to input
    // (unlike without residual where it could explode or vanish)
    let x = Tensor::randn(0.0f32, 1.0, (1, 4, hidden_size), &device).unwrap();
    let output = layer.forward(&x, 0, None, None).unwrap();

    // Check output is finite and reasonable
    let output_sum: f32 = output
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!(output_sum.is_finite());
    assert!(output_sum > 0.0);
}

#[test]
fn test_decoder_layer_with_qwen3_config() {
    // Qwen3-0.6B scaled down for testing
    let device = test_device();
    let hidden_size = 256;
    let intermediate_size = 768;
    let num_heads = 8;
    let num_kv_heads = 4;
    let head_dim = 32;

    let layer = Qwen3DecoderLayer::new_random(
        hidden_size,
        intermediate_size,
        num_heads,
        num_kv_heads,
        head_dim,
        4096,
        1000000.0, // Qwen3 uses 1M
        1e-6,
        false, // use_flash_attention
        DType::F32,
        &device,
    )
    .unwrap();

    let x = Tensor::randn(0.0f32, 0.1, (1, 4, hidden_size), &device).unwrap();
    let output = layer.forward(&x, 0, None, None).unwrap();

    assert_eq!(output.dims(), &[1, 4, hidden_size]);
}
