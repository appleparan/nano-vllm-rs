//! Integration tests for Qwen3Attention.

use candle_core::{DType, Device, Tensor};
use nano_vllm::model::Qwen3Attention;

fn test_device() -> Device {
    Device::Cpu
}

#[test]
fn test_attention_creation() {
    let device = test_device();
    let attn = Qwen3Attention::new_random(
        64,      // hidden_size
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

    assert_eq!(attn.num_heads(), 4);
    assert_eq!(attn.num_kv_heads(), 2);
    assert_eq!(attn.head_dim(), 16);
}

#[test]
fn test_attention_forward_shape() {
    let device = test_device();
    let hidden_size = 64;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 16;
    let batch_size = 2;
    let seq_len = 8;

    let attn = Qwen3Attention::new_random(
        hidden_size,
        num_heads,
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

    let x = Tensor::randn(0.0f32, 0.1, (batch_size, seq_len, hidden_size), &device).unwrap();

    let output = attn.forward(&x, 0, None, None).unwrap();

    assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
}

#[test]
fn test_attention_with_kv_cache() {
    let device = test_device();
    let hidden_size = 64;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 16;

    let attn = Qwen3Attention::new_random(
        hidden_size,
        num_heads,
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

    // Prefill: 4 tokens
    let x_prefill = Tensor::randn(0.0f32, 0.1, (1, 4, hidden_size), &device).unwrap();

    // Test with fresh KV cache
    let mut kv_cache = (
        Tensor::zeros((1, 0, num_kv_heads, head_dim), DType::F32, &device).unwrap(),
        Tensor::zeros((1, 0, num_kv_heads, head_dim), DType::F32, &device).unwrap(),
    );
    let _ = attn
        .forward(&x_prefill, 0, Some(&mut kv_cache), None)
        .unwrap();

    assert_eq!(kv_cache.0.dims(), &[1, 4, num_kv_heads, head_dim]);
    assert_eq!(kv_cache.1.dims(), &[1, 4, num_kv_heads, head_dim]);

    // Decode: 1 token at position 4
    let x_decode = Tensor::randn(0.0f32, 0.1, (1, 1, hidden_size), &device).unwrap();
    let output2 = attn
        .forward(&x_decode, 4, Some(&mut kv_cache), None)
        .unwrap();

    assert_eq!(output2.dims(), &[1, 1, hidden_size]);
    // Cache should now have 5 entries
    assert_eq!(kv_cache.0.dims(), &[1, 5, num_kv_heads, head_dim]);
}

#[test]
fn test_repeat_kv() {
    let device = test_device();
    let attn =
        Qwen3Attention::new_random(64, 4, 2, 16, 1024, 10000.0, 1e-6, false, DType::F32, &device)
            .unwrap();

    // Test via forward pass - GQA expansion happens internally
    let x = Tensor::randn(0.0f32, 0.1, (1, 4, 64), &device).unwrap();
    let output = attn.forward(&x, 0, None, None).unwrap();

    // Output should have correct shape
    assert_eq!(output.dims(), &[1, 4, 64]);
}

#[test]
fn test_causal_mask_prefill() {
    let device = test_device();
    let attn =
        Qwen3Attention::new_random(64, 4, 2, 16, 1024, 10000.0, 1e-6, false, DType::F32, &device)
            .unwrap();

    // Test via forward pass - causal masking is applied internally
    let x = Tensor::randn(0.0f32, 0.1, (1, 4, 64), &device).unwrap();
    let output = attn.forward(&x, 0, None, None).unwrap();

    // Verify output is finite
    let output_sum: f32 = output.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
    assert!(output_sum.is_finite());
}

#[test]
fn test_causal_mask_decode() {
    let device = test_device();
    let hidden_size = 64;
    let num_kv_heads = 2;
    let head_dim = 16;

    let attn =
        Qwen3Attention::new_random(hidden_size, 4, num_kv_heads, head_dim, 1024, 10000.0, 1e-6, false, DType::F32, &device)
            .unwrap();

    // Prefill first
    let x_prefill = Tensor::randn(0.0f32, 0.1, (1, 4, hidden_size), &device).unwrap();
    let mut kv_cache = (
        Tensor::zeros((1, 0, num_kv_heads, head_dim), DType::F32, &device).unwrap(),
        Tensor::zeros((1, 0, num_kv_heads, head_dim), DType::F32, &device).unwrap(),
    );
    let _ = attn.forward(&x_prefill, 0, Some(&mut kv_cache), None).unwrap();

    // Decode: single token can attend to all cached
    let x_decode = Tensor::randn(0.0f32, 0.1, (1, 1, hidden_size), &device).unwrap();
    let output = attn.forward(&x_decode, 4, Some(&mut kv_cache), None).unwrap();

    assert_eq!(output.dims(), &[1, 1, hidden_size]);
}

#[test]
fn test_attention_with_qwen3_config() {
    // Qwen3-0.6B config (scaled down for testing)
    let device = test_device();
    let hidden_size = 256; // Scaled down from 1024
    let num_heads = 8; // Scaled down from 16
    let num_kv_heads = 4; // Scaled down from 8
    let head_dim = 32; // Scaled down from 128

    let attn = Qwen3Attention::new_random(
        hidden_size,
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
    let output = attn.forward(&x, 0, None, None).unwrap();

    assert_eq!(output.dims(), &[1, 4, hidden_size]);
}

#[test]
fn test_attention_with_flash_attention() {
    // Test with Flash Attention enabled
    let device = test_device();
    let hidden_size = 64;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 16;
    let batch_size = 2;
    let seq_len = 8;

    let attn = Qwen3Attention::new_random(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        1024,
        10000.0,
        1e-6,
        true, // use_flash_attention
        DType::F32,
        &device,
    )
    .unwrap();

    let x = Tensor::randn(0.0f32, 0.1, (batch_size, seq_len, hidden_size), &device).unwrap();

    let output = attn.forward(&x, 0, None, None).unwrap();

    assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
}
