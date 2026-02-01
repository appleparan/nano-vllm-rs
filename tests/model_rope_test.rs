//! Integration tests for RotaryEmbedding.

use candle_core::{DType, Device, Tensor};
use nano_vllm::model::RotaryEmbedding;

fn test_device() -> Device {
    Device::Cpu
}

#[test]
fn test_rope_creation() {
    let rope = RotaryEmbedding::new(64, 1024, 10000.0, DType::F32, &test_device()).unwrap();

    assert_eq!(rope.dim(), 64);
    assert_eq!(rope.cos_cache().dims(), &[1024, 64]);
    assert_eq!(rope.sin_cache().dims(), &[1024, 64]);
}

#[test]
fn test_rope_apply_shape() {
    let device = test_device();
    let batch = 2;
    let seq_len = 8;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 64;

    let rope = RotaryEmbedding::new(head_dim, 1024, 10000.0, DType::F32, &device).unwrap();

    // Create random q and k
    let q = Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();
    let k = Tensor::randn(
        0.0f32,
        1.0,
        (batch, seq_len, num_kv_heads, head_dim),
        &device,
    )
    .unwrap();

    let (q_rot, k_rot) = rope.apply(&q, &k, 0).unwrap();

    // Check shapes preserved
    assert_eq!(q_rot.dims(), &[batch, seq_len, num_heads, head_dim]);
    assert_eq!(k_rot.dims(), &[batch, seq_len, num_kv_heads, head_dim]);
}

#[test]
fn test_rope_incremental_decode() {
    let device = test_device();
    let batch = 1;
    let num_heads = 2;
    let head_dim = 32;

    let rope = RotaryEmbedding::new(head_dim, 1024, 10000.0, DType::F32, &device).unwrap();

    // Prefill: positions 0-3
    let q_prefill = Tensor::randn(0.0f32, 1.0, (batch, 4, num_heads, head_dim), &device).unwrap();
    let k_prefill = Tensor::randn(0.0f32, 1.0, (batch, 4, num_heads, head_dim), &device).unwrap();
    let (_q1, _k1) = rope.apply(&q_prefill, &k_prefill, 0).unwrap();

    // Decode: position 4
    let q_decode = Tensor::randn(0.0f32, 1.0, (batch, 1, num_heads, head_dim), &device).unwrap();
    let k_decode = Tensor::randn(0.0f32, 1.0, (batch, 1, num_heads, head_dim), &device).unwrap();
    let (q2, k2) = rope.apply(&q_decode, &k_decode, 4).unwrap();

    // Check shapes for decode
    assert_eq!(q2.dims(), &[batch, 1, num_heads, head_dim]);
    assert_eq!(k2.dims(), &[batch, 1, num_heads, head_dim]);
}

#[test]
fn test_rotate_half() {
    let device = test_device();
    let rope = RotaryEmbedding::new(4, 1024, 10000.0, DType::F32, &device).unwrap();

    // Input: [1, 2, 3, 4]
    let x = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)
        .unwrap()
        .reshape((1, 1, 1, 4))
        .unwrap();

    // At position 0, cos=1 and sin=0 for most frequencies, so values stay similar.
    // Use position 5 where sine values are non-zero to verify rotation works.
    let (rotated, _) = rope.apply(&x, &x, 5).unwrap();

    // Check that rotation at non-zero position produces different values
    let rotated_vec: Vec<f32> = rotated.flatten_all().unwrap().to_vec1().unwrap();
    let orig_vec: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();

    // At position 5, rotation should change the values
    assert!(
        rotated_vec != orig_vec,
        "Rotation should produce different values at non-zero position"
    );
}

#[test]
fn test_rope_position_encoding_varies() {
    let device = test_device();
    let head_dim = 16;

    let rope = RotaryEmbedding::new(head_dim, 100, 10000.0, DType::F32, &device).unwrap();

    // Same input at different positions should give different outputs
    let x = Tensor::ones((1, 1, 1, head_dim), DType::F32, &device).unwrap();
    let x_clone = x.clone();

    let (q0, _) = rope.apply(&x, &x, 0).unwrap();
    let (q1, _) = rope.apply(&x_clone, &x_clone, 1).unwrap();

    // The rotated values should be different
    let diff = (&q0 - &q1).unwrap().abs().unwrap().sum_all().unwrap();
    let diff_val: f32 = diff.to_scalar().unwrap();

    // Difference should be non-zero
    assert!(diff_val > 1e-5);
}

#[test]
fn test_rope_with_qwen3_theta() {
    // Qwen3 uses theta=1000000
    let device = test_device();
    let rope = RotaryEmbedding::new(128, 40960, 1000000.0, DType::F32, &device).unwrap();

    assert_eq!(rope.dim(), 128);
    assert_eq!(rope.cos_cache().dims(), &[40960, 128]);
}
