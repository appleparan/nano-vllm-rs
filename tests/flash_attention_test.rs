//! Integration tests for Flash Attention.

use candle_core::{Device, Tensor, D};
use nano_vllm::{flash_attention, flash_attention_cpu, FlashAttentionConfig};

fn test_device() -> Device {
    Device::Cpu
}

#[test]
fn test_flash_attention_basic() {
    let device = test_device();
    let batch = 1;
    let seq_len = 16;
    let num_heads = 4;
    let head_dim = 32;

    let q = Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();

    let config = FlashAttentionConfig::new(head_dim, true);
    let output = flash_attention(&q, &k, &v, &config).unwrap();

    assert_eq!(output.dims(), &[batch, seq_len, num_heads, head_dim]);
}

#[test]
fn test_flash_attention_various_sizes() {
    let device = test_device();

    // Test various sequence lengths and head dimensions
    let test_cases = [
        (1, 8, 2, 16),   // Small
        (2, 32, 4, 32),  // Medium
        (1, 64, 8, 64),  // Larger
        (2, 128, 4, 64), // Long sequence
    ];

    for (batch, seq_len, num_heads, head_dim) in test_cases {
        let q =
            Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();
        let k =
            Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();
        let v =
            Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();

        let config = FlashAttentionConfig::new(head_dim, true);
        let output = flash_attention_cpu(&q, &k, &v, &config).unwrap();

        assert_eq!(
            output.dims(),
            &[batch, seq_len, num_heads, head_dim],
            "Failed for size ({batch}, {seq_len}, {num_heads}, {head_dim})"
        );
    }
}

#[test]
fn test_flash_attention_non_causal() {
    let device = test_device();
    let batch = 1;
    let seq_len = 16;
    let num_heads = 2;
    let head_dim = 32;

    let q = Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();

    // Non-causal attention
    let config = FlashAttentionConfig::new(head_dim, false);
    let output = flash_attention_cpu(&q, &k, &v, &config).unwrap();

    assert_eq!(output.dims(), &[batch, seq_len, num_heads, head_dim]);
}

#[test]
fn test_flash_attention_gqa() {
    // Test Grouped Query Attention (fewer KV heads than Q heads)
    let device = test_device();
    let batch = 1;
    let seq_len = 16;
    let num_heads = 8;
    let num_kv_heads = 2; // GQA ratio = 4
    let head_dim = 32;

    let q = Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();
    let k =
        Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_kv_heads, head_dim), &device).unwrap();
    let v =
        Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_kv_heads, head_dim), &device).unwrap();

    let config = FlashAttentionConfig::new(head_dim, true);
    let output = flash_attention_cpu(&q, &k, &v, &config).unwrap();

    // Output should have same shape as query
    assert_eq!(output.dims(), &[batch, seq_len, num_heads, head_dim]);
}

#[test]
fn test_flash_attention_numerical_stability() {
    // Test with extreme values to check numerical stability
    let device = test_device();
    let batch = 1;
    let seq_len = 8;
    let num_heads = 2;
    let head_dim = 16;

    // Create tensors with larger variance
    let q = Tensor::randn(0.0f32, 10.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();
    let k = Tensor::randn(0.0f32, 10.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();
    let v = Tensor::randn(0.0f32, 10.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();

    let config = FlashAttentionConfig::new(head_dim, true);
    let output = flash_attention_cpu(&q, &k, &v, &config).unwrap();

    // Check for NaN or Inf
    let output_flat: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
    for val in output_flat {
        assert!(!val.is_nan(), "Output contains NaN");
        assert!(!val.is_infinite(), "Output contains Inf");
    }
}

#[test]
fn test_flash_attention_different_block_sizes() {
    let device = test_device();
    let batch = 1;
    let seq_len = 32;
    let num_heads = 2;
    let head_dim = 16;

    let q = Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();

    // Test with different block sizes
    let block_sizes = [(8, 8), (16, 16), (4, 8), (8, 4)];

    let mut outputs = Vec::new();
    for (bq, bkv) in block_sizes {
        let config = FlashAttentionConfig::new(head_dim, true).with_block_sizes(bq, bkv);
        let output = flash_attention_cpu(&q, &k, &v, &config).unwrap();
        outputs.push(output);
    }

    // All outputs should be numerically close
    for i in 1..outputs.len() {
        let diff = (&outputs[0] - &outputs[i]).unwrap().abs().unwrap();
        let max_diff: f32 = diff
            .max(D::Minus1)
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .max(D::Minus1)
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            max_diff < 1e-4,
            "Block size {block_sizes:?}[{i}] differs by {max_diff}"
        );
    }
}

#[test]
fn test_flash_attention_vs_naive_sdpa() {
    let device = test_device();
    let batch = 1;
    let seq_len = 16;
    let num_heads = 4;
    let head_dim = 32;

    let q = Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();

    // Flash Attention
    let config = FlashAttentionConfig::new(head_dim, true);
    let flash_output = flash_attention_cpu(&q, &k, &v, &config).unwrap();

    // Naive SDPA
    let naive_output = naive_sdpa(&q, &k, &v, config.softmax_scale, true).unwrap();

    // Compare
    let diff = (&flash_output - &naive_output).unwrap().abs().unwrap();
    let max_diff: f32 = diff
        .max(D::Minus1)
        .unwrap()
        .max(D::Minus1)
        .unwrap()
        .max(D::Minus1)
        .unwrap()
        .max(D::Minus1)
        .unwrap()
        .to_scalar()
        .unwrap();

    assert!(
        max_diff < 1e-4,
        "Flash Attention differs from naive SDPA by {max_diff}"
    );
}

/// Naive SDPA implementation for comparison.
fn naive_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal: bool,
) -> candle_core::Result<Tensor> {
    let (_batch, seq_len, _num_heads, _head_dim) = q.dims4()?;

    // Transpose to [batch, num_heads, seq_len, head_dim]
    let q = q.transpose(1, 2)?.contiguous()?;
    let k = k.transpose(1, 2)?.contiguous()?;
    let v = v.transpose(1, 2)?.contiguous()?;

    // Q @ K^T
    let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
    let scores = (scores * scale as f64)?;

    // Causal mask
    let scores = if causal {
        let mask = create_causal_mask(seq_len, seq_len, q.device())?;
        scores.broadcast_add(&mask)?
    } else {
        scores
    };

    // Softmax
    let attn = candle_nn::ops::softmax_last_dim(&scores)?;

    // Attention @ V
    let output = attn.matmul(&v)?;

    // Transpose back
    output.transpose(1, 2)?.contiguous()
}

fn create_causal_mask(q_len: usize, kv_len: usize, device: &Device) -> candle_core::Result<Tensor> {
    let neg_inf = f32::NEG_INFINITY;
    let mask: Vec<f32> = (0..q_len)
        .flat_map(|i| (0..kv_len).map(move |j| if j > i { neg_inf } else { 0.0 }))
        .collect();
    Tensor::from_vec(mask, (1, 1, q_len, kv_len), device)
}
