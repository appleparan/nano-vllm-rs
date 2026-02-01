//! Unit tests for RejectionSampler.

use candle_core::{Device, Tensor};
use nano_vllm::RejectionSampler;

fn create_uniform_logits(
    k: usize,
    vocab_size: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    // Uniform logits (all zeros -> uniform after softmax)
    Tensor::zeros((k, vocab_size), candle_core::DType::F32, device)
}

#[test]
fn test_sampler_creation() {
    let sampler = RejectionSampler::new();
    assert!(format!("{sampler:?}").contains("RejectionSampler"));

    let seeded = RejectionSampler::with_seed(42);
    assert!(format!("{seeded:?}").contains("RejectionSampler"));
}

#[test]
fn test_accept_all_when_same_distribution() {
    let device = Device::Cpu;
    let mut sampler = RejectionSampler::with_seed(42);

    let k = 4;
    let vocab_size = 100;

    // Same distribution for draft and target -> high acceptance rate
    let draft_logits = create_uniform_logits(k, vocab_size, &device).unwrap();
    let target_logits = create_uniform_logits(k + 1, vocab_size, &device).unwrap();

    // Draft tokens (any valid tokens)
    let draft_tokens: Vec<u32> = vec![10, 20, 30, 40];

    let (accepted, final_token, num_accepted) = sampler
        .verify(&draft_tokens, &draft_logits, &target_logits, 1.0)
        .unwrap();

    // With uniform distribution, P_target = P_draft for all tokens
    // So α = min(1, 1) = 1, meaning 100% acceptance
    assert_eq!(num_accepted, k);
    assert_eq!(accepted.len(), k);
    assert!(final_token < vocab_size as u32);
}

#[test]
fn test_reject_when_target_prefers_different() {
    let device = Device::Cpu;
    let mut sampler = RejectionSampler::with_seed(42);

    let k = 1;

    // Draft strongly prefers token 0
    let draft_logits = Tensor::new(
        &[[10.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        &device,
    )
    .unwrap();

    // Target strongly prefers token 5
    let target_logits = Tensor::new(
        &[
            [0.0f32, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0f32, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
        ],
        &device,
    )
    .unwrap();

    // Draft generated token 0 (which target doesn't like)
    let draft_tokens: Vec<u32> = vec![0];

    let (accepted, final_token, num_accepted) = sampler
        .verify(&draft_tokens, &draft_logits, &target_logits, 1.0)
        .unwrap();

    // Target probability for token 0 is very low -> likely rejection
    // After rejection, should resample from adjusted distribution
    // which heavily favors token 5
    assert!(num_accepted <= k);
    assert!(accepted.len() == num_accepted);
    // Verify the function returns valid results
    assert!(final_token < 10);
}

#[test]
fn test_shape_validation() {
    let device = Device::Cpu;
    let mut sampler = RejectionSampler::new();

    // Mismatched shapes
    let draft_logits = Tensor::zeros((3, 10), candle_core::DType::F32, &device).unwrap();
    let target_logits = Tensor::zeros((3, 10), candle_core::DType::F32, &device).unwrap(); // Should be 4

    let draft_tokens: Vec<u32> = vec![1, 2, 3];

    let result = sampler.verify(&draft_tokens, &draft_logits, &target_logits, 1.0);
    assert!(result.is_err());
}

#[test]
fn test_temperature_effect() {
    let device = Device::Cpu;

    let k = 2;

    // Create logits with some variation
    let draft_logits = Tensor::new(&[[1.0f32; 10], [1.0f32; 10]], &device).unwrap();
    let target_logits = Tensor::new(&[[1.0f32; 10], [1.0f32; 10], [1.0f32; 10]], &device).unwrap();

    let draft_tokens: Vec<u32> = vec![0, 1];

    // Low temperature (more deterministic)
    let mut sampler_low = RejectionSampler::with_seed(123);
    let (_, _, num_low) = sampler_low
        .verify(&draft_tokens, &draft_logits, &target_logits, 0.1)
        .unwrap();

    // High temperature (more random)
    let mut sampler_high = RejectionSampler::with_seed(123);
    let (_, _, num_high) = sampler_high
        .verify(&draft_tokens, &draft_logits, &target_logits, 2.0)
        .unwrap();

    // Both should work without errors
    assert!(num_low <= k);
    assert!(num_high <= k);
}

#[test]
fn test_reproducibility_with_seed() {
    let device = Device::Cpu;

    let k = 4;
    let vocab_size = 50;

    let draft_logits = Tensor::randn(0.0f32, 1.0, (k, vocab_size), &device).unwrap();
    let target_logits = Tensor::randn(0.0f32, 1.0, (k + 1, vocab_size), &device).unwrap();

    let draft_tokens: Vec<u32> = vec![5, 10, 15, 20];

    // Same seed should give same results
    let mut sampler1 = RejectionSampler::with_seed(42);
    let mut sampler2 = RejectionSampler::with_seed(42);

    let result1 = sampler1
        .verify(&draft_tokens, &draft_logits, &target_logits, 1.0)
        .unwrap();
    let result2 = sampler2
        .verify(&draft_tokens, &draft_logits, &target_logits, 1.0)
        .unwrap();

    assert_eq!(result1.0, result2.0);
    assert_eq!(result1.1, result2.1);
    assert_eq!(result1.2, result2.2);
}
