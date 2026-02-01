//! Integration tests for Sampler.

use std::collections::HashSet;

use candle_core::{Device, Tensor};
use nano_vllm::Sampler;
use nano_vllm::config::SamplingConfig;

fn test_device() -> Device {
    Device::Cpu
}

fn test_sampling_config() -> SamplingConfig {
    SamplingConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        max_tokens: 100,
        stop_sequences: vec![],
    }
}

#[test]
fn test_greedy_sampling() {
    let config = SamplingConfig {
        temperature: 0.0,
        ..test_sampling_config()
    };
    let mut sampler = Sampler::with_seed(&config, 42);
    let device = test_device();

    // Create logits where token 3 has highest value
    let logits = Tensor::new(&[0.1f32, 0.2, 0.3, 10.0, 0.4], &device).unwrap();
    let tokens = sampler.sample(&logits).unwrap();

    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0], 3);
}

#[test]
fn test_temperature_sampling() {
    let config = test_sampling_config();
    let mut sampler = Sampler::with_seed(&config, 42);
    let device = test_device();

    // Create uniform-ish logits
    let logits = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0, 1.0], &device).unwrap();

    // Sample multiple times and verify we get different tokens
    let mut seen = HashSet::new();
    for _ in 0..100 {
        let tokens = sampler.sample(&logits).unwrap();
        seen.insert(tokens[0]);
    }

    // Should see multiple different tokens with uniform distribution
    assert!(seen.len() > 1, "Should sample different tokens");
}

#[test]
fn test_top_k_sampling() {
    let config = SamplingConfig {
        top_k: 2,
        ..test_sampling_config()
    };
    let mut sampler = Sampler::with_seed(&config, 42);
    let device = test_device();

    // Create logits where tokens 3 and 4 have highest values
    let logits = Tensor::new(&[0.1f32, 0.2, 0.3, 10.0, 9.0], &device).unwrap();

    // Sample multiple times and verify only top-k tokens are selected
    let mut seen = HashSet::new();
    for _ in 0..50 {
        let tokens = sampler.sample(&logits).unwrap();
        seen.insert(tokens[0]);
    }

    // Should only see tokens 3 and 4
    assert!(seen.contains(&3) || seen.contains(&4));
    assert!(!seen.contains(&0));
    assert!(!seen.contains(&1));
    assert!(!seen.contains(&2));
}

#[test]
fn test_top_p_sampling() {
    let config = SamplingConfig {
        top_p: 0.5, // Only top 50% probability mass
        ..test_sampling_config()
    };
    let mut sampler = Sampler::with_seed(&config, 42);
    let device = test_device();

    // Create logits where one token dominates
    let logits = Tensor::new(&[0.0f32, 0.0, 0.0, 10.0, 0.0], &device).unwrap();

    // With dominant token, should always sample that token
    for _ in 0..10 {
        let tokens = sampler.sample(&logits).unwrap();
        assert_eq!(tokens[0], 3);
    }
}

#[test]
fn test_batch_sampling() {
    let config = SamplingConfig {
        temperature: 0.0, // Greedy for determinism
        ..test_sampling_config()
    };
    let mut sampler = Sampler::with_seed(&config, 42);
    let device = test_device();

    // Create batch of logits [2, 5]
    let logits = Tensor::new(
        &[[0.1f32, 0.2, 0.3, 10.0, 0.4], [0.1, 10.0, 0.3, 0.4, 0.5]],
        &device,
    )
    .unwrap();

    let tokens = sampler.sample(&logits).unwrap();

    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0], 3); // Max in first row
    assert_eq!(tokens[1], 1); // Max in second row
}

#[test]
fn test_reproducibility_with_seed() {
    let config = test_sampling_config();
    let device = test_device();
    let logits = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0, 1.0], &device).unwrap();

    // Sample with same seed twice
    let mut sampler1 = Sampler::with_seed(&config, 12345);
    let mut sampler2 = Sampler::with_seed(&config, 12345);

    let mut tokens1 = Vec::new();
    let mut tokens2 = Vec::new();

    for _ in 0..10 {
        tokens1.push(sampler1.sample(&logits).unwrap()[0]);
        tokens2.push(sampler2.sample(&logits).unwrap()[0]);
    }

    assert_eq!(tokens1, tokens2, "Same seed should produce same sequence");
}
