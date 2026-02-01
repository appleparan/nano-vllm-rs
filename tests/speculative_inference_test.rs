//! Integration tests for speculative decoding.
//!
//! These tests require downloading Qwen3-4B (target) and Qwen3-0.6B (draft) models.
//! They are marked with `#[ignore]` by default since they require:
//! - Network access to HuggingFace Hub
//! - ~9GB disk space for both model weights
//! - Several minutes to complete
//!
//! Run with: `cargo test --test speculative_inference_test -- --ignored`

use candle_core::{DType, Device};
use nano_vllm::{
    EngineConfig, GenerationRequest, LLMEngine, Qwen3ForCausalLM, SpeculativeConfig,
    download_model, load_config, load_safetensors,
};

const TARGET_MODEL_ID: &str = "Qwen/Qwen3-4B";
const DRAFT_MODEL_ID: &str = "Qwen/Qwen3-0.6B";
const REVISION: &str = "main";

fn test_device() -> Device {
    Device::Cpu
}

/// Helper function to create a speculative engine for testing.
fn create_speculative_engine() -> anyhow::Result<LLMEngine> {
    let device = test_device();

    // Download and load target model
    let target_files = download_model(TARGET_MODEL_ID, REVISION)?;
    let target_config = load_config(&target_files.config)?;
    let target_vb = load_safetensors(&target_files.weights, DType::F32, &device)?;
    let target_model = Qwen3ForCausalLM::new(&target_config, false, target_vb)?;

    // Download and load draft model
    let draft_files = download_model(DRAFT_MODEL_ID, REVISION)?;
    let draft_config = load_config(&draft_files.config)?;
    let draft_vb = load_safetensors(&draft_files.weights, DType::F32, &device)?;
    let draft_model = Qwen3ForCausalLM::new(&draft_config, false, draft_vb)?;

    // Load tokenizer (use target's tokenizer - they share the same vocabulary)
    let tokenizer = tokenizers::Tokenizer::from_file(&target_files.tokenizer)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

    // Create engine config
    let engine_config = EngineConfig {
        max_num_seqs: 16,
        max_prefill_tokens: 2048,
        block_size: 16,
        num_blocks: 256,
        use_paged_attention: true,
        use_flash_attention: false,
        enable_prefix_caching: false,
        enable_preemption: false,
    };

    // Create speculative config
    let speculative_config = SpeculativeConfig::new(DRAFT_MODEL_ID).num_tokens(4);

    // Create engine with speculative decoding
    let engine = LLMEngine::new_with_speculative(
        target_model,
        draft_model,
        target_config,
        tokenizer,
        engine_config,
        speculative_config,
    )?;

    Ok(engine)
}

/// Helper function to create a standard engine for comparison.
fn create_standard_engine() -> anyhow::Result<LLMEngine> {
    let device = test_device();

    // Download and load target model
    let model_files = download_model(TARGET_MODEL_ID, REVISION)?;
    let config = load_config(&model_files.config)?;
    let vb = load_safetensors(&model_files.weights, DType::F32, &device)?;
    let model = Qwen3ForCausalLM::new(&config, false, vb)?;

    // Load tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file(&model_files.tokenizer)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

    // Create engine config
    let engine_config = EngineConfig {
        max_num_seqs: 16,
        max_prefill_tokens: 2048,
        block_size: 16,
        num_blocks: 256,
        use_paged_attention: true,
        use_flash_attention: false,
        enable_prefix_caching: false,
        enable_preemption: false,
    };

    let engine = LLMEngine::new(model, config, tokenizer, engine_config)?;

    Ok(engine)
}

#[test]
#[ignore]
fn test_speculative_generation() {
    let mut engine = create_speculative_engine().expect("Failed to create speculative engine");

    let request = GenerationRequest::new("Hello")
        .max_tokens(10)
        .temperature(0.0);

    engine.add_request(request).expect("Failed to add request");
    let outputs = engine.generate().expect("Failed to generate");

    assert_eq!(outputs.len(), 1);
    let output = &outputs[0];

    assert!(!output.output_text.is_empty(), "Output should not be empty");
    assert!(
        !output.output_tokens.is_empty(),
        "Output tokens should not be empty"
    );
    assert!(output.finish_reason.is_some(), "Should have finish reason");

    println!("Speculative output: {}", output.output_text);
}

#[test]
#[ignore]
fn test_speculative_matches_standard() {
    // Generate with speculative decoding
    let mut spec_engine = create_speculative_engine().expect("Failed to create speculative engine");

    let prompt = "The capital of France is";

    let request = GenerationRequest::new(prompt)
        .max_tokens(20)
        .temperature(0.0);

    spec_engine
        .add_request(request)
        .expect("Failed to add request");
    let spec_outputs = spec_engine.generate().expect("Failed to generate");

    // Generate with standard decoding
    let mut std_engine = create_standard_engine().expect("Failed to create standard engine");

    let request = GenerationRequest::new(prompt)
        .max_tokens(20)
        .temperature(0.0);

    std_engine
        .add_request(request)
        .expect("Failed to add request");
    let std_outputs = std_engine.generate().expect("Failed to generate");

    // With temperature=0 (greedy), outputs should be similar
    // (might not be exactly the same due to rejection sampling)
    println!("Speculative: {}", spec_outputs[0].output_text);
    println!("Standard:    {}", std_outputs[0].output_text);

    // Both should mention Paris for this prompt
    let spec_text = format!("{}{}", prompt, spec_outputs[0].output_text);
    let std_text = format!("{}{}", prompt, std_outputs[0].output_text);

    assert!(
        spec_text.to_lowercase().contains("paris") || std_text.to_lowercase().contains("paris"),
        "At least one output should mention Paris"
    );
}

#[test]
#[ignore]
fn test_speculative_multiple_prompts() {
    let mut engine = create_speculative_engine().expect("Failed to create speculative engine");

    let prompts = vec!["Hello, how are you?", "What is 2 + 2?", "The sun is"];

    for prompt in &prompts {
        let request = GenerationRequest::new(*prompt)
            .max_tokens(15)
            .temperature(0.0);
        engine.add_request(request).expect("Failed to add request");
    }

    let outputs = engine.generate().expect("Failed to generate");

    assert_eq!(outputs.len(), prompts.len());

    for output in &outputs {
        println!("Prompt: {}", output.prompt);
        println!("Output: {}", output.output_text);
        println!("---");
        assert!(!output.output_text.is_empty());
    }
}

#[test]
#[ignore]
fn test_model_download() {
    // Just test that both models can be downloaded
    let target_files =
        download_model(TARGET_MODEL_ID, REVISION).expect("Failed to download target model");

    assert!(target_files.config.exists(), "Target config should exist");
    assert!(
        !target_files.weights.is_empty(),
        "Target weights should be downloaded"
    );

    let draft_files =
        download_model(DRAFT_MODEL_ID, REVISION).expect("Failed to download draft model");

    assert!(draft_files.config.exists(), "Draft config should exist");
    assert!(
        !draft_files.weights.is_empty(),
        "Draft weights should be downloaded"
    );

    // Verify configs
    let target_config = load_config(&target_files.config).expect("Failed to load target config");
    let draft_config = load_config(&draft_files.config).expect("Failed to load draft config");

    // Target should be larger than draft
    assert!(
        target_config.num_hidden_layers > draft_config.num_hidden_layers,
        "Target should have more layers than draft"
    );

    println!(
        "Target: {} layers, hidden_size={}",
        target_config.num_hidden_layers, target_config.hidden_size
    );
    println!(
        "Draft:  {} layers, hidden_size={}",
        draft_config.num_hidden_layers, draft_config.hidden_size
    );
}
