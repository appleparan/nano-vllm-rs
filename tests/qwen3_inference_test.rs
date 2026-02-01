//! Integration tests for Qwen3 model inference.
//!
//! These tests download and run the actual Qwen/Qwen3-0.6B model.
//! They are marked with `#[ignore]` by default since they require:
//! - Network access to HuggingFace Hub
//! - ~1.2GB disk space for model weights
//! - Several minutes to complete
//!
//! Run with: `cargo test --test qwen3_inference_test -- --ignored`

use candle_core::{DType, Device};
use nano_vllm::{
    download_model, load_config, load_safetensors, EngineConfig, GenerationRequest, LLMEngine,
    Qwen3ForCausalLM,
};

const MODEL_ID: &str = "Qwen/Qwen3-0.6B";
const REVISION: &str = "main";

fn test_device() -> Device {
    Device::Cpu
}

/// Helper function to create a model and engine for testing.
fn create_engine() -> anyhow::Result<LLMEngine> {
    let device = test_device();

    // Download model
    let model_files = download_model(MODEL_ID, REVISION)?;

    // Load config
    let config = load_config(&model_files.config)?;

    // Load weights
    let vb = load_safetensors(&model_files.weights, DType::F32, &device)?;

    // Create model (use_flash_attention = false for CPU testing)
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

    // Create engine
    let engine = LLMEngine::new(model, config, tokenizer, engine_config)?;

    Ok(engine)
}

#[test]
#[ignore]
fn test_model_download_and_load() {
    let device = test_device();

    // Download model
    let model_files = download_model(MODEL_ID, REVISION).expect("Failed to download model");

    // Verify files exist
    assert!(model_files.config.exists(), "config.json should exist");
    assert!(
        !model_files.weights.is_empty(),
        "weights should be downloaded"
    );
    assert!(
        model_files.tokenizer.exists(),
        "tokenizer.json should exist"
    );

    // Load config
    let config = load_config(&model_files.config).expect("Failed to load config");
    assert_eq!(config.num_hidden_layers, 28);
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.hidden_size, 1024);

    // Load weights
    let vb =
        load_safetensors(&model_files.weights, DType::F32, &device).expect("Failed to load weights");

    // Create model
    let model = Qwen3ForCausalLM::new(&config, false, vb).expect("Failed to create model");

    // Verify model properties
    assert_eq!(model.model().num_layers(), 28);
}

#[test]
#[ignore]
fn test_simple_generation() {
    let mut engine = create_engine().expect("Failed to create engine");

    // Simple prompt
    let request = GenerationRequest::new("Hello")
        .max_tokens(10)
        .temperature(0.0); // Greedy decoding for deterministic output

    engine.add_request(request).expect("Failed to add request");

    // Generate
    let outputs = engine.generate().expect("Failed to generate");

    // Verify output
    assert_eq!(outputs.len(), 1);
    let output = &outputs[0];

    assert!(!output.output_text.is_empty(), "Output should not be empty");
    assert!(
        !output.output_tokens.is_empty(),
        "Output tokens should not be empty"
    );
    assert!(output.finish_reason.is_some(), "Should have finish reason");

    println!("Prompt: {}", output.prompt);
    println!("Output: {}", output.output_text);
    println!("Tokens generated: {}", output.output_tokens.len());
}

#[test]
#[ignore]
fn test_greedy_decoding() {
    let mut engine = create_engine().expect("Failed to create engine");

    // With temperature=0, output should be deterministic
    let prompt = "The capital of France is";

    let request = GenerationRequest::new(prompt)
        .max_tokens(20)
        .temperature(0.0);

    engine.add_request(request).expect("Failed to add request");

    let outputs = engine.generate().expect("Failed to generate");
    let output = &outputs[0];

    println!("Prompt: {prompt}");
    println!("Output: {}", output.output_text);

    // The model should mention Paris
    let full_text = format!("{}{}", prompt, output.output_text);
    assert!(
        full_text.to_lowercase().contains("paris"),
        "Output should mention Paris: {}",
        output.output_text
    );
}

#[test]
#[ignore]
fn test_max_tokens_limit() {
    let mut engine = create_engine().expect("Failed to create engine");

    let max_tokens = 5;
    let request = GenerationRequest::new("Once upon a time")
        .max_tokens(max_tokens)
        .temperature(0.0);

    engine.add_request(request).expect("Failed to add request");

    let outputs = engine.generate().expect("Failed to generate");
    let output = &outputs[0];

    // Should generate at most max_tokens
    assert!(
        output.output_tokens.len() <= max_tokens,
        "Generated {} tokens, expected at most {}",
        output.output_tokens.len(),
        max_tokens
    );

    println!("Tokens generated: {} (max: {})", output.output_tokens.len(), max_tokens);
}

#[test]
#[ignore]
fn test_multiple_prompts() {
    let mut engine = create_engine().expect("Failed to create engine");

    let prompts = vec![
        "Hello, how are you?",
        "What is 2 + 2?",
        "The sun is",
    ];

    for prompt in &prompts {
        let request = GenerationRequest::new(*prompt)
            .max_tokens(15)
            .temperature(0.0);
        engine.add_request(request).expect("Failed to add request");
    }

    let outputs = engine.generate().expect("Failed to generate");

    // Should get output for each prompt
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
fn test_generate_text_convenience() {
    let mut engine = create_engine().expect("Failed to create engine");

    let result = engine.generate_text("Hello", 10, 0.0);
    assert!(result.is_ok());

    let text = result.unwrap();
    assert!(!text.is_empty());
    println!("Generated text: {text}");
}

#[test]
#[ignore]
fn test_tokenizer_roundtrip() {
    let engine = create_engine().expect("Failed to create engine");
    let tokenizer = engine.tokenizer();

    let text = "Hello, world! This is a test.";

    // Encode
    let encoding = tokenizer.encode(text, false).expect("Failed to encode");
    let token_ids = encoding.get_ids();

    // Decode
    let decoded = tokenizer
        .decode(token_ids, true)
        .expect("Failed to decode");

    println!("Original: {text}");
    println!("Token IDs: {:?}", token_ids);
    println!("Decoded: {decoded}");

    // Should be similar (might have minor differences due to normalization)
    assert!(decoded.contains("Hello"));
    assert!(decoded.contains("world"));
}

#[test]
#[ignore]
fn test_model_config() {
    let engine = create_engine().expect("Failed to create engine");
    let config = engine.model_config();

    // Qwen3-0.6B specific config
    assert_eq!(config.vocab_size, 151936);
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.intermediate_size, 3072);
    assert_eq!(config.num_hidden_layers, 28);
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.num_key_value_heads, 8); // GQA ratio = 2
    assert_eq!(config.head_dim, 128);
    assert!((config.rope_theta - 1_000_000.0).abs() < 1.0);
}

#[test]
#[ignore]
fn test_sampling_with_temperature() {
    let mut engine1 = create_engine().expect("Failed to create engine");
    let mut engine2 = create_engine().expect("Failed to create engine");

    let prompt = "The meaning of life is";

    // With temperature 0 (greedy)
    let request1 = GenerationRequest::new(prompt)
        .max_tokens(10)
        .temperature(0.0);
    engine1.add_request(request1).expect("Failed to add request");
    let output1 = engine1.generate().expect("Failed to generate");

    // With temperature 1.0 (more random)
    let request2 = GenerationRequest::new(prompt)
        .max_tokens(10)
        .temperature(1.0);
    engine2.add_request(request2).expect("Failed to add request");
    let output2 = engine2.generate().expect("Failed to generate");

    println!("Greedy (temp=0): {}", output1[0].output_text);
    println!("Random (temp=1): {}", output2[0].output_text);

    // Both should produce output
    assert!(!output1[0].output_text.is_empty());
    assert!(!output2[0].output_text.is_empty());
}
