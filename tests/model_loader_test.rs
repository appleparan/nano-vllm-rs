//! Integration tests for model loader.

use nano_vllm::model::Qwen3Config;

#[test]
fn test_parse_qwen3_config() {
    let json = r#"{
        "vocab_size": 151936,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 40960,
        "tie_word_embeddings": true
    }"#;

    let config: Qwen3Config = serde_json::from_str(json).unwrap();

    assert_eq!(config.vocab_size, 151936);
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.num_hidden_layers, 28);
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.num_key_value_heads, 8);
    assert!(config.tie_word_embeddings);
}

#[test]
fn test_config_with_defaults() {
    // Minimal config - defaults should fill in
    let json = r#"{
        "vocab_size": 151936,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8
    }"#;

    let config: Qwen3Config = serde_json::from_str(json).unwrap();

    assert_eq!(config.head_dim, 128); // default
    assert_eq!(config.rms_norm_eps, 1e-6); // default
    assert_eq!(config.rope_theta, 1000000.0); // default
    assert!(config.tie_word_embeddings); // default
}

#[test]
fn test_to_model_config() {
    let json = r#"{
        "vocab_size": 151936,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128
    }"#;

    let qwen_config: Qwen3Config = serde_json::from_str(json).unwrap();
    let model_config = qwen_config.to_model_config();

    assert_eq!(model_config.vocab_size, 151936);
    assert_eq!(model_config.hidden_size, 1024);
    assert_eq!(model_config.num_kv_groups(), 2); // 16 / 8
}
