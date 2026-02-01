//! Integration tests for Qwen3Model.

use nano_vllm::model::Qwen3Config;

#[test]
fn test_qwen3_config_defaults() {
    let json = r#"{
        "vocab_size": 1000,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2
    }"#;

    let config: Qwen3Config = serde_json::from_str(json).unwrap();

    assert_eq!(config.vocab_size, 1000);
    assert_eq!(config.hidden_size, 64);
    assert_eq!(config.num_hidden_layers, 2);
    assert_eq!(config.head_dim, 128); // default
    assert!(config.tie_word_embeddings); // default
}
