//! Unit tests for SpeculativeConfig.

use nano_vllm::SpeculativeConfig;

#[test]
fn test_default_config() {
    let config = SpeculativeConfig::default();
    assert_eq!(config.num_speculative_tokens, 4);
    assert_eq!(config.draft_model_id, "Qwen/Qwen3-0.6B");
    assert_eq!(config.draft_revision, "main");
}

#[test]
fn test_builder_pattern() {
    let config = SpeculativeConfig::new("custom/model")
        .num_tokens(8)
        .revision("v1.0");

    assert_eq!(config.num_speculative_tokens, 8);
    assert_eq!(config.draft_model_id, "custom/model");
    assert_eq!(config.draft_revision, "v1.0");
}

#[test]
fn test_config_accessor() {
    let config = SpeculativeConfig::new("Qwen/Qwen3-0.6B").num_tokens(4);

    assert_eq!(config.num_speculative_tokens, 4);
    assert_eq!(config.draft_model_id, "Qwen/Qwen3-0.6B");
}
