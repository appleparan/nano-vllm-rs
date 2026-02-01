//! Integration tests for LLMEngine.

use nano_vllm::GenerationRequest;

// Note: Full engine tests require a loaded model.
// These tests verify the request handling logic.

#[test]
fn test_generation_request_builder() {
    let request = GenerationRequest::new("Hello, world!")
        .max_tokens(100)
        .temperature(0.7)
        .top_k(50)
        .top_p(0.9)
        .priority(5);

    assert_eq!(request.prompt, "Hello, world!");
    assert_eq!(request.sampling_config.max_tokens, 100);
    assert_eq!(request.sampling_config.temperature, 0.7);
    assert_eq!(request.sampling_config.top_k, 50);
    assert_eq!(request.sampling_config.top_p, 0.9);
    assert_eq!(request.priority, 5);
}

#[test]
fn test_generation_request_defaults() {
    let request = GenerationRequest::new("Test prompt");

    assert_eq!(request.prompt, "Test prompt");
    assert_eq!(request.request_id, None);
    assert_eq!(request.priority, 0);
    // Default sampling config values
    assert_eq!(request.sampling_config.temperature, 1.0);
    assert_eq!(request.sampling_config.top_k, 0);
    assert_eq!(request.sampling_config.top_p, 1.0);
    assert_eq!(request.sampling_config.max_tokens, 256);
}
