# nano-vllm-rs

Rust implementation of nano-vllm - an educational LLM inference engine.

## Goal

Implement core vLLM optimizations for learning purposes:

- PagedAttention (virtual memory for KV cache)
- Continuous Batching
- Prefix Caching
- Speculative Decoding

## Tech Stack

- `candle` - Tensor operations (Hugging Face Rust ML framework)
- `tokenizers` - Tokenization
- `hf-hub` - HuggingFace model loading
- `clap` - CLI

## Project Structure

```text
src/
├── core/       # Block, BlockManager, Sequence, KVCache
├── scheduler/  # Continuous batching, priority scheduling
├── attention/  # PagedAttention
├── model/      # Qwen3 (RMSNorm, RoPE, GQA, SwiGLU)
├── engine/     # LLMEngine, Sampler
└── speculative/ # Speculative decoding
```

## References

- [vLLM Paper](https://arxiv.org/abs/2309.06180)
- [Candle](https://huggingface.github.io/candle/)
