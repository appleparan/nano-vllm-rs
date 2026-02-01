# nano-vllm-rs

[![Crates.io](https://img.shields.io/crates/v/nano-vllm-rs.svg)](https://crates.io/crates/nano-vllm-rs)
[![Docs.rs](https://docs.rs/nano-vllm-rs/badge.svg)](https://docs.rs/nano-vllm-rs)
[![CI](https://github.com/appleparan/nano-vllm-rs/workflows/CI/badge.svg)](https://github.com/appleparan/nano-vllm-rs/actions)

A minimalistic LLM inference engine in Rust, ported from [nano-vllm](https://github.com/ovshake/nano-vllm) (Python).

This project implements core vLLM optimizations for **educational purposes**, helping you understand what happens inside LLM inference:

- **PagedAttention** for efficient KV cache management
- **Continuous batching** for high throughput
- **Prefix caching** for shared prompts
- **Speculative decoding** for faster generation

## Acknowledgments

This project is a Rust port of [nano-vllm](https://github.com/ovshake/nano-vllm) by [@ovshake](https://github.com/ovshake).
The original Python implementation provides an excellent educational foundation for understanding vLLM internals.

## Educational Modes

nano-vllm-rs includes four educational modes to help you learn LLM inference:

### Narrator Mode (`--narrate`)

Real-time plain-English commentary during inference. Watch the model "think" with explanations at each step.

```bash
cargo run -- --model Qwen/Qwen3-0.6B --prompt "Hello" --narrate
```

Output example:

```text
═══════════════════════════════════════════════════════════════════
  INFERENCE ANATOMY - Educational Mode
═══════════════════════════════════════════════════════════════════

ACT 1: TOKENIZATION
Converting your prompt into numbers the model understands...

  "Hello"
       ↓ Tokenizer (BPE algorithm)
  [Hello] → [9707]

  3 tokens

  Why? LLMs work with numbers, not text. The tokenizer splits text
       into subword units that balance vocabulary size with coverage.
```

### X-Ray Mode (`--xray`)

Mathematical and tensor operation visualizations. See the actual shapes and computations.

```bash
cargo run -- --model Qwen/Qwen3-0.6B --prompt "Hello" --xray
```

Output example:

```text
╔══════════════════════════════════════════════════════════════════╗
║  X-RAY: Layer 1 - Q/K/V Projections                              ║
╚══════════════════════════════════════════════════════════════════╝

Step 1: Project to Q, K, V
────────────────────────────────────────

  hidden_states: [1, 5, 1024]
       (batch=1, seq=5, hidden=1024)

       ↓ Linear projections (learned weights)

  Q: [1, 16, 5, 64]  (16 attention heads)
  K: [1, 4, 5, 64]   (4 KV heads - GQA)
  V: [1, 4, 5, 64]

  GQA: 16 Q heads share 4 KV heads (4:1)
     Memory saving: 4x less KV cache!
```

### Dashboard Mode (`--dashboard`)

Rich terminal UI with live throughput tracking and progress visualization.

```bash
cargo run -- --model Qwen/Qwen3-0.6B --prompt "Hello" --dashboard
```

Features:

- Real-time token generation progress
- Prefill/decode throughput (tokens/sec)
- KV cache memory usage
- Top-k probability distribution
- Generated text preview

### Tutorial Mode (`--tutorial`)

Interactive step-by-step learning experience with 12 chapters covering:

1. Welcome and Overview
2. Model Architecture
3. Tokenization
4. Embedding Lookup
5. Self-Attention
6. Prefill Phase
7. KV Cache
8. Decode Phase
9. Sampling
10. PagedAttention (Advanced)
11. Speculative Decoding (Advanced)
12. Putting It All Together

```bash
cargo run -- --tutorial
```

Each chapter includes:

- Conceptual explanations
- ASCII diagrams
- Interactive quizzes
- Progress tracking

## Installation

### From Source

```bash
git clone https://github.com/appleparan/nano-vllm-rs.git
cd nano-vllm-rs
cargo build --release
```

### Cargo

```bash
cargo install nano-vllm-rs
```

## Usage

### Basic Inference

```bash
# Generate text with Qwen3-0.6B (default)
cargo run -- --prompt "The capital of France is"

# Specify a different model
cargo run -- --model Qwen/Qwen3-4B --prompt "Hello, world"

# Adjust generation parameters
cargo run -- --prompt "Once upon a time" --max-tokens 100 --temperature 0.7
```

### Speculative Decoding

Use a smaller draft model to accelerate generation:

```bash
cargo run -- --model Qwen/Qwen3-4B \
              --draft-model Qwen/Qwen3-0.6B \
              --speculative \
              --num-speculative-tokens 4 \
              --prompt "Hello"
```

### Educational Modes

```bash
# Narrator: Real-time explanations
cargo run -- --prompt "Hello" --narrate

# X-Ray: Tensor visualizations
cargo run -- --prompt "Hello" --xray

# Dashboard: Live terminal UI
cargo run -- --prompt "Hello" --dashboard

# Tutorial: Interactive learning
cargo run -- --tutorial

# Combine modes
cargo run -- --prompt "Hello" --narrate --xray
```

## Library Usage

```rust
use nano_vllm::{
    LLMEngine, EngineConfig, ModelConfig, SamplingConfig,
    download_model, load_config,
};
use nano_vllm::educational::{
    EducationalConfig, InferenceNarrator, NarratorConfig,
    XRayVisualizer, XRayConfig, InteractiveTutorial,
};

// Download and load model
let model_id = "Qwen/Qwen3-0.6B";
let files = download_model(model_id, "main")?;
let config = load_config(&files.config)?;

// Create engine
let engine = LLMEngine::new(model, config, tokenizer, engine_config)?;

// Generate with educational narrator
let mut narrator = InferenceNarrator::new(NarratorConfig::default());
narrator.on_start(model_id, &model_config, "Hello");
// ... narrator hooks called during inference

// Or run the interactive tutorial
let mut tutorial = InteractiveTutorial::default();
tutorial.run();
```

## Supported Models

Currently supports Qwen3 family models:

| Model           | Parameters | Size (FP16) | Use Case                    |
|-----------------|------------|-------------|-----------------------------|
| Qwen/Qwen3-0.6B | 0.6B       | ~1.2GB      | Fast inference, draft model |
| Qwen/Qwen3-4B   | 4B         | ~8GB        | Target model                |

## Testing

```bash
# Run unit tests
cargo test

# Run integration tests with real Qwen3-0.6B model (requires ~1.2GB download)
cargo test --test qwen3_inference_test -- --ignored

# Run speculative decoding tests (requires ~9GB download)
cargo test --test speculative_inference_test -- --ignored
```

> **Note**: Integration tests are marked with `#[ignore]` by default and won't run in CI.
> They require network access to HuggingFace Hub and take several minutes to complete.

## Project Structure

```text
src/
├── lib.rs              # Library entry point
├── main.rs             # CLI entry point
├── config.rs           # Configuration types
├── error.rs            # Error handling
├── attention/          # Flash attention implementation
├── core/               # Core types (Block, Sequence, etc.)
├── engine/             # LLM engine and sampler
├── model/              # Qwen3 model implementation
├── scheduler/          # Request scheduling
├── speculative/        # Speculative decoding
└── educational/        # Educational modes
    ├── mod.rs          # Module entry point
    ├── narrator.rs     # Narrator mode
    ├── xray.rs         # X-Ray mode
    ├── dashboard.rs    # Dashboard mode
    ├── tutorial.rs     # Tutorial mode
    ├── explanations.rs # Educational text content
    └── visualizers.rs  # ASCII art generators
```

## References

- [nano-vllm](https://github.com/ovshake/nano-vllm) - Original Python implementation
- [vLLM](https://github.com/vllm-project/vllm) - Production LLM serving engine
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180) - Efficient Memory Management for LLMs
- [Speculative Decoding Paper](https://arxiv.org/abs/2211.17192) - Fast Inference from Transformers

## License

Licensed under the MIT license ([LICENSE](LICENSE) or http://opensource.org/licenses/MIT).
