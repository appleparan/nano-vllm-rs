# Sampler & LLM Engine

## Overview

Stage 8 implements the inference orchestration layer that ties together all previous components:

- **Sampler**: Token selection from probability distributions
- **LLMEngine**: Coordinates model, scheduler, sampler, and tokenizer for text generation

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           LLMEngine                                     │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Tokenizer  │  │    Model    │  │  Scheduler  │  │   Sampler   │    │
│  │             │  │  (Qwen3)    │  │             │  │             │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
│         │                │                │                │            │
│         ▼                │                │                │            │
│  ┌─────────────┐         │                │                │            │
│  │  Encode     │         │                │                │            │
│  │  prompt     │         │                │                │            │
│  └─────────────┘         │                │                │            │
│         │                │                │                │            │
│         ▼                ▼                ▼                │            │
│  ┌─────────────────────────────────────────────────────────┴───────┐   │
│  │                      Generation Loop                             │   │
│  │                                                                  │   │
│  │  1. schedule()  →  prefill/decode sequences                      │   │
│  │  2. forward()   →  compute logits                                │   │
│  │  3. sample()    →  select next token                             │   │
│  │  4. check EOS   →  finish or continue                            │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────┐                                                        │
│  │  Decode     │                                                        │
│  │  tokens     │                                                        │
│  └─────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Sampler

### Purpose

The Sampler converts raw model logits into token selections using configurable strategies.

### Sampling Pipeline

```text
Logits [vocab_size]
    │
    ▼ Temperature scaling
logits / temperature
    │
    ▼ Top-k filtering (optional)
Keep top k tokens
    │
    ▼ Softmax
Probabilities
    │
    ▼ Top-p filtering (optional)
Cumulative prob ≤ p
    │
    ▼ Renormalize + Sample
Selected token
```

### Implementation

```rust
pub struct Sampler {
    temperature: f32,    // 1.0 = no change, 0.0 = greedy
    top_k: usize,        // 0 = disabled
    top_p: f32,          // 1.0 = disabled
    rng: StdRng,         // For reproducibility
}

impl Sampler {
    pub fn sample(&mut self, logits: &Tensor) -> Result<Vec<u32>>;
}
```

### Sampling Strategies

#### 1. Greedy Decoding (temperature = 0)

Always select the token with highest logit:

```rust
fn argmax(&self, logits: &Tensor) -> Result<u32> {
    let max_idx = logits.argmax(D::Minus1)?;
    max_idx.to_scalar::<u32>()
}
```

**Use case**: Deterministic output, code generation, factual queries.

#### 2. Temperature Scaling

Controls randomness by scaling logits before softmax:

```rust
let scaled_logits = logits / temperature;
let probs = softmax(scaled_logits);
```

| Temperature | Effect |
| ----------- | ------ |
| 0.0 | Greedy (deterministic) |
| 0.1-0.5 | More focused, less random |
| 1.0 | Original distribution |
| 1.5-2.0 | More creative, more random |

#### 3. Top-k Filtering

Limits choices to the k most likely tokens:

```rust
fn apply_top_k(&self, logits: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    // Sort logits by value (descending)
    // Take top k tokens
    // Return filtered (logits, indices)
}
```

**Example with k=3:**

```text
Original: [0.1, 0.3, 0.05, 0.4, 0.15]  (tokens 0-4)
After:    [0.3, 0.4, 0.15]             (tokens 1, 3, 4)
```

#### 4. Top-p (Nucleus) Filtering

Keeps tokens until cumulative probability exceeds p:

```rust
fn apply_top_p(&self, probs: &Tensor, indices: &Tensor, p: f32)
    -> Result<(Tensor, Tensor)>
{
    // Sort by probability (descending)
    // Accumulate until sum > p
    // Return filtered (probs, indices)
}
```

**Example with p=0.9:**

```text
Sorted probs: [0.5, 0.3, 0.1, 0.05, 0.05]
Cumulative:   [0.5, 0.8, 0.9, 0.95, 1.0]
                            ↑ cutoff at 0.9
Keep: first 3 tokens
```

### Reproducibility

Use seeded RNG for deterministic sampling:

```rust
let sampler = Sampler::with_seed(&config, 42);

// Same seed + same logits = same output
```

## LLMEngine

### Purpose

Orchestrates the complete text generation pipeline.

### Components

```rust
pub struct LLMEngine {
    model: Qwen3ForCausalLM,           // Language model
    model_config: Qwen3Config,         // Model configuration
    scheduler: Scheduler,               // Request management
    tokenizer: Tokenizer,               // Text ↔ tokens
    sampling_configs: HashMap<SequenceId, SamplingConfig>,
    samplers: HashMap<SequenceId, Sampler>,
    prompts: HashMap<SequenceId, String>,
    next_request_id: SequenceId,
    device: Device,
    dtype: DType,
    eos_token_id: u32,
}
```

### API

#### Creating the Engine

```rust
let engine = LLMEngine::new(
    model,           // Loaded Qwen3ForCausalLM
    model_config,    // Qwen3Config
    tokenizer,       // HuggingFace tokenizer
    engine_config,   // EngineConfig
)?;
```

#### Adding Requests

```rust
// Builder pattern for requests
let request = GenerationRequest::new("Hello, world!")
    .max_tokens(100)
    .temperature(0.7)
    .top_k(50)
    .top_p(0.9)
    .priority(5);

let request_id = engine.add_request(request)?;
```

**Under the hood:**

1. Tokenize prompt → `Vec<u32>`
2. Create `Sequence` with tokens
3. Add to scheduler's waiting queue
4. Create dedicated `Sampler` for this request

#### Running Inference

**Single step:**

```rust
let outputs = engine.step()?;
// Returns completed sequences (if any)
```

**Until completion:**

```rust
let outputs = engine.generate()?;
// Returns all completed sequences
```

**Convenience method:**

```rust
let text = engine.generate_text(
    "Once upon a time",
    max_tokens: 50,
    temperature: 0.8,
)?;
```

### Generation Loop

```rust
pub fn generate(&mut self) -> Result<Vec<GenerationOutput>> {
    let mut all_outputs = Vec::new();

    loop {
        let outputs = self.step()?;
        all_outputs.extend(outputs);

        if !self.has_pending_requests() {
            break;
        }
    }

    Ok(all_outputs)
}
```

### Step Implementation

Each `step()` call:

```text
1. scheduler.schedule()
   │
   ├─► prefill_sequences: [seq_1, seq_2, ...]
   └─► decode_sequences: [seq_3, seq_4, ...]

2. For each prefill sequence:
   │
   ├─► Get tokens to process (chunked if needed)
   ├─► Forward pass: input_ids → logits
   ├─► If prefill complete: sample first token
   └─► Update prefill progress

3. For each decode sequence:
   │
   ├─► Get last token
   ├─► Forward pass: single token → logits
   ├─► Sample next token
   └─► Check completion (EOS, max_tokens)

4. Return completed sequences
```

### Prefill Processing

```rust
fn process_prefill(&mut self, seq_id: SequenceId, ...) -> Result<()> {
    let sequence = self.scheduler.get_sequence(seq_id)?;

    // Get chunk to prefill
    let start = sequence.num_prefilled_tokens();
    let end = (start + chunk_size).min(sequence.prompt_len());
    let tokens = sequence.all_token_ids()[start..end].to_vec();

    // Forward pass
    let input_ids = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
    let logits = self.model.forward(&input_ids, start)?;

    // If prefill complete, sample first generated token
    if end == sequence.prompt_len() {
        let sampler = self.samplers.get_mut(&seq_id)?;
        let new_token = sampler.sample(&logits)?[0];

        self.scheduler.append_token(seq_id, new_token)?;
        self.scheduler.mark_prefilled(seq_id, end)?;

        self.check_completion(seq_id, new_token)?;
    }

    Ok(())
}
```

### Decode Processing

```rust
fn process_decode(&mut self, seq_id: SequenceId) -> Result<Option<GenerationOutput>> {
    let sequence = self.scheduler.get_sequence(seq_id)?;

    // Get last token for forward pass
    let last_token = *sequence.all_token_ids().last()?;
    let start_pos = sequence.total_len() - 1;

    // Forward pass (single token)
    let input_ids = Tensor::new(&[[last_token]], &self.device)?;
    let logits = self.model.forward(&input_ids, start_pos)?;

    // Sample next token
    let sampler = self.samplers.get_mut(&seq_id)?;
    let new_token = sampler.sample(&logits)?[0];

    // Update sequence
    self.scheduler.append_token(seq_id, new_token)?;

    // Check for completion
    self.check_completion(seq_id, new_token)
}
```

### Completion Checking

```rust
fn check_completion(&mut self, seq_id: SequenceId, new_token: u32)
    -> Result<Option<GenerationOutput>>
{
    let config = self.sampling_configs.get(&seq_id)?;
    let sequence = self.scheduler.get_sequence(seq_id)?;

    let finish_reason = if new_token == self.eos_token_id {
        Some(FinishReason::EndOfSequence)
    } else if sequence.output_len() >= config.max_tokens {
        Some(FinishReason::MaxTokens)
    } else {
        None
    };

    if let Some(reason) = finish_reason {
        self.scheduler.finish_sequence(seq_id, reason);
        return Ok(Some(self.create_output(seq_id, reason)?));
    }

    Ok(None)
}
```

### Output Generation

```rust
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub request_id: SequenceId,
    pub prompt: String,
    pub output_text: String,
    pub output_tokens: Vec<u32>,
    pub finish_reason: Option<FinishReason>,
    pub total_tokens: usize,
}
```

## Configuration

### SamplingConfig

```rust
pub struct SamplingConfig {
    pub temperature: f32,        // Default: 1.0
    pub top_k: usize,            // Default: 0 (disabled)
    pub top_p: f32,              // Default: 1.0 (disabled)
    pub max_tokens: usize,       // Default: 256
    pub stop_sequences: Vec<String>,
}
```

### EngineConfig

```rust
pub struct EngineConfig {
    pub max_num_seqs: usize,           // Default: 256
    pub max_prefill_tokens: usize,     // Default: 4096
    pub block_size: usize,             // Default: 16
    pub num_blocks: usize,             // Default: 1024
    pub use_paged_attention: bool,     // Default: true
    pub enable_prefix_caching: bool,   // Default: true
    pub enable_preemption: bool,       // Default: false
}
```

## Usage Example

```rust
use nano_vllm::{
    LLMEngine, GenerationRequest, EngineConfig,
    model::{download_model, load_config, load_safetensors, Qwen3ForCausalLM},
};
use candle_core::{DType, Device};
use tokenizers::Tokenizer;

// Load model
let files = download_model("Qwen/Qwen3-0.6B", "main")?;
let config = load_config(&files.config)?;
let device = Device::Cpu;
let vb = load_safetensors(&files.weights, DType::F32, &device)?;
let model = Qwen3ForCausalLM::new(&config, vb)?;
let tokenizer = Tokenizer::from_file(&files.tokenizer)?;

// Create engine
let mut engine = LLMEngine::new(
    model,
    config,
    tokenizer,
    EngineConfig::default(),
)?;

// Add requests
engine.add_request(
    GenerationRequest::new("What is 2+2?")
        .max_tokens(50)
        .temperature(0.0)  // Greedy for math
)?;

engine.add_request(
    GenerationRequest::new("Write a poem about rust.")
        .max_tokens(100)
        .temperature(0.8)  // Creative
        .top_p(0.9)
)?;

// Generate
let outputs = engine.generate()?;

for output in outputs {
    println!("Prompt: {}", output.prompt);
    println!("Output: {}", output.output_text);
    println!("Reason: {:?}", output.finish_reason);
    println!();
}
```

## Implementation Files

| File | Description |
| ---- | ----------- |
| `src/engine/mod.rs` | Module exports |
| `src/engine/sampler.rs` | Sampler with temperature, top-k, top-p |
| `src/engine/llm.rs` | LLMEngine orchestration |
| `src/config.rs` | SamplingConfig, EngineConfig |

## Design Decisions

### Why Per-Request Samplers?

Each request can have different sampling parameters:

```rust
// Request 1: Deterministic code generation
request1.temperature(0.0)

// Request 2: Creative writing
request2.temperature(0.9).top_p(0.95)
```

Maintaining separate `Sampler` instances ensures:

- Isolated RNG state for reproducibility
- Independent configuration per request
- No cross-request interference

### Why Chunked Prefill Support?

Long prompts can be processed in chunks:

1. Prevents OOM on very long prompts
2. Allows interleaving with decode requests
3. Better utilization of `max_prefill_tokens` budget

### Why Check Completion After Each Token?

Early stopping improves efficiency:

```rust
// Don't continue generating after EOS
if new_token == eos_token_id {
    return finish_sequence();
}
```

## References

- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)
- [Sampling Strategies in LLMs](https://huggingface.co/blog/how-to-generate)
- [Top-p (Nucleus) Sampling Paper](https://arxiv.org/abs/1904.09751)
