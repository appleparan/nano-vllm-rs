# Model Loader & Qwen3 Model

## Overview

Stage 7 implements the model loading pipeline and full Qwen3 transformer model. This connects our low-level components (PagedAttention, Scheduler) with actual pretrained weights from HuggingFace.

## Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                    HuggingFace Hub                           │
│                   (Qwen/Qwen3-0.6B)                          │
└──────────────────────────────────────────────────────────────┘
                            │
                   download_model()
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                     ModelFiles                               │
│  ├── config.json      (Qwen3Config)                          │
│  ├── model.safetensors (weights)                             │
│  └── tokenizer.json   (for tokenization)                     │
└──────────────────────────────────────────────────────────────┘
                            │
            load_config() + load_safetensors()
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    Qwen3ForCausalLM                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                     Qwen3Model                         │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │  embed_tokens (Embedding)                        │  │  │
│  │  │       │                                          │  │  │
│  │  │       ▼                                          │  │  │
│  │  │  layers[0..N] (Qwen3DecoderLayer)                │  │  │
│  │  │       │                                          │  │  │
│  │  │       ▼                                          │  │  │
│  │  │  norm (RmsNorm)                                  │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
│                            │                                  │
│                            ▼                                  │
│                   lm_head (Linear)                            │
│             [hidden_size → vocab_size]                        │
└──────────────────────────────────────────────────────────────┘
```

## HuggingFace Model Download

### download_model()

Downloads model files from HuggingFace Hub using the `hf-hub` crate.

```rust
pub fn download_model(model_id: &str, revision: &str) -> Result<ModelFiles>

// Example usage
let files = download_model("Qwen/Qwen3-0.6B", "main")?;
```

**Files downloaded:**

- `config.json` - Model configuration
- `model.safetensors` - Model weights (or sharded files)
- `tokenizer.json` - Tokenizer configuration

### Sharded Model Support

For large models split across multiple files:

```text
model.safetensors.index.json
├── weight_map: { "layer.0.weight": "model-00001-of-00002.safetensors", ... }
└── Automatically downloads all required shards
```

## SafeTensors Loading

### load_safetensors()

Memory-mapped loading for efficient handling of large model files.

```rust
#[allow(unsafe_code)]
pub fn load_safetensors(
    paths: &[PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>>
```

**Why memory-mapped?**

- Avoids loading entire file into RAM
- OS handles page faults on-demand
- Critical for large models (7B+ parameters)

**Safety:**

- Uses `unsafe` for memory-mapped file access
- Safe as long as files aren't modified during loading
- Standard practice in ML frameworks

## Qwen3Config

Parses HuggingFace `config.json` format:

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    pub vocab_size: usize,           // 151936 for Qwen3
    pub hidden_size: usize,          // 1024 for 0.6B
    pub intermediate_size: usize,    // 3072 for 0.6B
    pub num_hidden_layers: usize,    // 28 for 0.6B
    pub num_attention_heads: usize,  // 16 (Q heads)
    pub num_key_value_heads: usize,  // 8 (KV heads, GQA ratio = 2)
    pub head_dim: usize,             // 128
    pub rms_norm_eps: f64,           // 1e-6
    pub rope_theta: f64,             // 1000000.0
    pub max_position_embeddings: usize, // 40960
    pub tie_word_embeddings: bool,   // true
}
```

### Default Values

Fields with sensible defaults for missing keys:

| Field | Default | Notes |
| ----- | ------- | ----- |
| `head_dim` | 128 | Standard for modern LLMs |
| `rms_norm_eps` | 1e-6 | Numerical stability |
| `rope_theta` | 1000000.0 | Extended context support |
| `max_position_embeddings` | 40960 | ~40K context window |
| `tie_word_embeddings` | true | Memory optimization |

## Qwen3Model

The base transformer without the language model head.

```rust
pub struct Qwen3Model {
    embed_tokens: Embedding,           // vocab_size → hidden_size
    layers: Vec<Qwen3DecoderLayer>,    // N transformer layers
    norm: RmsNorm,                     // Final normalization
    device: Device,
    dtype: DType,
}
```

### Forward Pass

```rust
pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor>
```

**Flow:**

1. Token IDs → Embedding lookup
2. Pass through N decoder layers (each has attention + MLP)
3. Final RmsNorm

**Input:** `[batch, seq_len]` token IDs
**Output:** `[batch, seq_len, hidden_size]` hidden states

## Qwen3ForCausalLM

Complete model for text generation.

```rust
pub struct Qwen3ForCausalLM {
    model: Qwen3Model,    // Base transformer
    lm_head: Linear,      // hidden_size → vocab_size
}
```

### Weight Tying

When `tie_word_embeddings = true` (default):

```rust
let lm_head = if config.tie_word_embeddings {
    // Reuse embedding weights (transposed) for lm_head
    let embed_weight = model.embed_tokens.embeddings();
    Linear::new(embed_weight.clone(), None)
} else {
    linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
};
```

**Benefits:**

- Reduces parameters by ~150M (vocab_size × hidden_size)
- Shared representation for input and output tokens
- Standard practice for modern LLMs

### Forward Methods

```rust
// Returns logits for last position only (efficient for generation)
pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor>
// Output: [batch, vocab_size]

// Returns logits for all positions (for training/evaluation)
pub fn forward_all(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor>
// Output: [batch, seq_len, vocab_size]
```

## Usage Example

```rust
use nano_vllm_rs::model::{download_model, load_config, load_safetensors, Qwen3ForCausalLM};
use candle_core::{DType, Device, Tensor};

// 1. Download model from HuggingFace
let files = download_model("Qwen/Qwen3-0.6B", "main")?;

// 2. Load configuration
let config = load_config(&files.config)?;

// 3. Load weights with memory mapping
let device = Device::Cpu; // or Device::new_cuda(0)?
let vb = load_safetensors(&files.weights, DType::F32, &device)?;

// 4. Create model
let mut model = Qwen3ForCausalLM::new(&config, vb)?;

// 5. Forward pass
let input_ids = Tensor::new(&[[1u32, 2, 3, 4]], &device)?;
let logits = model.forward(&input_ids, 0)?;  // [1, vocab_size]
```

## Qwen3-0.6B Specifications

| Parameter | Value |
| --------- | ----- |
| Parameters | ~600M |
| Hidden Size | 1024 |
| Layers | 28 |
| Attention Heads (Q) | 16 |
| KV Heads | 8 |
| Head Dim | 128 |
| Intermediate Size | 3072 |
| Vocab Size | 151936 |
| Max Context | 40960 tokens |
| GQA Ratio | 2:1 |

## Weight File Mapping

HuggingFace weight names → our module paths:

| HuggingFace Key | Module |
| --------------- | ------ |
| `model.embed_tokens.weight` | `Qwen3Model.embed_tokens` |
| `model.layers.{i}.input_layernorm.weight` | `DecoderLayer.input_layernorm` |
| `model.layers.{i}.self_attn.q_proj.weight` | `Attention.q_proj` |
| `model.layers.{i}.self_attn.k_proj.weight` | `Attention.k_proj` |
| `model.layers.{i}.self_attn.v_proj.weight` | `Attention.v_proj` |
| `model.layers.{i}.self_attn.o_proj.weight` | `Attention.o_proj` |
| `model.layers.{i}.self_attn.q_norm.weight` | `Attention.q_norm` |
| `model.layers.{i}.self_attn.k_norm.weight` | `Attention.k_norm` |
| `model.layers.{i}.post_attention_layernorm.weight` | `DecoderLayer.post_attention_layernorm` |
| `model.layers.{i}.mlp.gate_proj.weight` | `Mlp.gate_proj` |
| `model.layers.{i}.mlp.up_proj.weight` | `Mlp.up_proj` |
| `model.layers.{i}.mlp.down_proj.weight` | `Mlp.down_proj` |
| `model.norm.weight` | `Qwen3Model.norm` |
| `lm_head.weight` | (tied to embed_tokens) |

## Implementation Files

| File | Description |
| ---- | ----------- |
| `src/model/loader.rs` | download_model, load_safetensors, load_config, Qwen3Config |
| `src/model/qwen3.rs` | Qwen3Model, Qwen3ForCausalLM |
| `src/model/mod.rs` | Module exports |

## Dependencies

```toml
[dependencies]
hf-hub = "0.3"           # HuggingFace Hub API
candle-core = "0.8"      # Tensor operations
candle-nn = "0.8"        # Neural network primitives
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"       # Config parsing
```

## References

- [HuggingFace Hub Rust Client](https://github.com/huggingface/hf-hub)
- [SafeTensors Format](https://github.com/huggingface/safetensors)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [candle VarBuilder](https://github.com/huggingface/candle)
