# nano-vllm-rs Documentation

An educational Rust implementation of vLLM's core optimization techniques.

## Implementation Roadmap

This project follows a bottom-up implementation approach, building from low-level memory management to high-level inference engine.

```text
┌─────────────────────────────────────────────────────────────┐
│                    Stage 10: Speculative Decoding           │
│                    (Optional Advanced Feature)              │
├─────────────────────────────────────────────────────────────┤
│                    Stage 9: CLI                 ← Next      │
│                    (Command Line Interface)                 │
├─────────────────────────────────────────────────────────────┤
│                    Stage 8: LLM Engine          ✓ Complete  │
│                    (Orchestration)                          │
├─────────────────────────────────────────────────────────────┤
│                    Stage 7: Model Loader        ✓ Complete  │
│                    (HuggingFace Integration)                │
├─────────────────────────────────────────────────────────────┤
│        Stage 5-6: Qwen3 Model & PagedAttention  ✓ Complete  │
│        (Neural Network Components)                          │
├─────────────────────────────────────────────────────────────┤
│        Stage 4: Scheduler                                   │
│        (Continuous Batching)                                │
├─────────────────────────────────────────────────────────────┤
│        Stage 3: Sequence & KV Cache                         │
│        (Request Tracking & Tensor Storage)                  │
├─────────────────────────────────────────────────────────────┤
│        Stage 2: Block & BlockManager                        │
│        (PagedAttention Memory Management)                   │
├─────────────────────────────────────────────────────────────┤
│        Stage 1: Architecture, Config, Error                 │
│        (Project Foundation)                                 │
└─────────────────────────────────────────────────────────────┘
```

## Documentation by Stage

### Stage 1: Foundation

- [Architecture](architecture.md) - System structure and module relationships
- [Configuration](configuration.md) - Engine, Scheduler, Model settings
- [Errors](errors.md) - Error type definitions

**Why implement this first?**

These are shared types used by all modules. Error types enable `?` operator for error propagation, and Config types control component behavior. Defining these first ensures consistent error handling and configuration management throughout the project.

### Stage 2: PagedAttention Memory Management

- [PagedAttention](paged_attention.md) - Block, BlockTable, BlockManager

**Why this order?**

vLLM's core innovation is PagedAttention, inspired by OS virtual memory systems:

1. **Block** = Memory Page: Splits KV cache into fixed-size blocks
2. **BlockTable** = Page Table: Logical position → Physical block mapping
3. **BlockManager** = Memory Allocator: Block allocation/deallocation, reference counting

This abstraction enables Sequence, KVCache, and Scheduler to manage memory efficiently. It makes Prefix Caching (reusing common prompts) and Preemption (swapping low-priority requests) possible.

### Stage 3: Request Tracking & KV Storage

- [Sequence & KV Cache](sequence_kv_cache.md) - Sequence lifecycle, KV tensor storage

**Why this order?**

On top of the memory management infrastructure, we implement:

1. **Sequence**: Tracks the entire lifecycle of an inference request
   - State management (Waiting → Running → Finished)
   - Token management (prompt + generated)
   - KV cache connection via BlockTable

2. **KVCache**: Storage for Attention's Key/Value tensors
   - Based on Candle tensors
   - Block-level gather/scatter operations

The Scheduler needs Sequence state to decide "which request to process", and the model needs KVCache to compute Attention.

### Stage 4: Continuous Batching

- [Scheduler](scheduler.md) - Continuous batching, priority scheduling, preemption
- [Visual Guide](continuous_batching_visual.md) - Illustrated explanation with HuggingFace blog diagrams

**Why this order?**

With Sequence tracking requests and KVCache storing tensors, the Scheduler orchestrates them:

1. **Scheduling Algorithm**: Priority queue for waiting, running set for active sequences
2. **Continuous Batching**: Mix prefill and decode phases in single iteration
3. **Resource Management**: Track block allocations, enforce limits (max_num_seqs, max_prefill_tokens)
4. **Preemption**: Swap low-priority sequences when high-priority requests need memory

### Stage 5: Qwen3 Model Components

- [Qwen3 Architecture](qwen3_architecture.md) - Complete architecture explanation with diagrams

**Components implemented:**

1. **RmsNorm**: Root Mean Square Normalization (simpler than LayerNorm)
2. **RotaryEmbedding**: RoPE for position encoding via rotation
3. **Qwen3Mlp**: SwiGLU feed-forward with gating mechanism
4. **Qwen3Attention**: Grouped Query Attention with per-head normalization
5. **Qwen3DecoderLayer**: Pre-norm residual block combining attention and MLP

**Why this order?**

The model components build on each other:

- RmsNorm is used everywhere (attention, MLP, final output)
- RoPE is applied to Q and K in attention
- MLP and Attention are combined in DecoderLayer
- DecoderLayer is stacked N times to form the full model

### Stage 6: PagedAttention

- [PagedAttention](paged_attention.md) - Block-based attention operations

**Why this order?**

PagedAttention is vLLM's core optimization. Now that we have both:

- Memory management (Block, BlockTable, KVCache from Stage 2-3)
- Attention computation (Qwen3Attention from Stage 5)

We can combine them for efficient inference:

1. **prefill_attention**: Standard SDPA for processing prompts efficiently
2. **paged_attention**: Gather K/V from non-contiguous blocks for decode
3. **write_kv_to_cache**: Store K/V in block-based cache using slot mapping

### Stage 7: Model Loader & Full Qwen3 Model

- [Model Loader](model_loader.md) - HuggingFace download, SafeTensors loading, Qwen3 model assembly

**Components implemented:**

1. **Qwen3Config**: Parse HuggingFace config.json
2. **download_model()**: Download from HuggingFace Hub (hf-hub)
3. **load_safetensors()**: Memory-mapped SafeTensors loading
4. **Qwen3Model**: Token embedding + N decoder layers + final norm
5. **Qwen3ForCausalLM**: Adds language model head for token prediction

**Why this order?**

With all model components implemented in Stage 5-6, we can now:

- Load pretrained weights from HuggingFace
- Assemble the full Qwen3 transformer model
- Prepare for inference in Stage 8

### Stage 8: Sampler & LLM Engine

**Components implemented:**

1. **Sampler**: Token sampling with temperature, top-k, top-p
   - Greedy decoding (temperature=0)
   - Stochastic sampling with temperature scaling
   - Top-k and top-p (nucleus) filtering
   - Reproducible sampling with seed

2. **LLMEngine**: Orchestrates inference
   - Combines model, scheduler, sampler, and tokenizer
   - `add_request()`: Add generation requests
   - `step()`: Run single inference iteration
   - `generate()`: Run until completion

**Why this order?**

With the model, scheduler, and PagedAttention in place, the engine ties everything together for actual text generation.

### Stage 9-10: CLI & Speculative Decoding (Planned)

CLI and optional speculative decoding will be implemented next.

## Key Design Decisions

### Why Block-based Memory Management?

Traditional LLM inference allocates contiguous memory of `max_seq_len` size for each request. This causes:

- Memory waste far exceeding actual usage
- Difficulty with dynamic batching
- No prompt reuse

Block-based management provides:

- Allocate only as needed (proportional to token count)
- Efficient use of non-contiguous memory
- Shared common prompts via prefix caching

### Why Reference Counting?

When multiple requests share the same prefix (e.g., system prompt), we increment references instead of copying blocks. Blocks are only returned when the last reference is released.

### Why Cumulative Hash?

For prefix caching, we need fast lookup: "Has this token sequence already been processed?" Simple block hashing causes many collisions, so we use cumulative hash that includes the previous block's hash:

```text
Block 0: hash(tokens[0:16])
Block 1: hash(Block0.hash + tokens[16:32])
Block 2: hash(Block1.hash + tokens[32:48])
```

This chain captures the entire prefix history in a single hash.

## File Structure

```text
src/
├── lib.rs              # Public exports
├── main.rs             # CLI entry point
├── error.rs            # Error types
├── config.rs           # Configuration types
├── core/
│   ├── mod.rs
│   ├── block.rs        # Block, BlockTable
│   ├── block_manager.rs # BlockManager with prefix caching
│   ├── sequence.rs     # Sequence lifecycle
│   └── kv_cache.rs     # KV tensor storage
├── model/
│   ├── mod.rs
│   ├── norm.rs         # RmsNorm
│   ├── rope.rs         # RotaryEmbedding (RoPE)
│   ├── mlp.rs          # Qwen3Mlp (SwiGLU)
│   ├── attention.rs    # Qwen3Attention (GQA)
│   ├── decoder.rs      # Qwen3DecoderLayer
│   ├── loader.rs       # HuggingFace model download, SafeTensors loading
│   └── qwen3.rs        # Qwen3Model, Qwen3ForCausalLM
├── engine/
│   ├── mod.rs
│   ├── sampler.rs      # Token sampling (temperature, top_k, top_p)
│   └── llm.rs          # LLMEngine orchestration
└── scheduler/
    ├── mod.rs
    └── batch.rs        # Scheduler with continuous batching

docs/
├── README.md           # This file
├── architecture.md     # System overview
├── configuration.md    # Config types
├── errors.md           # Error handling
├── paged_attention.md  # Block/BlockTable/BlockManager
├── sequence_kv_cache.md # Sequence/KVCache
├── scheduler.md        # Continuous batching scheduler
├── continuous_batching_visual.md # Visual guide
├── qwen3_architecture.md # Qwen3 model architecture
└── model_loader.md     # HuggingFace integration, Qwen3 model
```

## References

- [vLLM Paper](https://arxiv.org/abs/2309.06180) - Efficient Memory Management for Large Language Model Serving with PagedAttention
- [nano-vllm (Python)](https://github.com/huggingface/nano-vllm) - Reference implementation
