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

**Implemented:**

| File | Type | Description |
| ---- | ---- | ----------- |
| `src/error.rs` | `Error` enum | Error types (Config, Model, Tokenization, etc.) |
| `src/error.rs` | `Result<T>` | Type alias for `std::result::Result<T, Error>` |
| `src/config.rs` | `ModelConfig` | Model hyperparameters |
| `src/config.rs` | `SchedulerConfig` | Scheduler settings |
| `src/config.rs` | `EngineConfig` | Engine settings |
| `src/config.rs` | `SamplingConfig` | Sampling parameters |

**Why implement this first?**

These are shared types used by all modules. Error types enable `?` operator for error propagation, and Config types control component behavior. Defining these first ensures consistent error handling and configuration management throughout the project.

### Stage 2: PagedAttention Memory Management

- [PagedAttention](paged_attention.md) - Block, BlockTable, BlockManager

**Implemented:**

| File | Type | Description |
| ---- | ---- | ----------- |
| `src/core/block.rs` | `Block` | Physical block with ref_count, prefix_hash |
| `src/core/block.rs` | `BlockTable` | Logical → physical block mapping |
| `src/core/block.rs` | `hash_token_block()` | Cumulative hash for prefix caching |
| `src/core/block.rs` | `compute_num_blocks()` | Calculate required blocks for token count |
| `src/core/block_manager.rs` | `BlockManager` | Free-list allocator with prefix cache |
| `src/core/block_manager.rs` | `BlockManager::allocate()` | Allocate blocks for sequence |
| `src/core/block_manager.rs` | `BlockManager::free()` | Release blocks (ref-count based) |
| `src/core/block_manager.rs` | `BlockManager::get_cached_block()` | Prefix cache lookup |

**Why this order?**

vLLM's core innovation is PagedAttention, inspired by OS virtual memory systems:

1. **Block** = Memory Page: Splits KV cache into fixed-size blocks
2. **BlockTable** = Page Table: Logical position → Physical block mapping
3. **BlockManager** = Memory Allocator: Block allocation/deallocation, reference counting

This abstraction enables Sequence, KVCache, and Scheduler to manage memory efficiently. It makes Prefix Caching (reusing common prompts) and Preemption (swapping low-priority requests) possible.

### Stage 3: Request Tracking & KV Storage

- [Sequence & KV Cache](sequence_kv_cache.md) - Sequence lifecycle, KV tensor storage

**Implemented:**

| File | Type | Description |
| ---- | ---- | ----------- |
| `src/core/sequence.rs` | `SequenceStatus` | Waiting, Running, Swapped, Finished |
| `src/core/sequence.rs` | `FinishReason` | EndOfSequence, MaxTokens, StopSequence, Aborted |
| `src/core/sequence.rs` | `Sequence` | Request lifecycle (tokens, state, priority) |
| `src/core/sequence.rs` | `Sequence::append_token()` | Add generated token |
| `src/core/sequence.rs` | `Sequence::set_status()` | State transition with validation |
| `src/core/kv_cache.rs` | `KVCacheConfig` | Cache configuration |
| `src/core/kv_cache.rs` | `LayerKVCache` | Per-layer K/V tensor storage |
| `src/core/kv_cache.rs` | `KVCache` | Multi-layer cache container |
| `src/core/kv_cache.rs` | `KVCache::gather_keys()` | Block-based key retrieval |
| `src/core/kv_cache.rs` | `KVCache::gather_values()` | Block-based value retrieval |

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

**Implemented:**

| File | Type | Description |
| ---- | ---- | ----------- |
| `src/scheduler/batch.rs` | `SchedulerOutputs` | Schedule decision (prefill/decode lists) |
| `src/scheduler/batch.rs` | `Scheduler` | Main scheduler with waiting/running/swapped queues |
| `src/scheduler/batch.rs` | `Scheduler::add_sequence()` | Add new request to waiting queue |
| `src/scheduler/batch.rs` | `Scheduler::schedule()` | Main scheduling algorithm |
| `src/scheduler/batch.rs` | `Scheduler::append_token()` | Add token to running sequence |
| `src/scheduler/batch.rs` | `Scheduler::mark_prefilled()` | Update prefill progress |
| `src/scheduler/batch.rs` | `Scheduler::finish_sequence()` | Complete sequence with reason |
| `src/scheduler/batch.rs` | `Scheduler::has_pending_requests()` | Check for unfinished work |

**Why this order?**

With Sequence tracking requests and KVCache storing tensors, the Scheduler orchestrates them:

1. **Scheduling Algorithm**: Priority queue for waiting, running set for active sequences
2. **Continuous Batching**: Mix prefill and decode phases in single iteration
3. **Resource Management**: Track block allocations, enforce limits (max_num_seqs, max_prefill_tokens)
4. **Preemption**: Swap low-priority sequences when high-priority requests need memory

### Stage 5: Qwen3 Model Components

- [Qwen3 Architecture](qwen3_architecture.md) - Complete architecture explanation with diagrams

**Implemented:**

| File | Type | Description |
| ---- | ---- | ----------- |
| `src/model/norm.rs` | `RmsNorm` | Root Mean Square Normalization |
| `src/model/norm.rs` | `RmsNorm::forward()` | x / rms(x) * weight |
| `src/model/rope.rs` | `RotaryEmbedding` | Rotary Position Embeddings |
| `src/model/rope.rs` | `RotaryEmbedding::apply()` | Apply rotation to Q, K |
| `src/model/mlp.rs` | `Qwen3Mlp` | SwiGLU feed-forward network |
| `src/model/mlp.rs` | `Qwen3Mlp::forward()` | down(gate * silu(up)) |
| `src/model/attention.rs` | `Qwen3Attention` | Grouped Query Attention (GQA) |
| `src/model/attention.rs` | `Qwen3Attention::forward()` | Q/K/V projection, RoPE, SDPA |
| `src/model/decoder.rs` | `Qwen3DecoderLayer` | Pre-norm residual block |
| `src/model/decoder.rs` | `Qwen3DecoderLayer::forward()` | attention + mlp + residuals |

**Why this order?**

The model components build on each other:

- RmsNorm is used everywhere (attention, MLP, final output)
- RoPE is applied to Q and K in attention
- MLP and Attention are combined in DecoderLayer
- DecoderLayer is stacked N times to form the full model

### Stage 6: PagedAttention

- [PagedAttention](paged_attention.md) - Block-based attention operations

**Implemented:**

| File | Type | Description |
| ---- | ---- | ----------- |
| `src/attention/paged.rs` | `prefill_attention()` | Standard SDPA for prompt processing |
| `src/attention/paged.rs` | `paged_attention()` | Block-based K/V gather + attention |
| `src/attention/paged.rs` | `write_kv_to_cache()` | Store K/V to block-based cache |
| `src/attention/paged.rs` | `gather_from_cache()` | Retrieve K/V from blocks |
| `src/attention/paged.rs` | `create_causal_mask()` | Causal attention mask |

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

**Implemented:**

| File | Type | Description |
| ---- | ---- | ----------- |
| `src/model/loader.rs` | `Qwen3Config` | HuggingFace config.json parsing |
| `src/model/loader.rs` | `ModelFiles` | Paths to downloaded model files |
| `src/model/loader.rs` | `download_model()` | Download from HuggingFace Hub |
| `src/model/loader.rs` | `load_config()` | Parse config.json to Qwen3Config |
| `src/model/loader.rs` | `load_safetensors()` | Memory-mapped weight loading |
| `src/model/qwen3.rs` | `Qwen3Model` | embed_tokens + layers + norm |
| `src/model/qwen3.rs` | `Qwen3Model::forward()` | Embedding → Layers → Norm |
| `src/model/qwen3.rs` | `Qwen3ForCausalLM` | Model + lm_head |
| `src/model/qwen3.rs` | `Qwen3ForCausalLM::forward()` | Returns logits for last position |
| `src/model/qwen3.rs` | `Qwen3ForCausalLM::forward_all()` | Returns logits for all positions |

**Why this order?**

With all model components implemented in Stage 5-6, we can now:

- Load pretrained weights from HuggingFace
- Assemble the full Qwen3 transformer model
- Prepare for inference in Stage 8

### Stage 8: Sampler & LLM Engine

- [Engine](engine.md) - Sampler and LLMEngine documentation

**Implemented:**

| File | Type | Description |
| ---- | ---- | ----------- |
| `src/engine/sampler.rs` | `Sampler` | Token sampling with RNG |
| `src/engine/sampler.rs` | `Sampler::new()` | Create with SamplingConfig |
| `src/engine/sampler.rs` | `Sampler::with_seed()` | Create with fixed seed |
| `src/engine/sampler.rs` | `Sampler::sample()` | Sample token(s) from logits |
| `src/engine/sampler.rs` | `Sampler::apply_top_k()` | Keep top-k tokens |
| `src/engine/sampler.rs` | `Sampler::apply_top_p()` | Nucleus sampling filter |
| `src/engine/llm.rs` | `GenerationRequest` | Request with prompt + config |
| `src/engine/llm.rs` | `GenerationOutput` | Result with text + tokens |
| `src/engine/llm.rs` | `LLMEngine` | Main inference orchestrator |
| `src/engine/llm.rs` | `LLMEngine::add_request()` | Add request (tokenize + schedule) |
| `src/engine/llm.rs` | `LLMEngine::step()` | Single inference iteration |
| `src/engine/llm.rs` | `LLMEngine::generate()` | Run until all complete |
| `src/engine/llm.rs` | `LLMEngine::generate_text()` | Convenience single-prompt method |

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
├── model_loader.md     # HuggingFace integration, Qwen3 model
└── engine.md           # Sampler & LLMEngine
```

## References

- [vLLM Paper](https://arxiv.org/abs/2309.06180) - Efficient Memory Management for Large Language Model Serving with PagedAttention
- [nano-vllm (Python)](https://github.com/huggingface/nano-vllm) - Reference implementation
