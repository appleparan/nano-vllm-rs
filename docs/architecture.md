# Architecture

## Overview

nano-vllm-rs implements core vLLM optimizations in Rust for educational purposes.

```
┌─────────────────────────────────────────────────────────┐
│                        CLI                              │
│                     (main.rs)                           │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                      Engine                             │
│              (engine/engine.rs)                         │
│  - Orchestrates inference                               │
│  - Manages prefill/decode phases                        │
└───────┬─────────────────┬─────────────────┬─────────────┘
        │                 │                 │
┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
│   Scheduler   │ │     Model     │ │    Sampler    │
│  (scheduler/) │ │    (model/)   │ │   (engine/)   │
│               │ │               │ │               │
│ - Continuous  │ │ - Qwen3 arch  │ │ - Temperature │
│   batching    │ │ - RMSNorm     │ │ - Top-k/p     │
│ - Priority    │ │ - RoPE        │ │               │
│ - Preemption  │ │ - GQA         │ │               │
└───────┬───────┘ │ - SwiGLU      │ └───────────────┘
        │         └───────┬───────┘
┌───────▼─────────────────▼───────────────────────────────┐
│                        Core                             │
│                      (core/)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │    Block    │  │  Sequence   │  │   KVCache   │     │
│  │ BlockTable  │  │             │  │             │     │
│  │BlockManager │  │             │  │             │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                     Attention                           │
│                   (attention/)                          │
│  - PagedAttention (block-based KV access)               │
│  - FlashAttention (optional)                            │
└─────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### `core/`
Memory management primitives for PagedAttention:
- **Block**: Fixed-size chunk of KV cache (like a memory page)
- **BlockTable**: Logical-to-physical block mapping (like a page table)
- **BlockManager**: Allocates/frees blocks using free list
- **Sequence**: Tracks request state and generated tokens
- **KVCache**: Tensor storage for key-value pairs

### `scheduler/`
Continuous batching scheduler:
- Admits new requests from waiting queue
- Manages running sequences
- Handles priority-based preemption
- Supports chunked prefill for long prompts

### `attention/`
Attention implementations:
- **PagedAttention**: Gathers K/V from non-contiguous blocks
- **FlashAttention**: Memory-efficient attention (optional)

### `model/`
Qwen3 architecture:
- **RMSNorm**: Root Mean Square normalization
- **RoPE**: Rotary Position Embeddings
- **Qwen3Attention**: Grouped Query Attention (GQA)
- **Qwen3MLP**: SwiGLU feed-forward network
- **Qwen3Model**: Full transformer stack

### `engine/`
Inference orchestration:
- **LLMEngine**: Coordinates scheduler, model, and sampler
- **Sampler**: Token sampling with temperature, top-k, top-p

### `speculative/`
Speculative decoding (optional):
- Draft model generates K candidate tokens
- Target model verifies in single forward pass
- Rejection sampling maintains output distribution

## Data Flow

### 1. Request Submission
```
User prompt → Tokenize → Sequence → Waiting queue
```

### 2. Scheduling
```
Waiting queue → Scheduler.schedule() → SchedulerOutputs
  - prefill_sequences: New requests to prefill
  - decode_sequences: Running requests to decode
  - chunked_prefill_sequences: Long prompts in chunks
```

### 3. Prefill Phase
```
Prompt tokens → Model forward → KV cache populated → First token sampled
```

### 4. Decode Phase
```
Last token → Model forward (with KV cache) → Next token sampled → Repeat
```

### 5. Completion
```
EOS or max_tokens → Sequence finished → Detokenize → Output
```
