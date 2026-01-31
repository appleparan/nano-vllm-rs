# Sequence & KV Cache

## Overview

This document explains how nano-vllm tracks inference requests (Sequences) and stores computed attention states (KV Cache).

## Sequence

A **Sequence** represents a single inference request throughout its lifecycle. It tracks:

- Input prompt tokens
- Generated output tokens
- KV cache block allocation
- Scheduling state and priority

### Sequence Lifecycle

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        Sequence Lifecycle                           │
└─────────────────────────────────────────────────────────────────────┘

  User Request                                              Response
       │                                                        ▲
       ▼                                                        │
  ┌─────────┐    schedule()    ┌─────────┐    EOS/max     ┌──────────┐
  │ Waiting │ ───────────────► │ Running │ ─────────────► │ Finished │
  └─────────┘                  └─────────┘                └──────────┘
       ▲                            │
       │         preempt()          │
       │                            ▼
       │                       ┌─────────┐
       └────── resume() ────── │ Swapped │
                               └─────────┘
```

### Sequence States

| State | Description | Next States |
| ----- | ----------- | ----------- |
| `Waiting` | In queue, waiting to be scheduled | Running |
| `Running` | Actively generating tokens (prefill or decode) | Waiting, Swapped, Finished |
| `Swapped` | Preempted, KV cache moved to CPU | Running |
| `Finished` | Generation complete | (terminal) |

### Finish Reasons

When a sequence reaches `Finished` state, it has a `FinishReason`:

| Reason | Description |
| ------ | ----------- |
| `EndOfSequence` | Model generated EOS token |
| `MaxTokens` | Reached `max_tokens` limit |
| `StopSequence` | Generated a stop sequence |
| `Aborted` | Cancelled by user or system |

### Sequence Structure

```rust
struct Sequence {
    // Identity
    seq_id: u64,              // Unique identifier

    // Tokens
    prompt_token_ids: Vec<u32>,   // Input tokens
    output_token_ids: Vec<u32>,   // Generated tokens

    // KV Cache
    block_table: BlockTable,      // Maps positions to physical blocks
    num_prefilled_tokens: usize,  // For chunked prefill progress

    // Scheduling
    status: SequenceStatus,
    priority: i32,                // Higher = more important
    arrival_time: Instant,        // For FIFO ordering

    // Completion
    finish_reason: Option<FinishReason>,
}
```

### Token Flow

```text
Prefill Phase (process entire prompt):

  Prompt: [t0, t1, t2, t3, t4, t5, t6, t7]  (8 tokens)

  Iteration 1 (chunked prefill, chunk_size=4):
    Process: [t0, t1, t2, t3]
    num_prefilled_tokens: 0 → 4

  Iteration 2:
    Process: [t4, t5, t6, t7]
    num_prefilled_tokens: 4 → 8
    is_prefill_complete: true

Decode Phase (generate one token at a time):

  Iteration 3:
    Input: t7 (last token)
    Output: t8
    output_token_ids: [] → [t8]

  Iteration 4:
    Input: t8
    Output: t9
    output_token_ids: [t8] → [t8, t9]

  ... continue until EOS or max_tokens ...
```

### Priority Scheduling

Sequences have a priority value for scheduling decisions:

```text
Priority values (higher = more important):

  priority = 10  →  VIP request, process first
  priority = 0   →  Normal request (default)
  priority = -10 →  Low priority, can be preempted

Within same priority, use FIFO ordering (arrival_time).

Scheduling order:
  1. [priority=10, arrived=12:00:01]  ← First
  2. [priority=10, arrived=12:00:02]
  3. [priority=0,  arrived=12:00:00]
  4. [priority=-5, arrived=11:59:00]  ← Last
```

## KV Cache

The **KV Cache** stores computed key and value tensors from the attention mechanism, avoiding recomputation during generation.

### Why KV Cache?

During autoregressive generation, each new token attends to all previous tokens:

```text
Without KV Cache (inefficient):

  Token 1: Compute K1, V1, attention
  Token 2: Recompute K1, V1, compute K2, V2, attention
  Token 3: Recompute K1, V1, K2, V2, compute K3, V3, attention
  ...
  Token N: Recompute all K1..K(N-1), V1..V(N-1), compute KN, VN

  Complexity: O(N²) in compute

With KV Cache (efficient):

  Token 1: Compute K1, V1, store in cache, attention
  Token 2: Load K1, V1 from cache, compute K2, V2, store, attention
  Token 3: Load K1..K2, V1..V2 from cache, compute K3, V3, store
  ...
  Token N: Load K1..K(N-1), V1..V(N-1), compute KN, VN, store

  Complexity: O(N) in compute, O(N) in memory
```

### Memory Layout

KV Cache is organized as a 4D tensor per layer:

```text
Shape: [num_blocks, block_size, num_kv_heads, head_dim]

For Qwen3-0.6B with 1024 blocks:
  - num_blocks: 1024
  - block_size: 16
  - num_kv_heads: 8 (GQA)
  - head_dim: 64

  Key cache:   [1024, 16, 8, 64]  →  32 MB (float32)
  Value cache: [1024, 16, 8, 64]  →  32 MB (float32)
  Per layer: 64 MB
  28 layers: 1.75 GB total
```

### Block-based Access

KV Cache integrates with PagedAttention's block system:

```text
Sequence A with BlockTable [5, 12, 3]:

  Token 0-15:  → Block 5,  slots 0-15
  Token 16-31: → Block 12, slots 0-15
  Token 32-47: → Block 3,  slots 0-15

Reading K/V for attention:

  1. Get block IDs from BlockTable: [5, 12, 3]
  2. Gather from KV cache:
     K_gathered = kv_cache.gather_keys([5, 12, 3])
     V_gathered = kv_cache.gather_values([5, 12, 3])
  3. Reshape for attention:
     K: [3 blocks, 16 slots, 8 heads, 64 dim] → [48 tokens, 8, 64]
     V: [3 blocks, 16 slots, 8 heads, 64 dim] → [48 tokens, 8, 64]
```

### Multi-Layer Structure

```text
┌─────────────────────────────────────────────────────────┐
│                       KVCache                           │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Layer 0                                         │    │
│  │  ┌─────────────┐  ┌─────────────┐               │    │
│  │  │  Key Cache  │  │ Value Cache │               │    │
│  │  │ [1024,16,   │  │ [1024,16,   │               │    │
│  │  │    8,64]    │  │    8,64]    │               │    │
│  │  └─────────────┘  └─────────────┘               │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Layer 1                                         │    │
│  │  ┌─────────────┐  ┌─────────────┐               │    │
│  │  │  Key Cache  │  │ Value Cache │               │    │
│  │  └─────────────┘  └─────────────┘               │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ... (28 layers total for Qwen3-0.6B) ...               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### KVCache Operations

```rust
// Configuration
let config = KVCacheConfig::new(
    1024,  // num_blocks
    16,    // block_size
    8,     // num_kv_heads
    64,    // head_dim
    28,    // num_layers
);

// Create cache
let cache = KVCache::new(config, Device::Cpu)?;

// Access layer
let layer_cache = cache.layer_mut(0)?;

// Gather K/V for specific blocks
let block_ids = vec![5, 12, 3];
let keys = layer_cache.gather_keys(&block_ids)?;    // [3, 16, 8, 64]
let values = layer_cache.gather_values(&block_ids)?; // [3, 16, 8, 64]

// Gather from all layers at once
let all_keys = cache.gather_keys_all_layers(&block_ids)?;   // Vec of 28 tensors
let all_values = cache.gather_values_all_layers(&block_ids)?;
```

## Integration: Sequence + BlockManager + KVCache

These components work together during inference:

```text
Request Arrival:

  1. Create Sequence with prompt tokens
  2. Sequence.status = Waiting
  3. Add to scheduler's waiting queue

Scheduling:

  4. Scheduler selects sequence for execution
  5. BlockManager allocates blocks for KV storage
  6. Sequence.block_table records the mapping
  7. Sequence.status = Running

Prefill:

  8. Model processes prompt tokens
  9. Write K/V to cache via block_table slot mapping:

     for (pos, &slot) in slot_mapping.iter().enumerate() {
         let (k, v) = model.compute_kv(hidden_states, pos);
         kv_cache.layer_mut(layer).write_key(block_id, slot_in_block, &k);
         kv_cache.layer_mut(layer).write_value(block_id, slot_in_block, &v);
     }

  10. Sequence.num_prefilled_tokens = prompt_len
  11. Sample first output token

Decode Loop:

  12. Model processes last token only
  13. Gather K/V from cache for attention:

      let keys = kv_cache.layer(layer).gather_keys(block_ids);
      let values = kv_cache.layer(layer).gather_values(block_ids);
      let output = attention(query, keys, values);

  14. Allocate new block if needed (current block full)
  15. Write new K/V to cache
  16. Sequence.output_token_ids.push(new_token)
  17. If EOS or max_tokens:
      - Sequence.status = Finished
      - BlockManager.free(block_ids)  // Release KV cache blocks

Preemption (if needed):

  18. Sequence.status = Swapped
  19. Copy KV blocks to CPU (or just drop and recompute later)
  20. BlockManager.free(block_ids)
  21. Later: Sequence.status = Running, reallocate blocks
```

## Memory Considerations

### KV Cache Memory Budget

```text
For Qwen3-0.6B serving:

  Per-token KV memory:
    = 2 (K+V) × 28 layers × 8 heads × 64 dim × 4 bytes
    = 114,688 bytes ≈ 112 KB per token

  For 1024 blocks × 16 tokens/block:
    = 16,384 tokens max
    = 1.75 GB KV cache

  GPU memory breakdown:
    - Model weights: ~1.2 GB (float16)
    - KV cache: ~1.75 GB
    - Activations: ~0.5 GB
    - Total: ~3.5 GB
```

### Sequence Memory

```text
Per-sequence overhead (excluding KV cache):

  - prompt_token_ids: 4 bytes × prompt_len
  - output_token_ids: 4 bytes × output_len
  - block_table: 8 bytes × num_blocks
  - metadata: ~100 bytes

  For 1000 concurrent sequences with avg 2048 tokens:
    ≈ 1000 × (2048 × 4 + 128 × 8 + 100)
    ≈ 9 MB (negligible compared to KV cache)
```

## Design Note: Sequence vs SequenceGroup

nano-vllm uses a simple `Sequence` structure, but the original vLLM introduces `SequenceGroup` for advanced sampling strategies.

### Why vLLM Has SequenceGroup

vLLM supports sampling methods that generate multiple candidate sequences from a single request:

```text
SequenceGroup in vLLM:

  ┌─────────────────────────────────────────────────────────┐
  │ SequenceGroup (request_id: "req-1")                     │
  │                                                         │
  │   Shared prompt: "Once upon a time"                     │
  │   Sampling params: n=3, beam_width=4                    │
  │                                                         │
  │   ┌──────────┐  ┌──────────┐  ┌──────────┐             │
  │   │ Seq 0    │  │ Seq 1    │  │ Seq 2    │   ...       │
  │   │ "there"  │  │ "in a"   │  │ "lived"  │             │
  │   └──────────┘  └──────────┘  └──────────┘             │
  └─────────────────────────────────────────────────────────┘

Use cases:
  - Beam search: Keep top-k candidates at each step
  - Parallel sampling (n > 1): Generate multiple completions
  - Best-of-n: Generate n, return the best one
```

### Benefits of SequenceGroup

1. **Shared Prompt Blocks**: All sequences in a group share the same prefix KV cache blocks (copy-on-write)
2. **Coordinated Scheduling**: The group is scheduled together
3. **Unified Completion**: The group finishes when sampling criteria are met

```text
Memory sharing with SequenceGroup:

  Prompt blocks (shared):     Divergent blocks (separate):
  ┌─────┬─────┬─────┐
  │ B0  │ B1  │ B2  │  ←  ref_count = 3 (shared by 3 seqs)
  └─────┴─────┴─────┘
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
  ┌───┐ ┌───┐ ┌───┐
  │B3 │ │B4 │ │B5 │  ←  Each sequence's own continuation
  └───┘ └───┘ └───┘
```

### Why nano-vllm Uses Simple Sequence

For educational clarity, nano-vllm implements only greedy/simple sampling:

| Feature | vLLM | nano-vllm |
| ------- | ---- | --------- |
| Basic generation | SequenceGroup with 1 Sequence | Sequence |
| Beam search | SequenceGroup with k Sequences | Not supported |
| Parallel sampling | SequenceGroup with n Sequences | Not supported |
| Prefix sharing | Copy-on-write via SequenceGroup | Prefix caching only |

This simplification allows focusing on the core concepts (PagedAttention, continuous batching) without the complexity of multi-sequence management.

### Extending to SequenceGroup

If you wanted to add SequenceGroup support:

```rust
// Hypothetical SequenceGroup structure
struct SequenceGroup {
    request_id: String,
    sequences: Vec<Sequence>,        // Multiple candidates
    sampling_params: SamplingParams, // n, beam_width, etc.
    shared_block_ids: Vec<usize>,    // Prefix blocks (copy-on-write)
}
```

Key implementation considerations:

1. **Fork operation**: Create new sequence sharing prefix blocks (increment ref_count)
2. **Copy-on-write**: When writing to shared block, copy first if ref_count > 1
3. **Group scheduling**: Schedule all sequences in group together
4. **Beam management**: Prune low-scoring candidates, keep top-k

## Summary

| Component | Role | Key Data |
| --------- | ---- | -------- |
| Sequence | Request lifecycle tracking | tokens, status, priority |
| BlockTable | Position → Block mapping | block_ids list |
| KVCache | K/V tensor storage | [blocks, slots, heads, dim] |
| BlockManager | Block allocation | free_list, ref_count |
