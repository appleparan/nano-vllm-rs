# PagedAttention

## Overview

PagedAttention is the core memory optimization in vLLM. It manages KV cache memory like an operating system manages virtual memory - dividing it into fixed-size blocks (pages) that can be allocated, shared, and freed independently.

## The Memory Problem

In traditional LLM inference, KV cache is pre-allocated contiguously for each sequence:

```text
Traditional approach:
Sequence A: [████████████████████░░░░░░░░░░░░░░░]  (used: 20, allocated: 36)
Sequence B: [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░]  (used: 8, allocated: 36)
Sequence C: [████████████░░░░░░░░░░░░░░░░░░░░░░░]  (used: 12, allocated: 36)
                                                   Total waste: 32 tokens
```

This leads to:

- **Internal fragmentation**: Pre-allocated space may not be fully used
- **External fragmentation**: Memory gaps between sequences
- **No sharing**: Common prefixes duplicated in memory

## PagedAttention Solution

PagedAttention divides KV cache into fixed-size blocks:

```text
PagedAttention approach:
Block Pool: [B0][B1][B2][B3][B4][B5][B6][B7][B8]...

Sequence A (20 tokens, block_size=16):
  BlockTable: [0, 3]     → Blocks B0 (16 tokens) + B3 (4 tokens)

Sequence B (8 tokens):
  BlockTable: [1]        → Block B1 (8 tokens)

Sequence C (12 tokens):
  BlockTable: [2]        → Block B2 (12 tokens)

Free blocks: [B4, B5, B6, B7, B8, ...]
```

## Block Structure

A **Block** is the unit of allocation, analogous to a memory page:

```rust
Block {
    block_id: 0,        // Unique identifier (physical address)
    block_size: 16,     // Tokens per block
    ref_count: 1,       // For sharing (prefix caching)
    prefix_hash: Some(hash),  // For identifying shared prefixes
    is_full: true,      // Whether all slots are used
}
```

### Block Sizing

The `block_size` parameter affects:

- **Smaller blocks (8-16)**: Fine-grained allocation, less waste, more metadata overhead
- **Larger blocks (32-64)**: Less overhead, potentially more internal fragmentation

Default: 16 tokens per block.

## BlockTable: The Page Table

A **BlockTable** maps a sequence's logical positions to physical blocks:

```text
Token position → Logical block index → Physical block ID → Global slot

Position Mapping:
  logical_block = position / block_size
  slot_in_block = position % block_size
  physical_block = block_table[logical_block]
  global_slot = physical_block * block_size + slot_in_block
```

### Example

```text
block_size = 16
sequence has 35 tokens

BlockTable.block_ids = [5, 12, 3]  # 3 physical blocks

Token 0:  block 5, slot 0   → global slot 80
Token 15: block 5, slot 15  → global slot 95
Token 16: block 12, slot 0  → global slot 192
Token 31: block 12, slot 15 → global slot 207
Token 32: block 3, slot 0   → global slot 48
Token 34: block 3, slot 2   → global slot 50
```

## Prefix Caching

### Why Prefix Caching?

In production LLM deployments, many requests share common prefixes:

1. **System prompts**: "You are a helpful assistant..." prepended to every request
2. **Few-shot examples**: Same examples used for in-context learning
3. **Document context**: Multiple questions about the same document
4. **Chat history**: Multi-turn conversations with shared context

Without prefix caching, each request recomputes and stores duplicate KV states:

```text
1000 requests × 2048 tokens system prompt × 2 (K+V) × hidden_dim × num_layers
= Massive memory waste and redundant computation
```

### How It Works

Prefix caching allows multiple sequences to **share the same physical blocks** for their common prefix:

```text
System prompt: "You are a helpful assistant..."  (48 tokens = 3 blocks)

Request 1: "You are a helpful assistant. What is 2+2?"
Request 2: "You are a helpful assistant. Explain quantum physics."
Request 3: "You are a helpful assistant. Write a poem."

Memory layout WITHOUT prefix caching (9 blocks for prefix alone):
  Request 1: [A0][A1][A2][A3]
  Request 2: [B0][B1][B2][B3]
  Request 3: [C0][C1][C2][C3]

Memory layout WITH prefix caching (3 blocks for prefix):
  Request 1: [S0][S1][S2] → [A3]    ← S0,S1,S2 shared
  Request 2: [S0][S1][S2] → [B3]    ← ref_count = 3
  Request 3: [S0][S1][S2] → [C3]

Savings: 6 blocks saved = 66% memory reduction for prefix!
```

### Hash-based Block Identification

The key question is: **How do we know if a block can be reused?**

We use **cumulative hashing** to create a unique identifier for each block that captures its entire prefix history:

```rust
// Block 0: Just the first 16 tokens
hash_0 = hash(tokens[0..16])

// Block 1: Includes Block 0's hash to form a chain
hash_1 = hash((hash_0, tokens[16..32]))

// Block 2: Includes the entire prefix chain
hash_2 = hash((hash_1, tokens[32..48]))
```

#### Why Cumulative Hashing?

Simple per-block hashing would be incorrect:

```text
Problem with simple hashing:

Sequence A: "Hello world. How are you?"
Sequence B: "Goodbye world. How are you?"
                           ↑
                    Block 2 has same tokens!
                    But different context!

Simple hash: hash("How are you?") → same hash → WRONG sharing!

Cumulative hash:
  A: hash((hash_block_1_A, "How are you?")) → hash_A
  B: hash((hash_block_1_B, "How are you?")) → hash_B
  hash_A ≠ hash_B → Correctly different!
```

The cumulative hash ensures we only share blocks when the **entire prefix** matches, not just the current block's tokens.

### Prefix Cache Lookup Flow

```text
New request arrives with tokens [t0, t1, t2, ..., t47]

1. Compute hash for block 0: h0 = hash(tokens[0..16])
2. Look up h0 in prefix_cache
   - HIT: Reuse block, increment ref_count
   - MISS: Allocate new block, store in cache

3. Compute hash for block 1: h1 = hash((h0, tokens[16..32]))
4. Look up h1 in prefix_cache
   - HIT: Reuse block, increment ref_count
   - MISS: Allocate new block, store in cache

5. Continue until all prefix blocks are resolved...

6. Allocate fresh blocks for the unique suffix
```

## Reference Counting

### Why Reference Counting?

When blocks are shared between sequences, we need to track **how many sequences are using each block**. This is critical for:

1. **Safe deallocation**: Don't free a block while another sequence still needs it
2. **Copy-on-write semantics**: Know when a block is exclusively owned
3. **Memory accounting**: Track actual memory usage

### The Sharing Lifecycle

```text
Timeline:

T0: Request A arrives, allocates blocks [B0, B1, B2, B3]
    B0.ref_count = 1, B1.ref_count = 1, B2.ref_count = 1, B3.ref_count = 1

T1: Request B arrives with same prefix, shares [B0, B1, B2], allocates [B4]
    B0.ref_count = 2, B1.ref_count = 2, B2.ref_count = 2  ← incremented
    B3.ref_count = 1  (A's suffix)
    B4.ref_count = 1  (B's suffix)

T2: Request A completes, frees its blocks
    B0.ref_count = 1  ← decremented, NOT freed (still used by B)
    B1.ref_count = 1
    B2.ref_count = 1
    B3.ref_count = 0  → returned to free list

T3: Request B completes, frees its blocks
    B0.ref_count = 0  → returned to free list
    B1.ref_count = 0  → returned to free list
    B2.ref_count = 0  → returned to free list
    B4.ref_count = 0  → returned to free list
```

### BlockManager Operations

```rust
// Sharing a cached block (prefix hit)
fn get_cached_block(&mut self, hash: u64) -> Option<usize> {
    if let Some(&block_id) = self.prefix_cache.get(&hash) {
        self.blocks[block_id].increment_ref();  // ref_count: 1 → 2
        return Some(block_id);
    }
    None
}

// Freeing a block (sequence completes)
fn free(&mut self, block_id: usize) -> bool {
    let block = &mut self.blocks[block_id];
    block.decrement_ref();  // ref_count: 2 → 1

    if block.ref_count() == 0 {
        // Last reference gone, actually free the block
        self.prefix_cache.remove(&block.prefix_hash());
        self.free_list.push(block_id);
        return true;
    }
    false  // Still in use by other sequences
}
```

### Copy-on-Write (Future Extension)

Reference counting also enables copy-on-write semantics for speculative decoding:

```text
Speculation scenario:

1. Target model and draft model share prefix blocks
2. Draft model speculatively extends the sequence
3. If speculation is rejected:
   - Draft's blocks are freed (ref_count goes to 0)
   - Target's blocks remain valid (ref_count > 0)
4. If speculation is accepted:
   - Ownership transfers, ref_counts adjusted accordingly
```

## Memory Layout

The KV cache is a large tensor accessed via block IDs:

```text
KV Cache Shape: [num_blocks, block_size, num_kv_heads, head_dim]

For Qwen3-0.6B (num_kv_heads=8, head_dim=64):
  Key cache:   [1024, 16, 8, 64]
  Value cache: [1024, 16, 8, 64]

Accessing token at position p in sequence with BlockTable [5, 12, 3]:
  logical_block = p // 16
  slot = p % 16
  physical_block = block_table[logical_block]

  K[physical_block, slot, :, :]  # Shape: [8, 64]
  V[physical_block, slot, :, :]  # Shape: [8, 64]
```

## Integration with Attention

During attention computation:

### Prefill Phase

All tokens are processed at once. KV pairs are written to allocated blocks:

```rust
// Write KV states to cache during prefill
let slot_mapping = block_table.get_slot_mapping(seq_len);
for (position, &slot) in slot_mapping.iter().enumerate() {
    let kv = compute_kv(&hidden_states, position);
    kv_cache.write(slot, &kv);
}
```

### Decode Phase

Only the last token is processed. Attention gathers K/V from blocks:

```rust
// Gather K/V from non-contiguous blocks for attention
let block_ids = block_table.get_physical_block_ids();
for &block_id in block_ids {
    let k_block = kv_cache.get_key_block(block_id);   // [block_size, num_kv_heads, head_dim]
    let v_block = kv_cache.get_value_block(block_id);
    // Compute attention with gathered K/V
}
```

## Summary

| Component | Analogy | Purpose |
| --------- | ------- | ------- |
| Block | Memory Page | Fixed-size KV storage unit |
| BlockTable | Page Table | Logical → Physical mapping |
| BlockManager | Memory Allocator | Allocation, freeing, sharing |
| prefix_hash | Content Hash | Identify shareable prefixes via cumulative hashing |
| ref_count | Reference Count | Track shared block usage for safe deallocation |

## References

- [vLLM Paper: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [OS Virtual Memory](https://en.wikipedia.org/wiki/Virtual_memory) - The inspiration for PagedAttention's design
