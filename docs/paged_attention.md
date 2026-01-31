# PagedAttention

## Overview

PagedAttention is the core memory optimization in vLLM. It manages KV cache memory like an operating system manages virtual memory - dividing it into fixed-size blocks (pages) that can be allocated, shared, and freed independently.

## The Memory Problem

In traditional LLM inference, KV cache is pre-allocated contiguously for each sequence:

```
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

```
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

```
Token position → Logical block index → Physical block ID → Global slot

Position Mapping:
  logical_block = position / block_size
  slot_in_block = position % block_size
  physical_block = block_table[logical_block]
  global_slot = physical_block * block_size + slot_in_block
```

### Example

```
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

When multiple sequences share a common prefix, they can share blocks:

```
System prompt: "You are a helpful assistant..."

Request 1: "You are a helpful assistant. What is 2+2?"
Request 2: "You are a helpful assistant. Explain quantum physics."

Without prefix caching:
  Request 1: [A0][A1][A2][A3]  # Duplicated prefix
  Request 2: [B0][B1][B2][B3]

With prefix caching:
  Request 1: [S0][S1][S2][A0]  # Shared prefix blocks
  Request 2: [S0][S1][S2][B0]
             ↑↑↑ Same physical blocks (ref_count = 2)
```

### Hash-based Identification

Prefix blocks are identified using cumulative hashes:

```rust
// First block
hash_1 = hash(tokens[0..16])

// Second block includes parent hash
hash_2 = hash((hash_1, tokens[16..32]))

// Third block
hash_3 = hash((hash_2, tokens[32..48]))
```

This ensures blocks are only shared when the **entire prefix** matches, not just the current block's tokens.

## Memory Layout

The KV cache is a large tensor accessed via block IDs:

```
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

```python
for position in range(seq_len):
    slot = slot_mapping[position]  # From BlockTable
    kv_cache[slot] = compute_kv(hidden_states[position])
```

### Decode Phase
Only the last token is processed. Attention gathers K/V from blocks:

```python
# Gather K/V from non-contiguous blocks
for block_id in block_table:
    K_block = kv_cache[block_id]  # [block_size, num_kv_heads, head_dim]
    V_block = ...
    # Compute attention with gathered K/V
```

## Summary

| Component | Analogy | Purpose |
|-----------|---------|---------|
| Block | Memory Page | Fixed-size KV storage unit |
| BlockTable | Page Table | Logical → Physical mapping |
| BlockManager | Memory Allocator | Allocation, freeing, sharing |
| prefix_hash | Content Hash | Identify shareable prefixes |
| ref_count | Reference Count | Track shared block usage |
