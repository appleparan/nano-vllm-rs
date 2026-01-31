# Scheduler: Continuous Batching

This document explains the Scheduler implementation in nano-vllm, which provides continuous batching for efficient LLM inference.

## Overview

The scheduler manages the lifecycle of inference requests, deciding which sequences to process in each iteration based on available resources and priorities.

```text
┌───────────────────────────────────────────────────────────────┐
│                       Scheduler Flow                          │
└───────────────────────────────────────────────────────────────┘

  add_sequence()                              schedule()
       │                                           │
       ▼                                           ▼
  ┌─────────┐                               ┌───────────┐
  │ Waiting │ ─────────────────────────────►│  Running  │
  │  Queue  │   (when resources available)  │    Set    │
  └─────────┘                               └───────────┘
       ▲                                           │
       │             preempt()                     │
       └───────────────────────────────────────────┘
                 (when memory pressure)
```

## Why Continuous Batching?

### The Problem with Static Batching

Traditional batching pads all prompts to match the longest sequence:

```text
Static Batching (with padding):
┌──────────────────────────────────────────────────────────────┐
│ Prompt 0: [<pad><pad><pad><pad> I  am sure this will ]       │
│ Prompt 1: [<bos> How are you doing today ?      <eos>]       │
└──────────────────────────────────────────────────────────────┘
           ↑ 4 padding tokens wasted

Problems:
- Compute wasted on padding tokens
- All sequences must wait for the longest one to finish
- New requests cannot join until entire batch completes
```

The padding cost grows quadratically with batch size and prompt length. With B=8 sequences and a new 100-token prompt, dynamic insertion requires ~693 padding tokens!

### Continuous Batching Solution

Continuous batching combines three key techniques:

1. **KV Caching**: Store computed K/V states to avoid recomputation
2. **Chunked Prefill**: Split long prompts into memory-fitting chunks
3. **Ragged Batching + Dynamic Scheduling**: Concatenate variable-length sequences without padding

```text
Continuous Batching (no padding):
┌──────────────────────────────────────────────────────────────┐
│ [<bos> I am sure this will] [<bos> How are you <eos>]        │
│         Seq 0 (decode)              Seq 1 (prefill)          │
└──────────────────────────────────────────────────────────────┘
           ↓ attention mask controls interaction

┌───────────────────────────┐
│ ■ ■ ■ ■ ■ ■ □ □ □ □ □     │  Seq 0 tokens see only Seq 0
│ ■ ■ ■ ■ ■ ■ □ □ □ □ □     │
│ □ □ □ □ □ □ ■ ■ ■ ■ ■     │  Seq 1 tokens see only Seq 1
│ □ □ □ □ □ □ ■ ■ ■ ■ ■     │
└───────────────────────────┘
■ = True (can attend)  □ = False (cannot attend)
```

## Prefill vs Decode Phases

Understanding these two phases is crucial for efficient scheduling.

### Prefill Phase

Process all prompt tokens to populate the KV cache:

```text
Input: "I am sure this project"

Tokens: [<bos>, I, am, sure, this, pro, ject]

Attention: Full causal attention over all tokens

  ┌───┬───┬───┬───┬───┬───┬───┐
  │ ■ │   │   │   │   │   │   │  <bos>
  │ ■ │ ■ │   │   │   │   │   │  I
  │ ■ │ ■ │ ■ │   │   │   │   │  am
  │ ■ │ ■ │ ■ │ ■ │   │   │   │  sure
  │ ■ │ ■ │ ■ │ ■ │ ■ │   │   │  this
  │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │   │  pro
  │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │  ject → predicts "will"
  └───┴───┴───┴───┴───┴───┴───┘

KV Cache: Stores K,V for all 7 tokens
Output: Next token prediction from last position

Complexity: O(n²) for attention computation
```

### Decode Phase

Generate one token at a time using cached K/V states:

```text
After prefill, generate "will":

New token: [will]

Q: Only compute for new token
K,V: Retrieve from cache + append new

  ┌───┬───┬───┬───┬───┬───┬───┬───┐
  │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │  will → "be"
  └───┴───┴───┴───┴───┴───┴───┴───┘
    ↑                           ↑
    └── From KV cache ──────────┘

Only 1 row of attention computed!

Complexity: O(n) - only 1 query token attends to n cached keys
```

### Why This Matters for Scheduling

```text
Token budget: m = 10 tokens per iteration

Strategy: Prioritize decode (1 token each), fill remainder with prefill

Iteration 1:
┌──────────────────────────────────────────────────┐
│ Decode: A(1) + B(1) + C(1) = 3 tokens            │
│ Prefill: D(7 tokens, chunked to fit) = 7 tokens  │
│ Total: 10 tokens                                 │
└──────────────────────────────────────────────────┘

Iteration 2:
┌──────────────────────────────────────────────────┐
│ Decode: A(1) + B(1) + C(1) + D(1) = 4 tokens     │
│ Prefill: E(6 tokens) = 6 tokens                  │
│ Total: 10 tokens                                 │
└──────────────────────────────────────────────────┘
```

## Continuous Batching in Action

Here's how continuous batching handles multiple requests over time:

```text
Time →
──────────────────────────────────────────────────────────────────

Iteration 1: Initial batch
┌────────────────────────────────────────────────────────────────┐
│  Seq A: [prefill ████████]                                     │
│  Seq B: [prefill ████]                                         │
└────────────────────────────────────────────────────────────────┘

Iteration 2: A,B in decode, C joins with prefill
┌────────────────────────────────────────────────────────────────┐
│  Seq A: [decode █]                                             │
│  Seq B: [decode █]                                             │
│  Seq C: [prefill ██████]  ← NEW! Joins immediately             │
└────────────────────────────────────────────────────────────────┘

Iteration 3: B finishes, D joins
┌────────────────────────────────────────────────────────────────┐
│  Seq A: [decode █]                                             │
│  Seq B: [<eos>] → FINISHED, resources released                 │
│  Seq C: [decode █]                                             │
│  Seq D: [prefill ████]  ← NEW! Fills B's slot                  │
└────────────────────────────────────────────────────────────────┘

Iteration 4: Continues...
┌────────────────────────────────────────────────────────────────┐
│  Seq A: [decode █]                                             │
│  Seq C: [decode █]                                             │
│  Seq D: [decode █]                                             │
│  Seq E: [prefill ██████]  ← More requests can join             │
└────────────────────────────────────────────────────────────────┘
```

**Key insight**: No sequence ever waits for another to finish. The GPU stays fully utilized.

## Chunked Prefill

When prompts are too long to fit in memory, split them into chunks:

```text
Long prompt: 100 tokens, but memory only fits 40 tokens

┌────────────────────────────────────────────────────────────────┐
│ Chunk 1: tokens[0:40]   → KV cache stores K,V[0:40]            │
│ Chunk 2: tokens[40:80]  → KV cache stores K,V[0:80]            │
│ Chunk 3: tokens[80:100] → KV cache stores K,V[0:100]           │
└────────────────────────────────────────────────────────────────┘

Each chunk uses cached K,V from previous chunks.
Other sequences can run between chunks!
```

Benefits:

- Long prompts don't block other requests
- Memory usage stays within limits
- Better latency distribution

## Priority Scheduling

Sequences are scheduled based on:

1. **Priority** (higher first) - User-defined importance
2. **Arrival Order** (earlier first) - FIFO within same priority

```rust
// Priority queue ordering: (priority DESC, arrival_time ASC)
impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => other.arrival_order.cmp(&self.arrival_order),
            ord => ord,
        }
    }
}
```

Example:

```text
Waiting Queue (BinaryHeap, highest priority first):
┌────────────────────────────────────────────────────────────────┐
│ 1. Seq X (priority=10, arrived=100) ← High priority first      │
│ 2. Seq Y (priority=0,  arrived=50)  ← Normal, early arrival    │
│ 3. Seq Z (priority=0,  arrived=80)  ← Normal, late arrival     │
│ 4. Seq W (priority=-5, arrived=10)  ← Low priority, even early │
└────────────────────────────────────────────────────────────────┘
```

## Scheduler Components

### SchedulerOutputs

Result of a scheduling step:

```rust
pub struct SchedulerOutputs {
    /// Sequences that need prefill
    pub prefill_sequences: Vec<SequenceId>,
    /// Sequences in decode phase
    pub decode_sequences: Vec<SequenceId>,
    /// Chunk sizes for chunked prefill
    pub prefill_chunk_sizes: HashMap<SequenceId, usize>,
    /// Preempted sequences
    pub preempted_sequences: Vec<SequenceId>,
    /// Total prefill tokens this iteration
    pub num_prefill_tokens: usize,
    /// Total decode tokens this iteration
    pub num_decode_tokens: usize,
}
```

### Scheduler State

```rust
pub struct Scheduler {
    config: SchedulerConfig,
    block_manager: BlockManager,
    sequences: HashMap<SequenceId, Sequence>,
    waiting_queue: BinaryHeap<PriorityEntry>,  // Priority queue
    running_ids: Vec<SequenceId>,              // Currently running
    swapped_ids: Vec<SequenceId>,              // Preempted to CPU
    arrival_counter: u64,                       // For FIFO ordering
}
```

## Scheduling Algorithm

### `schedule()` Method

```rust
pub fn schedule(&mut self) -> SchedulerOutputs {
    let mut outputs = SchedulerOutputs::new();

    // Step 1: Schedule decode for running sequences (1 token each)
    self.schedule_decode(&mut outputs);

    // Step 2: Allocate blocks for running sequences that need more
    self.allocate_running_blocks(&mut outputs);

    // Step 3: Admit new sequences from waiting queue (prefill)
    self.schedule_prefill(&mut outputs);

    // Step 4: Handle preemption if memory pressure
    if self.config.enable_preemption {
        self.handle_preemption(&mut outputs);
    }

    outputs
}
```

### Resource Constraints

The scheduler respects several limits:

```text
┌────────────────────────────────────────────────────────────────┐
│                     Resource Budget                            │
├────────────────────────────────────────────────────────────────┤
│ max_num_seqs: 4         │ Maximum concurrent sequences         │
│ max_prefill_tokens: 100 │ Maximum tokens to prefill/iteration  │
│ num_blocks: 1024        │ Total KV cache blocks available      │
└────────────────────────────────────────────────────────────────┘

Admission Decision:
┌────────────────────────────────────────────────────────────────┐
│ Can admit new sequence?                                        │
│                                                                │
│ ✓ num_sequences + pending < max_num_seqs                       │
│ ✓ num_prefill_tokens + new_tokens < max_prefill_tokens         │
│ ✓ block_manager.can_allocate(pending_blocks + blocks_needed)   │
│                                                                │
│ All conditions must be true to admit.                          │
└────────────────────────────────────────────────────────────────┘
```

### Preemption

When under memory pressure with high-priority requests waiting:

```text
Before Preemption:
┌────────────────────────────────────────────────────────────────┐
│ Running: [A(priority=-10), B(priority=0), C(priority=0)]       │
│ Waiting: [X(priority=10)]  ← High priority, but no memory!     │
│ Free Blocks: 0                                                 │
└────────────────────────────────────────────────────────────────┘

Preemption Decision:
┌────────────────────────────────────────────────────────────────┐
│ 1. Find lowest priority running: A (priority=-10)              │
│ 2. Free A's blocks → return to block pool                      │
│ 3. Move A to swapped state                                     │
│ 4. Now X can be admitted!                                      │
└────────────────────────────────────────────────────────────────┘

After Preemption:
┌────────────────────────────────────────────────────────────────┐
│ Running: [B(priority=0), C(priority=0), X(priority=10)]        │
│ Swapped: [A(priority=-10)]  ← Will resume when memory available│
│ Free Blocks: Available for X                                   │
└────────────────────────────────────────────────────────────────┘
```

## Configuration

```rust
pub struct SchedulerConfig {
    /// Maximum number of sequences to run concurrently
    pub max_num_seqs: usize,
    /// Maximum tokens to prefill per iteration
    pub max_prefill_tokens: usize,
    /// Enable chunked prefill for long sequences
    pub enable_chunked_prefill: bool,
    /// Chunk size for chunked prefill
    pub chunk_size: usize,
    /// Enable priority-based scheduling
    pub enable_priority: bool,
    /// Enable preemption for memory management
    pub enable_preemption: bool,
}
```

## Usage Example

```rust
use nano_vllm::scheduler::batch::{Scheduler, SchedulerOutputs};
use nano_vllm::core::sequence::Sequence;
use nano_vllm::SchedulerConfig;

// Create scheduler
let config = SchedulerConfig::default();
let mut scheduler = Scheduler::new(config, 16, 1024);

// Add sequences
let seq1 = Sequence::new(1, vec![1, 2, 3, 4]);
let seq2 = Sequence::new(2, vec![5, 6, 7, 8]);
scheduler.add_sequence(seq1);
scheduler.add_sequence(seq2);

// Main inference loop
while scheduler.has_unfinished_sequences() {
    let outputs = scheduler.schedule();

    // Process prefill sequences
    for seq_id in &outputs.prefill_sequences {
        // Run model forward pass for prefill
        scheduler.mark_prefilled(*seq_id, num_tokens)?;
    }

    // Process decode sequences
    for seq_id in &outputs.decode_sequences {
        // Run model forward pass for decode
        let token = sample_token();
        scheduler.append_token(*seq_id, token)?;

        if is_eos(token) {
            scheduler.finish_sequence(*seq_id, FinishReason::EndOfSequence);
        }
    }
}
```

## Comparison with vLLM

| Feature | vLLM | nano-vllm |
|---------|------|-----------|
| Continuous batching | Yes | Yes |
| Priority scheduling | Yes | Yes |
| Chunked prefill | Yes | Yes |
| Preemption | Yes (sophisticated) | Yes (basic) |
| Swapping to CPU | Yes | Partial (state only) |
| Beam search support | Yes | No |
| Multi-GPU | Yes | No |

## References

- [vLLM Paper](https://arxiv.org/abs/2309.06180) - Efficient Memory Management for Large Language Model Serving with PagedAttention
- [Continuous Batching from First Principles](https://huggingface.co/blog/continuous_batching) - HuggingFace's excellent deep dive
- [How continuous batching enables 23x throughput](https://www.anyscale.com/blog/continuous-batching-llm-inference) - AnyScale's analysis
