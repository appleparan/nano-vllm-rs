# Scheduler: Continuous Batching

This document explains the Scheduler implementation in nano-vllm, which provides continuous batching for efficient LLM inference.

## Overview

The scheduler manages the lifecycle of inference requests, deciding which sequences to process in each iteration based on available resources and priorities.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Scheduler Flow                               │
└─────────────────────────────────────────────────────────────────────┘

  add_sequence()                                  schedule()
       │                                              │
       ▼                                              ▼
  ┌─────────┐                                   ┌─────────────┐
  │ Waiting │ ──────────────────────────────────► │  Running   │
  │  Queue  │     (when resources available)      │    Set     │
  └─────────┘                                     └─────────────┘
       ▲                                              │
       │              preempt()                       │
       └──────────────────────────────────────────────┘
                  (when memory pressure)
```

## Key Concepts

### Continuous Batching vs Static Batching

**Static Batching** (traditional approach):
- Wait until batch is full
- Process entire batch together
- Wait for all sequences to complete before starting new batch
- Inefficient: short sequences wait for long sequences

**Continuous Batching** (vLLM approach):
- New sequences can join at any iteration
- Completed sequences immediately release resources
- Different sequences can be at different stages (prefill vs decode)
- Much higher throughput and lower latency

### Prefill vs Decode Phases

**Prefill Phase**:
- Process all prompt tokens in parallel
- Compute-bound (matrix multiplications)
- Higher memory bandwidth requirement
- Can be chunked for very long prompts

**Decode Phase**:
- Generate one token at a time (autoregressive)
- Memory-bound (KV cache access)
- Lower compute per token
- Always 1 token per sequence per iteration

### Priority Scheduling

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

    // Step 1: Schedule decode for running sequences
    self.schedule_decode(&mut outputs);

    // Step 2: Allocate blocks for running sequences
    self.allocate_running_blocks(&mut outputs);

    // Step 3: Admit new sequences from waiting queue
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

1. **max_num_seqs** - Maximum concurrent sequences
2. **max_prefill_tokens** - Maximum tokens to prefill per iteration
3. **Block availability** - KV cache memory limit

```rust
// Admission logic
while outputs.num_sequences() + to_admit.len() < max_seqs
    && outputs.num_prefill_tokens < max_prefill_tokens
{
    // Check block availability
    if !self.block_manager.can_allocate(pending_blocks + blocks_needed) {
        break;
    }
    // ... admit sequence
}
```

### Preemption

When under memory pressure with high-priority requests waiting:

1. Find lowest priority running sequence
2. Free its KV cache blocks
3. Move to swapped state
4. Later: restore when blocks available

```rust
fn handle_preemption(&mut self, outputs: &mut SchedulerOutputs) {
    while !self.waiting_queue.is_empty()
        && self.block_manager.num_free_blocks() == 0
    {
        let idx = self.find_lowest_priority_running()?;
        let seq_id = self.running_ids.remove(idx);

        // Free blocks and mark as swapped
        self.block_manager.free_many(&block_ids);
        seq.set_swapped();
        self.swapped_ids.push(seq_id);
    }
}
```

## Chunked Prefill

For very long prompts, prefill can be split into chunks:

```rust
// SchedulerConfig
enable_chunked_prefill: bool,
chunk_size: usize,  // e.g., 512 tokens

// In schedule_prefill()
let tokens_to_prefill = if self.config.enable_chunked_prefill {
    seq.num_tokens_to_prefill().min(self.config.chunk_size)
} else {
    seq.num_tokens_to_prefill()
};
```

Benefits:
- Prevents long prompts from blocking other requests
- Better interleaving of prefill and decode
- More consistent latency

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
- [Continuous Batching Blog](https://www.anyscale.com/blog/continuous-batching-llm-inference) - AnyScale's explanation of continuous batching
