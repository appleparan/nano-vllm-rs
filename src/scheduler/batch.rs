//! Continuous batching scheduler.
//!
//! The scheduler manages the lifecycle of inference requests, deciding which
//! sequences to process in each iteration based on available resources and priorities.
//!
//! ## Scheduling Flow
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                        Scheduler Flow                               │
//! └─────────────────────────────────────────────────────────────────────┘
//!
//!   add_sequence()                                  schedule()
//!        │                                              │
//!        ▼                                              ▼
//!   ┌─────────┐                                   ┌─────────────┐
//!   │ Waiting │ ──────────────────────────────────► │  Running   │
//!   │  Queue  │     (when resources available)      │    Set     │
//!   └─────────┘                                     └─────────────┘
//!        ▲                                              │
//!        │              preempt()                       │
//!        └──────────────────────────────────────────────┘
//!                   (when memory pressure)
//! ```
//!
//! ## Example
//!
//! ```
//! use nano_vllm::scheduler::batch::{Scheduler, SchedulerOutputs};
//! use nano_vllm::core::sequence::Sequence;
//! use nano_vllm::SchedulerConfig;
//!
//! let config = SchedulerConfig::default();
//! let mut scheduler = Scheduler::new(config, 16, 1024);
//!
//! // Add sequences
//! let seq1 = Sequence::new(1, vec![1, 2, 3, 4]);
//! let seq2 = Sequence::new(2, vec![5, 6, 7, 8]);
//! scheduler.add_sequence(seq1);
//! scheduler.add_sequence(seq2);
//!
//! // Schedule iteration
//! let outputs = scheduler.schedule();
//! assert!(!outputs.is_empty());
//! ```

use std::collections::{BinaryHeap, HashMap};

use crate::config::SchedulerConfig;
use crate::core::block::compute_num_blocks;
use crate::core::block_manager::BlockManager;
use crate::core::sequence::{Sequence, SequenceId, SequenceStatus};
use crate::error::{Error, Result};

/// Output of a scheduling step.
///
/// Contains the sequences to process in the current iteration,
/// separated by their processing type (prefill vs decode).
#[derive(Debug, Default)]
pub struct SchedulerOutputs {
    /// Sequences that need prefill (processing prompt tokens).
    pub prefill_sequences: Vec<SequenceId>,
    /// Sequences in decode phase (generating tokens).
    pub decode_sequences: Vec<SequenceId>,
    /// Number of tokens to prefill for chunked prefill sequences.
    /// Maps seq_id -> num_tokens_to_prefill_this_iteration
    pub prefill_chunk_sizes: HashMap<SequenceId, usize>,
    /// Sequences that were preempted this iteration.
    pub preempted_sequences: Vec<SequenceId>,
    /// Total number of prefill tokens this iteration.
    pub num_prefill_tokens: usize,
    /// Total number of decode tokens this iteration.
    pub num_decode_tokens: usize,
}

impl SchedulerOutputs {
    /// Create empty scheduler outputs.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if there are any sequences to process.
    pub fn is_empty(&self) -> bool {
        self.prefill_sequences.is_empty() && self.decode_sequences.is_empty()
    }

    /// Total number of sequences to process.
    pub fn num_sequences(&self) -> usize {
        self.prefill_sequences.len() + self.decode_sequences.len()
    }

    /// Get all sequence IDs to process.
    pub fn all_sequence_ids(&self) -> Vec<SequenceId> {
        let mut ids = self.prefill_sequences.clone();
        ids.extend(&self.decode_sequences);
        ids
    }
}

/// Entry in the priority queue for scheduling.
///
/// Ordered by (priority DESC, arrival_time ASC).
#[derive(Debug, Clone)]
struct PriorityEntry {
    seq_id: SequenceId,
    priority: i32,
    arrival_order: u64,
}

impl PartialEq for PriorityEntry {
    fn eq(&self, other: &Self) -> bool {
        self.seq_id == other.seq_id
    }
}

impl Eq for PriorityEntry {}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority first, then earlier arrival (lower order)
        match self.priority.cmp(&other.priority) {
            std::cmp::Ordering::Equal => other.arrival_order.cmp(&self.arrival_order),
            ord => ord,
        }
    }
}

/// Continuous batching scheduler.
///
/// Manages sequences through their lifecycle:
/// - Waiting: In queue, waiting for resources
/// - Running: Actively generating tokens
/// - Swapped: Preempted due to memory pressure
/// - Finished: Generation complete
pub struct Scheduler {
    /// Configuration.
    config: SchedulerConfig,
    /// Block manager for KV cache allocation.
    block_manager: BlockManager,
    /// All sequences (including waiting, running, swapped).
    sequences: HashMap<SequenceId, Sequence>,
    /// Priority queue for waiting sequences.
    waiting_queue: BinaryHeap<PriorityEntry>,
    /// Set of running sequence IDs.
    running_ids: Vec<SequenceId>,
    /// Set of swapped sequence IDs.
    swapped_ids: Vec<SequenceId>,
    /// Counter for arrival ordering.
    arrival_counter: u64,
}

impl Scheduler {
    /// Create a new scheduler.
    ///
    /// # Arguments
    ///
    /// * `config` - Scheduler configuration
    /// * `block_size` - Number of tokens per block
    /// * `num_blocks` - Total number of blocks available
    pub fn new(config: SchedulerConfig, block_size: usize, num_blocks: usize) -> Self {
        Self {
            config,
            block_manager: BlockManager::new(num_blocks, block_size),
            sequences: HashMap::new(),
            waiting_queue: BinaryHeap::new(),
            running_ids: Vec::new(),
            swapped_ids: Vec::new(),
            arrival_counter: 0,
        }
    }

    /// Add a new sequence to the scheduler.
    ///
    /// The sequence starts in `Waiting` status and is added to the priority queue.
    pub fn add_sequence(&mut self, seq: Sequence) {
        let seq_id = seq.seq_id();
        let priority = seq.priority();

        self.waiting_queue.push(PriorityEntry {
            seq_id,
            priority,
            arrival_order: self.arrival_counter,
        });
        self.arrival_counter += 1;

        self.sequences.insert(seq_id, seq);
    }

    /// Remove a finished sequence.
    ///
    /// Frees all blocks associated with the sequence.
    pub fn remove_sequence(&mut self, seq_id: SequenceId) -> Option<Sequence> {
        if let Some(seq) = self.sequences.remove(&seq_id) {
            // Free blocks
            let block_ids = seq.block_table().get_physical_block_ids().to_vec();
            self.block_manager.free_many(&block_ids);

            // Remove from running/swapped lists
            self.running_ids.retain(|&id| id != seq_id);
            self.swapped_ids.retain(|&id| id != seq_id);

            Some(seq)
        } else {
            None
        }
    }

    /// Get a reference to a sequence.
    pub fn get_sequence(&self, seq_id: SequenceId) -> Option<&Sequence> {
        self.sequences.get(&seq_id)
    }

    /// Get a mutable reference to a sequence.
    pub fn get_sequence_mut(&mut self, seq_id: SequenceId) -> Option<&mut Sequence> {
        self.sequences.get_mut(&seq_id)
    }

    /// Schedule the next iteration.
    ///
    /// Returns which sequences should be processed for prefill and decode.
    pub fn schedule(&mut self) -> SchedulerOutputs {
        let mut outputs = SchedulerOutputs::new();

        // Step 1: Schedule decode for running sequences
        self.schedule_decode(&mut outputs);

        // Step 2: Allocate blocks for running sequences that need more
        self.allocate_running_blocks(&mut outputs);

        // Step 3: Admit new sequences from waiting queue
        self.schedule_prefill(&mut outputs);

        // Step 4: If memory pressure, preempt low-priority sequences
        if self.config.enable_preemption {
            self.handle_preemption(&mut outputs);
        }

        outputs
    }

    /// Schedule decode tokens for running sequences.
    fn schedule_decode(&mut self, outputs: &mut SchedulerOutputs) {
        let max_seqs = self.config.max_num_seqs;

        for &seq_id in &self.running_ids {
            if outputs.num_sequences() >= max_seqs {
                break;
            }

            if let Some(seq) = self.sequences.get(&seq_id) {
                if seq.is_prefill_complete() {
                    outputs.decode_sequences.push(seq_id);
                    outputs.num_decode_tokens += 1; // Decode is always 1 token
                }
            }
        }
    }

    /// Allocate additional blocks for running sequences if needed.
    fn allocate_running_blocks(&mut self, _outputs: &mut SchedulerOutputs) {
        for &seq_id in &self.running_ids.clone() {
            if let Some(seq) = self.sequences.get_mut(&seq_id) {
                let total_tokens = seq.total_len();
                let blocks_needed =
                    compute_num_blocks(total_tokens, self.block_manager.block_size());
                let blocks_allocated = seq.block_table().num_blocks();

                if blocks_needed > blocks_allocated {
                    let new_blocks_needed = blocks_needed - blocks_allocated;
                    if self.block_manager.can_allocate(new_blocks_needed) {
                        if let Ok(block_ids) = self.block_manager.allocate_many(new_blocks_needed) {
                            for block_id in block_ids {
                                seq.block_table_mut().append_block(block_id);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Schedule prefill for waiting sequences.
    fn schedule_prefill(&mut self, outputs: &mut SchedulerOutputs) {
        let max_seqs = self.config.max_num_seqs;
        let max_prefill_tokens = self.config.max_prefill_tokens;

        // Collect sequences to admit
        let mut to_admit: Vec<(SequenceId, usize, usize)> = Vec::new();
        // Track pending block allocations (not yet committed)
        let mut pending_blocks = 0usize;

        while outputs.num_sequences() + to_admit.len() < max_seqs
            && outputs.num_prefill_tokens < max_prefill_tokens
        {
            // Pop from priority queue
            let entry = match self.waiting_queue.pop() {
                Some(e) => e,
                None => break,
            };

            let seq_id = entry.seq_id;

            // Check if sequence still exists and is waiting
            let seq = match self.sequences.get(&seq_id) {
                Some(s) if s.status() == SequenceStatus::Waiting => s,
                _ => continue,
            };

            // Calculate blocks needed for the prompt
            let tokens_to_prefill = if self.config.enable_chunked_prefill {
                seq.num_tokens_to_prefill().min(self.config.chunk_size)
            } else {
                seq.num_tokens_to_prefill()
            };

            // Would this exceed token budget?
            if outputs.num_prefill_tokens + tokens_to_prefill > max_prefill_tokens
                && !to_admit.is_empty()
            {
                // Put back and stop
                self.waiting_queue.push(entry);
                break;
            }

            // Calculate total blocks needed (for full prompt, not just chunk)
            let total_tokens = seq.prompt_len();
            let blocks_needed = compute_num_blocks(total_tokens, self.block_manager.block_size());

            // Check if we have enough blocks (including pending allocations)
            if !self
                .block_manager
                .can_allocate(pending_blocks + blocks_needed)
            {
                // Not enough blocks, put back
                self.waiting_queue.push(entry);
                break;
            }

            // Admit this sequence
            to_admit.push((seq_id, tokens_to_prefill, blocks_needed));
            outputs.num_prefill_tokens += tokens_to_prefill;
            pending_blocks += blocks_needed;
        }

        // Actually admit sequences
        for (seq_id, tokens_to_prefill, blocks_needed) in to_admit {
            // Allocate blocks
            if let Ok(block_ids) = self.block_manager.allocate_many(blocks_needed) {
                if let Some(seq) = self.sequences.get_mut(&seq_id) {
                    // Assign blocks
                    for block_id in block_ids {
                        seq.block_table_mut().append_block(block_id);
                    }

                    // Transition to running
                    let _ = seq.set_running();

                    // Track in outputs
                    outputs.prefill_sequences.push(seq_id);
                    outputs
                        .prefill_chunk_sizes
                        .insert(seq_id, tokens_to_prefill);

                    // Add to running set
                    self.running_ids.push(seq_id);
                }
            }
        }
    }

    /// Handle preemption when under memory pressure.
    fn handle_preemption(&mut self, outputs: &mut SchedulerOutputs) {
        // Simple strategy: if waiting queue is not empty and we have no free blocks,
        // preempt the lowest priority running sequence.

        while !self.waiting_queue.is_empty() && self.block_manager.num_free_blocks() == 0 {
            // Find lowest priority running sequence
            let lowest_priority_idx = self.find_lowest_priority_running();

            if let Some(idx) = lowest_priority_idx {
                let seq_id = self.running_ids.remove(idx);

                if let Some(seq) = self.sequences.get_mut(&seq_id) {
                    // Free blocks
                    let block_ids = seq.block_table().get_physical_block_ids().to_vec();
                    self.block_manager.free_many(&block_ids);
                    seq.block_table_mut().clear();

                    // Transition to swapped
                    let _ = seq.set_swapped();
                    self.swapped_ids.push(seq_id);

                    outputs.preempted_sequences.push(seq_id);

                    // Remove from decode outputs if it was there
                    outputs.decode_sequences.retain(|&id| id != seq_id);
                }
            } else {
                break;
            }
        }
    }

    /// Find the index of the lowest priority running sequence.
    fn find_lowest_priority_running(&self) -> Option<usize> {
        let mut lowest_priority = i32::MAX;
        let mut lowest_idx = None;

        for (idx, &seq_id) in self.running_ids.iter().enumerate() {
            if let Some(seq) = self.sequences.get(&seq_id) {
                if seq.priority() < lowest_priority {
                    lowest_priority = seq.priority();
                    lowest_idx = Some(idx);
                }
            }
        }

        lowest_idx
    }

    /// Mark a sequence as finished.
    pub fn finish_sequence(
        &mut self,
        seq_id: SequenceId,
        reason: crate::core::sequence::FinishReason,
    ) {
        if let Some(seq) = self.sequences.get_mut(&seq_id) {
            seq.set_finished(reason);

            // Remove from running
            self.running_ids.retain(|&id| id != seq_id);
        }
    }

    /// Append a token to a sequence.
    pub fn append_token(&mut self, seq_id: SequenceId, token_id: u32) -> Result<()> {
        let seq = self
            .sequences
            .get_mut(&seq_id)
            .ok_or_else(|| Error::Config(format!("Sequence {seq_id} not found")))?;
        seq.append_token(token_id);
        Ok(())
    }

    /// Mark tokens as prefilled for a sequence.
    pub fn mark_prefilled(&mut self, seq_id: SequenceId, num_tokens: usize) -> Result<()> {
        let seq = self
            .sequences
            .get_mut(&seq_id)
            .ok_or_else(|| Error::Config(format!("Sequence {seq_id} not found")))?;
        seq.mark_prefilled(num_tokens);
        Ok(())
    }

    /// Get number of waiting sequences.
    pub fn num_waiting(&self) -> usize {
        self.waiting_queue.len()
    }

    /// Get number of running sequences.
    pub fn num_running(&self) -> usize {
        self.running_ids.len()
    }

    /// Get number of swapped sequences.
    pub fn num_swapped(&self) -> usize {
        self.swapped_ids.len()
    }

    /// Get all running sequence IDs.
    pub fn running_sequence_ids(&self) -> &[SequenceId] {
        &self.running_ids
    }

    /// Check if scheduler has any active sequences.
    pub fn has_unfinished_sequences(&self) -> bool {
        !self.waiting_queue.is_empty()
            || !self.running_ids.is_empty()
            || !self.swapped_ids.is_empty()
    }

    /// Check if there are pending requests (waiting or running).
    pub fn has_pending_requests(&self) -> bool {
        !self.waiting_queue.is_empty() || !self.running_ids.is_empty()
    }

    /// Get block manager reference.
    pub fn block_manager(&self) -> &BlockManager {
        &self.block_manager
    }

    /// Get mutable block manager reference.
    pub fn block_manager_mut(&mut self) -> &mut BlockManager {
        &mut self.block_manager
    }

    /// Reset the scheduler.
    pub fn reset(&mut self) {
        self.sequences.clear();
        self.waiting_queue.clear();
        self.running_ids.clear();
        self.swapped_ids.clear();
        self.block_manager.reset();
        self.arrival_counter = 0;
    }
}
