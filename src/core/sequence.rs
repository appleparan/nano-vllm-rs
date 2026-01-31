//! Sequence tracking for inference requests.
//!
//! A sequence represents a single inference request, tracking its tokens,
//! state, and associated KV cache blocks.

use std::time::Instant;

use crate::core::block::BlockTable;
use crate::error::{Error, Result};

/// Unique identifier for a sequence.
pub type SequenceId = u64;

/// Status of a sequence in the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SequenceStatus {
    /// Waiting in queue to be scheduled.
    Waiting,
    /// Currently running (prefill or decode).
    Running,
    /// Swapped out to CPU memory (preempted).
    Swapped,
    /// Finished generation (EOS or max tokens).
    Finished,
}

impl SequenceStatus {
    /// Check if the sequence is active (waiting or running).
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Waiting | Self::Running)
    }

    /// Check if the sequence is finished.
    pub fn is_finished(&self) -> bool {
        matches!(self, Self::Finished)
    }

    /// Get the status name as a static string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Waiting => "Waiting",
            Self::Running => "Running",
            Self::Swapped => "Swapped",
            Self::Finished => "Finished",
        }
    }
}

/// Reason for sequence completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// End-of-sequence token generated.
    EndOfSequence,
    /// Maximum token limit reached.
    MaxTokens,
    /// Stop sequence encountered.
    StopSequence,
    /// Aborted by user or system.
    Aborted,
}

/// Priority level for scheduling.
///
/// Higher values mean higher priority.
pub type Priority = i32;

/// Default priority for new sequences.
pub const DEFAULT_PRIORITY: Priority = 0;

/// A sequence represents a single inference request.
///
/// It tracks:
/// - Input prompt tokens
/// - Generated output tokens
/// - KV cache block allocation (via BlockTable)
/// - Scheduling state and priority
///
/// # Example
///
/// ```
/// use nano_vllm::core::sequence::{Sequence, SequenceStatus};
///
/// let mut seq = Sequence::new(1, vec![1, 2, 3, 4]);
/// assert_eq!(seq.status(), SequenceStatus::Waiting);
/// assert_eq!(seq.prompt_len(), 4);
/// assert_eq!(seq.output_len(), 0);
///
/// seq.append_token(5);
/// assert_eq!(seq.output_len(), 1);
/// assert_eq!(seq.total_len(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct Sequence {
    /// Unique sequence identifier.
    seq_id: SequenceId,
    /// Prompt token IDs.
    prompt_token_ids: Vec<u32>,
    /// Generated output token IDs.
    output_token_ids: Vec<u32>,
    /// Block table for KV cache mapping.
    block_table: BlockTable,
    /// Number of tokens that have been prefilled.
    num_prefilled_tokens: usize,
    /// Current status.
    status: SequenceStatus,
    /// Scheduling priority (higher = more important).
    priority: Priority,
    /// Arrival time for FIFO ordering within same priority.
    arrival_time: Instant,
    /// Reason for finishing (if finished).
    finish_reason: Option<FinishReason>,
}

impl Sequence {
    /// Create a new sequence with the given prompt tokens.
    pub fn new(seq_id: SequenceId, prompt_token_ids: Vec<u32>) -> Self {
        Self {
            seq_id,
            prompt_token_ids,
            output_token_ids: Vec::new(),
            block_table: BlockTable::with_default_size(),
            num_prefilled_tokens: 0,
            status: SequenceStatus::Waiting,
            priority: DEFAULT_PRIORITY,
            arrival_time: Instant::now(),
            finish_reason: None,
        }
    }

    /// Create a new sequence with priority.
    pub fn with_priority(seq_id: SequenceId, prompt_token_ids: Vec<u32>, priority: Priority) -> Self {
        let mut seq = Self::new(seq_id, prompt_token_ids);
        seq.priority = priority;
        seq
    }

    // ========== Getters ==========

    /// Get the sequence ID.
    pub fn seq_id(&self) -> SequenceId {
        self.seq_id
    }

    /// Get the prompt token IDs.
    pub fn prompt_token_ids(&self) -> &[u32] {
        &self.prompt_token_ids
    }

    /// Get the output token IDs.
    pub fn output_token_ids(&self) -> &[u32] {
        &self.output_token_ids
    }

    /// Get all token IDs (prompt + output).
    pub fn all_token_ids(&self) -> Vec<u32> {
        let mut tokens = self.prompt_token_ids.clone();
        tokens.extend(&self.output_token_ids);
        tokens
    }

    /// Get the block table.
    pub fn block_table(&self) -> &BlockTable {
        &self.block_table
    }

    /// Get mutable access to the block table.
    pub fn block_table_mut(&mut self) -> &mut BlockTable {
        &mut self.block_table
    }

    /// Get the number of prefilled tokens.
    pub fn num_prefilled_tokens(&self) -> usize {
        self.num_prefilled_tokens
    }

    /// Get the current status.
    pub fn status(&self) -> SequenceStatus {
        self.status
    }

    /// Get the priority.
    pub fn priority(&self) -> Priority {
        self.priority
    }

    /// Get the arrival time.
    pub fn arrival_time(&self) -> Instant {
        self.arrival_time
    }

    /// Get the finish reason (if finished).
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.finish_reason
    }

    // ========== Length queries ==========

    /// Get the prompt length.
    pub fn prompt_len(&self) -> usize {
        self.prompt_token_ids.len()
    }

    /// Get the output length.
    pub fn output_len(&self) -> usize {
        self.output_token_ids.len()
    }

    /// Get the total length (prompt + output).
    pub fn total_len(&self) -> usize {
        self.prompt_len() + self.output_len()
    }

    /// Get the number of tokens remaining to prefill.
    pub fn num_tokens_to_prefill(&self) -> usize {
        self.prompt_len().saturating_sub(self.num_prefilled_tokens)
    }

    /// Check if prefill is complete.
    pub fn is_prefill_complete(&self) -> bool {
        self.num_prefilled_tokens >= self.prompt_len()
    }

    // ========== Token operations ==========

    /// Append a generated token.
    pub fn append_token(&mut self, token_id: u32) {
        self.output_token_ids.push(token_id);
    }

    /// Get the last token ID.
    pub fn last_token_id(&self) -> Option<u32> {
        self.output_token_ids.last().copied()
            .or_else(|| self.prompt_token_ids.last().copied())
    }

    /// Mark tokens as prefilled.
    pub fn mark_prefilled(&mut self, num_tokens: usize) {
        self.num_prefilled_tokens = (self.num_prefilled_tokens + num_tokens)
            .min(self.prompt_len());
    }

    // ========== State transitions ==========

    /// Transition to running state.
    ///
    /// # Errors
    ///
    /// Returns error if current state doesn't allow this transition.
    pub fn set_running(&mut self) -> Result<()> {
        match self.status {
            SequenceStatus::Waiting | SequenceStatus::Swapped => {
                self.status = SequenceStatus::Running;
                Ok(())
            }
            _ => Err(Error::InvalidStateTransition {
                from: self.status.as_str(),
                to: "Running",
            }),
        }
    }

    /// Transition to waiting state.
    ///
    /// # Errors
    ///
    /// Returns error if current state doesn't allow this transition.
    pub fn set_waiting(&mut self) -> Result<()> {
        match self.status {
            SequenceStatus::Running => {
                self.status = SequenceStatus::Waiting;
                Ok(())
            }
            _ => Err(Error::InvalidStateTransition {
                from: self.status.as_str(),
                to: "Waiting",
            }),
        }
    }

    /// Transition to swapped state (preempted).
    ///
    /// # Errors
    ///
    /// Returns error if current state doesn't allow this transition.
    pub fn set_swapped(&mut self) -> Result<()> {
        match self.status {
            SequenceStatus::Running => {
                self.status = SequenceStatus::Swapped;
                Ok(())
            }
            _ => Err(Error::InvalidStateTransition {
                from: self.status.as_str(),
                to: "Swapped",
            }),
        }
    }

    /// Mark the sequence as finished.
    pub fn set_finished(&mut self, reason: FinishReason) {
        self.status = SequenceStatus::Finished;
        self.finish_reason = Some(reason);
    }

    /// Set the priority.
    pub fn set_priority(&mut self, priority: Priority) {
        self.priority = priority;
    }
}

impl PartialEq for Sequence {
    fn eq(&self, other: &Self) -> bool {
        self.seq_id == other.seq_id
    }
}

impl Eq for Sequence {}

impl std::hash::Hash for Sequence {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.seq_id.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_creation() {
        let seq = Sequence::new(1, vec![10, 20, 30, 40]);

        assert_eq!(seq.seq_id(), 1);
        assert_eq!(seq.prompt_len(), 4);
        assert_eq!(seq.output_len(), 0);
        assert_eq!(seq.total_len(), 4);
        assert_eq!(seq.status(), SequenceStatus::Waiting);
        assert_eq!(seq.priority(), DEFAULT_PRIORITY);
    }

    #[test]
    fn test_sequence_with_priority() {
        let seq = Sequence::with_priority(2, vec![1, 2, 3], 10);

        assert_eq!(seq.seq_id(), 2);
        assert_eq!(seq.priority(), 10);
    }

    #[test]
    fn test_append_tokens() {
        let mut seq = Sequence::new(1, vec![1, 2, 3]);

        seq.append_token(100);
        seq.append_token(101);

        assert_eq!(seq.output_len(), 2);
        assert_eq!(seq.total_len(), 5);
        assert_eq!(seq.output_token_ids(), &[100, 101]);
        assert_eq!(seq.last_token_id(), Some(101));
    }

    #[test]
    fn test_prefill_tracking() {
        let mut seq = Sequence::new(1, vec![1, 2, 3, 4, 5, 6, 7, 8]);

        assert_eq!(seq.num_prefilled_tokens(), 0);
        assert_eq!(seq.num_tokens_to_prefill(), 8);
        assert!(!seq.is_prefill_complete());

        seq.mark_prefilled(4);
        assert_eq!(seq.num_prefilled_tokens(), 4);
        assert_eq!(seq.num_tokens_to_prefill(), 4);
        assert!(!seq.is_prefill_complete());

        seq.mark_prefilled(4);
        assert_eq!(seq.num_prefilled_tokens(), 8);
        assert_eq!(seq.num_tokens_to_prefill(), 0);
        assert!(seq.is_prefill_complete());
    }

    #[test]
    fn test_state_transitions() {
        let mut seq = Sequence::new(1, vec![1, 2, 3]);

        // Waiting -> Running
        assert!(seq.set_running().is_ok());
        assert_eq!(seq.status(), SequenceStatus::Running);

        // Running -> Swapped
        assert!(seq.set_swapped().is_ok());
        assert_eq!(seq.status(), SequenceStatus::Swapped);

        // Swapped -> Running
        assert!(seq.set_running().is_ok());
        assert_eq!(seq.status(), SequenceStatus::Running);

        // Running -> Finished
        seq.set_finished(FinishReason::EndOfSequence);
        assert_eq!(seq.status(), SequenceStatus::Finished);
        assert_eq!(seq.finish_reason(), Some(FinishReason::EndOfSequence));
    }

    #[test]
    fn test_invalid_state_transitions() {
        let mut seq = Sequence::new(1, vec![1, 2, 3]);

        // Waiting -> Swapped is invalid
        assert!(seq.set_swapped().is_err());

        // Waiting -> Waiting is invalid
        assert!(seq.set_waiting().is_err());
    }

    #[test]
    fn test_sequence_status_helpers() {
        assert!(SequenceStatus::Waiting.is_active());
        assert!(SequenceStatus::Running.is_active());
        assert!(!SequenceStatus::Swapped.is_active());
        assert!(!SequenceStatus::Finished.is_active());

        assert!(!SequenceStatus::Waiting.is_finished());
        assert!(SequenceStatus::Finished.is_finished());
    }

    #[test]
    fn test_all_token_ids() {
        let mut seq = Sequence::new(1, vec![1, 2, 3]);
        seq.append_token(10);
        seq.append_token(20);

        assert_eq!(seq.all_token_ids(), vec![1, 2, 3, 10, 20]);
    }
}
