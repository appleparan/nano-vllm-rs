//! Integration tests for Sequence.

use nano_vllm::core::sequence::{DEFAULT_PRIORITY, FinishReason, Sequence, SequenceStatus};

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
