//! Integration tests for Scheduler.

use nano_vllm::SchedulerConfig;
use nano_vllm::core::sequence::{FinishReason, Sequence, SequenceStatus};
use nano_vllm::scheduler::batch::{Scheduler, SchedulerOutputs};

fn test_config() -> SchedulerConfig {
    SchedulerConfig {
        max_num_seqs: 4,
        max_prefill_tokens: 100,
        enable_chunked_prefill: false,
        chunk_size: 512,
        enable_priority: true,
        enable_preemption: false,
    }
}

#[test]
fn test_scheduler_creation() {
    let config = test_config();
    let scheduler = Scheduler::new(config, 16, 100);

    assert_eq!(scheduler.num_waiting(), 0);
    assert_eq!(scheduler.num_running(), 0);
    assert_eq!(scheduler.num_swapped(), 0);
    assert!(!scheduler.has_unfinished_sequences());
}

#[test]
fn test_add_sequence() {
    let config = test_config();
    let mut scheduler = Scheduler::new(config, 16, 100);

    let seq = Sequence::new(1, vec![1, 2, 3, 4]);
    scheduler.add_sequence(seq);

    assert_eq!(scheduler.num_waiting(), 1);
    assert!(scheduler.has_unfinished_sequences());
}

#[test]
fn test_basic_scheduling() {
    let config = test_config();
    let mut scheduler = Scheduler::new(config, 16, 100);

    // Add sequence
    let seq = Sequence::new(1, vec![1, 2, 3, 4]);
    scheduler.add_sequence(seq);

    // Schedule
    let outputs = scheduler.schedule();

    assert_eq!(outputs.prefill_sequences.len(), 1);
    assert_eq!(outputs.prefill_sequences[0], 1);
    assert_eq!(outputs.decode_sequences.len(), 0);
    assert_eq!(scheduler.num_running(), 1);
    assert_eq!(scheduler.num_waiting(), 0);
}

#[test]
fn test_multiple_sequences() {
    let config = test_config();
    let mut scheduler = Scheduler::new(config, 16, 100);

    // Add multiple sequences
    for i in 1..=3 {
        let seq = Sequence::new(i, vec![1, 2, 3, 4]);
        scheduler.add_sequence(seq);
    }

    // Schedule - should admit all
    let outputs = scheduler.schedule();

    assert_eq!(outputs.prefill_sequences.len(), 3);
    assert_eq!(scheduler.num_running(), 3);
}

#[test]
fn test_max_num_seqs_limit() {
    let mut config = test_config();
    config.max_num_seqs = 2;
    let mut scheduler = Scheduler::new(config, 16, 100);

    // Add 3 sequences
    for i in 1..=3 {
        let seq = Sequence::new(i, vec![1, 2, 3, 4]);
        scheduler.add_sequence(seq);
    }

    // Schedule - should only admit 2
    let outputs = scheduler.schedule();

    assert_eq!(outputs.prefill_sequences.len(), 2);
    assert_eq!(scheduler.num_running(), 2);
    assert_eq!(scheduler.num_waiting(), 1);
}

#[test]
fn test_max_prefill_tokens_limit() {
    let mut config = test_config();
    config.max_prefill_tokens = 10;
    let mut scheduler = Scheduler::new(config, 16, 100);

    // Add sequences with 8 tokens each
    let seq1 = Sequence::new(1, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let seq2 = Sequence::new(2, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    scheduler.add_sequence(seq1);
    scheduler.add_sequence(seq2);

    // Schedule - should only admit 1 (8 < 10, but 8 + 8 > 10)
    let outputs = scheduler.schedule();

    assert_eq!(outputs.prefill_sequences.len(), 1);
    assert_eq!(outputs.num_prefill_tokens, 8);
    assert_eq!(scheduler.num_running(), 1);
    assert_eq!(scheduler.num_waiting(), 1);
}

#[test]
fn test_priority_scheduling() {
    let config = test_config();
    let mut scheduler = Scheduler::new(config, 16, 100);

    // Add sequences with different priorities
    let low_priority = Sequence::with_priority(1, vec![1, 2, 3, 4], -10);
    let high_priority = Sequence::with_priority(2, vec![5, 6, 7, 8], 10);
    let normal_priority = Sequence::new(3, vec![9, 10, 11, 12]); // priority = 0

    // Add in reverse priority order
    scheduler.add_sequence(low_priority);
    scheduler.add_sequence(normal_priority);
    scheduler.add_sequence(high_priority);

    // Schedule
    let outputs = scheduler.schedule();

    // Should be ordered by priority (high first)
    assert_eq!(outputs.prefill_sequences.len(), 3);
    assert_eq!(outputs.prefill_sequences[0], 2); // high priority
    assert_eq!(outputs.prefill_sequences[1], 3); // normal
    assert_eq!(outputs.prefill_sequences[2], 1); // low
}

#[test]
fn test_decode_scheduling() {
    let config = test_config();
    let mut scheduler = Scheduler::new(config, 16, 100);

    // Add and schedule sequence
    let seq = Sequence::new(1, vec![1, 2, 3, 4]);
    scheduler.add_sequence(seq);
    let _ = scheduler.schedule();

    // Mark as prefilled
    scheduler.mark_prefilled(1, 4).unwrap();

    // Schedule again - should be in decode
    let outputs = scheduler.schedule();

    assert_eq!(outputs.prefill_sequences.len(), 0);
    assert_eq!(outputs.decode_sequences.len(), 1);
    assert_eq!(outputs.decode_sequences[0], 1);
    assert_eq!(outputs.num_decode_tokens, 1);
}

#[test]
fn test_mixed_prefill_decode() {
    let config = test_config();
    let mut scheduler = Scheduler::new(config, 16, 100);

    // Add first sequence and run prefill
    let seq1 = Sequence::new(1, vec![1, 2, 3, 4]);
    scheduler.add_sequence(seq1);
    let _ = scheduler.schedule();
    scheduler.mark_prefilled(1, 4).unwrap();

    // Add second sequence
    let seq2 = Sequence::new(2, vec![5, 6, 7, 8]);
    scheduler.add_sequence(seq2);

    // Schedule - seq1 decode, seq2 prefill
    let outputs = scheduler.schedule();

    assert_eq!(outputs.decode_sequences.len(), 1);
    assert_eq!(outputs.decode_sequences[0], 1);
    assert_eq!(outputs.prefill_sequences.len(), 1);
    assert_eq!(outputs.prefill_sequences[0], 2);
}

#[test]
fn test_finish_sequence() {
    let config = test_config();
    let mut scheduler = Scheduler::new(config, 16, 100);

    // Add and run sequence
    let seq = Sequence::new(1, vec![1, 2, 3, 4]);
    scheduler.add_sequence(seq);
    let _ = scheduler.schedule();

    // Finish sequence
    scheduler.finish_sequence(1, FinishReason::EndOfSequence);

    assert_eq!(scheduler.num_running(), 0);

    // Verify sequence status
    let seq = scheduler.get_sequence(1).unwrap();
    assert_eq!(seq.status(), SequenceStatus::Finished);
    assert_eq!(seq.finish_reason(), Some(FinishReason::EndOfSequence));
}

#[test]
fn test_remove_sequence() {
    let config = test_config();
    let mut scheduler = Scheduler::new(config, 16, 100);

    // Add and run sequence
    let seq = Sequence::new(1, vec![1, 2, 3, 4]);
    scheduler.add_sequence(seq);
    let _ = scheduler.schedule();

    // Remove sequence
    let removed = scheduler.remove_sequence(1);

    assert!(removed.is_some());
    assert_eq!(scheduler.num_running(), 0);
    assert!(scheduler.get_sequence(1).is_none());
}

#[test]
fn test_append_token() {
    let config = test_config();
    let mut scheduler = Scheduler::new(config, 16, 100);

    let seq = Sequence::new(1, vec![1, 2, 3, 4]);
    scheduler.add_sequence(seq);
    let _ = scheduler.schedule();

    // Append token
    scheduler.append_token(1, 100).unwrap();

    let seq = scheduler.get_sequence(1).unwrap();
    assert_eq!(seq.output_len(), 1);
    assert_eq!(seq.output_token_ids(), &[100]);
}

#[test]
fn test_chunked_prefill() {
    let mut config = test_config();
    config.enable_chunked_prefill = true;
    config.chunk_size = 4;
    let mut scheduler = Scheduler::new(config, 16, 100);

    // Add sequence with 8 tokens
    let seq = Sequence::new(1, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    scheduler.add_sequence(seq);

    // First schedule - should prefill 4 tokens
    let outputs = scheduler.schedule();
    assert_eq!(outputs.prefill_sequences.len(), 1);
    assert_eq!(*outputs.prefill_chunk_sizes.get(&1).unwrap(), 4);

    // Mark first chunk as prefilled
    scheduler.mark_prefilled(1, 4).unwrap();

    // Schedule again - should prefill remaining 4 tokens
    // But since the sequence is already running, it will be in decode
    // Actually, since prefill is not complete, it should continue prefill
    let seq = scheduler.get_sequence(1).unwrap();
    assert!(!seq.is_prefill_complete());
}

#[test]
fn test_preemption() {
    let mut config = test_config();
    config.enable_preemption = true;
    config.max_num_seqs = 10;
    let mut scheduler = Scheduler::new(config, 16, 4); // Only 4 blocks!

    // Add low priority sequence that uses all blocks
    let low_priority = Sequence::with_priority(1, (0..64).map(|x| x as u32).collect(), -10);
    scheduler.add_sequence(low_priority);
    let _ = scheduler.schedule();

    // Now add high priority sequence
    let high_priority = Sequence::with_priority(2, vec![1, 2, 3, 4], 10);
    scheduler.add_sequence(high_priority);

    // Schedule - should preempt low priority
    let outputs = scheduler.schedule();

    assert!(outputs.preempted_sequences.contains(&1));
    assert_eq!(scheduler.num_swapped(), 1);
}

#[test]
fn test_scheduler_reset() {
    let config = test_config();
    let mut scheduler = Scheduler::new(config, 16, 100);

    // Add and run sequences
    for i in 1..=3 {
        let seq = Sequence::new(i, vec![1, 2, 3, 4]);
        scheduler.add_sequence(seq);
    }
    let _ = scheduler.schedule();

    // Reset
    scheduler.reset();

    assert_eq!(scheduler.num_waiting(), 0);
    assert_eq!(scheduler.num_running(), 0);
    assert_eq!(scheduler.num_swapped(), 0);
    assert!(!scheduler.has_unfinished_sequences());
}

#[test]
fn test_scheduler_outputs_helpers() {
    let mut outputs = SchedulerOutputs::new();
    assert!(outputs.is_empty());
    assert_eq!(outputs.num_sequences(), 0);

    outputs.prefill_sequences.push(1);
    outputs.decode_sequences.push(2);

    assert!(!outputs.is_empty());
    assert_eq!(outputs.num_sequences(), 2);

    let all_ids = outputs.all_sequence_ids();
    assert_eq!(all_ids.len(), 2);
    assert!(all_ids.contains(&1));
    assert!(all_ids.contains(&2));
}

#[test]
fn test_block_allocation() {
    let config = test_config();
    let mut scheduler = Scheduler::new(config, 16, 100);

    // Add sequence
    let seq = Sequence::new(1, vec![1, 2, 3, 4]); // 4 tokens = 1 block
    scheduler.add_sequence(seq);
    let _ = scheduler.schedule();

    // Check blocks allocated
    let seq = scheduler.get_sequence(1).unwrap();
    assert_eq!(seq.block_table().num_blocks(), 1);
    assert_eq!(scheduler.block_manager().num_used_blocks(), 1);
}

#[test]
fn test_out_of_blocks() {
    let config = test_config();
    let mut scheduler = Scheduler::new(config, 16, 2); // Only 2 blocks!

    // Add sequences that need 3 blocks total
    let seq1 = Sequence::new(1, vec![1, 2, 3, 4]); // 1 block
    let seq2 = Sequence::new(2, vec![5, 6, 7, 8]); // 1 block
    let seq3 = Sequence::new(3, vec![9, 10, 11, 12]); // 1 block - no room

    scheduler.add_sequence(seq1);
    scheduler.add_sequence(seq2);
    scheduler.add_sequence(seq3);

    // Schedule - should only admit 2
    let outputs = scheduler.schedule();

    assert_eq!(outputs.prefill_sequences.len(), 2);
    assert_eq!(scheduler.num_waiting(), 1);
}
