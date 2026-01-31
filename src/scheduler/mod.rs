//! Batch scheduling for continuous batching.
//!
//! This module handles:
//! - Priority-based request scheduling
//! - Continuous batching (iteration-level scheduling)
//! - Preemption of low-priority sequences
//!
//! ## Continuous Batching
//!
//! Unlike static batching where all requests in a batch must complete before
//! new requests can start, continuous batching schedules at the iteration level:
//!
//! ```text
//! Static Batching:
//!   Iteration 1: [A, B, C] → must wait for all to finish
//!   Iteration 2: [D, E, F] → can only start after above
//!
//! Continuous Batching:
//!   Iteration 1: [A(prefill), B(prefill)]
//!   Iteration 2: [A(decode), B(decode), C(prefill)]
//!   Iteration 3: [A(finish), B(decode), C(decode), D(prefill)]
//!   Iteration 4: [B(decode), C(decode), D(decode)]
//! ```
//!
//! This maximizes GPU utilization by always keeping the batch full.

pub mod batch;

pub use batch::{Scheduler, SchedulerOutputs};
