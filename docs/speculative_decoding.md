# Speculative Decoding: Accelerating LLM Inference

This document provides an intuitive explanation of the Speculative Decoding algorithm.
We focus on **why** the algorithm works rather than diving into mathematical details.

## Table of Contents

1. [Why Speculative Decoding?](#why-speculative-decoding)
2. [The Problem with Autoregressive Generation](#the-problem-with-autoregressive-generation)
3. [Core Idea: Draft-Verify Paradigm](#core-idea-draft-verify-paradigm)
4. [Rejection Sampling: The Mathematical Core](#rejection-sampling-the-mathematical-core)
5. [Speculative Decoding Algorithm](#speculative-decoding-algorithm)
6. [Implementation Details](#implementation-details)
7. [Model Selection Guidelines](#model-selection-guidelines)
8. [References](#references)

---

## Why Speculative Decoding?

### The Problem: Sequential Generation is Slow

Large Language Models (LLMs) generate text one token at a time.
Each token requires a full forward pass through the model.

| Model Size | Forward Pass Time | Tokens/sec (Batch=1) |
| ---------- | ----------------- | -------------------- |
| 0.6B       | ~10ms             | ~100                 |
| 4B         | ~50ms             | ~20                  |
| 70B        | ~500ms            | ~2                   |

For a 100-token response:
- 0.6B model: ~1 second
- 4B model: ~5 seconds
- 70B model: ~50 seconds

### The Solution: Speculative Decoding

Speculative decoding **uses a smaller model to speculate** what the larger model would generate,
then **verifies all speculated tokens in parallel**.

Expected speedup: **2-3x** with matching draft model.

---

## The Problem with Autoregressive Generation

### Standard Generation Flow

```text
Target Model (4B parameters):

Step 1: "The" -> forward pass (50ms) -> "capital"
Step 2: "The capital" -> forward pass (50ms) -> "of"
Step 3: "The capital of" -> forward pass (50ms) -> "France"
Step 4: "The capital of France" -> forward pass (50ms) -> "is"
Step 5: "The capital of France is" -> forward pass (50ms) -> "Paris"

Total: 5 steps × 50ms = 250ms for 5 tokens
```

### Why So Slow?

1. **Memory Bandwidth Bound**: Model weights must be loaded from memory for each token
2. **No Parallelization**: Each token depends on the previous one
3. **Underutilized Compute**: GPU is mostly waiting for memory

### The Key Insight

> "Small models agree with large models on most tokens"

For simple tokens like "of", "is", "the", a 0.6B model often predicts
the same token as a 4B model. We can exploit this!

---

## Core Idea: Draft-Verify Paradigm

### Two-Phase Approach

```text
Phase 1: DRAFT (Fast, Small Model)
+------------------+     +------------------+
| Draft Model      |     | Tokens Generated |
| (0.6B, ~10ms)    | --> | t1, t2, t3, t4   |
+------------------+     +------------------+

Phase 2: VERIFY (Parallel, Large Model)
+------------------+     +------------------+
| Target Model     |     | Verified Tokens  |
| (4B, ~50ms)      | --> | t1, t2, ✗, -     |
+------------------+     +------------------+
                          ↑ t3 rejected, resample
```

### Why This Works

1. **Draft phase**: Generate K tokens with small model (K × 10ms = 40ms for K=4)
2. **Verify phase**: Check all K tokens with large model in ONE forward pass (50ms)
3. **Total time**: 90ms for potentially 4+ tokens (vs 200ms with standard decoding)

### Best Case vs Worst Case

```text
Best Case: All K tokens accepted
- Draft: 4 tokens × 10ms = 40ms
- Verify: 1 forward pass = 50ms
- Result: 4+ tokens in 90ms = ~44 tok/s
- Speedup: 2.2x over standard (20 tok/s)

Worst Case: First token rejected
- Draft: 4 tokens × 10ms = 40ms
- Verify: 1 forward pass = 50ms
- Result: 1 token in 90ms = ~11 tok/s
- Slowdown: 0.55x vs standard

Average Case: ~2-3 tokens accepted
- Result: 2-3 tokens in 90ms = ~25-35 tok/s
- Speedup: 1.3-1.7x
```

---

## Rejection Sampling: The Mathematical Core

### The Challenge

How do we verify draft tokens while preserving the **exact output distribution**
of the target model?

**Naive approach**: Just check if draft token matches argmax of target.
**Problem**: This changes the distribution (forces greedy decoding).

### The Solution: Rejection Sampling

For each draft token $t$ at position $i$:

1. Get probability from draft model: $P_{draft}(t)$
2. Get probability from target model: $P_{target}(t)$
3. Compute acceptance ratio: $\alpha = \min\left(1, \frac{P_{target}(t)}{P_{draft}(t)}\right)$
4. Sample $u \sim \text{Uniform}(0, 1)$
5. If $u < \alpha$: **accept** $t$, continue to next token
6. Else: **reject** $t$, resample from adjusted distribution, **stop**

### Intuitive Understanding

```text
Case 1: Target agrees (P_target ≥ P_draft)
  P_draft("Paris") = 0.8
  P_target("Paris") = 0.9
  α = min(1, 0.9/0.8) = 1.0
  -> Always accept! Target likes it even more.

Case 2: Target disagrees slightly (P_target < P_draft)
  P_draft("Rome") = 0.6
  P_target("Rome") = 0.3
  α = min(1, 0.3/0.6) = 0.5
  -> Accept 50% of the time. Sometimes draft got lucky.

Case 3: Target strongly disagrees (P_target << P_draft)
  P_draft("Tokyo") = 0.5
  P_target("Tokyo") = 0.01
  α = min(1, 0.01/0.5) = 0.02
  -> Accept only 2% of the time. Draft was wrong.
```

### The Adjusted Distribution

When we reject, we resample from:

$$
P_{adjusted}(t) = \frac{\max(0, P_{target}(t) - P_{draft}(t))}{Z}
$$

where $Z = \sum_t \max(0, P_{target}(t) - P_{draft}(t))$ is the normalization constant.

**Why this works**: The adjusted distribution compensates for tokens
that the draft model underestimated.

```text
Example vocabulary: [A, B, C]

Draft:   P_draft  = [0.6, 0.3, 0.1]
Target:  P_target = [0.4, 0.5, 0.1]

If we rejected token A (which draft over-predicted):
  P_adjusted(A) = max(0, 0.4-0.6) = 0.0 (already over-sampled)
  P_adjusted(B) = max(0, 0.5-0.3) = 0.2 (target wants more)
  P_adjusted(C) = max(0, 0.1-0.1) = 0.0 (already correct)

After normalizing: P_adjusted = [0.0, 1.0, 0.0]
-> Resample will pick B (what target really wanted)
```

### Mathematical Guarantee

> **Theorem**: Speculative decoding with rejection sampling produces tokens
> from the **exact same distribution** as standard target model sampling.

This is not an approximation - the output is mathematically identical.

---

## Speculative Decoding Algorithm

### Algorithm Pseudocode

```python
def speculative_decode(target_model, draft_model, prompt, max_tokens, K=4):
    """
    target_model: Large model (e.g., 4B params)
    draft_model: Small model (e.g., 0.6B params)
    K: Number of tokens to speculate per iteration
    """
    tokens = tokenize(prompt)

    while len(tokens) < max_tokens:
        # Phase 1: Draft K tokens
        draft_tokens = []
        draft_probs = []

        for i in range(K):
            logits = draft_model.forward(tokens + draft_tokens)
            probs = softmax(logits[-1])
            token = sample(probs)
            draft_tokens.append(token)
            draft_probs.append(probs)

        # Phase 2: Verify with target model (single forward pass!)
        extended = tokens + draft_tokens
        target_logits = target_model.forward(extended)

        # Get target probs for positions we need to verify
        # target_logits[-K-1:-1] corresponds to draft token positions
        # target_logits[-1] is for the next token if all accepted

        # Phase 3: Rejection sampling
        accepted = []
        for i in range(K):
            t = draft_tokens[i]
            p_draft = draft_probs[i][t]
            p_target = softmax(target_logits[-(K+1)+i])[t]

            alpha = min(1.0, p_target / p_draft)
            u = random.uniform(0, 1)

            if u < alpha:
                accepted.append(t)
            else:
                # Reject: resample from adjusted distribution
                adjusted = compute_adjusted(
                    target_probs=softmax(target_logits[-(K+1)+i]),
                    draft_probs=draft_probs[i]
                )
                final_token = sample(adjusted)
                accepted.append(final_token)
                break  # Stop accepting more tokens
        else:
            # All K tokens accepted! Sample one more from target
            final_token = sample(softmax(target_logits[-1]))
            accepted.append(final_token)

        tokens.extend(accepted)

    return tokens
```

### Visual Flow

```text
Iteration 1:
  Input: "The capital of France is"

  Draft (K=4):
  ┌─────────────────────────────────────────────────────────┐
  │  t1="Paris" (0.9)  t2="." (0.7)  t3="It" (0.5)  t4="is" │
  └─────────────────────────────────────────────────────────┘

  Verify (single forward pass):
  ┌─────────────────────────────────────────────────────────┐
  │  Target: [P(Paris)=0.95] [P(.)=0.6] [P(It)=0.2] [...]   │
  └─────────────────────────────────────────────────────────┘

  Rejection Sampling:
  ┌─────────────────────────────────────────────────────────┐
  │  t1: α=min(1,0.95/0.9)=1.0  u=0.3  → Accept ✓           │
  │  t2: α=min(1,0.6/0.7)=0.86  u=0.4  → Accept ✓           │
  │  t3: α=min(1,0.2/0.5)=0.4   u=0.7  → Reject ✗           │
  │      Resample from adjusted → "The"                      │
  └─────────────────────────────────────────────────────────┘

  Result: ["Paris", ".", "The"] (3 tokens from 1 target forward pass!)
```

### Complexity Analysis

| Operation          | Standard Decoding | Speculative (K=4) |
| ------------------ | ----------------- | ----------------- |
| Target forwards    | N                 | N / (1 + avg_accept) |
| Draft forwards     | 0                 | N × K / (1 + avg_accept) |
| Tokens per target  | 1                 | 1 + avg_accept    |

With avg_accept ≈ 3 for well-matched models:
- Standard: N target forwards for N tokens
- Speculative: N/4 target forwards + N draft forwards for N tokens

If draft is 5x faster than target: **~2.5x speedup**

---

## Implementation Details

### Speculative Decoding in This Project

#### Configuration (`src/speculative/config.rs`)

```rust
pub struct SpeculativeConfig {
    /// Number of tokens to speculate per iteration (K)
    pub num_speculative_tokens: usize,  // default: 4

    /// Draft model identifier
    pub draft_model_id: String,

    /// Draft model revision
    pub draft_revision: String,
}

impl SpeculativeConfig {
    pub fn new(draft_model_id: &str) -> Self;
    pub fn num_tokens(self, k: usize) -> Self;
}
```

#### Rejection Sampler (`src/speculative/sampler.rs`)

```rust
pub struct RejectionSampler {
    rng: StdRng,
}

impl RejectionSampler {
    /// Verify draft tokens against target distribution
    pub fn verify(
        &mut self,
        draft_tokens: &[u32],
        draft_logits: &Tensor,    // [K, vocab_size]
        target_logits: &Tensor,   // [K+1, vocab_size]
        temperature: f32,
    ) -> Result<(Vec<u32>, u32, usize)>;
    // Returns: (accepted_tokens, final_token, num_accepted)
}
```

#### Speculative Engine (`src/speculative/engine.rs`)

```rust
pub struct SpeculativeEngine {
    target_model: Qwen3ForCausalLM,
    draft_model: Qwen3ForCausalLM,
    config: SpeculativeConfig,
    rejection_sampler: RejectionSampler,
    device: Device,
    dtype: DType,
}

impl SpeculativeEngine {
    /// Generate K draft tokens
    fn draft(&mut self, input_ids: &Tensor, start_pos: usize)
        -> Result<(Vec<u32>, Tensor)>;

    /// Verify with target model (single forward pass)
    fn verify(&mut self, input_ids: &Tensor, draft_tokens: &[u32], start_pos: usize)
        -> Result<Tensor>;

    /// Complete speculative step
    pub fn speculative_step(
        &mut self,
        input_ids: &Tensor,
        start_pos: usize,
        temperature: f32,
    ) -> Result<Vec<u32>>;
}
```

### Usage

```rust
// Create engine with speculative decoding
let engine = LLMEngine::new_with_speculative(
    target_model,  // Qwen3-4B
    draft_model,   // Qwen3-0.6B
    model_config,
    tokenizer,
    engine_config,
    SpeculativeConfig::new("Qwen/Qwen3-0.6B").num_tokens(4),
)?;

// Or via CLI
// cargo run -- --model Qwen/Qwen3-4B --speculative --draft-model Qwen/Qwen3-0.6B --prompt "Hello"
```

### CLI Options

```bash
# Enable speculative decoding (default draft: Qwen3-0.6B)
cargo run -- --model Qwen/Qwen3-4B --speculative --prompt "Hello"

# Custom draft model
cargo run -- --model Qwen/Qwen3-4B \
              --speculative \
              --draft-model Qwen/Qwen3-0.6B \
              --prompt "Hello"

# Adjust speculation depth
cargo run -- --model Qwen/Qwen3-4B \
              --speculative \
              --num-speculative-tokens 6 \
              --prompt "Hello"
```

---

## Model Selection Guidelines

### Choosing Draft and Target Models

| Target Model | Recommended Draft | Size Ratio | Expected Speedup |
| ------------ | ----------------- | ---------- | ---------------- |
| Qwen3-4B     | Qwen3-0.6B        | 6.7x       | 2-3x             |
| Llama-70B    | Llama-7B          | 10x        | 2-4x             |
| GPT-4 (1.8T?)| GPT-3.5 (~175B?)  | ~10x       | 2-3x             |

### Requirements for Draft-Target Compatibility

1. **Same vocabulary**: Must share exact tokenizer
2. **Same architecture family**: Helps with distribution matching
3. **Size ratio 5-10x**: Too similar = low speedup, too different = low acceptance

### Acceptance Rate vs Speedup

```text
Acceptance Rate:  ████████████████░░░░  80%  → Speedup: 2.5-3x
Acceptance Rate:  ████████████░░░░░░░░  60%  → Speedup: 1.5-2x
Acceptance Rate:  ████████░░░░░░░░░░░░  40%  → Speedup: 1-1.3x
Acceptance Rate:  ████░░░░░░░░░░░░░░░░  20%  → Slowdown possible
```

### When Speculative Decoding Helps Most

1. **Simple, predictable text**: News articles, code, structured data
2. **Low temperature (near-greedy)**: Draft and target agree more often
3. **Memory-bound scenarios**: Single batch, long sequences

### When to Avoid Speculative Decoding

1. **High temperature/creative text**: Draft-target disagreement increases
2. **Large batch sizes**: Already parallelized, adding draft adds overhead
3. **Very short generations**: Setup overhead not amortized

---

## References

1. **Fast Inference from Transformers via Speculative Decoding** (Leviathan et al., 2022)
   - [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
   - Original speculative decoding paper with rejection sampling guarantee

2. **Accelerating Large Language Model Decoding with Speculative Sampling** (Chen et al., 2023)
   - [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
   - DeepMind's independent discovery with similar approach

3. **SpecInfer: Accelerating Generative Large Language Model Serving** (Miao et al., 2023)
   - [arXiv:2305.09781](https://arxiv.org/abs/2305.09781)
   - Tree-based speculation for higher acceptance rates

4. **EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty** (Li et al., 2024)
   - [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)
   - Feature-level draft head approach (requires training)

---

## Summary

| Concept              | Description                                                   |
| -------------------- | ------------------------------------------------------------- |
| **Problem**          | Autoregressive generation is sequential and slow              |
| **Solution**         | Draft-verify paradigm with smaller model speculation          |
| **Key Insight**      | Small models agree with large models on most tokens           |
| **Rejection Sampling**| Mathematically guarantees same distribution as target        |
| **Speedup**          | 2-3x with well-matched draft model (same vocabulary)          |
| **Best Use Case**    | Low temperature, predictable text, single batch               |
