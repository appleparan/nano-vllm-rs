# Flash Attention: Memory-Efficient Attention Algorithm

This document provides an intuitive explanation of the Flash Attention algorithm.
We focus on **why** the algorithm works rather than diving into mathematical details.

## Table of Contents

1. [Why Flash Attention?](#why-flash-attention)
2. [The Problem with Standard Attention](#the-problem-with-standard-attention)
3. [Core Idea: IO-Aware Algorithm](#core-idea-io-aware-algorithm)
4. [Online Softmax: The Secret to Block-wise Computation](#online-softmax-the-secret-to-block-wise-computation)
5. [Flash Attention Algorithm](#flash-attention-algorithm)
6. [Flash Attention 2 Improvements](#flash-attention-2-improvements)
7. [Implementation Details](#implementation-details)
8. [References](#references)

---

## Why Flash Attention?

### The Problem: Memory Explosion with Long Sequences

Transformer attention uses $O(N^2)$ memory for sequence length $N$.

| Sequence Length | Attention Matrix Size | FP16 Memory |
| --------------- | --------------------- | ----------- |
| 1,024           | 1M elements           | 2 MB        |
| 4,096           | 16M elements          | 32 MB       |
| 16,384          | 268M elements         | 512 MB      |
| 65,536          | 4.3B elements         | 8 GB        |

Models like GPT-4 or Claude need to handle 100K+ tokens,
which would require **tens to hundreds of GB** just for the attention matrix.

### The Solution: Flash Attention

Flash Attention **reduces memory complexity to $O(N)$**.
Instead of storing the full attention matrix, it computes in blocks and discards intermediate results.

---

## The Problem with Standard Attention

### Standard Attention Formula

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

Where:

- $Q \in \mathbb{R}^{N \times d}$: Query matrix
- $K \in \mathbb{R}^{N \times d}$: Key matrix
- $V \in \mathbb{R}^{N \times d}$: Value matrix
- $d$: head dimension

### Step-by-Step Memory Usage

```text
Step 1: S = Q @ K^T        # [N, d] @ [d, N] = [N, N]  <- O(N^2) memory!
Step 2: P = softmax(S)     # [N, N]                    <- O(N^2) memory!
Step 3: O = P @ V          # [N, N] @ [N, d] = [N, d]
```

**Problem**: Step 1 requires storing the $N \times N$ matrix $S$ in HBM (GPU main memory).

### GPU Memory Hierarchy

```text
+-----------------------------------------------------------+
|                          GPU                              |
|  +-----------------------------------------------------+  |
|  |  SRAM (Shared Memory)                               |  |
|  |  - Size: ~100KB per SM                              |  |
|  |  - Bandwidth: ~19 TB/s                              |  |
|  |  - Register-level fast access                       |  |
|  +-----------------------------------------------------+  |
|                         | (fast)                          |
|  +-----------------------------------------------------+  |
|  |  HBM (High Bandwidth Memory)                        |  |
|  |  - Size: 24-80GB                                    |  |
|  |  - Bandwidth: ~2 TB/s (A100)                        |  |
|  |  - Main GPU memory                                  |  |
|  +-----------------------------------------------------+  |
+-----------------------------------------------------------+
```

**Key Insight**: SRAM is **~10x faster** than HBM, but **~1000x smaller**.

Standard attention stores intermediate results in HBM:

1. Write $S = QK^T$ to HBM
2. Read $S$ from HBM for softmax
3. Write result $P$ to HBM
4. Read $P$ from HBM for $P @ V$

**These HBM reads/writes are the actual bottleneck!**

---

## Core Idea: IO-Aware Algorithm

### The Paradigm Shift

> "Don't reduce computation — **reduce memory access**"

Flash Attention's core principles:

1. **Tiling**: Split Q, K, V into small blocks
2. **SRAM Utilization**: Load blocks into SRAM for computation
3. **Online Softmax**: Compute softmax without seeing the full row

### The Tiling Concept

Instead of computing the entire attention at once, we split it into blocks:

```text
Standard Attention:
+--------+     +--------+
|        |     |        |
|   Q    |  @  |  K^T   |  =  S (N x N, stored in HBM)
| (N x d)|     |(d x N) |
|        |     |        |
+--------+     +--------+

Flash Attention (Tiled):
+--+--+--+--+     +--+--+--+--+
|Q0|  |  |  |     |K0|K1|K2|K3|
+--+--+--+--+     +--+--+--+--+
|Q1|  |  |  |  @  |  |  |  |  |  ->  compute small blocks at a time
+--+--+--+--+     +--+--+--+--+
|Q2|  |  |  |     |  |  |  |  |
+--+--+--+--+     +--+--+--+--+
|Q3|  |  |  |     |  |  |  |  |
+--+--+--+--+     +--+--+--+--+
```

---

## Online Softmax: The Secret to Block-wise Computation

### The Challenge: Softmax Requires the Full Row?

Normally, softmax needs to see the entire row:

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}
$$

**Question**: If we process K in blocks, how do we compute the denominator sum?

### The Answer: Numerically Stable Online Softmax

For numerical stability, we typically compute:

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_{j=1}^{N} e^{x_j - m}}
$$

where $m = \max(x)$. This can be decomposed as:

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\ell}, \quad \text{where } \ell = \sum_{j=1}^{N} e^{x_j - m}
$$

### Key Insight: m and l Can Be Updated Incrementally

When a new block $x^{(2)}$ is added:

**Previous block** $(x^{(1)})$: $m^{(1)} = \max(x^{(1)})$, $\ell^{(1)} = \sum e^{x^{(1)}_j - m^{(1)}}$

**After adding new block**:
$$
m^{(new)} = \max(m^{(1)}, \max(x^{(2)}))
$$

$$
\ell^{(new)} = \ell^{(1)} \cdot e^{m^{(1)} - m^{(new)}} + \sum_j e^{x^{(2)}_j - m^{(new)}}
$$

### Intuitive Understanding

```text
Block 1 processed:   x = [3, 1, 4]
                     m = 4
                     l = e^(3-4) + e^(1-4) + e^(4-4)
                       = e^(-1) + e^(-3) + e^(0)
                       = 0.368 + 0.050 + 1.0 = 1.418

Block 2 added:       x_new = [1, 5, 2]
                     m_new = max(4, 5) = 5

                     # Adjust existing l to new m
                     l_adjusted = 1.418 * e^(4-5) = 1.418 * e^(-1) = 0.522

                     # Add new block contribution
                     l_new_block = e^(1-5) + e^(5-5) + e^(2-5)
                                 = 0.018 + 1.0 + 0.050 = 1.068

                     l_total = 0.522 + 1.068 = 1.590
```

**This is the key to Flash Attention!**
We can compute softmax incrementally without seeing the full sequence.

---

## Flash Attention Algorithm

### Algorithm Pseudocode

```python
def flash_attention(Q, K, V, block_size):
    """
    Q, K, V: [seq_len, head_dim]
    block_size: tokens to process at once
    """
    seq_len, head_dim = Q.shape
    scale = 1.0 / sqrt(head_dim)

    # Output storage: only O(N) memory
    O = zeros(seq_len, head_dim)  # output
    m = full(seq_len, -inf)        # per-row max
    l = zeros(seq_len)             # per-row softmax denominator

    # Split Q into blocks
    for i in range(0, seq_len, block_size):
        Q_block = Q[i:i+block_size]  # load to SRAM

        # For each Q block, iterate over all K, V blocks
        for j in range(0, seq_len, block_size):
            K_block = K[j:j+block_size]  # load to SRAM
            V_block = V[j:j+block_size]  # load to SRAM

            # 1. Compute block attention scores (in SRAM)
            S_block = Q_block @ K_block.T * scale

            # 2. Apply causal mask (if needed)
            if causal:
                S_block = apply_causal_mask(S_block, i, j)

            # 3. Online softmax update
            m_block = rowmax(S_block)
            m_new = max(m[i:i+block_size], m_block)

            # Rescale existing results
            scale_old = exp(m[i:i+block_size] - m_new)
            scale_new = exp(m_block - m_new)

            P_block = exp(S_block - m_new) * scale_new
            l_new = l[i:i+block_size] * scale_old + rowsum(P_block)

            # 4. Update output
            O[i:i+block_size] = (
                O[i:i+block_size] * (l[i:i+block_size] * scale_old / l_new) +
                P_block @ V_block / l_new
            )

            # 5. Update statistics
            m[i:i+block_size] = m_new
            l[i:i+block_size] = l_new

    return O
```

### Memory Analysis

| Item                  | Standard Attention | Flash Attention         |
| --------------------- | ------------------ | ----------------------- |
| Attention Matrix $S$  | $O(N^2)$           | $O(B^2)$ (block size)   |
| Softmax Result $P$    | $O(N^2)$           | $O(B^2)$ (block size)   |
| Output $O$            | $O(N \cdot d)$     | $O(N \cdot d)$          |
| Statistics $m, \ell$  | -                  | $O(N)$                  |
| **Total Memory**      | $O(N^2)$           | $O(N)$                  |

### Visual Understanding

```text
Standard Attention:                     Flash Attention:

  Q      K^T       S        P       O       Q_i    K_j^T   S_ij    O_i
+---+  +---+   +-----+  +-----+  +---+    +---+  +---+  +---+  +---+
|   |  |   |   |     |  |     |  |   |    |   |  |   |  |   |  |   |
|   |  |   |   |  S  |->|  P  |->| O |    |Q_i| @|K_j| =|S_ij|->|O_i|
|   |  |   |   |(NxN)|  |(NxN)|  |   |    |   |  |   |  |   |  |   |
|   |  |   |   |     |  |     |  |   |    +---+  +---+  +---+  +---+
+---+  +---+   +-----+  +-----+  +---+       |             |      ^
  |      |        |        |        |        |             |      |
 HBM    HBM      HBM      HBM      HBM       +--- SRAM ----+------+
                                                 (reused)
```

---

## Flash Attention 2 Improvements

### Flash Attention 1 Limitations

1. **Low GPU utilization**: Only 25-40% of theoretical max FLOPS
2. **Inefficient parallelization**: Parallelized only over heads
3. **Redundant operations**: Too many non-matmul operations

### Flash Attention 2 Improvements

#### 1. Parallelization over Query Blocks

**Flash Attention 1**: Outer loop over K/V blocks

```text
for each K_j, V_j block:        # Sequential
    for each Q_i block:          # Parallel
        compute attention
```

**Flash Attention 2**: Outer loop over Q blocks

```text
for each Q_i block:              # Parallel across thread blocks
    for each K_j, V_j block:     # Sequential within thread block
        compute attention
```

**Why is this better?**

- No dependencies between Q blocks -> fully parallelizable
- Each thread block works independently
- Increased GPU occupancy

#### 2. Minimizing Non-Matmul Operations

Flash Attention 1:

```text
P = exp(S - m)           # Element-wise exp
O = O * scale + P @ V    # Scaling
```

Flash Attention 2:

```text
# exp and scaling fused into kernel
# Fewer memory accesses
```

#### 3. Warp-Level Optimization

```text
Thread Block Structure:
+-------------------------------------------+
|  Warp 0    Warp 1    Warp 2    Warp 3    |
|  +-----+  +-----+  +-----+  +-----+      |
|  | t0  |  | t32 |  | t64 |  | t96 |      |
|  | ... |  | ... |  | ... |  | ... |      |
|  | t31 |  | t63 |  | t95 |  |t127 |      |
|  +-----+  +-----+  +-----+  +-----+      |
+-------------------------------------------+
```

- Flash Attention 1: Requires shared memory sync between warps
- Flash Attention 2: Each warp processes Q rows independently -> minimal sync

### Performance Comparison

| Metric                       | Flash Attention 1 | Flash Attention 2 |
| ---------------------------- | ----------------- | ----------------- |
| Theoretical FLOPS achieved   | 25-40%            | 50-73%            |
| A100 GPT training            | ~125 TFLOPs/s     | ~225 TFLOPs/s     |
| Speedup                      | Baseline          | ~2x               |

---

## Implementation Details

### Flash Attention in This Project

#### CPU Implementation (`src/attention/flash.rs`)

```rust
pub struct FlashAttentionConfig {
    /// Query block size
    pub block_size_q: usize,
    /// Key/Value block size
    pub block_size_kv: usize,
    /// Causal masking
    pub causal: bool,
    /// Softmax scale (1/sqrt(head_dim))
    pub softmax_scale: f32,
}

pub fn flash_attention_cpu(
    query: &Tensor,   // [batch, seq_len, num_heads, head_dim]
    key: &Tensor,
    value: &Tensor,
    config: &FlashAttentionConfig,
) -> Result<Tensor>
```

#### CUDA Kernel (`kernels/flash_attn_fwd.cu`)

```cuda
// Each thread block processes one Q block
// Q, K, V blocks loaded to shared memory
// Online softmax for incremental computation
__global__ void flash_attention_forward_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    float softmax_scale,
    bool causal
)
```

### Usage

```rust
// Enable Flash Attention in engine config
let engine_config = EngineConfig {
    use_flash_attention: true,
    ..Default::default()
};

// Or via CLI
// nano-vllm --model Qwen/Qwen3-0.6B --prompt "Hello" --flash-attention
```

---

## References

1. **Flash Attention** (Dao et al., 2022)
   - [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
   - Key contributions: IO-aware tiling, online softmax

2. **Flash Attention 2** (Dao, 2023)
   - [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
   - Key contributions: Q block parallelization, warp optimization

3. **Online Normalizer Calculation** (Milakov & Gimelshein, 2018)
   - Foundation for Flash Attention's online softmax technique

---

## Summary

| Concept              | Description                                                   |
| -------------------- | ------------------------------------------------------------- |
| **Problem**          | Standard attention uses $O(N^2)$ memory, HBM bottleneck       |
| **Solution**         | Tiling + Online Softmax achieves $O(N)$ memory                |
| **Key Insight**      | **Memory access**, not computation, is the actual bottleneck  |
| **Online Softmax**   | Incrementally update max and sum for block-wise computation   |
| **Flash Attention 2**| Q block parallelization + warp optimization = 2x speedup      |
