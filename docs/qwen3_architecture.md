# Qwen3 Architecture

This document explains the Qwen3 model architecture implemented in nano-vllm-rs.

## Overview

Qwen3 is a decoder-only transformer model from Alibaba, similar to LLaMA but with some key differences:

- **Grouped Query Attention (GQA)**: Reduces memory usage by sharing KV heads
- **Per-head Q/K Normalization**: RMSNorm applied to each attention head
- **SwiGLU Activation**: Gated linear unit with SiLU activation in MLP
- **RoPE**: Rotary Position Embeddings for position encoding

```text
┌─────────────────────────────────────────────────────────────────┐
│                      Qwen3 Model Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input Token IDs                                                │
│         │                                                        │
│         ▼                                                        │
│   ┌───────────────┐                                              │
│   │   Embedding   │  vocab_size → hidden_size                    │
│   └───────────────┘                                              │
│         │                                                        │
│         ▼                                                        │
│   ┌───────────────┐                                              │
│   │ DecoderLayer  │ ×N layers (28 for Qwen3-0.6B)               │
│   │   ┌───────┐   │                                              │
│   │   │RMSNorm│   │                                              │
│   │   └───┬───┘   │                                              │
│   │       ▼       │                                              │
│   │   ┌───────┐   │                                              │
│   │   │  GQA  │   │  Grouped Query Attention + RoPE              │
│   │   └───┬───┘   │                                              │
│   │       + ◄─────│── Residual                                   │
│   │       │       │                                              │
│   │   ┌───────┐   │                                              │
│   │   │RMSNorm│   │                                              │
│   │   └───┬───┘   │                                              │
│   │       ▼       │                                              │
│   │   ┌───────┐   │                                              │
│   │   │SwiGLU │   │  Gated MLP                                   │
│   │   └───┬───┘   │                                              │
│   │       + ◄─────│── Residual                                   │
│   └───────────────┘                                              │
│         │                                                        │
│         ▼                                                        │
│   ┌───────────────┐                                              │
│   │   RMSNorm     │  Final normalization                         │
│   └───────────────┘                                              │
│         │                                                        │
│         ▼                                                        │
│   ┌───────────────┐                                              │
│   │   LM Head     │  hidden_size → vocab_size                    │
│   └───────────────┘                                              │
│         │                                                        │
│         ▼                                                        │
│   Output Logits                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Model Configuration (Qwen3-0.6B)

```rust
ModelConfig {
    vocab_size: 151936,
    hidden_size: 1024,
    intermediate_size: 3072,
    num_hidden_layers: 28,
    num_attention_heads: 16,      // Q heads
    num_key_value_heads: 8,       // KV heads (GQA ratio = 2)
    head_dim: 128,
    rms_norm_eps: 1e-6,
    rope_theta: 1000000.0,
    max_position_embeddings: 40960,
}
```

## Components

### 1. RMSNorm (Root Mean Square Normalization)

Unlike LayerNorm which centers and scales, RMSNorm only scales by the root mean square.

```text
Formula: output = (x / rms(x)) * weight
where:  rms(x) = sqrt(mean(x²) + ε)
```

**Why RMSNorm over LayerNorm?**

- Computationally simpler (no mean calculation)
- Empirically works as well or better
- Used in LLaMA, Qwen, Mistral families

```text
Input: [batch, seq_len, hidden_size]
         │
         ▼
    ┌─────────┐
    │  x²     │  Square each element
    └────┬────┘
         ▼
    ┌─────────┐
    │  mean   │  Mean over hidden_size dimension
    └────┬────┘
         ▼
    ┌─────────┐
    │ √(+ε)   │  Square root with epsilon
    └────┬────┘
         ▼
    ┌─────────┐
    │  x/rms  │  Normalize
    └────┬────┘
         ▼
    ┌─────────┐
    │  ×weight│  Scale by learnable weight
    └────┬────┘
         ▼
Output: [batch, seq_len, hidden_size]
```

### 2. Rotary Position Embeddings (RoPE)

RoPE encodes position by rotating pairs of dimensions. The key insight is that the dot product of rotated vectors depends on their relative position.

```text
For position p and dimension pair (i, i+1):
  rotation angle θ = p / (10000^(2i/d))

  [x_i]     [cos(θ)  -sin(θ)] [x_i]
  [x_{i+1}] = [sin(θ)   cos(θ)] [x_{i+1}]
```

**Why RoPE?**

- No learned position embeddings needed
- Naturally handles relative positions
- Extrapolates to longer sequences
- Used by most modern LLMs

```text
Precomputation (done once):
┌──────────────────────────────────────────────────────────┐
│ For each position p ∈ [0, max_seq_len):                  │
│   For each dimension pair i ∈ [0, head_dim/2):           │
│     θ = p / (rope_theta^(2i/head_dim))                   │
│     cos_cache[p, 2i] = cos(θ)                            │
│     sin_cache[p, 2i+1] = sin(θ)                          │
└──────────────────────────────────────────────────────────┘

Application to Q and K:
┌──────────────────────────────────────────────────────────┐
│ q_rotated = q * cos + rotate_half(q) * sin               │
│ k_rotated = k * cos + rotate_half(k) * sin               │
│                                                          │
│ where rotate_half([a,b,c,d,...]) = [-b,a,-d,c,...]       │
└──────────────────────────────────────────────────────────┘
```

### 3. Grouped Query Attention (GQA)

GQA reduces memory by having multiple query heads share the same KV heads.

```text
Standard Multi-Head Attention (MHA):
  Q heads: 16    K heads: 16    V heads: 16
  Total KV cache per layer: 16 × 2 × seq_len × head_dim

Grouped Query Attention (GQA) in Qwen3:
  Q heads: 16    K heads: 8     V heads: 8
  Total KV cache per layer: 8 × 2 × seq_len × head_dim
  → 50% memory reduction!

Multi-Query Attention (MQA):
  Q heads: 16    K heads: 1     V heads: 1
  → Maximum compression but lower quality
```

**GQA Visualization:**

```text
Q Heads (16):  [Q0][Q1]  [Q2][Q3]  [Q4][Q5]  [Q6][Q7]  [Q8][Q9]  [Q10][Q11]  [Q12][Q13]  [Q14][Q15]
                 │  │      │  │      │  │      │  │      │  │       │   │       │   │       │   │
                 └──┴──┐   └──┴──┐   └──┴──┐   └──┴──┐   └──┴──┐    └───┴──┐    └───┴──┐    └───┴──┐
                       │         │         │         │         │           │           │           │
KV Heads (8):        [KV0]     [KV1]     [KV2]     [KV3]     [KV4]       [KV5]       [KV6]       [KV7]

Each KV head serves 2 query heads (num_heads / num_kv_heads = 2)
```

### 4. Qwen3 Attention (Complete Flow)

```text
Input: hidden_states [batch, seq_len, hidden_size]
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
┌───────┐   ┌───────┐   ┌───────┐
│Q Proj │   │K Proj │   │V Proj │
└───┬───┘   └───┬───┘   └───┬───┘
    │           │           │
    ▼           ▼           ▼
[b,s,16,128] [b,s,8,128] [b,s,8,128]
    │           │           │
    ▼           ▼           │
┌───────┐   ┌───────┐       │
│Q Norm │   │K Norm │       │   ← Per-head RMSNorm (Qwen3 specific!)
└───┬───┘   └───┬───┘       │
    │           │           │
    ▼           ▼           │
┌───────┐   ┌───────┐       │
│ RoPE  │   │ RoPE  │       │   ← Apply rotary embeddings
└───┬───┘   └───┬───┘       │
    │           │           │
    │       ┌───┴───┐   ┌───┴───┐
    │       │Expand │   │Expand │   ← Repeat KV for GQA
    │       │ ×2    │   │ ×2    │
    │       └───┬───┘   └───┬───┘
    │           │           │
    ▼           ▼           ▼
[b,16,s,128] [b,16,s,128] [b,16,s,128]
    │           │           │
    └─────┬─────┘           │
          ▼                 │
    ┌───────────┐           │
    │ Q @ K^T   │           │
    │ / √d      │           │
    └─────┬─────┘           │
          ▼                 │
    ┌───────────┐           │
    │  + Mask   │   ← Causal mask for autoregressive
    └─────┬─────┘           │
          ▼                 │
    ┌───────────┐           │
    │  Softmax  │           │
    └─────┬─────┘           │
          │                 │
          └────────┬────────┘
                   ▼
             ┌───────────┐
             │ Attn @ V  │
             └─────┬─────┘
                   ▼
             [b,16,s,128]
                   │
                   ▼
             ┌───────────┐
             │  O Proj   │
             └─────┬─────┘
                   ▼
Output: [batch, seq_len, hidden_size]
```

### 5. SwiGLU MLP

SwiGLU (Swish-Gated Linear Unit) uses a gating mechanism for better gradient flow.

```text
Standard FFN:
  output = W2 × GELU(W1 × x)

SwiGLU (Qwen3):
  output = W_down × (SiLU(W_gate × x) ⊙ (W_up × x))

where:
  SiLU(x) = x × sigmoid(x)  (also called Swish)
  ⊙ = element-wise multiplication
```

**Why SwiGLU?**

- Better training dynamics
- Improved performance on various benchmarks
- Worth the extra parameters (3 projections vs 2)

```text
Input: [batch, seq_len, hidden_size]
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌─────────┐
│gate_proj│ │ up_proj │
└────┬────┘ └────┬────┘
     │           │
     ▼           │
┌─────────┐      │
│  SiLU   │      │
└────┬────┘      │
     │           │
     └─────┬─────┘
           ▼
         ⊙ (element-wise multiply)
           │
           ▼
     ┌─────────┐
     │down_proj│
     └────┬────┘
           ▼
Output: [batch, seq_len, hidden_size]
```

### 6. Decoder Layer (Pre-Norm Residual)

Qwen3 uses pre-norm architecture where normalization happens before each sub-layer.

```text
Post-Norm (original Transformer):     Pre-Norm (Qwen3, LLaMA, etc.):
  x → Attention → +x → Norm             x → Norm → Attention → +x
  x → FFN → +x → Norm                   x → Norm → FFN → +x

Pre-Norm benefits:
- More stable training
- Better gradient flow
- Enables training deeper models
```

```text
Input: hidden_states
         │
         ├───────────────────┐ (residual)
         │                   │
         ▼                   │
   ┌───────────┐             │
   │ RMSNorm   │             │
   └─────┬─────┘             │
         ▼                   │
   ┌───────────┐             │
   │ Attention │             │
   └─────┬─────┘             │
         ▼                   │
         + ◄─────────────────┘
         │
         ├───────────────────┐ (residual)
         │                   │
         ▼                   │
   ┌───────────┐             │
   │ RMSNorm   │             │
   └─────┬─────┘             │
         ▼                   │
   ┌───────────┐             │
   │  SwiGLU   │             │
   └─────┬─────┘             │
         ▼                   │
         + ◄─────────────────┘
         │
         ▼
Output: hidden_states
```

## KV Cache for Incremental Decoding

During text generation, we cache K and V to avoid recomputation.

```text
Prefill Phase (process prompt):
┌────────────────────────────────────────────────────────────┐
│ Input: "The quick brown fox"                               │
│                                                            │
│ Compute Q, K, V for all tokens                             │
│ Store K, V in cache                                        │
│ Output: next token prediction                              │
└────────────────────────────────────────────────────────────┘

Decode Phase (generate each token):
┌────────────────────────────────────────────────────────────┐
│ Input: single new token                                    │
│                                                            │
│ Compute Q, K, V for new token only                         │
│ K = concat(cached_K, new_K)                                │
│ V = concat(cached_V, new_V)                                │
│ Update cache                                               │
│ Attend to all K, V                                         │
│ Output: next token prediction                              │
└────────────────────────────────────────────────────────────┘
```

## Memory Layout

```text
Per Layer KV Cache:
┌─────────────────────────────────────────────────────────────┐
│ K cache: [batch, seq_len, num_kv_heads, head_dim]           │
│          [1,     4096,    8,            128]                │
│          = 4096 × 8 × 128 × 4 bytes = 16 MB per layer       │
│                                                             │
│ V cache: same shape = 16 MB per layer                       │
│                                                             │
│ Total per layer: 32 MB                                      │
│ Total 28 layers: 896 MB (F32) or 448 MB (F16)               │
└─────────────────────────────────────────────────────────────┘
```

## Comparison with Other Models

| Feature | Qwen3 | LLaMA 2 | Mistral |
| ------- | ----- | ------- | ------- |
| Attention | GQA | GQA | GQA + Sliding Window |
| Q/K Norm | Per-head RMSNorm | None | None |
| MLP | SwiGLU | SwiGLU | SwiGLU |
| Position | RoPE | RoPE | RoPE |
| Normalization | Pre-norm | Pre-norm | Pre-norm |

## Implementation Files

| Component | File | Description |
| --------- | ---- | ----------- |
| RmsNorm | `src/model/norm.rs` | RMS Layer Normalization |
| RotaryEmbedding | `src/model/rope.rs` | Rotary Position Embeddings |
| Qwen3Mlp | `src/model/mlp.rs` | SwiGLU Feed-Forward |
| Qwen3Attention | `src/model/attention.rs` | GQA with per-head norm |
| Qwen3DecoderLayer | `src/model/decoder.rs` | Full decoder layer |

## References

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [RoFormer: Enhanced Transformer with RoPE](https://arxiv.org/abs/2104.09864)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
