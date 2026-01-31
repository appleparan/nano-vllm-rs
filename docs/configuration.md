# Configuration

## EngineConfig

Main engine configuration.

```rust
EngineConfig {
    max_num_seqs: 256,        // Max concurrent sequences
    max_prefill_tokens: 4096, // Max tokens to prefill per iteration
    block_size: 16,           // Tokens per KV cache block
    num_blocks: 1024,         // Total KV cache blocks
    use_paged_attention: true,
    enable_prefix_caching: true,
    enable_preemption: false,
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `max_num_seqs` | 256 | Maximum sequences to process concurrently |
| `max_prefill_tokens` | 4096 | Limits compute per iteration for prefill |
| `block_size` | 16 | Tokens stored per block (affects memory granularity) |
| `num_blocks` | 1024 | Total blocks available (determines max KV cache size) |
| `use_paged_attention` | true | Enable PagedAttention optimization |
| `enable_prefix_caching` | true | Share common prefixes across requests |
| `enable_preemption` | false | Allow preempting low-priority requests |

## SchedulerConfig

Batch scheduling behavior.

```rust
SchedulerConfig {
    max_num_seqs: 256,
    max_prefill_tokens: 4096,
    enable_chunked_prefill: true,
    chunk_size: 512,
    enable_priority: true,
    enable_preemption: false,
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `enable_chunked_prefill` | true | Break long prompts into chunks |
| `chunk_size` | 512 | Tokens per chunk for chunked prefill |
| `enable_priority` | true | Use priority-based scheduling |

## SamplingConfig

Token sampling parameters.

```rust
SamplingConfig {
    temperature: 1.0,  // 1.0 = no change
    top_k: 0,          // 0 = disabled
    top_p: 1.0,        // 1.0 = disabled
    max_tokens: 256,
    stop_sequences: vec![],
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `temperature` | 1.0 | Higher = more random, lower = more deterministic |
| `top_k` | 0 | Sample from top K tokens (0 = disabled) |
| `top_p` | 1.0 | Nucleus sampling threshold (1.0 = disabled) |
| `max_tokens` | 256 | Maximum tokens to generate |
| `stop_sequences` | [] | Stop generation on these strings |

## ModelConfig

Qwen3 model architecture.

```rust
ModelConfig {
    vocab_size: 151936,
    hidden_size: 1024,
    intermediate_size: 2816,
    num_hidden_layers: 28,
    num_attention_heads: 16,
    num_key_value_heads: 8,  // GQA
    rms_norm_eps: 1e-6,
    rope_theta: 1000000.0,
    max_position_embeddings: 40960,
}
```

Default values are for Qwen3-0.6B.

### Derived Values

```rust
head_dim = hidden_size / num_attention_heads  // 1024 / 16 = 64
num_kv_groups = num_attention_heads / num_key_value_heads  // 16 / 8 = 2
```
