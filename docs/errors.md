# Error Handling

## Error Types

All errors in nano-vllm are consolidated into the `Error` enum.

```rust
use nano_vllm::{Error, Result};

fn example() -> Result<()> {
    // Your code here
    Ok(())
}
```

## Error Variants

### `OutOfBlocks`

KV cache block allocation failed.

```rust
Error::OutOfBlocks
// "out of KV cache blocks"
```

**Cause**: All blocks are in use and no more sequences can be admitted.

**Solution**:

- Increase `num_blocks` in `EngineConfig`
- Enable preemption to free blocks from low-priority sequences
- Wait for running sequences to complete

### `SequenceNotFound`

Referenced sequence doesn't exist.

```rust
Error::SequenceNotFound(seq_id)
// "sequence {seq_id} not found"
```

**Cause**: Attempting to operate on a sequence that was already completed or never existed.

### `InvalidStateTransition`

Invalid sequence state change.

```rust
Error::InvalidStateTransition { from: "Running", to: "Waiting" }
// "invalid state transition: Running -> Waiting"
```

**Cause**: Attempting an invalid state transition (e.g., moving a running sequence back to waiting without preemption).

### `ModelLoad`

Model loading failed.

```rust
Error::ModelLoad("file not found".to_string())
// "failed to load model: file not found"
```

**Cause**: HuggingFace model download or safetensors loading failed.

### `Tokenization`

Tokenization error.

```rust
Error::Tokenization("unknown token".to_string())
// "tokenization error: unknown token"
```

**Cause**: Failed to tokenize input or decode output tokens.

### `Tensor`

Tensor operation error (from candle).

```rust
Error::Tensor(candle_error)
// "tensor error: {candle_error}"
```

**Cause**: Shape mismatch, device error, or other tensor operation failure.

### `Config`

Configuration error.

```rust
Error::Config("block_size must be positive".to_string())
// "configuration error: block_size must be positive"
```

**Cause**: Invalid configuration values.

### `Io`

I/O error.

```rust
Error::Io(io_error)
// "io error: {io_error}"
```

**Cause**: File system operations failed.

### `Json`

JSON parsing error.

```rust
Error::Json(serde_error)
// "json error: {serde_error}"
```

**Cause**: Failed to parse config.json or other JSON files.

## Error Propagation

Use `?` operator with `Result<T>`:

```rust
use nano_vllm::Result;

fn load_and_process() -> Result<()> {
    let config = load_config()?;  // Propagates any error
    let model = load_model(&config)?;
    Ok(())
}
```

For CLI/binary code, use `anyhow` for additional context:

```rust
use anyhow::{Context, Result};

fn main() -> Result<()> {
    let config = load_config()
        .context("failed to load configuration")?;
    Ok(())
}
```
