# nano-vllm-rs

[![Crates.io](https://img.shields.io/crates/v/nano-vllm-rs.svg)](https://crates.io/crates/nano-vllm-rs)
[![Docs.rs](https://docs.rs/nano-vllm-rs/badge.svg)](https://docs.rs/nano-vllm-rs)
[![CI](https://github.com/appleparan/nano-vllm-rs/workflows/CI/badge.svg)](https://github.com/appleparan/nano-vllm-rs/actions)

## Installation

### Cargo

* Install the rust toolchain in order to have cargo installed by following
  [this](https://www.rust-lang.org/tools/install) guide.
* run `cargo install nano-vllm-rs`

## Testing

```bash
# Run unit tests
cargo test

# Run integration tests with real Qwen3-0.6B model (requires ~1.2GB download)
cargo test --test qwen3_inference_test -- --ignored
```

> **Note**: Integration tests are marked with `#[ignore]` by default and won't run in CI.
> They require network access to HuggingFace Hub and take several minutes to complete.

## License

Licensed under the MIT license ([LICENSE](LICENSE) or http://opensource.org/licenses/MIT).
