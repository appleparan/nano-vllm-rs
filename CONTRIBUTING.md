# Contribution guidelines

First off, thank you for considering contributing to nano-vllm-rs.

If your contribution is not straightforward, please first discuss the change you
wish to make by creating a new issue before making the change.

## Reporting issues

Before reporting an issue on the
[issue tracker](https://github.com/appleparan/nano-vllm-rs/issues),
please check that it has not already been reported by searching for some related
keywords.

## Pull requests

Try to do one pull request per change.

### Updating the changelog

Update the changes you have made in
[CHANGELOG](https://github.com/appleparan/nano-vllm-rs/blob/main/CHANGELOG.md)
file under the **Unreleased** section.

Add the changes of your pull request to one of the following subsections,
depending on the types of changes defined by
[Keep a changelog](https://keepachangelog.com/en/1.0.0/):

- `Added` for new features.
- `Changed` for changes in existing functionality.
- `Deprecated` for soon-to-be removed features.
- `Removed` for now removed features.
- `Fixed` for any bug fixes.
- `Security` in case of vulnerabilities.

If the required subsection does not exist yet under **Unreleased**, create it!

## Developing

### Set up

This is no different than other Rust projects.

```shell
git clone https://github.com/appleparan/nano-vllm-rs
cd nano-vllm-rs
cargo test
```

### Useful Commands

- Build and run release version:

  ```shell
  cargo build --release && cargo run --release
  ```

- Run Clippy:

  ```shell
  cargo clippy --all-targets --all-features --workspace
  ```

- Run all tests:

  ```shell
  cargo test --all-features --workspace
  ```

- Run integration tests (requires model download, ~1.2GB):

  ```shell
  # Run all integration tests with real Qwen3-0.6B model
  cargo test --test qwen3_inference_test -- --ignored

  # Run specific integration test
  cargo test test_simple_generation -- --ignored

  # Run all tests including integration tests
  cargo test -- --include-ignored
  ```

  > **Note**: Integration tests are marked with `#[ignore]` and excluded from
  > normal `cargo test` runs. They require network access to download the model
  > from HuggingFace Hub and take several minutes to complete.

- Check to see if there are code formatting issues

  ```shell
  cargo fmt --all -- --check
  ```

- Format the code in the project

  ```shell
  cargo fmt --all
  ```
