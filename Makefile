.PHONY: build test doc clean

build:
	cargo build

release:
	cargo build --release

test:
	cargo test

doc:
	cargo doc --no-deps --open

doc-build:
	cargo doc --no-deps

clean:
	cargo clean

fmt:
	cargo fmt

lint:
	cargo clippy -- -D warnings
