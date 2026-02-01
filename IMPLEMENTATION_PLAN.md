# Implementation Plan

## Stage 1: Project Setup & Core Types

**Goal**: 기본 프로젝트 구조와 핵심 타입 정의
**Status**: Complete

### Tasks

1. Cargo.toml 의존성 설정 (candle, tokenizers, hf-hub, clap, anyhow, thiserror)
2. Error 타입 정의 (`src/error.rs`)
3. Config 타입 정의 (`src/config.rs`)
4. Core 모듈 구조 생성

### Success Criteria

- [x] `cargo build` 성공
- [x] `cargo test` 성공 (빈 테스트라도)

---

## Stage 2: Block & BlockManager (PagedAttention 기반)

**Goal**: OS 가상 메모리 스타일의 블록 관리 시스템 구현
**Status**: Complete

### Tasks

1. [x] `Block` 구조체 (block_id, block_size, ref_count, prefix_hash)
2. [x] `BlockTable` 구조체 (logical -> physical block mapping)
3. [x] `hash_token_block` 함수 (prefix caching용 해시)
4. [x] `BlockManager` 구현
   - Free list 기반 O(1) allocation/deallocation
   - Reference counting
   - Prefix caching (hash 기반)

### Success Criteria

- [x] Block 생성/ref_count 테스트 통과
- [x] BlockTable slot mapping 테스트 통과
- [x] BlockManager 할당/해제 테스트 통과
- [x] Prefix cache hit/miss 테스트 통과

---

## Stage 3: Sequence & KV Cache

**Goal**: 요청 추적 및 KV Cache 저장소 구현
**Status**: Complete

### Tasks

1. [x] `SequenceStatus` enum (Waiting, Running, Swapped, Finished)
2. [x] `FinishReason` enum (EndOfSequence, MaxTokens, StopSequence, Aborted)
3. [x] `Sequence` 구조체
   - prompt_token_ids, output_token_ids
   - block_table, num_prefilled_tokens
   - priority, arrival_time
   - State transition methods with validation
4. [x] `KVCacheConfig` 구조체
5. [x] `LayerKVCache` 구조체 (single layer K/V storage)
6. [x] `KVCache` 구조체
   - [num_blocks, block_size, num_kv_heads, head_dim] 형태 Tensor
   - gather_keys/values for block-based access

### Success Criteria

- [x] Sequence 상태 전이 테스트 통과
- [x] KV Cache creation/gather 테스트 통과

---

## Stage 4: Scheduler (Continuous Batching)

**Goal**: 연속 배칭 스케줄러 구현
**Status**: Complete

### Tasks

1. [x] `SchedulerConfig` (max_num_seqs, max_prefill_tokens, enable_preemption)
2. [x] `SchedulerOutputs` (prefill_seqs, decode_seqs, prefill_chunk_sizes, preempted_seqs)
3. [x] `Scheduler` 구현
   - Waiting queue (BinaryHeap priority queue)
   - Running set 관리
   - `schedule()` 메서드: schedule_decode → allocate_running_blocks → schedule_prefill → handle_preemption
   - Priority scheduling (priority DESC, arrival ASC)
   - Chunked prefill 지원
   - Preemption 로직 (lowest priority running → swapped)

### Success Criteria

- [x] 기본 스케줄링 테스트 통과 (18 tests)
- [x] Priority scheduling 테스트 통과
- [x] Preemption 테스트 통과
- [x] Chunked prefill 테스트 통과
- [x] Resource limit 테스트 통과 (max_num_seqs, max_prefill_tokens, out_of_blocks)

---

## Stage 5: Qwen3 Model Components

**Goal**: Qwen3 모델 구성요소 구현
**Status**: Complete

### Tasks

1. [x] `RmsNorm` (Root Mean Square Normalization) - `src/model/norm.rs`
2. [x] `RotaryEmbedding` (RoPE - Rotary Position Embeddings) - `src/model/rope.rs`
3. [x] `Qwen3Mlp` (SwiGLU: gate * silu(up) -> down) - `src/model/mlp.rs`
4. [x] `Qwen3Attention` (GQA - Grouped Query Attention) - `src/model/attention.rs`
   - Q/K/V projection (num_kv_heads < num_heads)
   - Per-head RMSNorm on Q and K (Qwen3 specific)
   - RoPE 적용
   - Scaled dot-product attention with causal mask
   - KV cache support
5. [x] `Qwen3DecoderLayer` (attention + mlp + residual) - `src/model/decoder.rs`

### Success Criteria

- [x] RmsNorm forward 테스트 통과 (5 tests)
- [x] RoPE 테스트 통과 (6 tests)
- [x] MLP 테스트 통과 (5 tests)
- [x] Attention output shape 테스트 통과 (7 tests)
- [x] DecoderLayer 테스트 통과 (5 tests)
- [x] 총 28개 모델 컴포넌트 테스트 통과

---

## Stage 6: PagedAttention

**Goal**: 블록 기반 attention 연산 구현
**Status**: Complete

### Tasks

1. [x] `paged_attention()` 함수 - `src/attention/paged.rs`
   - BlockTable에서 K/V gather
   - Attention 연산 (Q @ K^T / sqrt(d) -> softmax -> @ V)
   - Causal masking (prefill 및 chunked prefill 지원)
2. [x] `prefill_attention()` - Standard SDPA
3. [x] `write_kv_to_cache()` - 블록 기반 캐시에 K/V 저장

### Success Criteria

- [x] Prefill attention shape 테스트 통과
- [x] Paged attention 기본 동작 테스트 통과
- [x] Multi-block paged attention 테스트 통과
- [x] Chunked prefill 테스트 통과
- [x] Write KV to cache 테스트 통과
- [x] 총 10개 PagedAttention 테스트 통과

---

## Stage 7: Model Loader & Full Qwen3 Model

**Goal**: HuggingFace 모델 로딩 및 전체 Qwen3 모델 조립
**Status**: Complete

### Tasks

1. [x] `Qwen3Config` - HuggingFace config.json 파싱 (`src/model/loader.rs`)
2. [x] `download_model()` - HuggingFace Hub에서 모델 다운로드
3. [x] `load_safetensors()` - SafeTensors 파일 로딩 (memory-mapped)
4. [x] `Qwen3Model` 조립 (`src/model/qwen3.rs`)
   - Token embedding (`embed_tokens`)
   - N개 Qwen3DecoderLayer (`layers`)
   - Final RMSNorm (`norm`)
5. [x] `Qwen3ForCausalLM` - Language model head 포함

### Success Criteria

- [x] Qwen3Config 파싱 테스트 통과 (3 tests)
- [x] Qwen3Model 구조 정의 완료
- [x] 모든 테스트 통과 (100 tests)

---

## Stage 8: Sampler & Engine

**Goal**: 토큰 샘플링 및 추론 엔진 구현
**Status**: Complete

### Tasks

1. [x] `Sampler` (temperature, top_k, top_p) - `src/engine/sampler.rs`
   - Greedy decoding (temperature=0)
   - Temperature scaling
   - Top-k filtering
   - Top-p (nucleus) filtering
   - Reproducible sampling with seed
2. [x] `LLMEngine` - `src/engine/llm.rs`
   - model, scheduler, sampler, tokenizer 조합
   - `add_request()`: 새 요청 추가 (tokenization 포함)
   - `step()`: 한 iteration 실행 (prefill/decode)
   - `generate()`: 완료까지 실행
   - `generate_text()`: 단일 프롬프트 편의 메서드

### Success Criteria

- [x] Sampler 테스트 통과 (6 tests)
- [x] GenerationRequest 빌더 테스트 통과
- [x] 모든 테스트 통과 (107 tests)

---

## Stage 9: CLI

**Goal**: 커맨드라인 인터페이스 구현
**Status**: Complete

### Tasks

1. [x] Clap 기반 CLI 구조 (`src/main.rs`)
   - `--model` / `-m`: HuggingFace 모델 ID (required)
   - `--prompt` / `-p`: 입력 프롬프트 (여러개 가능, required)
   - `--max-tokens`: 최대 생성 토큰 수 (default: 256)
   - `--temperature` / `-t`: 샘플링 온도 (default: 1.0)
   - `--top-k`: Top-k 샘플링 (default: 0 = disabled)
   - `--top-p`: Top-p (nucleus) 샘플링 (default: 1.0 = disabled)
   - `--priority`: 요청 우선순위 (default: 0)
   - `--revision`: HuggingFace revision (default: "main")
   - `--cuda`: CUDA 사용 여부
   - `--block-size`: PagedAttention 블록 크기 (default: 16)
   - `--num-blocks`: KV cache 블록 수 (default: 512)
2. [x] HuggingFace 모델 다운로드 및 로딩
3. [x] 실행 결과 출력 (pretty formatting, summary)
4. [x] 여러 프롬프트 동시 처리

### Success Criteria

- [x] CLI로 텍스트 생성 성공 (Qwen/Qwen3-0.6B 테스트 완료)
- [x] 여러 프롬프트 동시 처리 성공

---

## Stage 10: Flash Attention

**Goal**: 메모리 효율적인 Flash Attention 알고리즘 구현 (CPU + CUDA)
**Status**: Not Started

### Background

Flash Attention은 Stanford에서 개발한 메모리 효율적 attention 알고리즘:
- **메모리**: O(n²) → O(n) 감소 (attention matrix 전체 저장 불필요)
- **속도**: HBM ↔ SRAM 간 IO 최소화로 2-4배 속도 향상
- **Tiling**: attention을 block 단위로 계산하여 on-chip SRAM 활용

### Target Hardware

- **CPU**: Reference implementation (교육 목적, 알고리즘 이해)
- **GPU**: NVIDIA RTX 4090 (Ada Lovelace, SM89, 24GB VRAM)
  - CUDA Compute Capability 8.9
  - L2 Cache: 72MB, Shared Memory: 100KB/SM

### Tasks

#### Phase 1: CPU Reference Implementation
1. [ ] `FlashAttentionConfig` 구조체
   - `block_size_q`, `block_size_kv`: Tiling 크기
   - `causal`: Causal masking 여부
   - `softmax_scale`: 1/sqrt(head_dim)
2. [ ] `flash_attention_cpu()` 함수 (`src/attention/flash.rs`)
   - Tiled matrix multiplication
   - Online softmax (numerically stable)
   - Incremental output accumulation
3. [ ] Causal masking 지원
4. [ ] 표준 SDPA와 수치 일치 테스트

#### Phase 2: Custom CUDA Kernel (RTX 4090)
5. [ ] CUDA 커널 빌드 환경 설정
   - `build.rs`에 CUDA 컴파일 추가
   - `nvcc` 또는 `cudarc` 크레이트 활용
6. [ ] Flash Attention Forward 커널 (`kernels/flash_attn_fwd.cu`)
   - Shared memory tiling (Q, K, V 블록 로드)
   - Online softmax with running max/sum
   - Warp-level reductions
7. [ ] Flash Attention Backward 커널 (Optional - training용)
8. [ ] Rust FFI 바인딩 (`src/attention/flash_cuda.rs`)
   - `flash_attention_cuda()` 함수
   - Device 자동 선택 로직
9. [ ] FP16/BF16 지원
   - `__half` / `__nv_bfloat16` 타입
   - Tensor Cores 활용 (`wmma` API 또는 `mma` PTX)

#### Phase 3: Integration
10. [ ] `Qwen3Attention`에 Flash Attention 옵션 추가
11. [ ] `EngineConfig`에 `use_flash_attention` 플래그 추가
12. [ ] Benchmark: 표준 SDPA vs Flash Attention (seq_len 별)

### Algorithm (Flash Attention 2)

```text
# Outer loop: parallelize over batch, heads, and query blocks
For each query block Q_i (parallelized):
    Initialize: O_i = 0, l_i = 0, m_i = -inf

    # Inner loop: sequential over key/value blocks
    For each key/value block K_j, V_j:
        # 1. Compute local attention scores
        S_ij = Q_i @ K_j^T * scale

        # 2. Apply causal mask (if needed)
        if causal and j > i:
            S_ij = masked_fill(S_ij, -inf)

        # 3. Online softmax update
        m_new = max(m_i, rowmax(S_ij))
        P_ij = exp(S_ij - m_new)
        l_new = l_i * exp(m_i - m_new) + rowsum(P_ij)

        # 4. Update output with rescaling
        O_i = O_i * (l_i * exp(m_i - m_new) / l_new) + (P_ij @ V_j) / l_new

        # 5. Update statistics
        m_i = m_new
        l_i = l_new
```

### File Structure

```
src/attention/
├── mod.rs              # Module exports
├── paged.rs            # PagedAttention (existing)
├── flash.rs            # Flash Attention Rust interface
│   ├── FlashAttentionConfig
│   ├── flash_attention()        # Dispatcher (CPU/CUDA 자동 선택)
│   └── flash_attention_cpu()    # CPU reference implementation
└── flash_cuda.rs       # CUDA FFI bindings
    └── flash_attention_cuda()   # CUDA kernel 호출

kernels/
├── flash_attn_fwd.cu   # Forward pass CUDA kernel
├── flash_attn_bwd.cu   # Backward pass CUDA kernel (optional)
└── utils.cuh           # Shared memory helpers, warp reductions

tests/
└── flash_attention_test.rs
```

### Dependencies & Build Setup

```toml
# Cargo.toml additions
[dependencies]
cudarc = { version = "0.12", optional = true }  # CUDA runtime bindings

[build-dependencies]
cc = { version = "1.0", features = ["parallel"] }

[features]
default = []
cuda = ["candle-core/cuda", "cudarc"]
```

```rust
// build.rs (커스텀 CUDA 커널 빌드)
fn main() {
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=kernels/");
        cc::Build::new()
            .cuda(true)
            .file("kernels/flash_attn_fwd.cu")
            .flag("-arch=sm_89")  // RTX 4090
            .flag("-O3")
            .compile("flash_attn_kernels");
    }
}
```

### Success Criteria

- [ ] CPU: Flash Attention 기본 동작 테스트 통과
- [ ] CPU: 표준 SDPA와 출력 일치 검증 (수치 오차 1e-5 이내)
- [ ] CUDA: RTX 4090에서 Flash Attention 동작 확인
- [ ] CUDA: FP16에서 정확도 검증
- [ ] Benchmark: seq_len=2048에서 2x+ 속도 향상 (vs naive SDPA)
- [ ] Benchmark: seq_len=8192에서 메모리 50%+ 감소

---

## Stage 11: Speculative Decoding (Optional)

**Goal**: Draft-Verify 기반 투기적 디코딩 구현
**Status**: Not Started

### Background

Speculative Decoding은 작은 draft 모델로 여러 토큰을 빠르게 생성하고, 큰 target 모델로 한번에 검증하여 처리량을 높이는 기법:
- Draft 모델이 K개 토큰 생성 (빠름)
- Target 모델이 K+1개 위치 동시 검증 (한번의 forward)
- Rejection sampling으로 accept/reject 결정
- 기대값: 평균 acceptance rate * K 토큰을 한번의 target forward로 처리

### Tasks

1. [ ] `SpeculativeConfig` (num_speculative_tokens, draft_model)
2. [ ] Draft 모델 로딩 (작은 모델 또는 동일 모델의 초기 레이어)
3. [ ] Speculative decoding loop
   - Draft 토큰 K개 생성
   - Target 모델로 한번에 검증
   - Rejection sampling으로 accept/reject
4. [ ] Batch speculative decoding (여러 시퀀스 동시 처리)

### Success Criteria

- [ ] Speculative decoding 기본 동작 테스트
- [ ] 출력 분포가 target model과 동일 검증
- [ ] Throughput 향상 측정 (tokens/sec 비교)

---

## Notes

- 각 Stage 완료 시 Status를 "Complete"로 변경
- 모든 Stage 완료 후 이 파일 삭제
- 막히면 3번 시도 후 다른 접근 방식 고려
