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
**Status**: Not Started

### Tasks

1. `Qwen3Config` (vocab_size, hidden_size, num_layers, num_heads, num_kv_heads, ...)
2. HuggingFace model 다운로드 (hf-hub)
3. SafeTensors 로딩
4. `Qwen3Model` 조립
   - Token embedding
   - N개 Qwen3DecoderLayer
   - Final RMSNorm
   - LM head

### Success Criteria

- [ ] Qwen3-0.6B 모델 로딩 성공
- [ ] Forward pass 실행 성공

---

## Stage 8: Sampler & Engine

**Goal**: 토큰 샘플링 및 추론 엔진 구현
**Status**: Not Started

### Tasks

1. `Sampler` (temperature, top_k, top_p)
2. `LLMEngine`
   - model, scheduler, block_manager 조합
   - `add_request()`: 새 요청 추가
   - `step()`: 한 iteration 실행 (prefill/decode)
   - `generate()`: 완료까지 실행

### Success Criteria

- [ ] 단일 프롬프트 생성 테스트
- [ ] 배치 생성 테스트
- [ ] 생성 결과가 유효한 텍스트인지 확인

---

## Stage 9: CLI

**Goal**: 커맨드라인 인터페이스 구현
**Status**: Not Started

### Tasks

1. Clap 기반 CLI 구조
   - `--model`: 모델 경로
   - `--prompt`: 입력 프롬프트 (여러개 가능)
   - `--max-tokens`: 최대 생성 토큰 수
   - `--priority`: 요청 우선순위
2. 실행 결과 출력

### Success Criteria

- [ ] CLI로 텍스트 생성 성공
- [ ] 여러 프롬프트 동시 처리 성공

---

## Stage 10: Speculative Decoding (Optional)

**Goal**: Draft-Verify 기반 투기적 디코딩 구현
**Status**: Not Started

### Tasks

1. `SpeculativeConfig` (num_speculative_tokens, draft_model)
2. Draft 모델 로딩 (작은 모델)
3. Speculative decoding loop
   - Draft 토큰 K개 생성
   - Target 모델로 한번에 검증
   - Rejection sampling으로 accept/reject

### Success Criteria

- [ ] Speculative decoding 기본 동작 테스트
- [ ] 출력 분포가 target model과 동일 검증

---

## Notes

- 각 Stage 완료 시 Status를 "Complete"로 변경
- 모든 Stage 완료 후 이 파일 삭제
- 막히면 3번 시도 후 다른 접근 방식 고려
