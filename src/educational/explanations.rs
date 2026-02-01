//! Educational text content for LLM inference explanations.
//!
//! This module contains all the educational text used by the narrator,
//! tutorial, and other educational modes. The explanations are organized
//! by topic and include multiple detail levels.

/// An educational explanation with multiple detail levels.
#[derive(Debug, Clone)]
pub struct Explanation {
    /// One-line summary.
    pub short: &'static str,
    /// 2-3 sentence explanation.
    pub medium: &'static str,
    /// Full explanation with examples.
    pub detailed: &'static str,
    /// Why this matters / intuition.
    pub why: &'static str,
}

/// Detail level for explanations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplanationLevel {
    /// One-line summary.
    Short,
    /// 2-3 sentence explanation.
    Medium,
    /// Full explanation with examples.
    Detailed,
    /// Why this matters / intuition.
    Why,
}

impl Explanation {
    /// Get the explanation at the specified detail level.
    pub fn get(&self, level: ExplanationLevel) -> &'static str {
        match level {
            ExplanationLevel::Short => self.short,
            ExplanationLevel::Medium => self.medium,
            ExplanationLevel::Detailed => self.detailed,
            ExplanationLevel::Why => self.why,
        }
    }
}

// =============================================================================
// TOKENIZATION
// =============================================================================

/// Explanation for tokenization.
pub const TOKENIZATION_EXPLANATION: Explanation = Explanation {
    short: "Converting text to numbers the model understands.",
    medium: "LLMs can't read text directly - they process numbers. Tokenization \
splits text into tokens (words or word-pieces) and converts each to a unique ID \
from a vocabulary of ~32,000 tokens.",
    detailed: "Tokenization is the first step in LLM inference. The model has a \
fixed vocabulary of tokens (typically 32,000 for LLaMA models) that was learned \
during training. Each token might be:
- A complete word: \"the\" -> 450
- A word piece: \"capital\" might be \"cap\" + \"ital\"
- A single character or punctuation

The tokenizer uses Byte-Pair Encoding (BPE) algorithm:
1. Start with individual characters
2. Repeatedly merge most frequent pairs
3. Stop when vocabulary size is reached

This allows handling any text, even words never seen in training.",
    why: "WHY TOKENS? Neural networks need numerical input. Token IDs let us \
map infinite text possibilities to a fixed vocabulary. The model learns \
embeddings (vector representations) for each token during training.",
};

// =============================================================================
// EMBEDDING
// =============================================================================

/// Explanation for embedding.
pub const EMBEDDING_EXPLANATION: Explanation = Explanation {
    short: "Looking up the learned vector representation for each token.",
    medium: "Each token ID is used to look up a learned embedding vector from a \
table. These embeddings encode semantic meaning - similar words have similar \
vectors. For LLaMA, each token becomes a 4096-dimensional vector.",
    detailed: "The embedding layer is a simple lookup table:
- Shape: [vocab_size, hidden_size] e.g., [32000, 4096]
- Token ID 450 -> Row 450 of the table -> 4096 numbers

These embeddings are LEARNED during training:
- Semantically similar words cluster together
- Classic example: king - man + woman ~ queen
- The model learns these relationships from massive text data

For a sequence of 5 tokens:
Input:  [450, 7483, 310, 3444, 338]  (5 token IDs)
Output: [5, 4096] tensor           (5 embedding vectors)",
    why: "WHY EMBEDDINGS? Raw token IDs (just integers) contain no meaning. \
Embeddings encode semantic relationships in a high-dimensional space where \
the model can learn patterns. Similar meanings -> similar vectors.",
};

// =============================================================================
// ATTENTION
// =============================================================================

/// Explanation for attention.
pub const ATTENTION_EXPLANATION: Explanation = Explanation {
    short: "Allowing each token to look at and gather information from other tokens.",
    medium: "Self-attention lets each position 'attend' to all previous positions. \
Each token creates Query (what it's looking for), Key (what it contains), and \
Value (information to share). Attention scores determine how much each token \
influences others.",
    detailed: "Self-attention is the core mechanism of transformers:

Step 1: Create Q, K, V projections
- Query: \"What am I looking for?\"
- Key:   \"What do I contain?\"
- Value: \"What information can I give?\"

Step 2: Compute attention scores
- scores = Q @ K^T / sqrt(head_dim)
- Higher score = more attention
- Causal mask: can only see past, not future

Step 3: Apply softmax (normalize to probabilities)
- Each row sums to 1.0
- Masked positions get 0 probability

Step 4: Weighted sum of values
- output = softmax(scores) @ V
- Each position gets a weighted mix of all values

Multi-head attention:
- Split into multiple heads (e.g., 32 heads)
- Each head learns different attention patterns
- Concatenate and project back",
    why: "WHY ATTENTION? It lets the model relate any two positions in the \
sequence, regardless of distance. In \"The capital of France is _\", the model \
can directly attend from \"_\" to \"France\" to predict \"Paris\", without the \
information having to pass through all intermediate positions.",
};

// =============================================================================
// KV CACHE
// =============================================================================

/// Explanation for KV cache.
pub const KV_CACHE_EXPLANATION: Explanation = Explanation {
    short: "Caching computed Key and Value vectors to avoid redundant computation.",
    medium: "During generation, we compute K and V for each new token. Without \
caching, we'd recompute K,V for ALL tokens at every step (quadratic cost!). \
KV cache stores past K,V, so we only compute for the new token (linear cost).",
    detailed: "The KV Cache optimization:

WITHOUT CACHE (naive approach):
Step 1: Process [The]                     - Compute K,V for 1 token
Step 2: Process [The, capital]            - Compute K,V for 2 tokens
Step 3: Process [The, capital, of]        - Compute K,V for 3 tokens
...
Total: 1+2+3+...+n = O(n^2) computations!

WITH CACHE:
Prefill: Process all prompt tokens at once
- Compute K,V for [The, capital, of, France, is]
- Store in cache

Decode: Only process NEW token
- Step 1: Compute K,V for [Paris] only, append to cache
- Step 2: Compute K,V for [.] only, append to cache
...
Total: n + 1 + 1 + ... = O(n) computations!

Memory usage for KV cache:
tokens x layers x 2 (K,V) x kv_heads x head_dim x dtype_size
5 tokens x 32 layers x 2 x 8 heads x 128 dim x 2 bytes = 2.6MB",
    why: "WHY CACHE? Attention needs to see all previous tokens. Without \
caching, we'd recompute the same K,V vectors over and over. The cache trades \
memory for compute, enabling fast autoregressive generation.",
};

// =============================================================================
// PREFILL
// =============================================================================

/// Explanation for prefill phase.
pub const PREFILL_EXPLANATION: Explanation = Explanation {
    short: "Processing the entire prompt at once (parallel, compute-bound).",
    medium: "Prefill is the 'reading' phase where the model processes all prompt \
tokens in parallel. This is compute-bound (lots of matrix multiplications) and \
produces the initial KV cache entries.",
    detailed: "The prefill phase:

1. All prompt tokens processed SIMULTANEOUSLY
   - [The, capital, of, France, is] -> forward pass together
   - Takes advantage of GPU parallelism

2. Characteristics:
   - Compute-bound: Many FLOPs, high GPU utilization
   - Memory efficient: One forward pass for all tokens
   - Produces initial KV cache state

3. Output:
   - KV cache populated for all prompt tokens
   - Logits for last position -> sample first generated token

Chunked Prefill (optimization):
- Very long prompts may overwhelm memory
- Process in chunks (e.g., 512 tokens at a time)
- Trade latency for memory efficiency",
    why: "WHY PARALLEL PREFILL? GPUs excel at parallel computation. Processing \
all prompt tokens together is much faster than one at a time. This is why \
prompts are 'read' quickly, but generation is slower (one token at a time).",
};

// =============================================================================
// DECODE
// =============================================================================

/// Explanation for decode phase.
pub const DECODE_EXPLANATION: Explanation = Explanation {
    short: "Generating tokens one at a time (sequential, memory-bound).",
    medium: "Decode is the 'writing' phase where tokens are generated one by one. \
Each new token depends on all previous ones, so we can't parallelize. This is \
memory-bound due to loading the large KV cache for each step.",
    detailed: "The decode phase:

1. Generate ONE token per step:
   - Can't parallelize: token N+1 depends on token N
   - Each step: forward pass -> sample -> append to sequence

2. Characteristics:
   - Memory-bound: Need to load entire KV cache for each step
   - Low GPU utilization: Small batch (seq_len=1)
   - Bottleneck: Memory bandwidth, not compute

3. Each decode step:
   a. Input: Last generated token [seq_len=1]
   b. Look up embedding
   c. For each layer:
      - Compute Q, K, V for new token
      - Append K, V to cache
      - Attend to ALL cached K, V
      - Apply FFN
   d. Compute logits
   e. Sample next token

Batching helps:
- Process multiple sequences together
- Better GPU utilization
- Same memory bandwidth, more useful work",
    why: "WHY ONE AT A TIME? Each token's probability depends on ALL previous \
tokens. There's no way to know token #7 without generating token #6 first. \
This sequential dependency is why generation is slower than prefill.",
};

// =============================================================================
// SAMPLING
// =============================================================================

/// Explanation for sampling.
pub const SAMPLING_EXPLANATION: Explanation = Explanation {
    short: "Choosing the next token from the probability distribution.",
    medium: "The model outputs logits (raw scores) for all 32,000 tokens. Softmax \
converts these to probabilities. Sampling strategies like greedy (pick highest), \
top-k, or temperature control the selection.",
    detailed: "Token sampling process:

1. Model outputs logits: [batch, seq, vocab_size]
   - Raw scores for each possible next token
   - Not probabilities yet!

2. Convert to probabilities: softmax(logits / temperature)
   - Temperature < 1: More confident, less random
   - Temperature > 1: More random, more creative
   - Temperature = 1: Original distribution

3. Sampling strategies:

   GREEDY (deterministic):
   - Pick argmax(probabilities)
   - Always same output for same input
   - Good for factual tasks

   TOP-K:
   - Only consider top K tokens
   - Sample randomly from those K
   - More diverse outputs

   TOP-P (nucleus):
   - Include tokens until cumulative prob > P
   - Adapts to distribution shape
   - Popular for creative tasks

   TEMPERATURE:
   - Scales logits before softmax
   - Lower = sharper, higher = flatter

4. Example:
   logits: [5.2, 2.1, 1.8, 0.3, ...]
   probs:  [87%, 4%, 2%, 1%, ...]  <- after softmax
   greedy: Always pick \"Paris\" (87%)
   top-k=3: Random from {Paris, the, a}",
    why: "WHY SAMPLE? The model produces a probability distribution, not a single \
answer. Sampling lets us control the creativity/determinism tradeoff. Greedy \
is predictable but boring; high temperature is creative but potentially incoherent.",
};

// =============================================================================
// PAGED ATTENTION
// =============================================================================

/// Explanation for PagedAttention.
pub const PAGED_ATTENTION_EXPLANATION: Explanation = Explanation {
    short: "Managing KV cache memory like virtual memory pages.",
    medium: "Traditional KV cache pre-allocates max_length for each sequence, \
wasting memory. PagedAttention divides KV cache into blocks, allocating them \
on-demand as sequences grow. This enables efficient memory sharing.",
    detailed: "PagedAttention (vLLM's innovation):

Problem: Variable-length sequences
- Seq A: 10 tokens, Seq B: 100 tokens, Seq C: 50 tokens
- Traditional: Allocate max_length (e.g., 2048) for each
- Waste: Most of that memory is empty!

Solution: Page-based allocation (like OS virtual memory)

1. Block Pool:
   - Divide GPU memory into fixed-size blocks
   - Each block holds K,V for N tokens (e.g., N=16)
   - [Block0][Block1][Block2]...[BlockM]

2. Block Tables:
   - Each sequence has a mapping: logical -> physical blocks
   - Seq A: [3, 7]        (uses blocks 3 and 7)
   - Seq B: [1, 4, 5, 8]  (uses blocks 1, 4, 5, 8)

3. Benefits:
   - No wasted memory: Allocate exactly what's needed
   - Dynamic growth: Add blocks as sequence grows
   - Prefix sharing: Same prefix -> share blocks!

4. Prefix caching:
   - \"The capital of France\" appears in many prompts
   - Cache its KV blocks, share across sequences
   - Hash: (token_ids) -> block_ids

Memory savings: Up to 4x more sequences in same memory!",
    why: "WHY PAGED? Without PagedAttention, serving many concurrent requests \
wastes enormous memory on padding. PagedAttention enables high-throughput \
serving by packing sequences efficiently and sharing common prefixes.",
};

// =============================================================================
// FLASH ATTENTION
// =============================================================================

/// Explanation for Flash Attention.
pub const FLASH_ATTENTION_EXPLANATION: Explanation = Explanation {
    short: "Memory-efficient attention using tiling and recomputation.",
    medium: "FlashAttention computes attention without materializing the full \
N*N attention matrix. It uses tiling to fit in GPU SRAM and recomputes \
attention during backprop instead of storing it.",
    detailed: "FlashAttention - IO-Aware Attention:

Problem: Standard attention is memory-bound
- Compute QK^T: [N, N] matrix (huge for long sequences!)
- Store for softmax: O(N^2) memory
- GPU compute is fast, but memory access is slow

Solution: Never materialize the full matrix

1. Tiling:
   - Process Q, K, V in blocks that fit in SRAM
   - SRAM: ~20MB, very fast (19TB/s)
   - HBM: ~80GB, slower (2TB/s)

2. Online softmax:
   - Compute softmax incrementally
   - Keep running max and sum statistics
   - Update as we process each K,V block

3. Kernel fusion:
   - One kernel: load Q block, iterate K/V blocks, write output
   - Avoid HBM reads/writes for intermediate results

Memory: O(N) instead of O(N^2)
Speed: 2-4x faster than standard attention

FlashAttention-2 improvements:
- Better parallelization across sequence length
- Reduced non-matmul FLOPs
- Even faster for long sequences",
    why: "WHY FLASH? Memory bandwidth is the bottleneck. Standard attention \
reads/writes O(N^2) data. FlashAttention reads O(N) data by keeping computation \
in fast SRAM. Enables longer sequences with less memory.",
};

// =============================================================================
// GQA
// =============================================================================

/// Explanation for Grouped Query Attention.
pub const GQA_EXPLANATION: Explanation = Explanation {
    short: "Sharing KV heads across multiple query heads to save memory.",
    medium: "GQA uses fewer KV heads than query heads. For example, 32 Q heads \
might share 8 KV heads (4:1 ratio). This dramatically reduces KV cache size \
while maintaining most of the model quality.",
    detailed: "Grouped Query Attention (GQA):

Standard MHA: Q heads = K heads = V heads (e.g., all 32)
GQA: Q heads = 32, K heads = V heads = 8 (4:1 ratio)

Memory savings:
- KV cache size: tokens x layers x 2 x kv_heads x head_dim x dtype
- GQA (8 heads): 4x smaller than MHA (32 heads)!

How it works:
- 32 Q heads grouped into 8 groups of 4
- Each group shares one K head and one V head
- Still compute full attention, just with repeated K/V

Code: repeat K,V to match Q heads
K_expanded = K.repeat_interleave(num_groups, dim=1)
// [batch, 8, seq, dim] -> [batch, 32, seq, dim]

Quality vs efficiency tradeoff:
- MHA: Best quality, 4x memory
- MQA: Least quality, 1x memory
- GQA: Near-MHA quality, ~1.5x memory (sweet spot!)",
    why: "WHY GQA? KV cache is the main memory bottleneck for long contexts. \
GQA dramatically reduces cache size while keeping most of MHA's expressiveness. \
This enables longer contexts or more concurrent requests.",
};

// =============================================================================
// SPECULATIVE DECODING
// =============================================================================

/// Explanation for Speculative Decoding.
pub const SPECULATIVE_DECODING_EXPLANATION: Explanation = Explanation {
    short: "Using a small model to draft tokens, verified by the large model.",
    medium: "Speculative decoding uses a smaller, faster draft model to predict \
multiple tokens ahead. The larger target model then verifies these in parallel. \
If drafts match what target would have generated, we accept them all at once.",
    detailed: "Speculative Decoding (Draft-Verify paradigm):

Problem: Autoregressive decoding is slow
- Generate one token at a time
- Large models = slow per-token latency
- GPU underutilized (memory-bound)

Solution: Speculate with small model, verify with large

1. Draft Phase (small model, e.g., Qwen3-0.6B):
   - Generate K tokens quickly (K=4 typical)
   - [draft_1, draft_2, draft_3, draft_4]

2. Verify Phase (large model, e.g., Qwen3-4B):
   - Process all K draft tokens + context in ONE forward pass
   - Get probabilities for each position

3. Rejection Sampling:
   - For each draft token:
     - If p_target >= p_draft: Always accept
     - Else: Accept with probability p_target/p_draft
   - On first rejection: Sample from adjusted distribution
   - Discard remaining drafts

4. Guarantees:
   - Output EXACTLY matches target model distribution
   - Never worse than normal decoding
   - Speedup depends on acceptance rate

Typical speedup: 2-3x for well-matched draft/target pairs",
    why: "WHY SPECULATE? Large models are slow but accurate. Small models are \
fast but less accurate. By having small model 'guess' and large model 'verify', \
we get large model quality at closer to small model speed.",
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Available explanation topics.
pub static TOPICS: &[&str] = &[
    "tokenization",
    "embedding",
    "attention",
    "kv_cache",
    "prefill",
    "decode",
    "sampling",
    "paged_attention",
    "flash_attention",
    "gqa",
    "speculative_decoding",
];

/// Get an explanation for a topic.
pub fn get_explanation(topic: &str) -> Option<&'static Explanation> {
    match topic {
        "tokenization" => Some(&TOKENIZATION_EXPLANATION),
        "embedding" => Some(&EMBEDDING_EXPLANATION),
        "attention" => Some(&ATTENTION_EXPLANATION),
        "kv_cache" => Some(&KV_CACHE_EXPLANATION),
        "prefill" => Some(&PREFILL_EXPLANATION),
        "decode" => Some(&DECODE_EXPLANATION),
        "sampling" => Some(&SAMPLING_EXPLANATION),
        "paged_attention" => Some(&PAGED_ATTENTION_EXPLANATION),
        "flash_attention" => Some(&FLASH_ATTENTION_EXPLANATION),
        "gqa" => Some(&GQA_EXPLANATION),
        "speculative_decoding" => Some(&SPECULATIVE_DECODING_EXPLANATION),
        _ => None,
    }
}

/// Get all available explanation topics.
pub fn get_all_topics() -> &'static [&'static str] {
    TOPICS
}

/// Quiz question for tutorial mode.
#[derive(Debug, Clone)]
pub struct QuizQuestion {
    /// The question text.
    pub question: &'static str,
    /// Available answer options.
    pub options: &'static [&'static str],
    /// Correct answer (A, B, C, or D).
    pub answer: char,
    /// Explanation of the correct answer.
    pub explanation: &'static str,
}

/// Quiz questions indexed by topic.
pub fn get_quiz_question(topic: &str) -> Option<QuizQuestion> {
    match topic {
        "tokenization" => Some(QuizQuestion {
            question: "Why do we need tokens?",
            options: &[
                "A) Computers can only process numbers",
                "B) It's faster than processing characters",
                "C) It reduces vocabulary size",
                "D) All of the above",
            ],
            answer: 'D',
            explanation: "All these reasons apply! Neural networks need numerical input, \
tokens are more efficient than characters, and a fixed vocabulary size \
allows the model to handle any text.",
        }),
        "attention" => Some(QuizQuestion {
            question: "Why is the attention mask 'causal' (triangular)?",
            options: &[
                "A) To save memory",
                "B) So tokens can only see past tokens, not future",
                "C) To make computation faster",
                "D) Because GPUs prefer triangular matrices",
            ],
            answer: 'B',
            explanation: "Language models predict the next token, so they should only \
see past context. Looking at future tokens would be 'cheating'!",
        }),
        "kv_cache" => Some(QuizQuestion {
            question: "What does the KV cache store?",
            options: &[
                "A) Model weights",
                "B) Token embeddings",
                "C) Key and Value vectors from previous tokens",
                "D) Attention scores",
            ],
            answer: 'C',
            explanation: "The KV cache stores the Key and Value projections from all \
previous tokens. This avoids recomputing them at every generation step.",
        }),
        "prefill_decode" => Some(QuizQuestion {
            question: "Why is decode slower than prefill (per token)?",
            options: &[
                "A) Prefill uses simpler math",
                "B) Decode loads the entire KV cache for each token",
                "C) Decode uses more GPU memory",
                "D) Prefill runs on CPU",
            ],
            answer: 'B',
            explanation: "Decode must load the entire KV cache from GPU memory for each \
new token, making it memory-bound. Prefill processes all tokens in parallel, \
making better use of GPU compute.",
        }),
        _ => None,
    }
}
