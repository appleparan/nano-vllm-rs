//! X-Ray mode - Mathematical and tensor operation visualizations.
//!
//! Shows the actual mathematics and tensor operations happening during
//! inference, helping users understand the computational flow.

/// Configuration for X-Ray visualizations.
#[derive(Debug, Clone)]
pub struct XRayConfig {
    /// Show tensor dimensions.
    pub show_tensor_shapes: bool,
    /// Show matrix multiplication details.
    pub show_matrix_operations: bool,
    /// Show actual numbers.
    pub show_numerical_examples: bool,
    /// Show memory calculations.
    pub show_memory_layout: bool,
    /// Decimal places for numbers.
    pub precision: usize,
    /// Max elements to show from tensors.
    pub max_display_elements: usize,
}

impl Default for XRayConfig {
    fn default() -> Self {
        Self {
            show_tensor_shapes: true,
            show_matrix_operations: true,
            show_numerical_examples: true,
            show_memory_layout: true,
            precision: 3,
            max_display_elements: 5,
        }
    }
}

/// Trait for X-Ray visualizer implementations.
pub trait XRay {
    /// Show tokenization details.
    fn show_tokenization(&self, text: &str, token_ids: &[u32], token_strs: &[String]);

    /// Show embedding lookup operation.
    fn show_embedding_lookup(
        &self,
        token_ids: &[u32],
        vocab_size: usize,
        embedding_dim: usize,
        sample_embedding: Option<&[f32]>,
    );

    /// Show Q, K, V projection operations.
    fn show_qkv_projection(
        &self,
        layer_idx: usize,
        input_shape: (usize, usize, usize),
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    );

    /// Show attention score computation.
    fn show_attention_scores(
        &self,
        layer_idx: usize,
        q_shape: (usize, usize, usize, usize),
        k_shape: (usize, usize, usize, usize),
        head_dim: usize,
        sample_scores: Option<&[Vec<f32>]>,
    );

    /// Show softmax with causal mask.
    fn show_softmax(
        &self,
        seq_len: usize,
        sample_before: Option<&[Vec<f32>]>,
        sample_after: Option<&[Vec<f32>]>,
    );

    /// Show attention output computation.
    fn show_attention_output(
        &self,
        scores_shape: (usize, usize, usize, usize),
        v_shape: (usize, usize, usize, usize),
        output_shape: (usize, usize, usize, usize),
    );

    /// Show RoPE (Rotary Position Embedding).
    fn show_rope(&self, head_dim: usize, max_positions: usize, base: f32);

    /// Show FFN (SwiGLU) computation.
    fn show_ffn(
        &self,
        layer_idx: usize,
        input_shape: (usize, usize, usize),
        hidden_size: usize,
        intermediate_size: usize,
    );

    /// Show KV cache structure and memory usage.
    fn show_kv_cache_structure(
        &self,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        current_seq_len: usize,
        dtype_bytes: usize,
    );

    /// Show LM head projection and sampling.
    fn show_lm_head(
        &self,
        input_shape: (usize, usize, usize),
        vocab_size: usize,
        sample_logits: Option<&[f32]>,
        sample_probs: Option<&[f32]>,
        top_tokens: Option<&[(String, f32)]>,
    );
}

/// Visualizes tensor operations and mathematics during inference.
pub struct XRayVisualizer {
    config: XRayConfig,
}

impl XRayVisualizer {
    /// Create a new X-Ray visualizer.
    pub fn new(config: XRayConfig) -> Self {
        Self { config }
    }

    fn print(&self, text: &str) {
        println!("{text}");
    }

    fn header(&self, title: &str) {
        self.print(&format!("\n╔{}╗", "═".repeat(62)));
        self.print(&format!("║  X-RAY: {title:52} ║"));
        self.print(&format!("╚{}╝", "═".repeat(62)));
    }

    fn shape_str(shape: &[usize]) -> String {
        format!(
            "[{}]",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    fn format_tensor_preview(&self, values: &[f32]) -> String {
        let max_show = self.config.max_display_elements;
        let formatted = if values.len() <= max_show {
            values
                .iter()
                .map(|v| format!("{:.prec$}", v, prec = self.config.precision))
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            let half = max_show / 2;
            let first: Vec<_> = values[..half]
                .iter()
                .map(|v| format!("{:.prec$}", v, prec = self.config.precision))
                .collect();
            let last: Vec<_> = values[values.len() - half..]
                .iter()
                .map(|v| format!("{:.prec$}", v, prec = self.config.precision))
                .collect();
            format!("{}, ..., {}", first.join(", "), last.join(", "))
        };
        format!("[{formatted}]")
    }
}

impl Default for XRayVisualizer {
    fn default() -> Self {
        Self::new(XRayConfig::default())
    }
}

impl XRay for XRayVisualizer {
    fn show_tokenization(&self, text: &str, token_ids: &[u32], token_strs: &[String]) {
        self.header("Tokenization");
        self.print(&format!("\nInput text: \"{text}\""));
        self.print(&format!("Characters: {}", text.len()));
        self.print("\n→ Tokenizer (BPE)\n");

        for (i, (tid, tstr)) in token_ids.iter().zip(token_strs.iter()).enumerate() {
            self.print(&format!("  Token {i}: \"{tstr}\" → ID {tid}"));
        }

        self.print(&format!(
            "\nOutput: {} tensor of token IDs",
            Self::shape_str(&[token_ids.len()])
        ));
    }

    fn show_embedding_lookup(
        &self,
        token_ids: &[u32],
        vocab_size: usize,
        embedding_dim: usize,
        sample_embedding: Option<&[f32]>,
    ) {
        self.header("Embedding Lookup");

        let batch_size = 1;
        let seq_len = token_ids.len();

        self.print("\nEmbedding Table:");
        self.print(&format!(
            "  Shape: {}",
            Self::shape_str(&[vocab_size, embedding_dim])
        ));
        self.print(&format!(
            "  (Each of {vocab_size} tokens has a {embedding_dim}-d vector)"
        ));

        self.print("\nLookup Operation:");
        self.print(&format!("  Input token IDs: {token_ids:?}"));
        self.print("  → Index into embedding table");
        let first_few: Vec<_> = token_ids.iter().take(3).map(|id| id.to_string()).collect();
        self.print(&format!("  → Gather rows: {}, ...", first_few.join(", ")));

        self.print("\nOutput:");
        self.print(&format!(
            "  Shape: {}",
            Self::shape_str(&[batch_size, seq_len, embedding_dim])
        ));
        self.print(&format!(
            "  (batch={batch_size}, seq_len={seq_len}, hidden={embedding_dim})"
        ));

        if let Some(embedding) = sample_embedding
            && self.config.show_numerical_examples
        {
            self.print(&format!("\nSample embedding (token {}):", token_ids[0]));
            self.print(&format!("  {}", self.format_tensor_preview(embedding)));
        }
    }

    fn show_qkv_projection(
        &self,
        layer_idx: usize,
        input_shape: (usize, usize, usize),
        _hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        self.header(&format!("Layer {} - Q/K/V Projections", layer_idx + 1));

        let (batch, seq_len, hidden) = input_shape;

        self.print("\nStep 1: Project to Q, K, V");
        self.print(&"─".repeat(40));

        self.print(&format!(
            "\n  hidden_states: {}",
            Self::shape_str(&[batch, seq_len, hidden])
        ));
        self.print(&format!(
            "       (batch={batch}, seq={seq_len}, hidden={hidden})"
        ));
        self.print("\n       ↓ Linear projections (learned weights)");

        let q_shape = [batch, num_heads, seq_len, head_dim];
        let k_shape = [batch, num_kv_heads, seq_len, head_dim];
        let v_shape = [batch, num_kv_heads, seq_len, head_dim];

        self.print(&format!(
            "\n  Q: {}  ({} heads, each {} dims)",
            Self::shape_str(&q_shape),
            num_heads,
            head_dim
        ));
        self.print(&format!(
            "  K: {}   ({} KV heads for GQA)",
            Self::shape_str(&k_shape),
            num_kv_heads
        ));
        self.print(&format!("  V: {}", Self::shape_str(&v_shape)));

        if num_heads != num_kv_heads {
            let ratio = num_heads / num_kv_heads;
            self.print(&format!(
                "\n  💡 GQA: {num_heads} Q heads share {num_kv_heads} KV heads ({ratio}:1)"
            ));
            self.print(&format!("     Memory saving: {ratio}x less KV cache!"));
        }
    }

    fn show_attention_scores(
        &self,
        layer_idx: usize,
        q_shape: (usize, usize, usize, usize),
        k_shape: (usize, usize, usize, usize),
        head_dim: usize,
        sample_scores: Option<&[Vec<f32>]>,
    ) {
        self.header(&format!("Layer {} - Attention Scores", layer_idx + 1));

        let (batch, num_heads, q_len, _) = q_shape;
        let (_, _, k_len, _) = k_shape;
        let _scale = (head_dim as f32).sqrt();

        self.print("\nStep 2: Compute Attention Scores");
        self.print(&"─".repeat(40));

        self.print(&format!("\n  scores = Q @ K^T / √{head_dim}"));
        self.print(&format!(
            "\n  {} @ {} → {}",
            Self::shape_str(&[batch, num_heads, q_len, head_dim]),
            Self::shape_str(&[batch, num_heads, head_dim, k_len]),
            Self::shape_str(&[batch, num_heads, q_len, k_len])
        ));
        self.print("       Q           K^T           scores");

        self.print(&format!(
            "\n  Why √{head_dim}? Prevents scores from getting too large before softmax."
        ));

        if let Some(scores) = sample_scores
            && self.config.show_numerical_examples
        {
            self.print("\n  Sample scores (head 0):");
            for (i, row) in scores.iter().take(3).enumerate() {
                let row_str: String = row
                    .iter()
                    .take(5)
                    .map(|v| format!("{v:5.2}"))
                    .collect::<Vec<_>>()
                    .join("  ");
                let suffix = if row.len() > 5 { " ..." } else { "" };
                self.print(&format!("    Row {i}: [{row_str}{suffix}]"));
            }
            if scores.len() > 3 {
                self.print("    ...");
            }
        }
    }

    fn show_softmax(
        &self,
        _seq_len: usize,
        sample_before: Option<&[Vec<f32>]>,
        sample_after: Option<&[Vec<f32>]>,
    ) {
        self.print("\nStep 3: Softmax (normalize to probabilities)");
        self.print(&"─".repeat(40));

        self.print("\n  Apply causal mask: positions can only see past + self");
        self.print("  Then softmax each row (sums to 1.0)");

        if let (Some(before), Some(after)) = (sample_before, sample_after)
            && self.config.show_numerical_examples
        {
            self.print("\n  Before softmax:    After softmax (with causal mask):");

            for i in 0..3.min(before.len()) {
                let before_str: String = before[i]
                    .iter()
                    .take(5)
                    .map(|v| format!("{v:5.2}"))
                    .collect::<Vec<_>>()
                    .join("  ");
                let after_str: String = after[i]
                    .iter()
                    .take(5)
                    .map(|v| format!("{v:5.2}"))
                    .collect::<Vec<_>>()
                    .join("  ");
                let indicator = if i == 0 { "← row sums = 1" } else { "" };
                self.print(&format!("  [{before_str}]   [{after_str}]  {indicator}"));
            }
        }
    }

    fn show_attention_output(
        &self,
        scores_shape: (usize, usize, usize, usize),
        v_shape: (usize, usize, usize, usize),
        output_shape: (usize, usize, usize, usize),
    ) {
        self.print("\nStep 4: Weighted sum of Values");
        self.print(&"─".repeat(40));

        self.print("\n  output = softmax(scores) @ V");
        self.print(&format!(
            "\n  {} @ {} → {}",
            Self::shape_str(&[
                scores_shape.0,
                scores_shape.1,
                scores_shape.2,
                scores_shape.3
            ]),
            Self::shape_str(&[v_shape.0, v_shape.1, v_shape.2, v_shape.3]),
            Self::shape_str(&[
                output_shape.0,
                output_shape.1,
                output_shape.2,
                output_shape.3
            ])
        ));
        self.print("   weights       V              output");
    }

    fn show_rope(&self, head_dim: usize, _max_positions: usize, base: f32) {
        self.header("Rotary Position Embedding (RoPE)");

        self.print("\nRoPE encodes position by rotating Q and K vectors");
        self.print("\nConfiguration:");
        self.print(&format!("  Head dim: {head_dim}"));
        self.print(&format!("  Base frequency: {base}"));

        self.print("\nRotation formula:");
        self.print("  q_rotated = q * cos(θ) + rotate_half(q) * sin(θ)");
        self.print("\n  where θ depends on position and dimension");

        self.print("\nKey insight:");
        self.print("  q_rotated · k_rotated depends on (pos_q - pos_k)");
        self.print("  → Relative position naturally emerges!");
    }

    fn show_ffn(
        &self,
        layer_idx: usize,
        input_shape: (usize, usize, usize),
        hidden_size: usize,
        intermediate_size: usize,
    ) {
        self.header(&format!(
            "Layer {} - Feed-Forward Network (SwiGLU)",
            layer_idx + 1
        ));

        let (batch, seq_len, _hidden) = input_shape;

        self.print("\nSwiGLU: output = down(SiLU(gate(x)) * up(x))");
        self.print("\nProjections:");
        self.print(&format!(
            "  gate_proj: [{hidden_size}] → [{intermediate_size}]"
        ));
        self.print(&format!(
            "  up_proj:   [{hidden_size}] → [{intermediate_size}]"
        ));
        self.print(&format!(
            "  down_proj: [{intermediate_size}] → [{hidden_size}]"
        ));

        self.print("\nComputation flow:");
        self.print(&format!(
            "  1. gate = gate_proj(x)     → {}",
            Self::shape_str(&[batch, seq_len, intermediate_size])
        ));
        self.print("  2. gate = SiLU(gate)       → element-wise activation");
        self.print(&format!(
            "  3. up = up_proj(x)         → {}",
            Self::shape_str(&[batch, seq_len, intermediate_size])
        ));
        self.print("  4. hidden = gate * up      → element-wise multiply");
        self.print(&format!(
            "  5. output = down_proj(hidden) → {}",
            Self::shape_str(&[batch, seq_len, hidden_size])
        ));

        self.print("\n  💡 SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))");
    }

    fn show_kv_cache_structure(
        &self,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        current_seq_len: usize,
        dtype_bytes: usize,
    ) {
        self.header("KV Cache Structure");

        self.print("\nCache shape per layer:");
        self.print(&format!(
            "  Keys:   [{num_kv_heads}, {max_seq_len}, {head_dim}]"
        ));
        self.print(&format!(
            "  Values: [{num_kv_heads}, {max_seq_len}, {head_dim}]"
        ));

        self.print("\nTotal structure:");
        self.print(&format!(
            "  [{num_layers} layers, 2 (K/V), {num_kv_heads} heads, {max_seq_len} positions, {head_dim} dim]"
        ));

        // Memory calculation
        let elements = num_layers * 2 * num_kv_heads * max_seq_len * head_dim;
        let bytes_total = elements * dtype_bytes;
        let mb_total = bytes_total as f32 / (1024.0 * 1024.0);

        let current_elements = num_layers * 2 * num_kv_heads * current_seq_len * head_dim;
        let current_bytes = current_elements * dtype_bytes;
        let current_mb = current_bytes as f32 / (1024.0 * 1024.0);

        self.print("\nMemory:");
        self.print(&format!(
            "  Max capacity: {mb_total:.2} MB ({max_seq_len} tokens)"
        ));
        self.print(&format!(
            "  Current used: {current_mb:.2} MB ({current_seq_len} tokens)"
        ));
        self.print(&format!(
            "  Utilization:  {:.1}%",
            current_seq_len as f32 / max_seq_len as f32 * 100.0
        ));

        self.print("\nCalculation:");
        self.print(&format!(
            "  {num_layers} layers × 2 × {num_kv_heads} heads × {current_seq_len} tokens × {head_dim} dim × {dtype_bytes} bytes"
        ));
        self.print(&format!("  = {current_bytes} bytes = {current_mb:.2} MB"));
    }

    fn show_lm_head(
        &self,
        input_shape: (usize, usize, usize),
        vocab_size: usize,
        _sample_logits: Option<&[f32]>,
        _sample_probs: Option<&[f32]>,
        top_tokens: Option<&[(String, f32)]>,
    ) {
        self.header("LM Head - Predicting Next Token");

        let (batch, seq_len, hidden) = input_shape;

        self.print("\nProject to vocabulary:");
        self.print(&format!(
            "  Input:  {}",
            Self::shape_str(&[batch, seq_len, hidden])
        ));
        self.print(&format!(
            "  Weight: {}",
            Self::shape_str(&[hidden, vocab_size])
        ));
        self.print(&format!(
            "  Output: {}",
            Self::shape_str(&[batch, seq_len, vocab_size])
        ));

        self.print("\nTake last position logits:");
        self.print(&format!(
            "  logits: {}",
            Self::shape_str(&[batch, vocab_size])
        ));
        self.print(&format!("  ({vocab_size} scores, one per possible token)"));

        self.print("\nSoftmax → probabilities:");
        self.print("  probs = softmax(logits)");

        if let Some(tokens) = top_tokens
            && self.config.show_numerical_examples
        {
            self.print("\nTop predictions:");
            for (token, prob) in tokens.iter().take(5) {
                let bar_len = (prob * 30.0) as usize;
                let bar = "█".repeat(bar_len);
                self.print(&format!("  {:12} {:5.1}%  {}", token, prob * 100.0, bar));
            }
        }
    }
}

/// A no-op X-Ray visualizer that doesn't print anything.
pub struct SilentXRay;

impl Default for SilentXRay {
    fn default() -> Self {
        Self
    }
}

impl XRay for SilentXRay {
    fn show_tokenization(&self, _text: &str, _token_ids: &[u32], _token_strs: &[String]) {}
    fn show_embedding_lookup(
        &self,
        _token_ids: &[u32],
        _vocab_size: usize,
        _embedding_dim: usize,
        _sample_embedding: Option<&[f32]>,
    ) {
    }
    fn show_qkv_projection(
        &self,
        _layer_idx: usize,
        _input_shape: (usize, usize, usize),
        _hidden_size: usize,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
    ) {
    }
    fn show_attention_scores(
        &self,
        _layer_idx: usize,
        _q_shape: (usize, usize, usize, usize),
        _k_shape: (usize, usize, usize, usize),
        _head_dim: usize,
        _sample_scores: Option<&[Vec<f32>]>,
    ) {
    }
    fn show_softmax(
        &self,
        _seq_len: usize,
        _sample_before: Option<&[Vec<f32>]>,
        _sample_after: Option<&[Vec<f32>]>,
    ) {
    }
    fn show_attention_output(
        &self,
        _scores_shape: (usize, usize, usize, usize),
        _v_shape: (usize, usize, usize, usize),
        _output_shape: (usize, usize, usize, usize),
    ) {
    }
    fn show_rope(&self, _head_dim: usize, _max_positions: usize, _base: f32) {}
    fn show_ffn(
        &self,
        _layer_idx: usize,
        _input_shape: (usize, usize, usize),
        _hidden_size: usize,
        _intermediate_size: usize,
    ) {
    }
    fn show_kv_cache_structure(
        &self,
        _num_layers: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _max_seq_len: usize,
        _current_seq_len: usize,
        _dtype_bytes: usize,
    ) {
    }
    fn show_lm_head(
        &self,
        _input_shape: (usize, usize, usize),
        _vocab_size: usize,
        _sample_logits: Option<&[f32]>,
        _sample_probs: Option<&[f32]>,
        _top_tokens: Option<&[(String, f32)]>,
    ) {
    }
}
