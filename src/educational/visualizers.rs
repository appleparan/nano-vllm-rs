//! ASCII art generators for educational visualizations.
//!
//! This module provides ASCII-based visualizations for understanding
//! LLM inference, including attention heatmaps, memory usage bars,
//! probability distributions, and architectural diagrams.

/// Unicode block characters for different intensities.
const BLOCKS: [char; 5] = [' ', '░', '▒', '▓', '█'];

/// Convert a value [0, max_val] to a block character.
fn intensity_to_block(value: f32, max_val: f32) -> char {
    if max_val == 0.0 {
        return BLOCKS[0];
    }
    let normalized = (value / max_val).clamp(0.0, 1.0);
    let idx = (normalized * (BLOCKS.len() - 1) as f32) as usize;
    BLOCKS[idx.min(BLOCKS.len() - 1)]
}

/// Generate ASCII heatmap of attention weights.
///
/// # Arguments
///
/// * `weights` - 2D slice of attention weights [query_len, key_len]
/// * `row_labels` - Labels for rows (query tokens)
/// * `col_labels` - Labels for columns (key tokens)
/// * `max_label_width` - Maximum width for token labels
/// * `title` - Optional title for the heatmap
///
/// # Returns
///
/// ASCII art string representing the attention heatmap.
pub fn attention_heatmap_ascii(
    weights: &[Vec<f32>],
    row_labels: &[String],
    col_labels: &[String],
    max_label_width: usize,
    title: Option<&str>,
) -> String {
    let mut lines = Vec::new();

    // Truncate labels
    let row_labels: Vec<String> = row_labels
        .iter()
        .map(|l| {
            let truncated: String = l.chars().take(max_label_width).collect();
            format!("{truncated:max_label_width$}")
        })
        .collect();
    let col_labels: Vec<String> = col_labels
        .iter()
        .map(|l| l.chars().take(3).collect())
        .collect();

    // Title
    if let Some(t) = title {
        lines.push(t.to_string());
        lines.push("─".repeat(max_label_width + 2 + col_labels.len() * 4));
    }

    // Header row (column labels)
    let mut header = " ".repeat(max_label_width + 2);
    for label in &col_labels {
        header.push_str(&format!(" {label} "));
    }
    lines.push(header);

    // Data rows
    for (i, row_label) in row_labels.iter().enumerate() {
        let mut row_str = format!("{row_label}  ");
        for j in 0..col_labels.len() {
            if i < weights.len() && j < weights[i].len() {
                let val = weights[i][j];
                let block = intensity_to_block(val, 1.0);
                row_str.push_str(&format!(" {block}{block}{block}"));
            } else {
                row_str.push_str("    ");
            }
        }
        lines.push(row_str);
    }

    lines.join("\n")
}

/// Generate memory usage bar.
///
/// # Arguments
///
/// * `used` - Used memory/blocks
/// * `total` - Total memory/blocks
/// * `width` - Bar width in characters
/// * `label` - Optional label prefix
/// * `show_percentage` - Show percentage at end
///
/// # Returns
///
/// ASCII bar string.
pub fn memory_bar(
    used: usize,
    total: usize,
    width: usize,
    label: &str,
    show_percentage: bool,
) -> String {
    let pct = if total == 0 {
        0.0
    } else {
        used as f32 / total as f32
    };

    let filled = (pct * width as f32) as usize;
    let empty = width - filled;

    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));

    let mut result = format!("{label}{bar}");
    if show_percentage {
        result.push_str(&format!(" {:.0}%", pct * 100.0));
    }

    result
}

/// Generate probability distribution bars.
///
/// # Arguments
///
/// * `probs` - List of probabilities
/// * `labels` - List of labels
/// * `max_width` - Maximum bar width
/// * `top_k` - Number of top items to show
///
/// # Returns
///
/// ASCII bars showing probability distribution.
pub fn probability_bars(
    probs: &[f32],
    labels: &[String],
    max_width: usize,
    top_k: usize,
) -> String {
    // Sort by probability
    let mut items: Vec<(f32, &String)> = probs.iter().copied().zip(labels.iter()).collect();
    items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let items: Vec<_> = items.into_iter().take(top_k).collect();

    let max_label_len = items.iter().map(|(_, l)| l.len()).max().unwrap_or(0);

    let mut lines = Vec::new();

    for (prob, label) in &items {
        let bar_len = (prob * max_width as f32) as usize;
        let bar = "█".repeat(bar_len);
        let padded_label = format!("{label:max_label_len$}");
        lines.push(format!(
            "│ {} │ {:5.1}%  {}",
            padded_label,
            prob * 100.0,
            bar
        ));
    }

    // Create table border
    let border_len = max_label_len + 2;
    let header = format!("┌{}┬{}┐", "─".repeat(border_len), "─".repeat(10));
    let footer = format!("└{}┴{}┘", "─".repeat(border_len), "─".repeat(10));

    format!("{}\n{}\n{}", header, lines.join("\n"), footer)
}

/// Display tokens in a box with optional highlighting.
///
/// # Arguments
///
/// * `tokens` - List of token strings
/// * `highlight_idx` - Index of token to highlight with cursor
/// * `prefix` - Optional prefix text
///
/// # Returns
///
/// Boxed token sequence.
pub fn token_sequence_box(tokens: &[String], highlight_idx: Option<usize>, prefix: &str) -> String {
    if tokens.is_empty() {
        return "╭──────────────────────────────────────────╮\n\
                │ (empty)                                  │\n\
                ╰──────────────────────────────────────────╯"
            .to_string();
    }

    // Build token string with brackets
    let token_strs: Vec<String> = tokens
        .iter()
        .enumerate()
        .map(|(i, tok)| {
            if highlight_idx == Some(i) {
                format!("[{tok}]")
            } else {
                tok.clone()
            }
        })
        .collect();

    let mut content = token_strs.join(" ");
    if highlight_idx == Some(tokens.len()) {
        content.push('█'); // Cursor at end
    }

    // Box it
    let width = 40.max(content.len() + prefix.len() + 4);
    let inner_width = width - 2;
    let content_padded = format!(
        "{}{:width$}",
        prefix,
        content,
        width = inner_width - prefix.len()
    );

    format!(
        "╭{}╮\n│ {} │\n╰{}╯",
        "─".repeat(inner_width),
        content_padded,
        "─".repeat(inner_width)
    )
}

/// Return the full model architecture ASCII diagram.
pub fn model_architecture_diagram() -> &'static str {
    r#"┌─────────────────────────────────────────────────────────────────┐
│                    LLaMA Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: "The capital of France is"                              │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐                                            │
│  │   Tokenizer     │  "The" → 450, "capital" → 7483, ...        │
│  └────────┬────────┘                                            │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │   Embedding     │  450 → [0.12, -0.34, 0.87, ...]  (4096-d)  │
│  │   (Lookup)      │                                            │
│  └────────┬────────┘                                            │
│           ▼                                                     │
│  ╔═════════════════╗                                            │
│  ║  Decoder Layer  ║ ×32                                        │
│  ║  ┌───────────┐  ║                                            │
│  ║  │ RMSNorm   │  ║                                            │
│  ║  └─────┬─────┘  ║                                            │
│  ║        ▼        ║                                            │
│  ║  ┌───────────┐  ║                                            │
│  ║  │ Attention │◄─╬──── KV Cache (stores past K,V)             │
│  ║  └─────┬─────┘  ║                                            │
│  ║        │+       ║  ◄── Residual connection                   │
│  ║        ▼        ║                                            │
│  ║  ┌───────────┐  ║                                            │
│  ║  │ RMSNorm   │  ║                                            │
│  ║  └─────┬─────┘  ║                                            │
│  ║        ▼        ║                                            │
│  ║  ┌───────────┐  ║                                            │
│  ║  │    FFN    │  ║  (SwiGLU: up_proj, gate, down_proj)        │
│  ║  └─────┬─────┘  ║                                            │
│  ║        │+       ║  ◄── Residual connection                   │
│  ╚════════╪════════╝                                            │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │   Final Norm    │                                            │
│  └────────┬────────┘                                            │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │    LM Head      │  Project to vocabulary (32000 tokens)      │
│  └────────┬────────┘                                            │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │    Softmax      │  → Probability distribution                │
│  └────────┬────────┘                                            │
│           ▼                                                     │
│      "Paris" (87%)                                              │
└─────────────────────────────────────────────────────────────────┘"#
}

/// Return the attention mechanism ASCII diagram.
pub fn attention_mechanism_diagram() -> &'static str {
    r#"┌─────────────────────────────────────────────────────────────────┐
│                  Self-Attention Explained                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: [The] [capital] [of] [France] [is]                      │
│                                                                 │
│  Step 1: Create Q, K, V for each token                          │
│  ─────────────────────────────────────                          │
│         ┌───┐     ┌───┐     ┌───┐                               │
│  The ──►│ Q │     │ K │     │ V │                               │
│         └───┘     └───┘     └───┘                               │
│         ┌───┐     ┌───┐     ┌───┐                               │
│  cap ──►│ Q │     │ K │     │ V │     Q = "What am I looking for?"
│         └───┘     └───┘     └───┘     K = "What do I contain?"  │
│         ┌───┐     ┌───┐     ┌───┐     V = "What info do I give?"│
│  of  ──►│ Q │     │ K │     │ V │                               │
│         └───┘     └───┘     └───┘                               │
│          ...       ...       ...                                │
│                                                                 │
│  Step 2: Compute attention scores (Q @ K^T)                     │
│  ──────────────────────────────────────────                     │
│                                                                 │
│            Keys:  The  cap   of  Fra   is                       │
│  Queries:      ┌────────────────────────────┐                   │
│     The        │ 0.8  0.1  0.0  0.0  0.0    │  Can only see     │
│     capital    │ 0.3  0.6  0.0  0.0  0.0    │  itself & past    │
│     of         │ 0.1  0.4  0.4  0.0  0.0    │  (causal mask!)   │
│     France     │ 0.0  0.5  0.1  0.3  0.0    │                   │
│     is         │ 0.0  0.2  0.0  0.7  0.1    │                   │
│                └────────────────────────────┘                   │
│                  ▲                                              │
│                  │ Higher = pays more attention                 │
│                                                                 │
│  Step 3: Weighted sum of Values                                 │
│  ─────────────────────────────                                  │
│     output["is"] = 0.2×V[cap] + 0.7×V[Fra] + 0.1×V[is]          │
│                                                                 │
│     "is" pays most attention to "France" - makes sense!         │
│     This helps it predict "Paris" next.                         │
└─────────────────────────────────────────────────────────────────┘"#
}

/// Return the KV cache explanation diagram.
pub fn kv_cache_diagram() -> &'static str {
    r#"┌─────────────────────────────────────────────────────────────────┐
│                     KV Cache: Why We Cache                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  WITHOUT CACHE (Quadratic complexity):                          │
│  ──────────────────────────────────────                         │
│  Step 1: Process [The]                    → Compute K,V for 1   │
│  Step 2: Process [The][capital]           → Compute K,V for 2   │
│  Step 3: Process [The][capital][of]       → Compute K,V for 3   │
│  Step 4: Process [The][capital][of][France] → Compute K,V for 4 │
│                                                                 │
│  Total K,V computations: 1+2+3+4 = 10 (O(n²) for n tokens!)     │
│                                                                 │
│  WITH CACHE (Linear complexity):                                │
│  ────────────────────────────────                               │
│                                                                 │
│  Prefill: Compute K,V for all prompt tokens at once             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Layer 1 Cache    │ K₁ │ K₂ │ K₃ │ K₄ │ K₅ │    │    │ │    │
│  │                  │ V₁ │ V₂ │ V₃ │ V₄ │ V₅ │    │    │ │    │
│  ├──────────────────┴────┴────┴────┴────┴────┴────┴────┴─┤    │
│  │ Layer 2 Cache    │ K₁ │ K₂ │ K₃ │ K₄ │ K₅ │    │    │ │    │
│  │                  │ V₁ │ V₂ │ V₃ │ V₄ │ V₅ │    │    │ │    │
│  ├──────────────────┴────┴────┴────┴────┴────┴────┴────┴─┤    │
│  │ ...              │    │    │    │    │    │    │    │ │    │
│  └───────────────────────────────────────────────────────────┘  │
│                                   ▲                             │
│                                   └── Empty slots for decode    │
│                                                                 │
│  Decode: Only compute K,V for the NEW token                     │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Layer 1 Cache    │ K₁ │ K₂ │ K₃ │ K₄ │ K₅ │ K₆ │    │ │    │
│  │                  │ V₁ │ V₂ │ V₃ │ V₄ │ V₅ │ V₆ │    │ │    │
│  └───────────────────────────────────────────────────────────┘  │
│                                            ▲                    │
│                                            └── New token added! │
│                                                                 │
│  Total K,V computations: 5 + 1 + 1 + 1 = 8 (O(n) linear!)       │
│                                                                 │
│  Memory: 5 tokens × 32 layers × 2 (K,V) × 4096 dims × 2 bytes   │
│        = 2.6 MB for just 5 tokens!                              │
└─────────────────────────────────────────────────────────────────┘"#
}

/// Return the PagedAttention explanation diagram.
pub fn paged_attention_diagram() -> &'static str {
    r#"┌─────────────────────────────────────────────────────────────────┐
│                    PagedAttention Explained                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PROBLEM: Sequences have variable lengths                       │
│  ─────────────────────────────────────────                      │
│                                                                 │
│  Seq A: "Hello"           (1 token)                             │
│  Seq B: "The quick brown fox jumps over the lazy dog" (9 tokens)│
│                                                                 │
│  Traditional: Pre-allocate max_length for each → WASTE!         │
│  ┌────────────────────────────────────────────────────┐         │
│  │ Seq A: █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (1/32 used) │        │
│  │ Seq B: █████████░░░░░░░░░░░░░░░░░░░░░░░░  (9/32 used) │        │
│  └────────────────────────────────────────────────────┘         │
│                                                                 │
│  SOLUTION: PagedAttention (like OS virtual memory)              │
│  ─────────────────────────────────────────────────              │
│                                                                 │
│  Block Pool (GPU memory):                                       │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐           │
│  │ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │ B8 │ B9 │           │
│  │████│████│░░░░│████│████│░░░░│████│░░░░│░░░░│░░░░│           │
│  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘           │
│    ▲    ▲         ▲    ▲         ▲                              │
│    │    │         │    │         │                              │
│    └────┼─────────┘    │         │                              │
│   Seq A │              │         │                              │
│  (1 block)             └─────────┘                              │
│                         Seq B                                   │
│                      (3 blocks)                                 │
│                                                                 │
│  Block Table (mapping):                                         │
│  ┌───────────┬───────────────────────┐                          │
│  │ Seq A     │ [0, 1]                │  Logical → Physical      │
│  │ Seq B     │ [3, 4, 6]             │                          │
│  └───────────┴───────────────────────┘                          │
│                                                                 │
│  Benefits:                                                      │
│  • No wasted memory (blocks allocated on-demand)                │
│  • Sequences can grow dynamically                               │
│  • Prefix sharing (same prefix = share blocks!)                 │
└─────────────────────────────────────────────────────────────────┘"#
}

/// Return the speculative decoding explanation diagram.
pub fn speculative_decoding_diagram() -> &'static str {
    r#"┌─────────────────────────────────────────────────────────────────┐
│                  Speculative Decoding Explained                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PROBLEM: Large models are slow at autoregressive decoding      │
│  ────────────────────────────────────────────────────           │
│                                                                 │
│  Traditional (one token at a time):                             │
│    Context: "The capital of France is"                          │
│    Step 1: [Large Model] → "Paris"    (slow: 100ms)             │
│    Step 2: [Large Model] → "."        (slow: 100ms)             │
│    Step 3: [Large Model] → "It"       (slow: 100ms)             │
│    Total: 300ms for 3 tokens                                    │
│                                                                 │
│  SOLUTION: Draft with small model, verify with large            │
│  ──────────────────────────────────────────────────             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ DRAFT PHASE (Small Model - Qwen3-0.6B)                  │    │
│  │                                                         │    │
│  │  Context → [Draft] → [Draft] → [Draft] → [Draft]        │    │
│  │            "Paris"   "."       "It"      "is"           │    │
│  │                                                         │    │
│  │  Fast: 10ms × 4 = 40ms total                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ VERIFY PHASE (Large Model - Qwen3-4B)                   │    │
│  │                                                         │    │
│  │  Process all 4 drafts in ONE forward pass               │    │
│  │                                                         │    │
│  │  Position:    0         1        2        3             │    │
│  │  Draft:      "Paris"   "."      "It"     "is"           │    │
│  │  P(target):   0.87     0.72     0.65     0.40           │    │
│  │  P(draft):    0.82     0.70     0.60     0.80           │    │
│  │  Accept?:     ✓        ✓        ✓        ✗ (reject)     │    │
│  │                                                         │    │
│  │  Time: 100ms (same as 1 token)                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Result: 3 tokens accepted + 1 bonus token sampled              │
│  Total time: 40ms + 100ms = 140ms for 4 tokens                  │
│  Speedup: 300ms → 140ms = 2.1x faster!                          │
│                                                                 │
│  KEY INSIGHT: Rejection sampling guarantees output matches      │
│  what the large model would have generated alone.               │
└─────────────────────────────────────────────────────────────────┘"#
}

/// Create an act header for narrator mode.
pub fn act_header(act_num: usize, title: &str) -> String {
    format!(
        "\n🎬 ACT {}: {}\n┌{}┐\n│ {:59} │\n└{}┘",
        act_num,
        title,
        "─".repeat(61),
        title,
        "─".repeat(61)
    )
}

/// Create an insight/tip box.
pub fn insight_box(text: &str, emoji: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut result = format!("\n  {emoji} \n");
    for line in lines {
        result.push_str(&format!("     {line}\n"));
    }
    result
}

/// Create a box around text.
pub fn box_text(text: &str, title: &str, width: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let content_width = width - 4;

    let mut result = Vec::new();

    // Top border with optional title
    if !title.is_empty() {
        let title_part = format!(" {title} ");
        let remaining = width - 2 - title_part.len();
        let left = remaining / 2;
        let right = remaining - left;
        result.push(format!(
            "┌{}{}{}┐",
            "─".repeat(left),
            title_part,
            "─".repeat(right)
        ));
    } else {
        result.push(format!("┌{}┐", "─".repeat(width - 2)));
    }

    // Content
    for line in lines {
        let mut remaining = line;
        while remaining.len() > content_width {
            let (chunk, rest) = remaining.split_at(content_width);
            result.push(format!("│ {chunk} │"));
            remaining = rest;
        }
        result.push(format!("│ {remaining:content_width$} │"));
    }

    // Bottom border
    result.push(format!("└{}┘", "─".repeat(width - 2)));

    result.join("\n")
}

/// Format tensor operation with shapes.
pub fn format_tensor_operation(
    op_name: &str,
    input_shapes: &[(&str, &[usize])],
    output_shape: (&str, &[usize]),
    formula: Option<&str>,
) -> String {
    let mut lines = vec![format!("  {}:", op_name)];

    // Inputs
    for (name, shape) in input_shapes {
        let shape_str = format!(
            "[{}]",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        lines.push(format!("    {name}: {shape_str}"));
    }

    // Arrow
    lines.push("       ↓".to_string());

    // Output
    let (out_name, out_shape) = output_shape;
    let out_shape_str = format!(
        "[{}]",
        out_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    lines.push(format!("    {out_name}: {out_shape_str}"));

    // Formula
    if let Some(f) = formula {
        lines.push(format!("    Formula: {f}"));
    }

    lines.join("\n")
}

/// Format memory statistics.
pub fn format_memory_stats(
    kv_cache_mb: f32,
    total_blocks: usize,
    used_blocks: usize,
    _block_size: usize,
    num_tokens: usize,
) -> String {
    let bar = memory_bar(used_blocks, total_blocks, 15, "", true);
    format!(
        "┌─ Memory ──────────┐\n\
         │ KV Cache: {kv_cache_mb:5.1}MB  │\n\
         │ {bar} │\n\
         │ Blocks: {used_blocks}/{total_blocks}   │\n\
         │ Tokens: {num_tokens}       │\n\
         └───────────────────┘"
    )
}

/// Layer-by-layer progress bar.
pub fn layer_progress_bar(current_layer: usize, total_layers: usize, width: usize) -> String {
    let pct = (current_layer + 1) as f32 / total_layers as f32;
    let filled = (pct * width as f32) as usize;
    let empty = width - filled;

    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));
    format!("Layer: {} {}/{}", bar, current_layer + 1, total_layers)
}
