//! Narrator mode - Real-time plain-English commentary during inference.
//!
//! The narrator provides educational explanations as each step of inference
//! happens, making it feel like watching surgery with an expert explaining
//! each step.

use super::explanations::{ExplanationLevel, get_explanation};
use super::visualizers::{
    act_header, attention_heatmap_ascii, box_text, insight_box, model_architecture_diagram,
    probability_bars,
};

/// Configuration for narrator output.
#[derive(Debug, Clone)]
pub struct NarratorConfig {
    /// Show model architecture at start.
    pub show_architecture: bool,
    /// Show attention heatmaps.
    pub show_attention_patterns: bool,
    /// Show top-k predictions.
    pub show_token_probabilities: bool,
    /// Show KV cache memory usage.
    pub show_memory_stats: bool,
    /// Show "WHY" insights.
    pub show_insights: bool,
    /// Verbosity level: 0=minimal, 1=normal, 2=detailed.
    pub verbosity: u8,
    /// Use ANSI colors (if terminal supports).
    pub color: bool,
}

impl Default for NarratorConfig {
    fn default() -> Self {
        Self {
            show_architecture: true,
            show_attention_patterns: true,
            show_token_probabilities: true,
            show_memory_stats: true,
            show_insights: true,
            verbosity: 2,
            color: true,
        }
    }
}

/// Model configuration for educational display.
#[derive(Debug, Clone, Default)]
pub struct ModelConfig {
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Hidden dimension size.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of KV heads (for GQA).
    pub num_kv_heads: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
}

/// Trait for narrator implementations.
pub trait Narrator {
    /// Called at the start of inference.
    fn on_start(&mut self, model_name: &str, model_config: &ModelConfig, prompt: &str);

    /// Called after tokenization.
    fn on_tokenization(&mut self, text: &str, token_ids: &[u32], token_strs: &[String]);

    /// Called after embedding lookup.
    fn on_embedding_lookup(&mut self, num_tokens: usize, embedding_dim: usize);

    /// Called at the start of prefill phase.
    fn on_prefill_start(&mut self, num_tokens: usize, num_layers: usize);

    /// Called at the start of each layer.
    fn on_layer_start(&mut self, layer_idx: usize, total_layers: usize);

    /// Called after attention is computed.
    fn on_attention_computed(
        &mut self,
        layer_idx: usize,
        attention_weights: Option<&[Vec<f32>]>,
        tokens: Option<&[String]>,
    );

    /// Called after KV cache is updated.
    fn on_kv_cache_update(&mut self, layer_idx: usize, num_tokens: usize, memory_bytes: usize);

    /// Called after prefill is complete.
    fn on_prefill_complete(&mut self, num_tokens: usize, total_memory_mb: f32);

    /// Called at the start of decode phase.
    fn on_decode_start(&mut self);

    /// Called after each decode step.
    fn on_decode_step(
        &mut self,
        step: usize,
        input_token: &str,
        top_predictions: &[(String, f32)],
        sampled_token: &str,
        sampled_prob: f32,
    );

    /// Called when generation is complete.
    fn on_generation_complete(
        &mut self,
        prompt: &str,
        generated_text: &str,
        num_generated_tokens: usize,
        total_time_ms: f32,
    );
}

/// Provides real-time educational commentary during inference.
pub struct InferenceNarrator {
    config: NarratorConfig,
    act_number: usize,
    current_prompt: String,
    model_config: Option<ModelConfig>,
    generated_tokens: Vec<String>,
    prompt_tokens: Vec<String>,
}

impl InferenceNarrator {
    /// Create a new inference narrator.
    pub fn new(config: NarratorConfig) -> Self {
        Self {
            config,
            act_number: 0,
            current_prompt: String::new(),
            model_config: None,
            generated_tokens: Vec::new(),
            prompt_tokens: Vec::new(),
        }
    }

    fn print(&self, text: &str) {
        println!("{text}");
    }

    fn section(&self, title: &str) {
        self.print(&format!("\n{}", "─".repeat(65)));
        self.print(&format!("  {title}"));
        self.print(&"─".repeat(65));
    }

    fn act(&mut self, title: &str) -> String {
        self.act_number += 1;
        act_header(self.act_number, title)
    }

    fn insight(&self, text: &str) {
        if self.config.show_insights {
            self.print(&insight_box(text, "💡"));
        }
    }
}

impl Default for InferenceNarrator {
    fn default() -> Self {
        Self::new(NarratorConfig::default())
    }
}

impl Narrator for InferenceNarrator {
    fn on_start(&mut self, model_name: &str, model_config: &ModelConfig, prompt: &str) {
        self.model_config = Some(model_config.clone());
        self.current_prompt = prompt.to_string();
        self.act_number = 0;
        self.generated_tokens.clear();

        // Welcome message
        self.print(&format!("\n{}", "═".repeat(65)));
        self.print("  🎓 INFERENCE ANATOMY - Educational Mode");
        self.print("  Understanding what happens inside an LLM");
        self.print(&"═".repeat(65));

        self.print(&format!("\n📖 Prompt: \"{prompt}\""));
        self.print(&format!("🤖 Model: {model_name}"));

        // Model stats
        self.print("\n📊 Model Architecture:");
        self.print(&format!(
            "   • {} transformer layers",
            model_config.num_hidden_layers
        ));
        self.print(&format!(
            "   • {}-dimensional embeddings",
            model_config.hidden_size
        ));
        self.print(&format!(
            "   • {} attention heads ({} KV heads)",
            model_config.num_attention_heads, model_config.num_kv_heads
        ));
        self.print(&format!(
            "   • {} token vocabulary",
            model_config.vocab_size
        ));

        if self.config.show_architecture && self.config.verbosity >= 2 {
            self.print(&format!("\n{}", model_architecture_diagram()));
        }
    }

    fn on_tokenization(&mut self, text: &str, token_ids: &[u32], token_strs: &[String]) {
        self.prompt_tokens = token_strs.to_vec();

        let header = self.act("TOKENIZATION");
        self.print(&header);

        self.print("Converting your prompt into numbers the model understands...");
        self.print(&format!("\n  \"{text}\""));
        self.print("       ↓ Tokenizer (BPE algorithm)");

        // Show token breakdown
        let token_display: String = token_strs
            .iter()
            .map(|t| format!("[{t}]"))
            .collect::<Vec<_>>()
            .join(" ");
        let id_display = format!("{token_ids:?}");
        self.print(&format!("  {token_display} → {id_display}"));

        self.print(&format!("\n  📊 {} tokens", token_ids.len()));

        if let Some(explanation) = get_explanation("tokenization") {
            self.insight(explanation.get(ExplanationLevel::Why));
        }
    }

    fn on_embedding_lookup(&mut self, num_tokens: usize, embedding_dim: usize) {
        if self.config.verbosity >= 2 {
            self.section("EMBEDDING LOOKUP");
            self.print("  Looking up learned vector for each token...");
            self.print(&format!(
                "  {num_tokens} tokens × {embedding_dim} dimensions"
            ));
            self.print(&format!("  = [{num_tokens}, {embedding_dim}] tensor"));

            if let Some(explanation) = get_explanation("embedding") {
                self.insight(explanation.get(ExplanationLevel::Why));
            }
        }
    }

    fn on_prefill_start(&mut self, num_tokens: usize, num_layers: usize) {
        let header = self.act("PREFILL PHASE (The 'Reading' Phase)");
        self.print(&header);

        self.print("The model reads your entire prompt at once...");
        self.print(&format!(
            "\n  📊 Processing {num_tokens} tokens through {num_layers} layers"
        ));
        self.print("  ✓ Parallel computation (all tokens at once)");
        self.print("  ✓ Building the KV cache");

        if let Some(explanation) = get_explanation("prefill") {
            self.insight(explanation.get(ExplanationLevel::Why));
        }
    }

    fn on_layer_start(&mut self, layer_idx: usize, total_layers: usize) {
        if self.config.verbosity >= 2 {
            let filled = layer_idx + 1;
            let empty = total_layers - layer_idx - 1;
            let progress = format!("{}{}", "█".repeat(filled), "░".repeat(empty));
            print!(
                "  Layer {}/{}: [{}]\r",
                layer_idx + 1,
                total_layers,
                progress
            );
        }
    }

    fn on_attention_computed(
        &mut self,
        layer_idx: usize,
        attention_weights: Option<&[Vec<f32>]>,
        tokens: Option<&[String]>,
    ) {
        if self.config.show_attention_patterns
            && self.config.verbosity >= 2
            && layer_idx == 0 // Only show first layer to avoid spam
            && let (Some(weights), Some(toks)) = (attention_weights, tokens)
        {
            self.print(&format!(
                "\n  🧠 Layer {}: Attention Pattern (who looks at whom)",
                layer_idx + 1
            ));
            let heatmap = attention_heatmap_ascii(weights, toks, toks, 8, None);
            self.print(&heatmap);
            self.insight("Each row shows what that token attends to.\nCausal mask: tokens can only see past, not future.");
        }
    }

    fn on_kv_cache_update(&mut self, layer_idx: usize, num_tokens: usize, memory_bytes: usize) {
        if self.config.show_memory_stats && self.config.verbosity >= 2 && layer_idx == 0 {
            let memory_mb = memory_bytes as f32 / (1024.0 * 1024.0);
            self.print(&format!(
                "\n  💾 KV Cache: {num_tokens} tokens, {memory_mb:.2}MB"
            ));
        }
    }

    fn on_prefill_complete(&mut self, num_tokens: usize, total_memory_mb: f32) {
        self.print("\n  ✅ Prefill complete!");
        self.print(&format!("     • {num_tokens} tokens processed"));
        self.print(&format!("     • KV cache: {total_memory_mb:.1}MB"));

        if let Some(explanation) = get_explanation("kv_cache") {
            self.insight(explanation.get(ExplanationLevel::Why));
        }
    }

    fn on_decode_start(&mut self) {
        let header = self.act("DECODE PHASE (Token-by-Token Generation)");
        self.print(&header);
        self.print("Now generating one token at a time...");

        if let Some(explanation) = get_explanation("decode") {
            self.insight(explanation.get(ExplanationLevel::Why));
        }
    }

    fn on_decode_step(
        &mut self,
        step: usize,
        input_token: &str,
        top_predictions: &[(String, f32)],
        sampled_token: &str,
        sampled_prob: f32,
    ) {
        self.generated_tokens.push(sampled_token.to_string());

        self.print(&format!(
            "\n  Step {}: Predicting token #{}",
            step,
            self.prompt_tokens.len() + step
        ));
        self.print(&format!(
            "  ├── Input: Previous tokens (via KV cache) + \"{input_token}\""
        ));

        if self.config.show_token_probabilities {
            self.print("  ├── Output: Probability distribution over vocabulary");
            self.print("  │");

            // Show top predictions
            let probs: Vec<f32> = top_predictions.iter().map(|(_, p)| *p).collect();
            let labels: Vec<String> = top_predictions.iter().map(|(t, _)| t.clone()).collect();
            self.print("  │   Top 5 predictions:");
            let bars = probability_bars(&probs, &labels, 25, 5);
            for line in bars.lines() {
                self.print(&format!("  │   {line}"));
            }
        }

        self.print("  │");
        self.print(&format!(
            "  └── 🎲 Sampled: \"{}\" ({:.1}%)",
            sampled_token,
            sampled_prob * 100.0
        ));

        // Show current generated text
        let all_tokens: Vec<&str> = self
            .prompt_tokens
            .iter()
            .chain(self.generated_tokens.iter())
            .map(|s| s.as_str())
            .collect();
        self.print(&format!("\n  Current output: {}", all_tokens.join(" ")));
    }

    fn on_generation_complete(
        &mut self,
        _prompt: &str,
        generated_text: &str,
        num_generated_tokens: usize,
        total_time_ms: f32,
    ) {
        let header = self.act("GENERATION COMPLETE");
        self.print(&header);

        self.print("\n📝 Final Output:");
        self.print(&box_text(generated_text, "Generated Text", 65));

        let tokens_per_sec = if total_time_ms > 0.0 {
            num_generated_tokens as f32 / (total_time_ms / 1000.0)
        } else {
            0.0
        };
        self.print("\n📊 Statistics:");
        self.print(&format!("   • Generated {num_generated_tokens} tokens"));
        self.print(&format!("   • Time: {total_time_ms:.1}ms"));
        self.print(&format!("   • Speed: {tokens_per_sec:.1} tokens/sec"));

        self.print(&format!("\n{}", "═".repeat(65)));
        self.print("  🎓 End of Inference Anatomy");
        self.print(&format!("{}\n", "═".repeat(65)));
    }
}

/// A no-op narrator that doesn't print anything.
pub struct SilentNarrator;

impl Default for SilentNarrator {
    fn default() -> Self {
        Self
    }
}

impl Narrator for SilentNarrator {
    fn on_start(&mut self, _model_name: &str, _model_config: &ModelConfig, _prompt: &str) {}
    fn on_tokenization(&mut self, _text: &str, _token_ids: &[u32], _token_strs: &[String]) {}
    fn on_embedding_lookup(&mut self, _num_tokens: usize, _embedding_dim: usize) {}
    fn on_prefill_start(&mut self, _num_tokens: usize, _num_layers: usize) {}
    fn on_layer_start(&mut self, _layer_idx: usize, _total_layers: usize) {}
    fn on_attention_computed(
        &mut self,
        _layer_idx: usize,
        _attention_weights: Option<&[Vec<f32>]>,
        _tokens: Option<&[String]>,
    ) {
    }
    fn on_kv_cache_update(&mut self, _layer_idx: usize, _num_tokens: usize, _memory_bytes: usize) {}
    fn on_prefill_complete(&mut self, _num_tokens: usize, _total_memory_mb: f32) {}
    fn on_decode_start(&mut self) {}
    fn on_decode_step(
        &mut self,
        _step: usize,
        _input_token: &str,
        _top_predictions: &[(String, f32)],
        _sampled_token: &str,
        _sampled_prob: f32,
    ) {
    }
    fn on_generation_complete(
        &mut self,
        _prompt: &str,
        _generated_text: &str,
        _num_generated_tokens: usize,
        _total_time_ms: f32,
    ) {
    }
}
