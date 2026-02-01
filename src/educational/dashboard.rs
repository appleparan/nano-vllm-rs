//! Dashboard mode - Rich terminal UI for live inference visualization.
//!
//! Provides a real-time dashboard showing inference progress, memory usage,
//! attention patterns, and generated tokens.

use std::io::{self, Write};
use std::time::Instant;

/// Current state of the dashboard.
#[derive(Debug, Clone, Default)]
pub struct DashboardState {
    // Model info
    /// Model name/path.
    pub model_name: String,
    /// Device (cuda/cpu).
    pub device: String,
    /// Data type.
    pub dtype: String,
    /// Number of layers.
    pub num_layers: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,

    // Progress
    /// Current inference phase.
    pub phase: Phase,
    /// Current layer being processed.
    pub current_layer: usize,
    /// Total number of layers.
    pub total_layers: usize,
    /// Prefilled tokens so far.
    pub prefill_tokens: usize,
    /// Total tokens to prefill.
    pub total_prefill_tokens: usize,
    /// Current decode step.
    pub decode_step: usize,
    /// Maximum decode steps.
    pub max_decode_steps: usize,

    // Memory
    /// KV cache memory in MB.
    pub kv_cache_mb: f32,
    /// Maximum KV cache memory in MB.
    pub kv_cache_max_mb: f32,
    /// Number of used blocks.
    pub used_blocks: usize,
    /// Total number of blocks.
    pub total_blocks: usize,

    // Throughput
    /// Prefill tokens per second.
    pub prefill_tokens_per_sec: f32,
    /// Decode tokens per second.
    pub decode_tokens_per_sec: f32,
    /// Total tokens per second.
    pub total_tokens_per_sec: f32,

    // Tokens
    /// Prompt token strings.
    pub prompt_tokens: Vec<String>,
    /// Generated token strings.
    pub generated_tokens: Vec<String>,
    /// Top predictions with probabilities.
    pub top_predictions: Vec<(String, f32)>,
}

/// Inference phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Phase {
    /// Idle state.
    #[default]
    Idle,
    /// Loading model.
    Loading,
    /// Prefill phase.
    Prefill,
    /// Decode phase.
    Decode,
    /// Generation complete.
    Complete,
}

impl Phase {
    /// Get emoji for the phase.
    pub fn emoji(&self) -> &'static str {
        match self {
            Phase::Idle => "⏸️",
            Phase::Loading => "📥",
            Phase::Prefill => "📖",
            Phase::Decode => "✏️",
            Phase::Complete => "✅",
        }
    }

    /// Get name for the phase.
    pub fn name(&self) -> &'static str {
        match self {
            Phase::Idle => "IDLE",
            Phase::Loading => "LOADING",
            Phase::Prefill => "PREFILL",
            Phase::Decode => "DECODE",
            Phase::Complete => "COMPLETE",
        }
    }
}

/// Trait for dashboard implementations.
#[allow(clippy::too_many_arguments)]
pub trait Dashboard {
    /// Start the live dashboard display.
    fn start(&mut self);

    /// Stop the live dashboard display.
    fn stop(&mut self);

    /// Force a dashboard refresh.
    fn update(&mut self);

    /// Set model information.
    fn set_model_info(
        &mut self,
        model_name: &str,
        device: &str,
        dtype: &str,
        num_layers: usize,
        hidden_size: usize,
        vocab_size: usize,
        num_blocks: usize,
        kv_cache_max_mb: f32,
    );

    /// Set current inference phase.
    fn set_phase(&mut self, phase: Phase);

    /// Update progress indicators.
    fn update_progress(
        &mut self,
        current_layer: Option<usize>,
        prefill_tokens: Option<usize>,
        total_prefill_tokens: Option<usize>,
        decode_step: Option<usize>,
        max_decode_steps: Option<usize>,
    );

    /// Update memory statistics.
    fn update_memory(&mut self, kv_cache_mb: Option<f32>, used_blocks: Option<usize>);

    /// Update token information.
    fn update_tokens(
        &mut self,
        prompt_tokens: Option<Vec<String>>,
        generated_tokens: Option<Vec<String>>,
        top_predictions: Option<Vec<(String, f32)>>,
    );
}

/// Real-time terminal dashboard for inference visualization.
pub struct InferenceDashboard {
    state: DashboardState,
    start_time: Option<Instant>,
    prefill_start_time: Option<Instant>,
    decode_start_time: Option<Instant>,
    is_running: bool,
}

impl InferenceDashboard {
    /// Create a new inference dashboard.
    pub fn new() -> Self {
        Self {
            state: DashboardState::default(),
            start_time: None,
            prefill_start_time: None,
            decode_start_time: None,
            is_running: false,
        }
    }

    /// Get the current state.
    pub fn state(&self) -> &DashboardState {
        &self.state
    }

    fn render(&self) {
        let mut output = String::new();

        // Clear screen and move cursor to top
        output.push_str("\x1B[2J\x1B[H");

        // Header
        output.push_str(&format!("╔{}╗\n", "═".repeat(62)));
        output.push_str(&format!("║  {:^58}  ║\n", "nano-vllm Inference Dashboard"));
        output.push_str(&format!("╚{}╝\n\n", "═".repeat(62)));

        // Model info
        let model_name = if self.state.model_name.len() > 40 {
            format!(
                "...{}",
                &self.state.model_name[self.state.model_name.len() - 37..]
            )
        } else {
            self.state.model_name.clone()
        };
        output.push_str(&format!("Model:  {model_name}\n"));
        output.push_str(&format!(
            "Device: {}  Dtype: {}\n\n",
            self.state.device, self.state.dtype
        ));

        // Progress section
        output.push_str("┌─ Progress ─────────────────────────────────────────────────┐\n");

        // Phase
        output.push_str(&format!(
            "│ Phase:   {} {:54} │\n",
            self.state.phase.emoji(),
            self.state.phase.name()
        ));

        // Prefill progress
        let prefill_pct = if self.state.total_prefill_tokens > 0 {
            self.state.prefill_tokens as f32 / self.state.total_prefill_tokens as f32
        } else {
            0.0
        };
        let prefill_filled = (prefill_pct * 40.0) as usize;
        let prefill_bar = format!(
            "{}{}",
            "█".repeat(prefill_filled),
            "░".repeat(40 - prefill_filled)
        );
        output.push_str(&format!(
            "│ Prefill: {} {:5.0}%                      │\n",
            prefill_bar,
            prefill_pct * 100.0
        ));

        // Layer progress
        let layer_pct = if self.state.total_layers > 0 {
            (self.state.current_layer + 1) as f32 / self.state.total_layers as f32
        } else {
            0.0
        };
        let layer_filled = (layer_pct * 40.0) as usize;
        let layer_bar = format!(
            "{}{}",
            "█".repeat(layer_filled),
            "░".repeat(40 - layer_filled)
        );
        output.push_str(&format!(
            "│ Layer:   {} {}/{}                    │\n",
            layer_bar,
            self.state.current_layer + 1,
            self.state.total_layers
        ));

        // Decode progress
        let decode_pct = if self.state.max_decode_steps > 0 {
            self.state.decode_step as f32 / self.state.max_decode_steps as f32
        } else {
            0.0
        };
        let decode_filled = (decode_pct * 40.0) as usize;
        let decode_bar = format!(
            "{}{}",
            "█".repeat(decode_filled),
            "░".repeat(40 - decode_filled)
        );
        output.push_str(&format!(
            "│ Decode:  {} {}/{}                    │\n",
            decode_bar, self.state.decode_step, self.state.max_decode_steps
        ));

        output.push_str("└────────────────────────────────────────────────────────────┘\n\n");

        // Stats row (memory + throughput)
        output.push_str("┌─ Memory ────────────────┐  ┌─ Throughput ────────────────┐\n");
        output.push_str(&format!(
            "│ KV Cache: {:6.1}MB      │  │ Prefill: {:8.0} tok/s     │\n",
            self.state.kv_cache_mb, self.state.prefill_tokens_per_sec
        ));

        if self.state.total_blocks > 0 {
            let block_pct = self.state.used_blocks as f32 / self.state.total_blocks as f32;
            let block_filled = (block_pct * 10.0) as usize;
            let block_bar = format!(
                "{}{}",
                "█".repeat(block_filled),
                "░".repeat(10 - block_filled)
            );
            output.push_str(&format!(
                "│ {} {:3.0}%          │  │ Decode:  {:8.1} tok/s     │\n",
                block_bar,
                block_pct * 100.0,
                self.state.decode_tokens_per_sec
            ));
            output.push_str(&format!(
                "│ Blocks: {}/{}       │  │ Total:   {:8.0} tok/s     │\n",
                self.state.used_blocks, self.state.total_blocks, self.state.total_tokens_per_sec
            ));
        } else {
            output.push_str(&format!(
                "│                        │  │ Decode:  {:8.1} tok/s     │\n",
                self.state.decode_tokens_per_sec
            ));
            output.push_str(&format!(
                "│                        │  │ Total:   {:8.0} tok/s     │\n",
                self.state.total_tokens_per_sec
            ));
        }
        output.push_str("└────────────────────────┘  └─────────────────────────────┘\n\n");

        // Generated text
        if !self.state.prompt_tokens.is_empty() || !self.state.generated_tokens.is_empty() {
            output.push_str("┌─ Generated Text ────────────────────────────────────────────┐\n");

            let prompt: String = self.state.prompt_tokens.join(" ");
            let generated: String = self.state.generated_tokens.join(" ");

            // Truncate if too long
            let max_len = 55;
            let display_prompt = if prompt.len() > max_len / 2 {
                format!("...{}", &prompt[prompt.len() - max_len / 2..])
            } else {
                prompt
            };
            let display_generated = if generated.len() > max_len / 2 {
                format!("{}...", &generated[..max_len / 2])
            } else {
                generated
            };

            output.push_str(&format!(
                "│ {:60} │\n",
                format!("{} {}", display_prompt, display_generated)
            ));
            output.push_str("└────────────────────────────────────────────────────────────┘\n\n");
        }

        // Top predictions
        if !self.state.top_predictions.is_empty() {
            output.push_str("┌─ Top Predictions ───────────────────────────────────────────┐\n");
            for (token, prob) in self.state.top_predictions.iter().take(5) {
                let bar_len = (prob * 30.0) as usize;
                let bar = "█".repeat(bar_len);
                output.push_str(&format!(
                    "│ {:12} {:5.1}% {:30}      │\n",
                    token,
                    prob * 100.0,
                    bar
                ));
            }
            output.push_str("└────────────────────────────────────────────────────────────┘\n");
        }

        // Print the output
        print!("{output}");
        let _ = io::stdout().flush();
    }

    fn update_throughput(&mut self) {
        let now = Instant::now();

        // Prefill throughput
        if let Some(prefill_start) = self.prefill_start_time
            && self.state.prefill_tokens > 0
        {
            let elapsed = now.duration_since(prefill_start).as_secs_f32();
            if elapsed > 0.0 {
                self.state.prefill_tokens_per_sec = self.state.prefill_tokens as f32 / elapsed;
            }
        }

        // Decode throughput
        if let Some(decode_start) = self.decode_start_time
            && self.state.decode_step > 0
        {
            let elapsed = now.duration_since(decode_start).as_secs_f32();
            if elapsed > 0.0 {
                self.state.decode_tokens_per_sec = self.state.decode_step as f32 / elapsed;
            }
        }

        // Total throughput
        if let Some(start) = self.start_time {
            let total_tokens = self.state.prefill_tokens + self.state.decode_step;
            let elapsed = now.duration_since(start).as_secs_f32();
            if elapsed > 0.0 {
                self.state.total_tokens_per_sec = total_tokens as f32 / elapsed;
            }
        }
    }
}

impl Default for InferenceDashboard {
    fn default() -> Self {
        Self::new()
    }
}

impl Dashboard for InferenceDashboard {
    fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.is_running = true;
        self.render();
    }

    fn stop(&mut self) {
        self.is_running = false;
        println!(); // Add newline after dashboard
    }

    fn update(&mut self) {
        if self.is_running {
            self.update_throughput();
            self.render();
        }
    }

    fn set_model_info(
        &mut self,
        model_name: &str,
        device: &str,
        dtype: &str,
        num_layers: usize,
        hidden_size: usize,
        vocab_size: usize,
        num_blocks: usize,
        kv_cache_max_mb: f32,
    ) {
        self.state.model_name = model_name.to_string();
        self.state.device = device.to_string();
        self.state.dtype = dtype.to_string();
        self.state.num_layers = num_layers;
        self.state.total_layers = num_layers;
        self.state.hidden_size = hidden_size;
        self.state.vocab_size = vocab_size;
        self.state.total_blocks = num_blocks;
        self.state.kv_cache_max_mb = kv_cache_max_mb;
        self.update();
    }

    fn set_phase(&mut self, phase: Phase) {
        self.state.phase = phase;

        if phase == Phase::Prefill {
            self.prefill_start_time = Some(Instant::now());
        } else if phase == Phase::Decode {
            self.decode_start_time = Some(Instant::now());
        }

        self.update();
    }

    fn update_progress(
        &mut self,
        current_layer: Option<usize>,
        prefill_tokens: Option<usize>,
        total_prefill_tokens: Option<usize>,
        decode_step: Option<usize>,
        max_decode_steps: Option<usize>,
    ) {
        if let Some(layer) = current_layer {
            self.state.current_layer = layer;
        }
        if let Some(tokens) = prefill_tokens {
            self.state.prefill_tokens = tokens;
        }
        if let Some(total) = total_prefill_tokens {
            self.state.total_prefill_tokens = total;
        }
        if let Some(step) = decode_step {
            self.state.decode_step = step;
        }
        if let Some(max) = max_decode_steps {
            self.state.max_decode_steps = max;
        }
        self.update();
    }

    fn update_memory(&mut self, kv_cache_mb: Option<f32>, used_blocks: Option<usize>) {
        if let Some(mb) = kv_cache_mb {
            self.state.kv_cache_mb = mb;
        }
        if let Some(blocks) = used_blocks {
            self.state.used_blocks = blocks;
        }
        self.update();
    }

    fn update_tokens(
        &mut self,
        prompt_tokens: Option<Vec<String>>,
        generated_tokens: Option<Vec<String>>,
        top_predictions: Option<Vec<(String, f32)>>,
    ) {
        if let Some(tokens) = prompt_tokens {
            self.state.prompt_tokens = tokens;
        }
        if let Some(tokens) = generated_tokens {
            self.state.generated_tokens = tokens;
        }
        if let Some(predictions) = top_predictions {
            self.state.top_predictions = predictions;
        }
        self.update();
    }
}

/// A no-op dashboard that doesn't display anything.
pub struct SilentDashboard;

impl Default for SilentDashboard {
    fn default() -> Self {
        Self
    }
}

impl Dashboard for SilentDashboard {
    fn start(&mut self) {}
    fn stop(&mut self) {}
    fn update(&mut self) {}
    fn set_model_info(
        &mut self,
        _model_name: &str,
        _device: &str,
        _dtype: &str,
        _num_layers: usize,
        _hidden_size: usize,
        _vocab_size: usize,
        _num_blocks: usize,
        _kv_cache_max_mb: f32,
    ) {
    }
    fn set_phase(&mut self, _phase: Phase) {}
    fn update_progress(
        &mut self,
        _current_layer: Option<usize>,
        _prefill_tokens: Option<usize>,
        _total_prefill_tokens: Option<usize>,
        _decode_step: Option<usize>,
        _max_decode_steps: Option<usize>,
    ) {
    }
    fn update_memory(&mut self, _kv_cache_mb: Option<f32>, _used_blocks: Option<usize>) {}
    fn update_tokens(
        &mut self,
        _prompt_tokens: Option<Vec<String>>,
        _generated_tokens: Option<Vec<String>>,
        _top_predictions: Option<Vec<(String, f32)>>,
    ) {
    }
}
