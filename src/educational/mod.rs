//! Educational module for nano-vllm inference visualization.
//!
//! This module provides four educational modes to help users understand
//! what happens inside an LLM during inference:
//!
//! 1. **Narrator Mode**: Real-time plain-English commentary
//! 2. **X-Ray Mode**: Mathematical/tensor visualizations
//! 3. **Dashboard Mode**: Rich terminal UI with live updates
//! 4. **Tutorial Mode**: Interactive step-by-step learning
//!
//! ## Usage
//!
//! ```rust,ignore
//! use nano_vllm::educational::{EducationalConfig, InferenceNarrator};
//!
//! let config = EducationalConfig::default().with_narrate(true);
//! let narrator = InferenceNarrator::new(NarratorConfig::default());
//! narrator.on_start("Qwen/Qwen3-0.6B", &model_config, "Hello world");
//! ```

pub mod dashboard;
pub mod explanations;
pub mod narrator;
pub mod tutorial;
pub mod visualizers;
pub mod xray;

// Re-export main types
pub use dashboard::{DashboardState, InferenceDashboard, SilentDashboard};
pub use explanations::{Explanation, get_all_topics, get_explanation};
pub use narrator::{InferenceNarrator, NarratorConfig, SilentNarrator};
pub use tutorial::{InteractiveTutorial, TutorialChapter};
pub use visualizers::{
    attention_heatmap_ascii, attention_mechanism_diagram, kv_cache_diagram, memory_bar,
    model_architecture_diagram, paged_attention_diagram, probability_bars, token_sequence_box,
};
pub use xray::{SilentXRay, XRayConfig, XRayVisualizer};

/// Configuration for educational modes.
#[derive(Debug, Clone, Default)]
pub struct EducationalConfig {
    /// Enable narrator mode (real-time plain-English explanations).
    pub narrate: bool,
    /// Enable X-Ray mode (tensor/math visualizations).
    pub xray: bool,
    /// Enable dashboard mode (live terminal UI).
    pub dashboard: bool,
    /// Enable tutorial mode (interactive learning).
    pub tutorial: bool,
}

impl EducationalConfig {
    /// Create a new educational config with narrator mode enabled.
    pub fn with_narrate(mut self, enable: bool) -> Self {
        self.narrate = enable;
        self
    }

    /// Create a new educational config with X-Ray mode enabled.
    pub fn with_xray(mut self, enable: bool) -> Self {
        self.xray = enable;
        self
    }

    /// Create a new educational config with dashboard mode enabled.
    pub fn with_dashboard(mut self, enable: bool) -> Self {
        self.dashboard = enable;
        self
    }

    /// Create a new educational config with tutorial mode enabled.
    pub fn with_tutorial(mut self, enable: bool) -> Self {
        self.tutorial = enable;
        self
    }

    /// Check if any educational mode is enabled.
    pub fn is_enabled(&self) -> bool {
        self.narrate || self.xray || self.dashboard || self.tutorial
    }
}
