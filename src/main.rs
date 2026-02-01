//! nano-vllm CLI - A minimalistic LLM inference engine.
//!
//! ## Usage
//!
//! ```bash
//! # Basic usage
//! nano-vllm --model Qwen/Qwen3-0.6B --prompt "Hello, world!"
//!
//! # Multiple prompts with priority
//! nano-vllm -m Qwen/Qwen3-0.6B -p "First prompt" -p "Second prompt" --priority 5
//!
//! # With sampling parameters
//! nano-vllm -m Qwen/Qwen3-0.6B -p "Tell me a story" --temperature 0.8 --top-k 50 --top-p 0.9
//! ```

use std::time::Instant;

use candle_core::{DType, Device};
use clap::Parser;
use tracing::{info, warn};

use nano_vllm::{
    download_model, load_config, load_safetensors, EngineConfig, GenerationRequest, LLMEngine,
    Qwen3ForCausalLM,
};

/// nano-vllm: A minimalistic LLM inference engine
#[derive(Parser, Debug)]
#[command(name = "nano-vllm")]
#[command(version, about, long_about = None)]
struct Args {
    /// Model path or HuggingFace model ID (e.g., "Qwen/Qwen3-0.6B")
    #[arg(short, long)]
    model: String,

    /// Input prompt(s) - can be specified multiple times
    #[arg(short, long, required = true)]
    prompt: Vec<String>,

    /// Maximum tokens to generate per prompt
    #[arg(long, default_value = "256")]
    max_tokens: usize,

    /// Sampling temperature (0.0 = greedy, higher = more random)
    #[arg(short, long, default_value = "1.0")]
    temperature: f32,

    /// Top-k sampling (0 = disabled)
    #[arg(long, default_value = "0")]
    top_k: usize,

    /// Top-p (nucleus) sampling (1.0 = disabled)
    #[arg(long, default_value = "1.0")]
    top_p: f32,

    /// Request priority (higher = more important)
    #[arg(long, default_value = "0")]
    priority: i32,

    /// HuggingFace revision (branch, tag, or commit)
    #[arg(long, default_value = "main")]
    revision: String,

    /// Random seed for reproducible sampling
    #[arg(long)]
    seed: Option<u64>,

    /// Use CUDA device if available
    #[arg(long)]
    cuda: bool,

    /// Block size for PagedAttention
    #[arg(long, default_value = "16")]
    block_size: usize,

    /// Number of KV cache blocks
    #[arg(long, default_value = "512")]
    num_blocks: usize,
}

fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let args = Args::parse();

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║            nano-vllm v{}                            ║", env!("CARGO_PKG_VERSION"));
    println!("║      A minimalistic LLM inference engine in Rust          ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // Determine device
    let device = if args.cuda {
        Device::new_cuda(0).unwrap_or_else(|e| {
            warn!("CUDA not available: {e}, falling back to CPU");
            Device::Cpu
        })
    } else {
        Device::Cpu
    };
    info!("Using device: {:?}", device);

    // Download model from HuggingFace
    info!("Downloading model: {} (revision: {})", args.model, args.revision);
    let start = Instant::now();
    let model_files = download_model(&args.model, &args.revision)?;
    info!("Model downloaded in {:?}", start.elapsed());

    // Load config
    info!("Loading configuration...");
    let config = load_config(&model_files.config)?;
    info!(
        "Model config: {} layers, {} heads, hidden_size={}",
        config.num_hidden_layers, config.num_attention_heads, config.hidden_size
    );

    // Load weights
    info!("Loading model weights...");
    let start = Instant::now();
    let dtype = DType::F32; // Use F16 for GPU, F32 for CPU
    let vb = load_safetensors(&model_files.weights, dtype, &device)?;
    info!("Weights loaded in {:?}", start.elapsed());

    // Create model
    info!("Creating Qwen3 model...");
    let start = Instant::now();
    let model = Qwen3ForCausalLM::new(&config, vb)?;
    info!("Model created in {:?}", start.elapsed());

    // Load tokenizer
    info!("Loading tokenizer...");
    let tokenizer = tokenizers::Tokenizer::from_file(&model_files.tokenizer)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

    // Create engine config
    let engine_config = EngineConfig {
        max_num_seqs: 256,
        max_prefill_tokens: 4096,
        block_size: args.block_size,
        num_blocks: args.num_blocks,
        use_paged_attention: true,
        enable_prefix_caching: true,
        enable_preemption: false,
    };

    // Create LLM engine
    info!("Creating LLM engine...");
    let mut engine = LLMEngine::new(model, config.clone(), tokenizer, engine_config)?;

    // Add requests
    info!("Adding {} prompt(s)...", args.prompt.len());
    for (i, prompt) in args.prompt.iter().enumerate() {
        let request = GenerationRequest::new(prompt.clone())
            .max_tokens(args.max_tokens)
            .temperature(args.temperature)
            .top_k(args.top_k)
            .top_p(args.top_p)
            .priority(args.priority);

        let request_id = engine.add_request(request)?;
        info!("Request {} added with id={}", i + 1, request_id);
    }

    // Generate
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("                        GENERATION                             ");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let start = Instant::now();
    let outputs = engine.generate()?;
    let elapsed = start.elapsed();

    // Print outputs
    for output in &outputs {
        println!("┌─────────────────────────────────────────────────────────────┐");
        println!("│ Request ID: {:3}                                            │", output.request_id);
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ Prompt: {}", truncate_str(&output.prompt, 50));
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ Output:");
        for line in output.output_text.lines() {
            println!("│   {}", line);
        }
        println!("├─────────────────────────────────────────────────────────────┤");
        println!(
            "│ Tokens: {} total, {} generated  Finish: {:?}",
            output.total_tokens,
            output.output_tokens.len(),
            output.finish_reason.as_ref().map(|r| format!("{:?}", r)).unwrap_or_default()
        );
        println!("└─────────────────────────────────────────────────────────────┘");
        println!();
    }

    // Print summary
    let total_tokens: usize = outputs.iter().map(|o| o.output_tokens.len()).sum();
    let tokens_per_sec = total_tokens as f64 / elapsed.as_secs_f64();

    println!("═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Prompts processed: {}", outputs.len());
    println!("  Total tokens generated: {}", total_tokens);
    println!("  Time: {:.2?}", elapsed);
    println!("  Throughput: {:.2} tokens/sec", tokens_per_sec);
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

/// Truncate a string to a maximum length, adding "..." if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}
