use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "nano-vllm")]
#[command(about = "A minimalistic LLM inference engine")]
struct Args {
    /// Model path or HuggingFace model ID
    #[arg(short, long)]
    model: Option<String>,

    /// Input prompt
    #[arg(short, long)]
    prompt: Option<String>,

    /// Maximum tokens to generate
    #[arg(long, default_value = "256")]
    max_tokens: usize,
}

fn main() {
    let args = Args::parse();

    println!("nano-vllm v{}", env!("CARGO_PKG_VERSION"));

    if let Some(model) = &args.model {
        println!("Model: {}", model);
    }

    if let Some(prompt) = &args.prompt {
        println!("Prompt: {}", prompt);
    }

    println!("Max tokens: {}", args.max_tokens);
    println!("\n[Engine not yet implemented - see IMPLEMENTATION_PLAN.md]");
}
