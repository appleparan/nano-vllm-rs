//! Interactive Tutorial mode - Step-by-step guided learning experience.
//!
//! Provides an interactive tutorial that walks users through LLM inference
//! with pauses, explanations, and quizzes.

use std::io::{self, BufRead, Write};

use super::explanations::{ExplanationLevel, QuizQuestion, get_explanation, get_quiz_question};
use super::visualizers::{
    attention_mechanism_diagram, kv_cache_diagram, model_architecture_diagram,
    paged_attention_diagram, speculative_decoding_diagram,
};

/// A chapter in the tutorial.
#[derive(Debug, Clone)]
pub struct TutorialChapter {
    /// Chapter title.
    pub title: &'static str,
    /// Chapter content.
    pub content: &'static str,
    /// Optional ASCII diagram.
    pub diagram: Option<fn() -> &'static str>,
    /// Optional quiz topic.
    pub quiz_topic: Option<&'static str>,
}

/// State of the tutorial progress.
#[derive(Debug, Clone, Default)]
pub struct TutorialState {
    /// Current chapter index.
    pub current_chapter: usize,
    /// Total number of chapters.
    pub total_chapters: usize,
    /// Number of correct quiz answers.
    pub quiz_correct: usize,
    /// Total number of quiz questions answered.
    pub quiz_total: usize,
    /// Whether the tutorial is completed.
    pub completed: bool,
}

/// Interactive step-by-step tutorial for learning LLM inference.
pub struct InteractiveTutorial {
    /// Example prompt used throughout the tutorial.
    #[allow(dead_code)]
    prompt: String,
    state: TutorialState,
    chapters: Vec<TutorialChapter>,
}

impl InteractiveTutorial {
    /// Create a new interactive tutorial.
    pub fn new(prompt: &str) -> Self {
        let chapters = Self::build_chapters();
        let total_chapters = chapters.len();
        Self {
            prompt: prompt.to_string(),
            state: TutorialState {
                total_chapters,
                ..Default::default()
            },
            chapters,
        }
    }

    fn build_chapters() -> Vec<TutorialChapter> {
        vec![
            // Introduction
            TutorialChapter {
                title: "Welcome to the LLM Inference Tutorial",
                content: r#"This interactive tutorial will walk you through exactly what happens
when a Large Language Model (LLM) generates text.

By the end, you'll understand:
• How text is converted to numbers (tokenization)
• How the model "reads" your prompt (prefill)
• How tokens are generated one by one (decode)
• Why KV caching matters for speed
• How attention helps the model understand context

Let's begin!"#,
                diagram: None,
                quiz_topic: None,
            },
            // Model Architecture Overview
            TutorialChapter {
                title: "Model Architecture Overview",
                content: r#"Before we dive into inference, let's look at the overall architecture
of a transformer-based LLM like LLaMA.

The model is a stack of "decoder layers", each containing:
1. Self-Attention: Lets tokens communicate with each other
2. Feed-Forward Network (FFN): Processes each token independently

The key insight: information flows bottom-to-top through layers,
with each layer refining the representation."#,
                diagram: Some(model_architecture_diagram),
                quiz_topic: None,
            },
            // Tokenization
            TutorialChapter {
                title: "Chapter 1: Tokenization",
                content: r#"The first step is converting your text into numbers.

LLMs don't read text directly - they process sequences of integers called
"token IDs". Each token might be a word, part of a word, or punctuation.

Your prompt:
  "The capital of France is"

Gets tokenized into something like:
  [The] [capital] [of] [France] [is] → [450, 7483, 310, 3444, 338]

The model has a vocabulary of ~32,000 tokens. Each unique token has a
unique ID, which is learned during training."#,
                diagram: None,
                quiz_topic: Some("tokenization"),
            },
            // Embedding
            TutorialChapter {
                title: "Chapter 2: Embedding Lookup",
                content: r#"Now each token ID needs to become a vector the model can process.

The embedding layer is a giant lookup table:
• Shape: [32000 tokens, 4096 dimensions]
• Token ID 450 ("The") → Row 450 → a 4096-dimensional vector

These embeddings are LEARNED during training:
• Semantically similar words have similar vectors
• Classic example: king - man + woman ≈ queen

After embedding:
  5 token IDs → [5, 4096] tensor (5 vectors, each 4096 dimensions)"#,
                diagram: None,
                quiz_topic: None,
            },
            // Attention Mechanism
            TutorialChapter {
                title: "Chapter 3: Self-Attention",
                content: r#"Self-attention is the core mechanism that lets tokens "talk" to each other.

For each token, we compute:
• Query (Q): "What am I looking for?"
• Key (K): "What do I contain?"
• Value (V): "What information can I share?"

Attention scores = Q @ K^T (how much each token attends to others)
Output = softmax(scores) @ V (weighted sum of values)

The CAUSAL mask ensures tokens can only see the past, not the future.
This is crucial for language modeling - we predict the NEXT token."#,
                diagram: Some(attention_mechanism_diagram),
                quiz_topic: Some("attention"),
            },
            // Prefill Phase
            TutorialChapter {
                title: "Chapter 4: Prefill Phase",
                content: r#"Now let's see how the model "reads" your prompt.

Prefill processes ALL prompt tokens at once (in parallel):
  [The capital of France is] → Forward pass → KV cache populated

Why parallel?
• GPUs excel at parallel computation
• Processing 5 tokens together is almost as fast as 1 token
• This is why prompts are "read" quickly

After prefill:
• The KV cache contains Key and Value vectors for all prompt tokens
• We get logits for the last position → sample first generated token"#,
                diagram: None,
                quiz_topic: None,
            },
            // KV Cache
            TutorialChapter {
                title: "Chapter 5: KV Cache",
                content: r#"The KV Cache is crucial for fast generation.

WITHOUT cache (naive approach):
  Step 1: Process [The]               → compute K,V for 1 token
  Step 2: Process [The, capital]      → compute K,V for 2 tokens
  Step 3: Process [The, capital, of]  → compute K,V for 3 tokens
  ...
  Total: O(n²) computations!

WITH cache:
  Prefill: Compute K,V for all prompt tokens, store in cache
  Decode:  Only compute K,V for the NEW token, append to cache
  Total: O(n) computations!

The tradeoff: We use memory to store the cache, but save massive compute."#,
                diagram: Some(kv_cache_diagram),
                quiz_topic: Some("kv_cache"),
            },
            // Decode Phase
            TutorialChapter {
                title: "Chapter 6: Decode Phase",
                content: r#"Now we generate tokens one at a time.

Each decode step:
1. Input: Just the last generated token (seq_len=1)
2. Compute Q, K, V for this single token
3. Append K, V to the cache
4. Attend to ALL cached K, V (the entire history)
5. Pass through FFN
6. Compute logits → sample next token

Why one at a time?
• Token N+1 depends on token N
• There's no way to know what comes next without generating it first
• This sequential dependency is why generation is slower than prefill"#,
                diagram: None,
                quiz_topic: Some("prefill_decode"),
            },
            // Sampling
            TutorialChapter {
                title: "Chapter 7: Sampling",
                content: r#"The model outputs "logits" - raw scores for each possible next token.

Converting to probabilities:
  logits = [5.2, 2.1, 1.8, 0.3, ...]  (32000 values)
  probs  = softmax(logits) = [87%, 4%, 2%, 1%, ...]

Sampling strategies:

GREEDY: Always pick the highest probability token
  → Deterministic, good for factual tasks

TOP-K: Sample randomly from the top K tokens
  → More diverse outputs

TEMPERATURE: Scale logits before softmax
  → Low temp = more confident, High temp = more random

For our example:
  "The capital of France is ___"
  → Top predictions: Paris (87%), the (4%), a (2%), ...
  → Greedy picks: "Paris""#,
                diagram: None,
                quiz_topic: None,
            },
            // PagedAttention (Optional Advanced)
            TutorialChapter {
                title: "Chapter 8: PagedAttention (Advanced)",
                content: r#"PagedAttention is vLLM's innovation for efficient memory management.

Problem: Sequences have variable lengths
  Seq A: 10 tokens, Seq B: 100 tokens, Seq C: 50 tokens
  Traditional: Pre-allocate max_length for each → massive waste!

Solution: Page-based allocation (like OS virtual memory)
  • Divide GPU memory into fixed-size blocks
  • Allocate blocks on-demand as sequences grow
  • Block table maps logical → physical blocks

Benefits:
  • No wasted memory
  • Support more concurrent sequences
  • Share blocks for common prefixes (system prompts!)"#,
                diagram: Some(paged_attention_diagram),
                quiz_topic: None,
            },
            // Speculative Decoding
            TutorialChapter {
                title: "Chapter 9: Speculative Decoding (Advanced)",
                content: r#"Speculative Decoding accelerates generation using a draft-verify paradigm.

Problem: Large models are slow at autoregressive decoding
  → One token at a time, GPU underutilized

Solution: Use a small model to "guess" multiple tokens
  1. DRAFT: Small model (Qwen3-0.6B) generates K tokens quickly
  2. VERIFY: Large model (Qwen3-4B) checks all K tokens in parallel
  3. ACCEPT: Use rejection sampling to decide which to keep

Key insight: Verification is parallel (efficient on GPU)
  → If all K drafts are accepted, we get K tokens for cost of 1!

Guarantee: Output EXACTLY matches what large model would generate
  → Rejection sampling ensures statistical correctness"#,
                diagram: Some(speculative_decoding_diagram),
                quiz_topic: None,
            },
            // Putting It All Together
            TutorialChapter {
                title: "Putting It All Together",
                content: r#"Let's trace through the full inference for our example:

1. TOKENIZE: "The capital of France is" → [450, 7483, 310, 3444, 338]

2. EMBED: Look up 5 vectors → [5, 4096] tensor

3. PREFILL: Process all 5 tokens through 32 layers
   • Each layer: Attention → FFN
   • KV cache populated with 5 entries per layer

4. DECODE STEP 1:
   • Logits for position 5 → "Paris" (87%)
   • Sample "Paris", add to sequence

5. DECODE STEP 2:
   • Input: "Paris" (using cached K,V for context)
   • Logits → "." (72%)
   • Sample ".", add to sequence

6. Continue until max_tokens or EOS token...

Final output: "The capital of France is Paris."

Congratulations! You now understand LLM inference!"#,
                diagram: None,
                quiz_topic: None,
            },
            // Summary and Next Steps
            TutorialChapter {
                title: "Summary and Next Steps",
                content: r#"Key Takeaways:

1. TOKENIZATION: Text → numbers via learned vocabulary
2. EMBEDDING: Token IDs → dense vectors
3. ATTENTION: Tokens communicate via Q, K, V
4. KV CACHE: Trade memory for O(n) instead of O(n²) compute
5. PREFILL: Process prompt in parallel (fast)
6. DECODE: Generate tokens one by one (sequential)
7. SAMPLING: Convert logits → probabilities → token

Next Steps:
• Run inference with --narrate to see it live
• Use --xray for tensor-level details
• Explore the source code in src/

Thank you for completing the tutorial!"#,
                diagram: None,
                quiz_topic: None,
            },
        ]
    }

    fn print(&self, text: &str) {
        println!("{text}");
    }

    fn wait_for_enter(&self, prompt: &str) {
        print!("\n{prompt}");
        let _ = io::stdout().flush();
        let stdin = io::stdin();
        let mut line = String::new();
        let _ = stdin.lock().read_line(&mut line);
    }

    fn ask_quiz(&mut self, quiz: &QuizQuestion) -> bool {
        self.print("\n🧠 POP QUIZ:");
        self.print(&format!("  {}", quiz.question));
        for option in quiz.options {
            self.print(&format!("    {option}"));
        }

        print!("\nYour answer (A/B/C/D): ");
        let _ = io::stdout().flush();

        let stdin = io::stdin();
        let mut answer = String::new();
        let _ = stdin.lock().read_line(&mut answer);
        let answer = answer.trim().to_uppercase();

        let correct = answer.starts_with(quiz.answer);

        if correct {
            self.print("\n✅ Correct!");
        } else {
            self.print(&format!("\n❌ Not quite. The answer is {}.", quiz.answer));
        }

        self.print(&format!("\n📝 {}", quiz.explanation));

        self.state.quiz_total += 1;
        if correct {
            self.state.quiz_correct += 1;
        }

        correct
    }

    /// Show a single chapter.
    pub fn show_chapter(&mut self, chapter_idx: usize) -> bool {
        if chapter_idx >= self.chapters.len() {
            return false;
        }

        let chapter = &self.chapters[chapter_idx];

        // Clear some space
        self.print(&format!("\n{}", "═".repeat(65)));
        self.print(&format!("  🎓 {}", chapter.title));
        self.print(&"═".repeat(65));

        // Content
        self.print("");
        for line in chapter.content.lines() {
            self.print(line);
        }

        // Diagram
        if let Some(diagram_fn) = chapter.diagram {
            self.print("");
            self.print(diagram_fn());
        }

        // Quiz
        if let Some(topic) = chapter.quiz_topic
            && let Some(quiz) = get_quiz_question(topic)
        {
            self.wait_for_enter("[Press Enter for quiz...]");
            self.ask_quiz(&quiz);
        }

        true
    }

    /// Run the full interactive tutorial.
    pub fn run(&mut self) {
        self.print(&format!("\n{}", "═".repeat(65)));
        self.print("  🎓 NANO-VLLM INTERACTIVE TUTORIAL");
        self.print("  Understanding LLM Inference from the Inside Out");
        self.print(&"═".repeat(65));

        self.print(&format!(
            "\nThis tutorial has {} chapters.",
            self.state.total_chapters
        ));
        self.print("Navigate with Enter, or type 'q' to quit.\n");

        self.wait_for_enter("[Press Enter to begin...]");

        for i in 0..self.chapters.len() {
            self.state.current_chapter = i;

            // Show progress
            self.print(&format!(
                "\n[Chapter {}/{}]",
                i + 1,
                self.state.total_chapters
            ));

            self.show_chapter(i);

            if i < self.chapters.len() - 1 {
                print!("\n[Press Enter for next chapter, 'q' to quit] ");
                let _ = io::stdout().flush();

                let stdin = io::stdin();
                let mut response = String::new();
                let _ = stdin.lock().read_line(&mut response);

                if response.trim().to_lowercase() == "q" {
                    break;
                }
            }
        }

        // Completion summary
        self.state.completed = true;
        self.print(&format!("\n{}", "═".repeat(65)));
        self.print("  🎓 TUTORIAL COMPLETE!");
        self.print(&"═".repeat(65));

        if self.state.quiz_total > 0 {
            let pct = self.state.quiz_correct as f32 / self.state.quiz_total as f32 * 100.0;
            self.print(&format!(
                "\n📊 Quiz Score: {}/{} ({:.0}%)",
                self.state.quiz_correct, self.state.quiz_total, pct
            ));
        }

        self.print("\nTry these commands next:");
        self.print("  nano-vllm --prompt \"Hello\" --narrate");
        self.print("  nano-vllm --prompt \"Hello\" --xray");
        self.print("  nano-vllm --prompt \"Hello\" --dashboard");
        self.print("");
    }

    /// Run a specific chapter by index.
    pub fn run_chapter(&mut self, chapter_idx: usize) {
        if chapter_idx < self.chapters.len() {
            self.show_chapter(chapter_idx);
        } else {
            self.print(&format!(
                "Chapter {} not found. Available: 0-{}",
                chapter_idx,
                self.chapters.len() - 1
            ));
        }
    }

    /// Print a list of all chapters.
    pub fn list_chapters(&self) {
        self.print("\n📚 Tutorial Chapters:");
        for (i, chapter) in self.chapters.iter().enumerate() {
            self.print(&format!("  {}. {}", i + 1, chapter.title));
        }
        self.print("");
    }

    /// Show explanation for a specific topic.
    pub fn show_topic(&self, topic: &str, level: ExplanationLevel) {
        if let Some(explanation) = get_explanation(topic) {
            self.print(&format!("\n📖 {}:", topic.to_uppercase()));
            self.print(&"─".repeat(40));
            self.print(explanation.get(level));
            self.print("");
        } else {
            self.print(&format!("Topic not found: {topic}"));
            self.print(&format!(
                "Available topics: {}",
                super::explanations::get_all_topics().join(", ")
            ));
        }
    }

    /// Get the current tutorial state.
    pub fn state(&self) -> &TutorialState {
        &self.state
    }
}

impl Default for InteractiveTutorial {
    fn default() -> Self {
        Self::new("The capital of France is")
    }
}

/// Entry point for running the tutorial from command line.
pub fn run_tutorial() {
    let mut tutorial = InteractiveTutorial::default();
    tutorial.run();
}
