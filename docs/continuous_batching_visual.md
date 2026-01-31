# Continuous Batching: Visual Guide

This document provides visual explanations of continuous batching concepts using diagrams from the [HuggingFace blog](https://huggingface.co/blog/continuous_batching).

## Attention Mechanism

The attention mechanism is where tokens interact with each other. Understanding this is key to understanding why continuous batching works.

### Q, K, V Projections

![Projection and Multiplication](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/proj_and_mul.png)

Input tensor is projected by three matrices (Wq, Wk, Wv) to produce Q, K, V tensors. Then Q and K are multiplied to compute attention scores.

### Attention Masking and Softmax

![Masking and Softmax](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/masking_and_softmax.png)

The attention mask controls which tokens can interact:

- **Green**: Token can attend (True)
- **White**: Token cannot attend (False)

Causal masking ensures each token only sees previous tokens.

### Complete Attention Flow

![Full Attention](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/attention.png)

The complete attention computation: project → multiply → mask → softmax → output.

### Simplified Attention Visualization

![Simple Attention](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/simple_attention.png)

We can simplify by showing only Q, K, and the attention mask. Each row shows what one token can attend to.

## KV Cache

### Why KV Cache?

Without caching, generating token n+1 would recompute K and V for all previous tokens:

![Naive Generation](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/naive_generate.png)

The grey tokens are recomputed unnecessarily!

### How KV Cache Saves Compute

![KV Cache](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/kv_cache.png)

With KV caching:

- Only compute K, V for the **new token** (white)
- Retrieve previous K, V from cache (grey)
- Complexity reduces from O(n²) to O(n)

### Why New Token Can't Affect Previous Computations

![Can't See Me](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/cant_see_me.png)

Due to causal masking, the new token doesn't affect attention for previous tokens. Their computations remain valid from the cache.

## Chunked Prefill

When prompts are too long for GPU memory, split into chunks:

![Chunked Prefill](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/chunked_prefill.png)

- First chunk: Process tokens[0:4], store K,V in cache
- Second chunk: Process tokens[4:7], prepend cached K,V
- The attention mask adapts to show the chunked structure

**Key insight**: Cached KV states let us process prompts incrementally without losing information.

## Batching Strategies

### Static (Padded) Batching

![Padding](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/padding.png)

Traditional batching pads all prompts to match the longest sequence:

- Orange tokens are padding (`<pad>`)
- Attention mask ensures padding doesn't affect computation
- But compute is still wasted on padding!

### Batched Generation Timeline

![Batched Generation](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/batched_generation.png)

Static batching over multiple iterations:

1. **Top**: Prefill both prompts (padded to equal length)
2. **Below**: Decode iterations generate tokens one at a time
3. When one finishes (`<eos>`), its compute is wasted

### Dynamic Batching Problem

![Dynamic Batching](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/dynamic_batching.png)

Dynamic batching swaps finished requests with new ones, but requires massive padding:

- New prompt needs full prefill
- Other prompts only decode (1 token)
- With B=8 and n=100, we need ~693 padding tokens!

### Ragged Batching Solution

![Concatenate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/concatenate.png)

Instead of padding, concatenate prompts directly.

![Ragged Batching](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/ragged_batching.png)

The attention mask prevents cross-sequence interaction:

- Different tints of green show different sequences
- White blocks prevent tokens from different sequences seeing each other
- No padding tokens needed!

## Continuous Batching

![Continuous Batching](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/continuous_batching.png)

The complete continuous batching algorithm:

1. Maintain a token budget (m tokens per batch)
2. First: Add all decode sequences (1 token each)
3. Then: Fill remaining space with prefill (chunked if needed)
4. When a sequence finishes, immediately replace it

**Result**: GPU always stays fully utilized, no wasted compute on padding!

## Summary

Continuous batching combines three techniques:

| Technique                                | Purpose                                       |
| ---------------------------------------- | --------------------------------------------- |
| **KV Caching**                           | Avoid recomputing past token representations  |
| **Chunked Prefill**                      | Handle variable-length prompts within memory  |
| **Ragged Batching + Dynamic Scheduling** | Eliminate padding, keep GPU utilized          |

This is why services like ChatGPT can efficiently serve thousands of concurrent users.

## References

- [Continuous Batching from First Principles](https://huggingface.co/blog/continuous_batching) - HuggingFace Blog (source of all images)
- [vLLM Paper](https://arxiv.org/abs/2309.06180) - PagedAttention paper
- [KV Caching Explained](https://huggingface.co/blog/kv-cache) - Detailed KV cache implementation
