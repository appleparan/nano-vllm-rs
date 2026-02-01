/**
 * Flash Attention Forward CUDA Kernel
 *
 * Educational implementation of Flash Attention for learning purposes.
 * Optimized for NVIDIA RTX 4090 (Ada Lovelace, SM89).
 *
 * References:
 * - FlashAttention: https://arxiv.org/abs/2205.14135
 * - FlashAttention-2: https://arxiv.org/abs/2307.08691
 *
 * Key optimizations:
 * - Tiled computation to fit in shared memory (100KB per SM on RTX 4090)
 * - Online softmax for numerical stability and memory efficiency
 * - Warp-level reductions for max/sum operations
 * - Coalesced memory access patterns
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

// Block sizes for tiling
// RTX 4090 has 100KB shared memory per SM
// For head_dim=64: Q block = 64*64*2 = 8KB (FP16)
// For head_dim=128: Q block = 64*128*2 = 16KB (FP16)
constexpr int BLOCK_SIZE_Q = 64;   // Query block size
constexpr int BLOCK_SIZE_KV = 64;  // Key/Value block size
constexpr int WARP_SIZE = 32;

// Thread block configuration
// Each thread block handles one (batch, head, q_block) combination
constexpr int THREADS_PER_BLOCK = 128;

/**
 * Warp-level reduction for maximum value.
 */
__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

/**
 * Warp-level reduction for sum.
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Block-level reduction for maximum value using shared memory.
 */
__device__ float block_reduce_max(float val, float *shared_mem) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // Warp-level reduction
    val = warp_reduce_max(val);

    // Write warp results to shared memory
    if (lane == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across all warps
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared_mem[lane] : -FLT_MAX;
        val = warp_reduce_max(val);
    }

    __syncthreads();
    return shared_mem[0];
}

/**
 * Block-level reduction for sum using shared memory.
 */
__device__ float block_reduce_sum(float val, float *shared_mem) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // Warp-level reduction
    val = warp_reduce_sum(val);

    // Write warp results to shared memory
    if (lane == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across all warps
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared_mem[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) {
            shared_mem[0] = val;
        }
    }

    __syncthreads();
    return shared_mem[0];
}

/**
 * Flash Attention Forward Kernel (FP32 version for simplicity)
 *
 * Input shapes:
 *   Q: [batch_size, num_heads, seq_len_q, head_dim]
 *   K: [batch_size, num_heads, seq_len_kv, head_dim]
 *   V: [batch_size, num_heads, seq_len_kv, head_dim]
 *   O: [batch_size, num_heads, seq_len_q, head_dim]
 *
 * Grid: (num_q_blocks, batch_size * num_heads, 1)
 * Block: (THREADS_PER_BLOCK, 1, 1)
 */
__global__ void flash_attention_forward_f32(
    const float *__restrict__ Q, const float *__restrict__ K,
    const float *__restrict__ V, float *__restrict__ O,
    float *__restrict__ L,  // Log-sum-exp for backward (optional)
    const int batch_size, const int num_heads, const int seq_len_q,
    const int seq_len_kv, const int head_dim, const float softmax_scale,
    const bool causal) {
    // Shared memory for:
    // - Q block: [BLOCK_SIZE_Q, head_dim]
    // - K block: [BLOCK_SIZE_KV, head_dim]
    // - V block: [BLOCK_SIZE_KV, head_dim]
    // - Scratch for reductions
    extern __shared__ float shared_mem[];

    const int q_block_idx = blockIdx.x;
    const int batch_head_idx = blockIdx.y;
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    // Calculate offsets for this batch/head
    const int qkv_offset = batch_idx * num_heads * seq_len_q * head_dim +
                           head_idx * seq_len_q * head_dim;
    const int kv_offset = batch_idx * num_heads * seq_len_kv * head_dim +
                          head_idx * seq_len_kv * head_dim;

    // Query block start/end
    const int q_start = q_block_idx * BLOCK_SIZE_Q;
    const int q_end = min(q_start + BLOCK_SIZE_Q, seq_len_q);
    const int q_len = q_end - q_start;

    // Allocate shared memory regions
    float *s_Q = shared_mem;                                  // [BLOCK_SIZE_Q, head_dim]
    float *s_K = s_Q + BLOCK_SIZE_Q * head_dim;               // [BLOCK_SIZE_KV, head_dim]
    float *s_V = s_K + BLOCK_SIZE_KV * head_dim;              // [BLOCK_SIZE_KV, head_dim]
    float *s_S = s_V + BLOCK_SIZE_KV * head_dim;              // [BLOCK_SIZE_Q, BLOCK_SIZE_KV] scores
    // Note: Additional scratch space available after s_S for future optimizations

    const int tid = threadIdx.x;

    // Load Q block to shared memory
    for (int i = tid; i < q_len * head_dim; i += blockDim.x) {
        int q_row = i / head_dim;
        int q_col = i % head_dim;
        s_Q[q_row * head_dim + q_col] =
            Q[qkv_offset + (q_start + q_row) * head_dim + q_col];
    }
    __syncthreads();

    // Initialize output accumulator and statistics per query row
    // Each thread handles specific rows
    float m_i[BLOCK_SIZE_Q / THREADS_PER_BLOCK + 1];  // Running max
    float l_i[BLOCK_SIZE_Q / THREADS_PER_BLOCK + 1];  // Running sum
    float o_i[BLOCK_SIZE_Q / THREADS_PER_BLOCK + 1]
             [128];  // Output accumulator (max head_dim=128)

    // Initialize
    for (int q = tid; q < q_len; q += blockDim.x) {
        int local_idx = q / blockDim.x;
        m_i[local_idx] = -FLT_MAX;
        l_i[local_idx] = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            o_i[local_idx][d] = 0.0f;
        }
    }

    // Number of KV blocks
    const int num_kv_blocks = (seq_len_kv + BLOCK_SIZE_KV - 1) / BLOCK_SIZE_KV;

    // Loop over KV blocks
    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
        const int kv_start = kv_block_idx * BLOCK_SIZE_KV;
        const int kv_end = min(kv_start + BLOCK_SIZE_KV, seq_len_kv);
        const int kv_len = kv_end - kv_start;

        // Skip entirely masked blocks (causal)
        if (causal && kv_start > q_end - 1) {
            continue;
        }

        // Load K block to shared memory
        for (int i = tid; i < kv_len * head_dim; i += blockDim.x) {
            int k_row = i / head_dim;
            int k_col = i % head_dim;
            s_K[k_row * head_dim + k_col] =
                K[kv_offset + (kv_start + k_row) * head_dim + k_col];
        }
        __syncthreads();

        // Load V block to shared memory
        for (int i = tid; i < kv_len * head_dim; i += blockDim.x) {
            int v_row = i / head_dim;
            int v_col = i % head_dim;
            s_V[v_row * head_dim + v_col] =
                V[kv_offset + (kv_start + v_row) * head_dim + v_col];
        }
        __syncthreads();

        // Compute attention scores: S = Q @ K^T * scale
        // Each thread computes a subset of the scores
        for (int i = tid; i < q_len * kv_len; i += blockDim.x) {
            int q_row = i / kv_len;
            int k_row = i % kv_len;

            // Causal mask check
            int global_q = q_start + q_row;
            int global_k = kv_start + k_row;

            float score = 0.0f;
            if (!causal || global_k <= global_q) {
                // Dot product Q[q_row] @ K[k_row]
                for (int d = 0; d < head_dim; d++) {
                    score += s_Q[q_row * head_dim + d] * s_K[k_row * head_dim + d];
                }
                score *= softmax_scale;
            } else {
                score = -FLT_MAX;  // Masked
            }
            s_S[q_row * BLOCK_SIZE_KV + k_row] = score;
        }
        __syncthreads();

        // Online softmax update per query row
        for (int q = tid; q < q_len; q += blockDim.x) {
            int local_idx = q / blockDim.x;

            // Find max in this row
            float row_max = -FLT_MAX;
            for (int k = 0; k < kv_len; k++) {
                row_max = fmaxf(row_max, s_S[q * BLOCK_SIZE_KV + k]);
            }

            // New running max
            float m_new = fmaxf(m_i[local_idx], row_max);

            // Compute exp(S - m_new) and sum
            float row_sum = 0.0f;
            for (int k = 0; k < kv_len; k++) {
                float score = s_S[q * BLOCK_SIZE_KV + k];
                float p = expf(score - m_new);
                s_S[q * BLOCK_SIZE_KV + k] = p;  // Overwrite with softmax numerator
                row_sum += p;
            }

            // Rescale previous output
            float alpha = expf(m_i[local_idx] - m_new);
            float l_new = l_i[local_idx] * alpha + row_sum;

            // Update output: O = O * (l_old * alpha / l_new) + P @ V / l_new
            float scale_old = (l_i[local_idx] * alpha) / l_new;
            float scale_new = 1.0f / l_new;

            for (int d = 0; d < head_dim; d++) {
                // Rescale old output
                o_i[local_idx][d] *= scale_old;

                // Add P @ V contribution
                float pv = 0.0f;
                for (int k = 0; k < kv_len; k++) {
                    pv += s_S[q * BLOCK_SIZE_KV + k] * s_V[k * head_dim + d];
                }
                o_i[local_idx][d] += pv * scale_new;
            }

            // Update statistics
            m_i[local_idx] = m_new;
            l_i[local_idx] = l_new;
        }
        __syncthreads();
    }

    // Write output back to global memory
    for (int q = tid; q < q_len; q += blockDim.x) {
        int local_idx = q / blockDim.x;
        for (int d = 0; d < head_dim; d++) {
            O[qkv_offset + (q_start + q) * head_dim + d] = o_i[local_idx][d];
        }

        // Optionally write log-sum-exp for backward pass
        if (L != nullptr) {
            L[batch_head_idx * seq_len_q + q_start + q] =
                m_i[local_idx] + logf(l_i[local_idx]);
        }
    }
}

/**
 * Flash Attention Forward Kernel (FP16 version)
 *
 * Uses half precision for compute with FP32 accumulation for accuracy.
 */
#ifdef ENABLE_FP16
__global__ void flash_attention_forward_f16(
    const __half *__restrict__ Q, const __half *__restrict__ K,
    const __half *__restrict__ V, __half *__restrict__ O,
    float *__restrict__ L, const int batch_size, const int num_heads,
    const int seq_len_q, const int seq_len_kv, const int head_dim,
    const float softmax_scale, const bool causal) {
    // Similar to FP32 version but with half precision loads/stores
    // and FP32 accumulation for numerical stability

    extern __shared__ char shared_mem_bytes[];
    __half *s_Q = reinterpret_cast<__half *>(shared_mem_bytes);
    __half *s_K = s_Q + BLOCK_SIZE_Q * head_dim;
    __half *s_V = s_K + BLOCK_SIZE_KV * head_dim;
    float *s_S = reinterpret_cast<float *>(s_V + BLOCK_SIZE_KV * head_dim);

    const int q_block_idx = blockIdx.x;
    const int batch_head_idx = blockIdx.y;
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    const int qkv_offset = batch_idx * num_heads * seq_len_q * head_dim +
                           head_idx * seq_len_q * head_dim;
    const int kv_offset = batch_idx * num_heads * seq_len_kv * head_dim +
                          head_idx * seq_len_kv * head_dim;

    const int q_start = q_block_idx * BLOCK_SIZE_Q;
    const int q_end = min(q_start + BLOCK_SIZE_Q, seq_len_q);
    const int q_len = q_end - q_start;

    const int tid = threadIdx.x;

    // Load Q block
    for (int i = tid; i < q_len * head_dim; i += blockDim.x) {
        int q_row = i / head_dim;
        int q_col = i % head_dim;
        s_Q[q_row * head_dim + q_col] =
            Q[qkv_offset + (q_start + q_row) * head_dim + q_col];
    }
    __syncthreads();

    // Initialize accumulators (FP32)
    float m_i[BLOCK_SIZE_Q / THREADS_PER_BLOCK + 1];
    float l_i[BLOCK_SIZE_Q / THREADS_PER_BLOCK + 1];
    float o_i[BLOCK_SIZE_Q / THREADS_PER_BLOCK + 1][128];

    for (int q = tid; q < q_len; q += blockDim.x) {
        int local_idx = q / blockDim.x;
        m_i[local_idx] = -FLT_MAX;
        l_i[local_idx] = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            o_i[local_idx][d] = 0.0f;
        }
    }

    const int num_kv_blocks = (seq_len_kv + BLOCK_SIZE_KV - 1) / BLOCK_SIZE_KV;

    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
        const int kv_start = kv_block_idx * BLOCK_SIZE_KV;
        const int kv_end = min(kv_start + BLOCK_SIZE_KV, seq_len_kv);
        const int kv_len = kv_end - kv_start;

        if (causal && kv_start > q_end - 1) {
            continue;
        }

        // Load K, V blocks
        for (int i = tid; i < kv_len * head_dim; i += blockDim.x) {
            int row = i / head_dim;
            int col = i % head_dim;
            s_K[row * head_dim + col] =
                K[kv_offset + (kv_start + row) * head_dim + col];
            s_V[row * head_dim + col] =
                V[kv_offset + (kv_start + row) * head_dim + col];
        }
        __syncthreads();

        // Compute scores (FP32 accumulation)
        for (int i = tid; i < q_len * kv_len; i += blockDim.x) {
            int q_row = i / kv_len;
            int k_row = i % kv_len;
            int global_q = q_start + q_row;
            int global_k = kv_start + k_row;

            float score = 0.0f;
            if (!causal || global_k <= global_q) {
                for (int d = 0; d < head_dim; d++) {
                    score += __half2float(s_Q[q_row * head_dim + d]) *
                             __half2float(s_K[k_row * head_dim + d]);
                }
                score *= softmax_scale;
            } else {
                score = -FLT_MAX;
            }
            s_S[q_row * BLOCK_SIZE_KV + k_row] = score;
        }
        __syncthreads();

        // Online softmax update
        for (int q = tid; q < q_len; q += blockDim.x) {
            int local_idx = q / blockDim.x;

            float row_max = -FLT_MAX;
            for (int k = 0; k < kv_len; k++) {
                row_max = fmaxf(row_max, s_S[q * BLOCK_SIZE_KV + k]);
            }

            float m_new = fmaxf(m_i[local_idx], row_max);
            float row_sum = 0.0f;
            for (int k = 0; k < kv_len; k++) {
                float p = expf(s_S[q * BLOCK_SIZE_KV + k] - m_new);
                s_S[q * BLOCK_SIZE_KV + k] = p;
                row_sum += p;
            }

            float alpha = expf(m_i[local_idx] - m_new);
            float l_new = l_i[local_idx] * alpha + row_sum;
            float scale_old = (l_i[local_idx] * alpha) / l_new;
            float scale_new = 1.0f / l_new;

            for (int d = 0; d < head_dim; d++) {
                o_i[local_idx][d] *= scale_old;
                float pv = 0.0f;
                for (int k = 0; k < kv_len; k++) {
                    pv += s_S[q * BLOCK_SIZE_KV + k] *
                          __half2float(s_V[k * head_dim + d]);
                }
                o_i[local_idx][d] += pv * scale_new;
            }

            m_i[local_idx] = m_new;
            l_i[local_idx] = l_new;
        }
        __syncthreads();
    }

    // Write output (convert to FP16)
    for (int q = tid; q < q_len; q += blockDim.x) {
        int local_idx = q / blockDim.x;
        for (int d = 0; d < head_dim; d++) {
            O[qkv_offset + (q_start + q) * head_dim + d] =
                __float2half(o_i[local_idx][d]);
        }
        if (L != nullptr) {
            L[batch_head_idx * seq_len_q + q_start + q] =
                m_i[local_idx] + logf(l_i[local_idx]);
        }
    }
}
#endif  // ENABLE_FP16

// C interface for Rust FFI
extern "C" {

/**
 * Launch Flash Attention forward kernel (FP32).
 *
 * @param q Query tensor pointer
 * @param k Key tensor pointer
 * @param v Value tensor pointer
 * @param o Output tensor pointer
 * @param l Log-sum-exp output (can be null)
 * @param batch_size Batch size
 * @param num_heads Number of attention heads
 * @param seq_len_q Query sequence length
 * @param seq_len_kv Key/Value sequence length
 * @param head_dim Head dimension
 * @param softmax_scale Softmax scale (1/sqrt(head_dim))
 * @param causal Whether to apply causal mask
 * @param stream CUDA stream
 */
void flash_attention_forward_f32_launch(const float *q, const float *k,
                                        const float *v, float *o, float *l,
                                        int batch_size, int num_heads,
                                        int seq_len_q, int seq_len_kv,
                                        int head_dim, float softmax_scale,
                                        bool causal, cudaStream_t stream) {
    const int num_q_blocks = (seq_len_q + BLOCK_SIZE_Q - 1) / BLOCK_SIZE_Q;

    dim3 grid(num_q_blocks, batch_size * num_heads, 1);
    dim3 block(THREADS_PER_BLOCK, 1, 1);

    // Shared memory size
    size_t shared_mem_size =
        (BLOCK_SIZE_Q * head_dim + BLOCK_SIZE_KV * head_dim * 2 +
         BLOCK_SIZE_Q * BLOCK_SIZE_KV + THREADS_PER_BLOCK) *
        sizeof(float);

    flash_attention_forward_f32<<<grid, block, shared_mem_size, stream>>>(
        q, k, v, o, l, batch_size, num_heads, seq_len_q, seq_len_kv, head_dim,
        softmax_scale, causal);
}

#ifdef ENABLE_FP16
/**
 * Launch Flash Attention forward kernel (FP16).
 */
void flash_attention_forward_f16_launch(const __half *q, const __half *k,
                                        const __half *v, __half *o, float *l,
                                        int batch_size, int num_heads,
                                        int seq_len_q, int seq_len_kv,
                                        int head_dim, float softmax_scale,
                                        bool causal, cudaStream_t stream) {
    const int num_q_blocks = (seq_len_q + BLOCK_SIZE_Q - 1) / BLOCK_SIZE_Q;

    dim3 grid(num_q_blocks, batch_size * num_heads, 1);
    dim3 block(THREADS_PER_BLOCK, 1, 1);

    // Shared memory size (Q, K, V in FP16, scores in FP32)
    size_t shared_mem_size = (BLOCK_SIZE_Q * head_dim + BLOCK_SIZE_KV * head_dim * 2) *
                                 sizeof(__half) +
                             BLOCK_SIZE_Q * BLOCK_SIZE_KV * sizeof(float);

    flash_attention_forward_f16<<<grid, block, shared_mem_size, stream>>>(
        q, k, v, o, l, batch_size, num_heads, seq_len_q, seq_len_kv, head_dim,
        softmax_scale, causal);
}
#endif  // ENABLE_FP16

/**
 * Get last CUDA error message.
 */
const char *flash_attention_get_last_error() {
    cudaError_t err = cudaGetLastError();
    return cudaGetErrorString(err);
}

/**
 * Synchronize CUDA device.
 */
int flash_attention_synchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? 0 : -1;
}

}  // extern "C"
