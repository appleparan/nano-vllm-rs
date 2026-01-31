//! Rotary Position Embeddings (RoPE) implementation.
//!
//! RoPE encodes position information by rotating pairs of elements in the
//! query and key vectors. This allows the model to understand relative
//! positions through the dot product of rotated vectors.
//!
//! Reference: <https://arxiv.org/abs/2104.09864>

use candle_core::{DType, Device, Result, Tensor};

/// Rotary Position Embedding.
///
/// RoPE applies position-dependent rotation to query and key vectors.
/// The rotation angle is determined by the position and frequency.
///
/// Key insight: `dot(rotate(q, pos_q), rotate(k, pos_k))` depends on
/// `pos_q - pos_k`, naturally capturing relative position.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    /// Precomputed cosine values [max_seq_len, head_dim].
    cos_cache: Tensor,
    /// Precomputed sine values [max_seq_len, head_dim].
    sin_cache: Tensor,
    /// Head dimension (must be even for rotation pairs).
    dim: usize,
}

impl RotaryEmbedding {
    /// Creates a new RotaryEmbedding with precomputed cos/sin caches.
    ///
    /// # Arguments
    ///
    /// * `dim` - Head dimension (must be even)
    /// * `max_seq_len` - Maximum sequence length to support
    /// * `theta` - Base frequency (typically 10000 or 1000000 for Qwen3)
    /// * `dtype` - Data type for the cache tensors
    /// * `device` - Device to create tensors on
    ///
    /// # Panics
    ///
    /// Panics if `dim` is not even (rotation requires pairs).
    pub fn new(
        dim: usize,
        max_seq_len: usize,
        theta: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        assert!(dim % 2 == 0, "RoPE dimension must be even");

        // Compute inverse frequencies: 1 / (theta^(2i/dim)) for i in 0..dim/2
        // This determines the rotation frequency for each dimension pair
        let half_dim = dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?; // [dim/2]

        // Compute position indices: [0, 1, 2, ..., max_seq_len-1]
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?.reshape((max_seq_len, 1))?; // [max_seq_len, 1]

        // Compute freqs: positions * inv_freq -> [max_seq_len, dim/2]
        let freqs = positions.broadcast_mul(&inv_freq)?;

        // Duplicate for full dimension: [max_seq_len, dim]
        // Each position has [freq_0, freq_0, freq_1, freq_1, ..., freq_{dim/2-1}, freq_{dim/2-1}]
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?;

        // Compute cos and sin caches
        let cos_cache = freqs.cos()?.to_dtype(dtype)?;
        let sin_cache = freqs.sin()?.to_dtype(dtype)?;

        Ok(Self {
            cos_cache,
            sin_cache,
            dim,
        })
    }

    /// Returns the head dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns a reference to the cosine cache.
    pub fn cos_cache(&self) -> &Tensor {
        &self.cos_cache
    }

    /// Returns a reference to the sine cache.
    pub fn sin_cache(&self) -> &Tensor {
        &self.sin_cache
    }

    /// Applies rotary embedding to query and key tensors.
    ///
    /// # Arguments
    ///
    /// * `q` - Query tensor [batch, seq_len, num_heads, head_dim]
    /// * `k` - Key tensor [batch, seq_len, num_kv_heads, head_dim]
    /// * `start_pos` - Starting position index (for incremental decoding)
    ///
    /// # Returns
    ///
    /// Tuple of rotated (q, k) tensors with same shapes as input.
    pub fn apply(&self, q: &Tensor, k: &Tensor, start_pos: usize) -> Result<(Tensor, Tensor)> {
        let (_, seq_len, _, _) = q.dims4()?;

        // Get cos/sin for the relevant positions [seq_len, dim]
        let cos = self.cos_cache.narrow(0, start_pos, seq_len)?;
        let sin = self.sin_cache.narrow(0, start_pos, seq_len)?;

        // Add dimensions for broadcasting: [1, seq_len, 1, dim]
        let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

        // Apply rotation to q and k
        let q_rot = self.apply_rotation(q, &cos, &sin)?;
        let k_rot = self.apply_rotation(k, &cos, &sin)?;

        Ok((q_rot, k_rot))
    }

    /// Applies rotation to a single tensor.
    ///
    /// Formula: x_rot = x * cos + rotate_half(x) * sin
    /// where rotate_half([a, b, c, d, ...]) = [-b, a, -d, c, ...]
    fn apply_rotation(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x_rotated = self.rotate_half(x)?;
        let x_cos = x.broadcast_mul(cos)?;
        let x_sin = x_rotated.broadcast_mul(sin)?;
        x_cos.add(&x_sin)
    }

    /// Rotates half of the tensor dimensions.
    ///
    /// For input [a, b, c, d, ...], produces [-b, a, -d, c, ...]
    /// This pairs consecutive elements and rotates each pair by 90 degrees.
    fn rotate_half(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let last_dim = dims[dims.len() - 1];
        let half = last_dim / 2;

        // Split into first half and second half
        let x1 = x.narrow(dims.len() - 1, 0, half)?;
        let x2 = x.narrow(dims.len() - 1, half, half)?;

        // Negate x2 and concatenate as [-x2, x1]
        let neg_x2 = x2.neg()?;
        Tensor::cat(&[&neg_x2, &x1], dims.len() - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> Device {
        Device::Cpu
    }

    #[test]
    fn test_rope_creation() {
        let rope = RotaryEmbedding::new(64, 1024, 10000.0, DType::F32, &test_device()).unwrap();

        assert_eq!(rope.dim(), 64);
        assert_eq!(rope.cos_cache().dims(), &[1024, 64]);
        assert_eq!(rope.sin_cache().dims(), &[1024, 64]);
    }

    #[test]
    fn test_rope_apply_shape() {
        let device = test_device();
        let batch = 2;
        let seq_len = 8;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 64;

        let rope = RotaryEmbedding::new(head_dim, 1024, 10000.0, DType::F32, &device).unwrap();

        // Create random q and k
        let q = Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_heads, head_dim), &device).unwrap();
        let k =
            Tensor::randn(0.0f32, 1.0, (batch, seq_len, num_kv_heads, head_dim), &device).unwrap();

        let (q_rot, k_rot) = rope.apply(&q, &k, 0).unwrap();

        // Check shapes preserved
        assert_eq!(q_rot.dims(), &[batch, seq_len, num_heads, head_dim]);
        assert_eq!(k_rot.dims(), &[batch, seq_len, num_kv_heads, head_dim]);
    }

    #[test]
    fn test_rope_incremental_decode() {
        let device = test_device();
        let batch = 1;
        let num_heads = 2;
        let head_dim = 32;

        let rope = RotaryEmbedding::new(head_dim, 1024, 10000.0, DType::F32, &device).unwrap();

        // Prefill: positions 0-3
        let q_prefill =
            Tensor::randn(0.0f32, 1.0, (batch, 4, num_heads, head_dim), &device).unwrap();
        let k_prefill =
            Tensor::randn(0.0f32, 1.0, (batch, 4, num_heads, head_dim), &device).unwrap();
        let (_q1, _k1) = rope.apply(&q_prefill, &k_prefill, 0).unwrap();

        // Decode: position 4
        let q_decode =
            Tensor::randn(0.0f32, 1.0, (batch, 1, num_heads, head_dim), &device).unwrap();
        let k_decode =
            Tensor::randn(0.0f32, 1.0, (batch, 1, num_heads, head_dim), &device).unwrap();
        let (q2, k2) = rope.apply(&q_decode, &k_decode, 4).unwrap();

        // Check shapes for decode
        assert_eq!(q2.dims(), &[batch, 1, num_heads, head_dim]);
        assert_eq!(k2.dims(), &[batch, 1, num_heads, head_dim]);
    }

    #[test]
    fn test_rotate_half() {
        let device = test_device();
        let rope = RotaryEmbedding::new(4, 1024, 10000.0, DType::F32, &device).unwrap();

        // Input: [1, 2, 3, 4]
        let x = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)
            .unwrap()
            .reshape((1, 1, 1, 4))
            .unwrap();

        let rotated = rope.rotate_half(&x).unwrap();
        let values: Vec<f32> = rotated.flatten_all().unwrap().to_vec1().unwrap();

        // Expected: [-3, -4, 1, 2] (split at half, negate first, swap)
        // Actually: first half = [1,2], second half = [3,4]
        // rotate_half: [-second_half, first_half] = [-3, -4, 1, 2]
        assert_eq!(values, vec![-3.0, -4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_rope_position_encoding_varies() {
        let device = test_device();
        let head_dim = 16;

        let rope = RotaryEmbedding::new(head_dim, 100, 10000.0, DType::F32, &device).unwrap();

        // Same input at different positions should give different outputs
        let x = Tensor::ones((1, 1, 1, head_dim), DType::F32, &device).unwrap();
        let x_clone = x.clone();

        let (q0, _) = rope.apply(&x, &x, 0).unwrap();
        let (q1, _) = rope.apply(&x_clone, &x_clone, 1).unwrap();

        // The rotated values should be different
        let diff = (&q0 - &q1).unwrap().abs().unwrap().sum_all().unwrap();
        let diff_val: f32 = diff.to_scalar().unwrap();

        // Difference should be non-zero
        assert!(diff_val > 1e-5);
    }

    #[test]
    fn test_rope_with_qwen3_theta() {
        // Qwen3 uses theta=1000000
        let device = test_device();
        let rope = RotaryEmbedding::new(128, 40960, 1000000.0, DType::F32, &device).unwrap();

        assert_eq!(rope.dim(), 128);
        assert_eq!(rope.cos_cache().dims(), &[40960, 128]);
    }
}
