//! KV Cache for storing key-value states.
//!
//! The KVCache stores computed key and value tensors for the attention mechanism,
//! organized into fixed-size blocks for PagedAttention.
//!
//! ## Memory Layout
//!
//! Each layer has separate key and value caches with shape:
//! `[num_blocks, block_size, num_kv_heads, head_dim]`
//!
//! For Qwen3-0.6B with 1024 blocks:
//! - Key cache: `[1024, 16, 8, 64]`
//! - Value cache: `[1024, 16, 8, 64]`

use candle_core::{DType, Device, Tensor};

use crate::error::{Error, Result};

/// Configuration for KV cache.
#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    /// Number of blocks.
    pub num_blocks: usize,
    /// Tokens per block.
    pub block_size: usize,
    /// Number of KV heads (for GQA, typically fewer than Q heads).
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Data type for cache tensors.
    pub dtype: DType,
}

impl KVCacheConfig {
    /// Create a new KV cache configuration.
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_layers: usize,
    ) -> Self {
        Self {
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            num_layers,
            dtype: DType::F32,
        }
    }

    /// Set the data type.
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Calculate memory size in bytes for one layer's cache (K or V).
    pub fn layer_cache_size_bytes(&self) -> usize {
        let elements = self.num_blocks * self.block_size * self.num_kv_heads * self.head_dim;
        elements * self.dtype.size_in_bytes()
    }

    /// Calculate total memory size in bytes (all layers, K and V).
    pub fn total_cache_size_bytes(&self) -> usize {
        self.layer_cache_size_bytes() * self.num_layers * 2 // K and V
    }
}

/// KV cache for a single transformer layer.
#[derive(Debug)]
pub struct LayerKVCache {
    /// Key cache: [num_blocks, block_size, num_kv_heads, head_dim]
    key_cache: Tensor,
    /// Value cache: [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: Tensor,
    /// Configuration.
    config: KVCacheConfig,
}

impl LayerKVCache {
    /// Create a new layer KV cache.
    pub fn new(config: &KVCacheConfig, device: &Device) -> Result<Self> {
        let shape = (
            config.num_blocks,
            config.block_size,
            config.num_kv_heads,
            config.head_dim,
        );

        let key_cache = Tensor::zeros(shape, config.dtype, device)?;
        let value_cache = Tensor::zeros(shape, config.dtype, device)?;

        Ok(Self {
            key_cache,
            value_cache,
            config: config.clone(),
        })
    }

    /// Get the key cache tensor.
    pub fn key_cache(&self) -> &Tensor {
        &self.key_cache
    }

    /// Get the value cache tensor.
    pub fn value_cache(&self) -> &Tensor {
        &self.value_cache
    }

    /// Get a slice of the key cache for specific blocks.
    ///
    /// # Arguments
    ///
    /// * `block_ids` - Block IDs to gather
    ///
    /// # Returns
    ///
    /// Tensor of shape `[num_blocks, block_size, num_kv_heads, head_dim]`
    pub fn gather_keys(&self, block_ids: &[usize]) -> Result<Tensor> {
        if block_ids.is_empty() {
            return Err(Error::Config("block_ids cannot be empty".into()));
        }

        let indices: Vec<u32> = block_ids.iter().map(|&id| id as u32).collect();
        let index_tensor = Tensor::new(indices, self.key_cache.device())?;
        let gathered = self.key_cache.index_select(&index_tensor, 0)?;
        Ok(gathered)
    }

    /// Get a slice of the value cache for specific blocks.
    pub fn gather_values(&self, block_ids: &[usize]) -> Result<Tensor> {
        if block_ids.is_empty() {
            return Err(Error::Config("block_ids cannot be empty".into()));
        }

        let indices: Vec<u32> = block_ids.iter().map(|&id| id as u32).collect();
        let index_tensor = Tensor::new(indices, self.value_cache.device())?;
        let gathered = self.value_cache.index_select(&index_tensor, 0)?;
        Ok(gathered)
    }

    /// Write key states to a specific slot.
    ///
    /// # Arguments
    ///
    /// * `block_id` - Block to write to
    /// * `slot` - Slot within the block (0 to block_size-1)
    /// * `key` - Key tensor of shape `[num_kv_heads, head_dim]`
    pub fn write_key(&mut self, block_id: usize, slot: usize, key: &Tensor) -> Result<()> {
        self.validate_slot(block_id, slot)?;

        // Create a new tensor with the key written at the specified position
        let key_expanded = key.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, num_kv_heads, head_dim]

        // Use slice_scatter to write to the specific position
        let key_cache = self.key_cache.slice_scatter(
            &key_expanded,
            0, // dim for block_id
            block_id,
        )?;
        let key_cache = key_cache.slice_scatter(
            &key_expanded.squeeze(0)?, // Remove the block dim we just scattered
            1, // dim for slot
            slot,
        )?;

        self.key_cache = key_cache;
        Ok(())
    }

    /// Write value states to a specific slot.
    pub fn write_value(&mut self, block_id: usize, slot: usize, value: &Tensor) -> Result<()> {
        self.validate_slot(block_id, slot)?;

        let value_expanded = value.unsqueeze(0)?.unsqueeze(0)?;

        let value_cache = self.value_cache.slice_scatter(
            &value_expanded,
            0,
            block_id,
        )?;
        let value_cache = value_cache.slice_scatter(
            &value_expanded.squeeze(0)?,
            1,
            slot,
        )?;

        self.value_cache = value_cache;
        Ok(())
    }

    /// Validate block_id and slot are within bounds.
    fn validate_slot(&self, block_id: usize, slot: usize) -> Result<()> {
        if block_id >= self.config.num_blocks {
            return Err(Error::Config(format!(
                "block_id {} out of bounds (max {})",
                block_id, self.config.num_blocks
            )));
        }
        if slot >= self.config.block_size {
            return Err(Error::Config(format!(
                "slot {} out of bounds (max {})",
                slot, self.config.block_size
            )));
        }
        Ok(())
    }
}

/// Full KV cache for all transformer layers.
#[derive(Debug)]
pub struct KVCache {
    /// Per-layer caches.
    layers: Vec<LayerKVCache>,
    /// Configuration.
    config: KVCacheConfig,
    /// Device.
    device: Device,
}

impl KVCache {
    /// Create a new KV cache for all layers.
    pub fn new(config: KVCacheConfig, device: Device) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(LayerKVCache::new(&config, &device)?);
        }

        Ok(Self {
            layers,
            config,
            device,
        })
    }

    /// Get the cache for a specific layer.
    pub fn layer(&self, layer_idx: usize) -> Option<&LayerKVCache> {
        self.layers.get(layer_idx)
    }

    /// Get mutable access to a layer's cache.
    pub fn layer_mut(&mut self, layer_idx: usize) -> Option<&mut LayerKVCache> {
        self.layers.get_mut(layer_idx)
    }

    /// Get the number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get the configuration.
    pub fn config(&self) -> &KVCacheConfig {
        &self.config
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Gather keys from all layers for specific blocks.
    ///
    /// # Returns
    ///
    /// Vector of tensors, one per layer, each of shape
    /// `[num_gathered_blocks, block_size, num_kv_heads, head_dim]`
    pub fn gather_keys_all_layers(&self, block_ids: &[usize]) -> Result<Vec<Tensor>> {
        self.layers
            .iter()
            .map(|layer| layer.gather_keys(block_ids))
            .collect()
    }

    /// Gather values from all layers for specific blocks.
    pub fn gather_values_all_layers(&self, block_ids: &[usize]) -> Result<Vec<Tensor>> {
        self.layers
            .iter()
            .map(|layer| layer.gather_values(block_ids))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> KVCacheConfig {
        KVCacheConfig::new(
            16,  // num_blocks
            4,   // block_size
            8,   // num_kv_heads
            64,  // head_dim
            2,   // num_layers
        )
    }

    #[test]
    fn test_kv_cache_config() {
        let config = test_config();

        assert_eq!(config.num_blocks, 16);
        assert_eq!(config.block_size, 4);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.num_layers, 2);

        // 16 * 4 * 8 * 64 * 4 bytes = 131072 bytes per layer cache
        assert_eq!(config.layer_cache_size_bytes(), 131072);

        // 2 layers * 2 (K+V) * 131072 = 524288 bytes total
        assert_eq!(config.total_cache_size_bytes(), 524288);
    }

    #[test]
    fn test_layer_kv_cache_creation() {
        let config = test_config();
        let device = Device::Cpu;
        let layer_cache = LayerKVCache::new(&config, &device).unwrap();

        let key_shape = layer_cache.key_cache().dims();
        assert_eq!(key_shape, &[16, 4, 8, 64]);

        let value_shape = layer_cache.value_cache().dims();
        assert_eq!(value_shape, &[16, 4, 8, 64]);
    }

    #[test]
    fn test_kv_cache_creation() {
        let config = test_config();
        let device = Device::Cpu;
        let cache = KVCache::new(config, device).unwrap();

        assert_eq!(cache.num_layers(), 2);
        assert!(cache.layer(0).is_some());
        assert!(cache.layer(1).is_some());
        assert!(cache.layer(2).is_none());
    }

    #[test]
    fn test_gather_keys_values() {
        let config = test_config();
        let device = Device::Cpu;
        let layer_cache = LayerKVCache::new(&config, &device).unwrap();

        let block_ids = vec![0, 2, 5];
        let keys = layer_cache.gather_keys(&block_ids).unwrap();
        let values = layer_cache.gather_values(&block_ids).unwrap();

        // Should gather 3 blocks
        assert_eq!(keys.dims(), &[3, 4, 8, 64]);
        assert_eq!(values.dims(), &[3, 4, 8, 64]);
    }

    #[test]
    fn test_validate_slot() {
        let config = test_config();
        let device = Device::Cpu;
        let layer_cache = LayerKVCache::new(&config, &device).unwrap();

        // Valid slot
        assert!(layer_cache.validate_slot(0, 0).is_ok());
        assert!(layer_cache.validate_slot(15, 3).is_ok());

        // Invalid block_id
        assert!(layer_cache.validate_slot(16, 0).is_err());

        // Invalid slot
        assert!(layer_cache.validate_slot(0, 4).is_err());
    }

    #[test]
    fn test_gather_all_layers() {
        let config = test_config();
        let device = Device::Cpu;
        let cache = KVCache::new(config, device).unwrap();

        let block_ids = vec![0, 1];
        let all_keys = cache.gather_keys_all_layers(&block_ids).unwrap();
        let all_values = cache.gather_values_all_layers(&block_ids).unwrap();

        assert_eq!(all_keys.len(), 2);
        assert_eq!(all_values.len(), 2);

        for keys in &all_keys {
            assert_eq!(keys.dims(), &[2, 4, 8, 64]);
        }
    }
}
