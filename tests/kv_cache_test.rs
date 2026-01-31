//! Integration tests for KVCache.

use candle_core::Device;
use nano_vllm::core::kv_cache::{KVCache, KVCacheConfig, LayerKVCache};

fn test_config() -> KVCacheConfig {
    KVCacheConfig::new(
        16, // num_blocks
        4,  // block_size
        8,  // num_kv_heads
        64, // head_dim
        2,  // num_layers
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
