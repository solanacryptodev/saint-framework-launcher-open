/// KV Cache Layer for The SAINT Framework Game System
/// 
/// This module provides efficient key-value cache management for transformer models,
/// enabling incremental token generation with memory optimization.

use ort::value::Value;
use crate::config::{ModelConfig, CacheConfig};
use crate::error::{Result, LlmError};

/// Storage for a single transformer layer's key-value cache
/// 
/// Manages the cached key and value tensors for one transformer layer,
/// handling storage, retrieval, and reset operations.
#[derive(Debug)]
pub struct LayerCache {
    /// Cached key tensor: [batch, num_heads, sequence_length, head_dim]
    key_cache: Option<Value>,
    
    /// Cached value tensor: [batch, num_heads, sequence_length, head_dim]
    value_cache: Option<Value>,
    
    /// Current number of tokens cached
    current_length: usize,
    
    /// Index of the transformer layer this cache belongs to
    layer_idx: usize,
    
    /// Maximum number of tokens this cache can hold before eviction
    max_length: usize,
    
    /// Number of attention heads (used for shape validation)
    num_heads: usize,
    
    /// Dimension per attention head (used for shape validation)
    head_dim: usize,
}

impl LayerCache {
    /// Creates a new LayerCache for a specific transformer layer
    /// 
    /// # Arguments
    /// 
    /// * `layer_idx` - Index of the transformer layer (0-based)
    /// * `max_length` - Maximum tokens to cache before eviction
    /// * `num_heads` - Number of attention heads in this layer
    /// * `head_dim` - Dimension per attention head
    /// 
    /// # Example
    /// 
    /// ```
    /// let cache = LayerCache::new(0, 2048, 1, 256);
    /// ```
    pub fn new(layer_idx: usize, max_length: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            key_cache: None,
            value_cache: None,
            current_length: 0,
            layer_idx,
            max_length,
            num_heads,
            head_dim,
        }
    }

    /// Updates the cache with new key and value tensors
    /// 
    /// For the initial implementation, this simply stores the most recent key/value pair.
    /// Future enhancements will support concatenation along the sequence dimension.
    /// 
    /// # Arguments
    /// 
    /// * `key_input` - New key tensor to cache: [batch, num_heads, seq_len, head_dim]
    /// * `value_input` - New value tensor to cache: [batch, num_heads, seq_len, head_dim]
    /// 
    /// # Returns
    /// 
    /// `Result<()>` - Success or error during tensor operations
    /// 
    /// # Example
    /// 
    /// ```
    /// cache.update(&key_value, &value_value)?;
    /// ```
    pub fn update(&mut self, key_input: Value, value_input: Value) -> Result<()> {
        // Get the shape to determine sequence length
        let key_shape = key_input.shape();
        if key_shape.len() < 3 {
            return Err(LlmError::InvalidInput {
                reason: format!("Key input must have at least 3 dimensions, got {}", key_shape.len()),
            });
        }
        
        let value_shape = value_input.shape();
        if value_shape.len() < 3 {
            return Err(LlmError::InvalidInput {
                reason: format!("Value input must have at least 3 dimensions, got {}", value_shape.len()),
            });
        }
        
        let seq_len = key_shape[2] as usize;
        
        // Validate shapes match expected dimensions
        if key_shape != value_shape {
            return Err(LlmError::cache_shape_mismatch(
                self.layer_idx,
                format!("key shape {:?}", key_shape),
                format!("value shape {:?}", value_shape),
            ));
        }
        
        // Store the input Values (take ownership)
        self.key_cache = Some(key_input);
        self.value_cache = Some(value_input);
        self.current_length = seq_len;
        
        Ok(())
    }

    /// Trims the cache to the maximum window size using sliding window eviction
    /// 
    /// Note: For the initial implementation, trimming is a placeholder.
    /// Future implementations will slice tensors to keep only recent tokens.
    /// 
    /// # Returns
    /// 
    /// `Result<()>` - Success or error during tensor operations
    /// 
    /// # Example
    /// 
    /// ```
    /// cache.trim_to_window()?;
    /// ```
    pub fn trim_to_window(&mut self) -> Result<()> {
        if self.current_length <= self.max_length {
            // No trimming needed
            return Ok(());
        }
        
        // TODO: Implement actual trimming by slicing the tensors
        // For now, we just mark that trimming would occur
        println!("⚠️  Cache length ({}) exceeds max ({}), trimming would occur here", 
                 self.current_length, self.max_length);
        
        Ok(())
    }

    /// Retrieves references to the cached key and value tensors
    /// 
    /// # Returns
    /// 
    /// - `Some((&Value, &Value))` - References to key and value caches if available
    /// - `None` - If cache is empty (current_length == 0)
    /// 
    /// # Example
    /// 
    /// ```
    /// if let Some((key, value)) = cache.get_cache() {
    ///     // Use cached key and value
    /// }
    /// ```
    pub fn get_cache(&self) -> Option<(&Value, &Value)> {
        if self.current_length == 0 {
            None
        } else {
            match (&self.key_cache, &self.value_cache) {
                (Some(key), Some(value)) => Some((key, value)),
                _ => None,
            }
        }
    }

    /// Resets the cache to empty state
    /// 
    /// Clears all cached tensors and resets the current length to 0.
    /// Useful for starting a fresh conversation or generation sequence.
    /// 
    /// # Example
    /// 
    /// ```
    /// cache.reset();
    /// ```
    pub fn reset(&mut self) {
        self.key_cache = None;
        self.value_cache = None;
        self.current_length = 0;
    }

    /// Returns the current number of cached tokens
    pub fn len(&self) -> usize {
        self.current_length
    }

    /// Returns whether the cache is empty
    pub fn is_empty(&self) -> bool {
        self.current_length == 0
    }

    /// Returns the layer index this cache belongs to
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Returns the maximum cache capacity
    pub fn max_length(&self) -> usize {
        self.max_length
    }
}

/// Manager for all transformer layer caches
/// 
/// Coordinates caching across all transformer layers, handling batch updates,
/// sliding window enforcement, and state management.
#[derive(Debug)]
pub struct CacheManager {
    /// One cache per transformer layer
    layers: Vec<LayerCache>,
    
    /// Reference to cache configuration
    config: CacheConfig,
    
    /// Lifetime token counter
    total_tokens_processed: usize,
}

impl CacheManager {
    /// Creates a new CacheManager for a transformer model
    /// 
    /// # Arguments
    /// 
    /// * `model_config` - Reference to the model configuration containing layer count and dimensions
    /// 
    /// # Returns
    /// 
    /// A new CacheManager with initialized layer caches
    /// 
    /// # Example
    /// 
    /// ```
    /// let model_config = ModelConfig::default();
    /// let cache_manager = CacheManager::new(&model_config);
    /// ```
    pub fn new(model_config: &ModelConfig) -> Self {
        let num_layers = model_config.num_layers;
        let max_length = model_config.cache_config.max_cache_tokens;
        let num_heads = model_config.num_attention_heads;
        let head_dim = model_config.head_dim;
        
        // Create a Vec with capacity for all layers
        let mut layers = Vec::with_capacity(num_layers);
        
        // Initialize each layer cache
        for layer_idx in 0..num_layers {
            layers.push(LayerCache::new(layer_idx, max_length, num_heads, head_dim));
        }
        
        Self {
            layers,
            config: model_config.cache_config.clone(),
            total_tokens_processed: 0,
        }
    }
    
    /// Updates all layers with new key-value pairs
    /// 
    /// # Arguments
    /// 
    /// * `kv_pairs` - Vec of (key, value) pairs, one per layer (takes ownership)
    /// 
    /// # Returns
    /// 
    /// `Result<()>` - Success or error if length doesn't match or update fails
    /// 
    /// # Example
    /// 
    /// ```
    /// let kv_pairs: Vec<(Value, Value)> = get_kv_from_model();
    /// cache_manager.update_all_layers(kv_pairs)?;
    /// ```
    pub fn update_all_layers(&mut self, kv_pairs: Vec<(Value, Value)>) -> Result<()> {
        // Verify length matches number of layers
        if kv_pairs.len() != self.layers.len() {
            return Err(LlmError::LayerCountMismatch {
                expected: self.layers.len(),
                actual: kv_pairs.len(),
            });
        }
        
        // Get the new token count from the first layer (all should be the same)
        let new_token_count = if !kv_pairs.is_empty() {
            let shape = kv_pairs[0].0.shape();
            if shape.len() >= 3 {
                shape[2] as usize
            } else {
                0
            }
        } else {
            0
        };
        
        // Update each layer with its corresponding KV pair
        for (layer, (key, value)) in self.layers.iter_mut().zip(kv_pairs.into_iter()) {
            // Move the Values into each layer
            layer.update(key, value)?;
        }
        
        // Increment total tokens processed
        self.total_tokens_processed += new_token_count;
        
        Ok(())
    }
    
    /// Enforces sliding window by trimming caches that exceed max length
    /// 
    /// # Returns
    /// 
    /// `Result<()>` - Success or error during trimming
    /// 
    /// # Example
    /// 
    /// ```
    /// cache_manager.enforce_window()?;
    /// ```
    pub fn enforce_window(&mut self) -> Result<()> {
        // Check if any layer exceeds max_cache_tokens
        let needs_trimming = self.layers.iter()
            .any(|layer| layer.current_length > self.config.max_cache_tokens);
        
        if needs_trimming {
            // Trim all layers to maintain consistency
            for layer in self.layers.iter_mut() {
                layer.trim_to_window()?;
            }
        }
        
        Ok(())
    }
    
    /// Retrieves all cached KV pairs for inference
    /// 
    /// # Returns
    /// 
    /// Vec of Option<(&Value, &Value)> - One entry per layer, maintaining layer order
    /// 
    /// # Example
    /// 
    /// ```
    /// let all_caches = cache_manager.get_all_caches();
    /// for (layer_idx, cache) in all_caches.iter().enumerate() {
    ///     if let Some((key, value)) = cache {
    ///         // Use cached KV for this layer
    ///     }
    /// }
    /// ```
    pub fn get_all_caches(&self) -> Vec<Option<(&Value, &Value)>> {
        self.layers.iter()
            .map(|layer| layer.get_cache())
            .collect()
    }
    
    /// Checks if the cache is empty
    /// 
    /// # Returns
    /// 
    /// `true` if all layers are empty, `false` otherwise
    /// 
    /// # Example
    /// 
    /// ```
    /// if cache_manager.is_empty() {
    ///     println!("Cache is empty");
    /// }
    /// ```
    pub fn is_empty(&self) -> bool {
        // Check first layer - all layers should have the same length
        self.layers.first()
            .map(|layer| layer.current_length == 0)
            .unwrap_or(true)
    }
    
    /// Returns the current number of cached tokens
    /// 
    /// # Returns
    /// 
    /// Number of tokens currently cached (same across all layers)
    /// 
    /// # Example
    /// 
    /// ```
    /// let cached_tokens = cache_manager.current_length();
    /// println!("Currently caching {} tokens", cached_tokens);
    /// ```
    pub fn current_length(&self) -> usize {
        // Return first layer's length - all layers should have the same length
        self.layers.first()
            .map(|layer| layer.current_length)
            .unwrap_or(0)
    }
    
    /// Resets all layer caches and token counter
    /// 
    /// # Example
    /// 
    /// ```
    /// cache_manager.reset_all();
    /// assert!(cache_manager.is_empty());
    /// ```
    pub fn reset_all(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.reset();
        }
        self.total_tokens_processed = 0;
    }
    
    /// Returns the total number of tokens processed over the lifetime
    /// 
    /// # Returns
    /// 
    /// Total tokens processed since creation or last reset
    pub fn total_tokens_processed(&self) -> usize {
        self.total_tokens_processed
    }
    
    /// Returns the number of layers managed by this cache
    /// 
    /// # Returns
    /// 
    /// Number of transformer layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_cache_new() {
        let cache = LayerCache::new(0, 2048, 1, 256);
        assert_eq!(cache.layer_idx(), 0);
        assert_eq!(cache.max_length(), 2048);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert!(cache.get_cache().is_none());
    }

    #[test]
    fn test_layer_cache_reset() {
        let mut cache = LayerCache::new(0, 2048, 1, 256);
        
        // Create dummy tensors for testing using f32 (supported by ort)
        let key_data: Vec<f32> = vec![1.0; 1 * 1 * 10 * 256];
        let value_data: Vec<f32> = vec![2.0; 1 * 1 * 10 * 256];
        
        let key_value = Value::from_array((vec![1, 1, 10, 256], key_data)).unwrap();
        let value_value = Value::from_array((vec![1, 1, 10, 256], value_data)).unwrap();
        
        // Update cache
        cache.update(key_value.into(), value_value.into()).unwrap();
        assert_eq!(cache.len(), 10);
        assert!(!cache.is_empty());
        
        // Reset cache
        cache.reset();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert!(cache.get_cache().is_none());
    }

    #[test]
    fn test_first_update() {
        let mut cache = LayerCache::new(0, 2048, 1, 256);
        
        // Create dummy tensors: [batch=1, num_heads=1, seq_len=5, head_dim=256]
        let key_data: Vec<f32> = vec![1.0; 1 * 1 * 5 * 256];
        let value_data: Vec<f32> = vec![2.0; 1 * 1 * 5 * 256];
        
        let key_value = Value::from_array((vec![1, 1, 5, 256], key_data)).unwrap();
        let value_value = Value::from_array((vec![1, 1, 5, 256], value_data)).unwrap();
        
        // First update
        cache.update(key_value.into(), value_value.into()).unwrap();
        
        assert_eq!(cache.len(), 5);
        assert!(!cache.is_empty());
        assert!(cache.get_cache().is_some());
    }

    #[test]
    fn test_subsequent_update() {
        let mut cache = LayerCache::new(0, 2048, 1, 256);
        
        // First update: 5 tokens
        let key1_data: Vec<f32> = vec![1.0; 1 * 1 * 5 * 256];
        let value1_data: Vec<f32> = vec![2.0; 1 * 1 * 5 * 256];
        let key1 = Value::from_array((vec![1, 1, 5, 256], key1_data)).unwrap();
        let value1 = Value::from_array((vec![1, 1, 5, 256], value1_data)).unwrap();
        cache.update(key1.into(), value1.into()).unwrap();
        assert_eq!(cache.len(), 5);
        
        // Second update: 3 more tokens (for now, this replaces rather than concatenates)
        let key2_data: Vec<f32> = vec![3.0; 1 * 1 * 3 * 256];
        let value2_data: Vec<f32> = vec![4.0; 1 * 1 * 3 * 256];
        let key2 = Value::from_array((vec![1, 1, 3, 256], key2_data)).unwrap();
        let value2 = Value::from_array((vec![1, 1, 3, 256], value2_data)).unwrap();
        cache.update(key2.into(), value2.into()).unwrap();
        
        // Current implementation replaces, so length is 3
        assert_eq!(cache.len(), 3);
        
        let (cached_key, _cached_value) = cache.get_cache().unwrap();
        let shape = cached_key.shape();
        assert_eq!(shape[2], 3); // sequence dimension should be 3
    }

    #[test]
    fn test_trim_to_window() {
        let mut cache = LayerCache::new(0, 10, 1, 256); // max_length = 10
        
        // Add 15 tokens
        let key_data: Vec<f32> = vec![1.0; 1 * 1 * 15 * 256];
        let value_data: Vec<f32> = vec![2.0; 1 * 1 * 15 * 256];
        let key = Value::from_array((vec![1, 1, 15, 256], key_data)).unwrap();
        let value = Value::from_array((vec![1, 1, 15, 256], value_data)).unwrap();
        cache.update(key.into(), value.into()).unwrap();
        assert_eq!(cache.len(), 15);
        
        // Trim to window (currently a no-op but doesn't error)
        cache.trim_to_window().unwrap();
        
        // Length stays 15 in current implementation (trimming not implemented yet)
        assert_eq!(cache.len(), 15);
    }

    #[test]
    fn test_trim_when_under_max() {
        let mut cache = LayerCache::new(0, 100, 1, 256);
        
        // Add only 5 tokens (under max)
        let key_data: Vec<f32> = vec![1.0; 1 * 1 * 5 * 256];
        let value_data: Vec<f32> = vec![2.0; 1 * 1 * 5 * 256];
        let key = Value::from_array((vec![1, 1, 5, 256], key_data)).unwrap();
        let value = Value::from_array((vec![1, 1, 5, 256], value_data)).unwrap();
        cache.update(key.into(), value.into()).unwrap();
        
        // Trim should do nothing
        cache.trim_to_window().unwrap();
        assert_eq!(cache.len(), 5);
    }

    // CacheManager Tests
    
    #[test]
    fn test_cache_manager_new() {
        let config = ModelConfig::default();
        let manager = CacheManager::new(&config);
        
        assert_eq!(manager.num_layers(), 18); // Gemma 270M has 18 layers
        assert!(manager.is_empty());
        assert_eq!(manager.current_length(), 0);
        assert_eq!(manager.total_tokens_processed(), 0);
    }
    
    #[test]
    fn test_cache_manager_update_all_layers() {
        let config = ModelConfig::default();
        let mut manager = CacheManager::new(&config);
        
        // Create KV pairs for all 18 layers
        let mut kv_pairs = Vec::new();
        for _ in 0..18 {
            let key_data: Vec<f32> = vec![1.0; 1 * 1 * 5 * 256];
            let value_data: Vec<f32> = vec![2.0; 1 * 1 * 5 * 256];
            let key = Value::from_array((vec![1, 1, 5, 256], key_data)).unwrap();
            let value = Value::from_array((vec![1, 1, 5, 256], value_data)).unwrap();
            kv_pairs.push((key.into(), value.into()));
        }
        
        // Update all layers
        manager.update_all_layers(kv_pairs).unwrap();
        
        assert_eq!(manager.current_length(), 5);
        assert!(!manager.is_empty());
        assert_eq!(manager.total_tokens_processed(), 5);
    }
    
    #[test]
    fn test_cache_manager_update_wrong_length() {
        let config = ModelConfig::default();
        let mut manager = CacheManager::new(&config);
        
        // Create KV pairs for only 5 layers (should be 18)
        let mut kv_pairs = Vec::new();
        for _ in 0..5 {
            let key_data: Vec<f32> = vec![1.0; 1 * 1 * 5 * 256];
            let value_data: Vec<f32> = vec![2.0; 1 * 1 * 5 * 256];
            let key = Value::from_array((vec![1, 1, 5, 256], key_data)).unwrap();
            let value = Value::from_array((vec![1, 1, 5, 256], value_data)).unwrap();
            kv_pairs.push((key.into(), value.into()));
        }
        
        // Should error due to length mismatch
        let result = manager.update_all_layers(kv_pairs);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_cache_manager_get_all_caches() {
        let config = ModelConfig::default();
        let mut manager = CacheManager::new(&config);
        
        // Initially all caches should be None
        let caches = manager.get_all_caches();
        assert_eq!(caches.len(), 18);
        assert!(caches.iter().all(|c| c.is_none()));
        
        // Update all layers
        let mut kv_pairs = Vec::new();
        for _ in 0..18 {
            let key_data: Vec<f32> = vec![1.0; 1 * 1 * 5 * 256];
            let value_data: Vec<f32> = vec![2.0; 1 * 1 * 5 * 256];
            let key = Value::from_array((vec![1, 1, 5, 256], key_data)).unwrap();
            let value = Value::from_array((vec![1, 1, 5, 256], value_data)).unwrap();
            kv_pairs.push((key.into(), value.into()));
        }
        manager.update_all_layers(kv_pairs).unwrap();
        
        // Now all caches should have values
        let caches = manager.get_all_caches();
        assert_eq!(caches.len(), 18);
        assert!(caches.iter().all(|c| c.is_some()));
    }
    
    #[test]
    fn test_cache_manager_enforce_window() {
        let config = ModelConfig::default();
        let mut manager = CacheManager::new(&config);
        
        // Create KV pairs with length that exceeds max_cache_tokens (2048)
        let mut kv_pairs = Vec::new();
        for _ in 0..18 {
            let key_data: Vec<f32> = vec![1.0; 1 * 1 * 2500 * 256]; // 2500 > 2048
            let value_data: Vec<f32> = vec![2.0; 1 * 1 * 2500 * 256];
            let key = Value::from_array((vec![1, 1, 2500, 256], key_data)).unwrap();
            let value = Value::from_array((vec![1, 1, 2500, 256], value_data)).unwrap();
            kv_pairs.push((key.into(), value.into()));
        }
        
        manager.update_all_layers(kv_pairs).unwrap();
        assert_eq!(manager.current_length(), 2500);
        
        // Enforce window (currently just logs warning)
        manager.enforce_window().unwrap();
    }
    
    #[test]
    fn test_cache_manager_reset_all() {
        let config = ModelConfig::default();
        let mut manager = CacheManager::new(&config);
        
        // Update all layers
        let mut kv_pairs = Vec::new();
        for _ in 0..18 {
            let key_data: Vec<f32> = vec![1.0; 1 * 1 * 5 * 256];
            let value_data: Vec<f32> = vec![2.0; 1 * 1 * 5 * 256];
            let key = Value::from_array((vec![1, 1, 5, 256], key_data)).unwrap();
            let value = Value::from_array((vec![1, 1, 5, 256], value_data)).unwrap();
            kv_pairs.push((key.into(), value.into()));
        }
        manager.update_all_layers(kv_pairs).unwrap();
        
        assert_eq!(manager.current_length(), 5);
        assert_eq!(manager.total_tokens_processed(), 5);
        assert!(!manager.is_empty());
        
        // Reset all
        manager.reset_all();
        
        assert_eq!(manager.current_length(), 0);
        assert_eq!(manager.total_tokens_processed(), 0);
        assert!(manager.is_empty());
        
        // All caches should be None
        let caches = manager.get_all_caches();
        assert!(caches.iter().all(|c| c.is_none()));
    }
    
    #[test]
    fn test_cache_manager_total_tokens_accumulation() {
        let config = ModelConfig::default();
        let mut manager = CacheManager::new(&config);
        
        // First update: 5 tokens
        let mut kv_pairs = Vec::new();
        for _ in 0..18 {
            let key_data: Vec<f32> = vec![1.0; 1 * 1 * 5 * 256];
            let value_data: Vec<f32> = vec![2.0; 1 * 1 * 5 * 256];
            let key = Value::from_array((vec![1, 1, 5, 256], key_data)).unwrap();
            let value = Value::from_array((vec![1, 1, 5, 256], value_data)).unwrap();
            kv_pairs.push((key.into(), value.into()));
        }
        manager.update_all_layers(kv_pairs).unwrap();
        assert_eq!(manager.total_tokens_processed(), 5);
        
        // Second update: 3 more tokens
        let mut kv_pairs2 = Vec::new();
        for _ in 0..18 {
            let key_data: Vec<f32> = vec![3.0; 1 * 1 * 3 * 256];
            let value_data: Vec<f32> = vec![4.0; 1 * 1 * 3 * 256];
            let key = Value::from_array((vec![1, 1, 3, 256], key_data)).unwrap();
            let value = Value::from_array((vec![1, 1, 3, 256], value_data)).unwrap();
            kv_pairs2.push((key.into(), value.into()));
        }
        manager.update_all_layers(kv_pairs2).unwrap();
        
        // Total should be 5 + 3 = 8
        assert_eq!(manager.total_tokens_processed(), 8);
        // Current length is 3 (because update replaces)
        assert_eq!(manager.current_length(), 3);
    }
}
