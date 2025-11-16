/// Configuration for model-specific parameters and KV cache management
/// 
/// This module provides the core configuration structures for the custom KV cache layer
/// designed for local LLM game systems.

/// Input/Output tensor name mappings for the ONNX model
#[derive(Debug, Clone)]
pub struct IONames {
    /// Name of the input_ids tensor (e.g., "input_ids")
    pub input_ids_name: String,
    
    /// Name of the attention_mask tensor (e.g., "attention_mask")
    pub attention_mask_name: String,
    
    /// Name of the position_ids tensor (optional for some models)
    pub position_ids_name: Option<String>,
    
    /// Template for past key inputs (e.g., "past_key_values.{}.key")
    pub past_key_format: String,
    
    /// Template for past value inputs (e.g., "past_key_values.{}.value")
    pub past_value_format: String,
    
    /// Template for present key outputs (e.g., "present.{}.key")
    pub present_key_format: String,
    
    /// Template for present value outputs (e.g., "present.{}.value")
    pub present_value_format: String,
    
    /// Name of the logits output tensor (e.g., "logits")
    pub logits_output_name: String,
}

impl Default for IONames {
    fn default() -> Self {
        Self {
            input_ids_name: "input_ids".to_string(),
            attention_mask_name: "attention_mask".to_string(),
            position_ids_name: Some("position_ids".to_string()),
            past_key_format: "past_key_values.{}.key".to_string(),
            past_value_format: "past_key_values.{}.value".to_string(),
            present_key_format: "present.{}.key".to_string(),
            present_value_format: "present.{}.value".to_string(),
            logits_output_name: "logits".to_string(),
        }
    }
}

/// Configuration for KV cache management behavior
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of tokens to cache before eviction (e.g., 2048)
    pub max_cache_tokens: usize,
    
    /// Number of recent tokens to always keep (e.g., 512)
    pub keep_recent_tokens: usize,
    
    /// Whether to use sliding window eviction strategy
    pub enable_sliding_window: bool,
    
    /// Batch size (always 1 for demo, but kept for future expansion)
    pub batch_size: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_cache_tokens: 2048,
            keep_recent_tokens: 512,
            enable_sliding_window: true,
            batch_size: 1,
        }
    }
}

/// Core model configuration containing all model-specific parameters
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Number of transformer layers in the model
    pub num_layers: usize,
    
    /// Number of attention heads per layer
    pub num_attention_heads: usize,
    
    /// Dimension per attention head
    pub head_dim: usize,
    
    /// Maximum sequence length the model can handle
    pub max_sequence_length: usize,
    
    /// Size of the token vocabulary
    pub vocab_size: usize,
    
    /// Model's hidden dimension (num_attention_heads * head_dim)
    pub hidden_size: usize,
    
    /// Input/Output tensor name mappings
    pub io_names: IONames,
    
    /// Cache management configuration
    pub cache_config: CacheConfig,
}

impl ModelConfig {
    /// Creates a new ModelConfig with the specified parameters
    /// 
    /// # Arguments
    /// 
    /// * `num_layers` - Number of transformer layers (from model inspection)
    /// * `num_attention_heads` - Number of attention heads per layer
    /// * `head_dim` - Dimension per attention head
    /// * `max_sequence_length` - Maximum tokens to support
    /// * `vocab_size` - Size of token vocabulary
    /// * `hidden_size` - Model's hidden dimension
    /// * `max_cache_tokens` - Optional override for max cache tokens (default: 2048)
    /// * `keep_recent_tokens` - Optional override for keep recent tokens (default: 512)
    /// * `enable_sliding_window` - Optional override for sliding window (default: true)
    /// * `batch_size` - Optional override for batch size (default: 1)
    /// 
    /// # Panics
    /// 
    /// Panics if validation fails:
    /// - `hidden_size` must equal `num_attention_heads * head_dim`
    /// - `max_sequence_length` must be greater than `keep_recent_tokens`
    /// 
    /// # Example
    /// 
    /// ```
    /// let config = ModelConfig::new(
    ///     18,     // num_layers
    ///     1,      // num_attention_heads
    ///     256,    // head_dim
    ///     2048,   // max_sequence_length
    ///     262144, // vocab_size
    ///     256,    // hidden_size
    ///     None,   // use default max_cache_tokens
    ///     None,   // use default keep_recent_tokens
    ///     None,   // use default enable_sliding_window
    ///     None,   // use default batch_size
    /// );
    /// ```
    pub fn new(
        num_layers: usize,
        num_attention_heads: usize,
        head_dim: usize,
        max_sequence_length: usize,
        vocab_size: usize,
        hidden_size: usize,
        max_cache_tokens: Option<usize>,
        keep_recent_tokens: Option<usize>,
        enable_sliding_window: Option<bool>,
        batch_size: Option<usize>,
    ) -> Self {
        // Validate hidden_size matches num_attention_heads * head_dim
        let expected_hidden_size = num_attention_heads * head_dim;
        assert_eq!(
            hidden_size, expected_hidden_size,
            "hidden_size ({}) must equal num_attention_heads * head_dim ({})",
            hidden_size, expected_hidden_size
        );

        // Create cache config with defaults or provided values
        let cache_config = CacheConfig {
            max_cache_tokens: max_cache_tokens.unwrap_or(2048),
            keep_recent_tokens: keep_recent_tokens.unwrap_or(512),
            enable_sliding_window: enable_sliding_window.unwrap_or(true),
            batch_size: batch_size.unwrap_or(1),
        };

        // Validate max_sequence_length is greater than keep_recent_tokens
        assert!(
            max_sequence_length > cache_config.keep_recent_tokens,
            "max_sequence_length ({}) must be greater than keep_recent_tokens ({})",
            max_sequence_length, cache_config.keep_recent_tokens
        );

        Self {
            num_layers,
            num_attention_heads,
            head_dim,
            max_sequence_length,
            vocab_size,
            hidden_size,
            io_names: IONames::default(),
            cache_config,
        }
    }

    /// Generates the input tensor name for past KV cache at a specific layer
    /// 
    /// # Arguments
    /// 
    /// * `layer_idx` - Index of the transformer layer (0-based)
    /// * `is_key` - true for key tensor, false for value tensor
    /// 
    /// # Returns
    /// 
    /// Formatted string like "past_key_values.0.key" or "past_key_values.0.value"
    /// 
    /// # Example
    /// 
    /// ```
    /// let config = ModelConfig::new(...);
    /// let key_name = config.input_name_for_past_kv(0, true);    // "past_key_values.0.key"
    /// let value_name = config.input_name_for_past_kv(0, false); // "past_key_values.0.value"
    /// ```
    pub fn input_name_for_past_kv(&self, layer_idx: usize, is_key: bool) -> String {
        if is_key {
            self.io_names.past_key_format.replace("{}", &layer_idx.to_string())
        } else {
            self.io_names.past_value_format.replace("{}", &layer_idx.to_string())
        }
    }

    /// Generates the output tensor name for present KV cache at a specific layer
    /// 
    /// # Arguments
    /// 
    /// * `layer_idx` - Index of the transformer layer (0-based)
    /// * `is_key` - true for key tensor, false for value tensor
    /// 
    /// # Returns
    /// 
    /// Formatted string like "present.0.key" or "present.0.value"
    /// 
    /// # Example
    /// 
    /// ```
    /// let config = ModelConfig::new(...);
    /// let key_name = config.output_name_for_present_kv(0, true);    // "present.0.key"
    /// let value_name = config.output_name_for_present_kv(0, false); // "present.0.value"
    /// ```
    pub fn output_name_for_present_kv(&self, layer_idx: usize, is_key: bool) -> String {
        if is_key {
            self.io_names.present_key_format.replace("{}", &layer_idx.to_string())
        } else {
            self.io_names.present_value_format.replace("{}", &layer_idx.to_string())
        }
    }

    /// Validates the configuration and logs all values
    /// 
    /// This method checks for obvious errors such as zero values or mismatched dimensions,
    /// and logs all configuration parameters for debugging.
    /// 
    /// # Panics
    /// 
    /// Panics if any critical validation checks fail
    pub fn validate_shapes(&self) {
        println!("=== Model Configuration ===");
        println!("Model Architecture:");
        println!("  num_layers: {}", self.num_layers);
        println!("  num_attention_heads: {}", self.num_attention_heads);
        println!("  head_dim: {}", self.head_dim);
        println!("  hidden_size: {}", self.hidden_size);
        println!("  vocab_size: {}", self.vocab_size);
        println!("  max_sequence_length: {}", self.max_sequence_length);
        
        println!("\nCache Configuration:");
        println!("  max_cache_tokens: {}", self.cache_config.max_cache_tokens);
        println!("  keep_recent_tokens: {}", self.cache_config.keep_recent_tokens);
        println!("  enable_sliding_window: {}", self.cache_config.enable_sliding_window);
        println!("  batch_size: {}", self.cache_config.batch_size);
        
        println!("\nI/O Names:");
        println!("  input_ids: {}", self.io_names.input_ids_name);
        println!("  attention_mask: {}", self.io_names.attention_mask_name);
        println!("  position_ids: {:?}", self.io_names.position_ids_name);
        println!("  past_key_format: {}", self.io_names.past_key_format);
        println!("  past_value_format: {}", self.io_names.past_value_format);
        println!("  present_key_format: {}", self.io_names.present_key_format);
        println!("  present_value_format: {}", self.io_names.present_value_format);
        println!("  logits_output: {}", self.io_names.logits_output_name);
        
        // Perform validation checks
        assert!(self.num_layers > 0, "num_layers must be greater than 0");
        assert!(self.num_attention_heads > 0, "num_attention_heads must be greater than 0");
        assert!(self.head_dim > 0, "head_dim must be greater than 0");
        assert!(self.hidden_size > 0, "hidden_size must be greater than 0");
        assert!(self.vocab_size > 0, "vocab_size must be greater than 0");
        assert!(self.max_sequence_length > 0, "max_sequence_length must be greater than 0");
        
        // Verify hidden_size matches num_attention_heads * head_dim
        let expected_hidden_size = self.num_attention_heads * self.head_dim;
        assert_eq!(
            self.hidden_size, expected_hidden_size,
            "hidden_size mismatch: expected {}, got {}",
            expected_hidden_size, self.hidden_size
        );
        
        // Verify cache configuration is sensible
        assert!(
            self.cache_config.max_cache_tokens >= self.cache_config.keep_recent_tokens,
            "max_cache_tokens must be >= keep_recent_tokens"
        );
        
        assert!(
            self.max_sequence_length >= self.cache_config.max_cache_tokens,
            "max_sequence_length should be >= max_cache_tokens"
        );
        
        println!("\n✓ All validation checks passed!");
        println!("===========================\n");
    }
}

impl Default for ModelConfig {
    /// Creates a default ModelConfig for the Gemma 270M model
    /// based on the specifications in MODEL_SPECS.md
    fn default() -> Self {
        Self::new(
            18,      // num_layers
            1,       // num_attention_heads (KV heads)
            256,     // head_dim
            2048,    // max_sequence_length
            262144,  // vocab_size
            256,     // hidden_size (1 * 256)
            None,    // use default max_cache_tokens (2048)
            None,    // use default keep_recent_tokens (512)
            None,    // use default enable_sliding_window (true)
            None,    // use default batch_size (1)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validates() {
        let config = ModelConfig::default();
        config.validate_shapes();
    }

    #[test]
    fn test_input_name_for_past_kv() {
        let config = ModelConfig::default();
        assert_eq!(config.input_name_for_past_kv(0, true), "past_key_values.0.key");
        assert_eq!(config.input_name_for_past_kv(0, false), "past_key_values.0.value");
        assert_eq!(config.input_name_for_past_kv(17, true), "past_key_values.17.key");
        assert_eq!(config.input_name_for_past_kv(17, false), "past_key_values.17.value");
    }

    #[test]
    fn test_output_name_for_present_kv() {
        let config = ModelConfig::default();
        assert_eq!(config.output_name_for_present_kv(0, true), "present.0.key");
        assert_eq!(config.output_name_for_present_kv(0, false), "present.0.value");
        assert_eq!(config.output_name_for_present_kv(17, true), "present.17.key");
        assert_eq!(config.output_name_for_present_kv(17, false), "present.17.value");
    }

    #[test]
    #[should_panic(expected = "hidden_size")]
    fn test_hidden_size_validation_fails() {
        ModelConfig::new(
            18,
            1,
            256,
            2048,
            262144,
            512, // Wrong! Should be 256 (1 * 256)
            None,
            None,
            None,
            None,
        );
    }

    #[test]
    #[should_panic(expected = "max_sequence_length")]
    fn test_max_sequence_length_validation_fails() {
        ModelConfig::new(
            18,
            1,
            256,
            100, // Too small! Less than default keep_recent_tokens (512)
            262144,
            256,
            None,
            None,
            None,
            None,
        );
    }

    #[test]
    fn test_custom_cache_config() {
        let config = ModelConfig::new(
            18,
            1,
            256,
            4096,
            262144,
            256,
            Some(4096),
            Some(1024),
            Some(false),
            Some(2),
        );
        
        assert_eq!(config.cache_config.max_cache_tokens, 4096);
        assert_eq!(config.cache_config.keep_recent_tokens, 1024);
        assert_eq!(config.cache_config.enable_sliding_window, false);
        assert_eq!(config.cache_config.batch_size, 2);
    }
}
