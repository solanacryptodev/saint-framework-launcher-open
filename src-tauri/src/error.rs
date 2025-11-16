//! Error types for LLM inference with KV caching
//!
//! This module defines all error types that can occur during model loading,
//! inference, cache management, and tokenization.

use std::fmt;

/// Result type alias for LLM operations
pub type Result<T> = std::result::Result<T, LlmError>;

/// Main error type for LLM inference operations
#[derive(Debug)]
pub enum LlmError {
    /// Errors from the ONNX Runtime
    OrtError(ort::Error),

    /// Cache-related errors
    CacheShapeMismatch {
        layer: usize,
        expected: String,
        actual: String,
    },

    CacheConcatenationFailed {
        layer: usize,
        reason: String,
    },

    CacheEmpty {
        layer: usize,
    },

    CacheSliceFailed {
        layer: usize,
        reason: String,
    },

    InvalidCacheState {
        reason: String,
    },

    /// Model input/output errors
    InvalidModelOutput {
        output_name: String,
        reason: String,
    },

    MissingInput {
        input_name: String,
    },

    MissingOutput {
        output_name: String,
    },

    InvalidOutputShape {
        output_name: String,
        expected: String,
        actual: String,
    },

    /// Configuration errors
    InvalidConfiguration {
        field: String,
        reason: String,
    },

    ConfigurationMismatch {
        expected: String,
        actual: String,
    },

    /// Tokenization errors
    TokenizationError(String),

    /// Generation errors
    GenerationFailed {
        reason: String,
    },

    SamplingFailed {
        reason: String,
    },

    /// Tensor operation errors
    TensorCreationFailed {
        reason: String,
    },

    TensorExtractionFailed {
        tensor_name: String,
        reason: String,
    },

    ShapeConversionFailed {
        reason: String,
    },

    /// Layer count mismatch
    LayerCountMismatch {
        expected: usize,
        actual: usize,
    },

    /// Invalid input
    InvalidInput {
        reason: String,
    },

    /// End of sequence reached
    EndOfSequence,
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmError::OrtError(err) => write!(f, "ONNX Runtime error: {}", err),

            LlmError::CacheShapeMismatch { layer, expected, actual } => {
                write!(
                    f,
                    "Cache shape mismatch at layer {}: expected {}, got {}",
                    layer, expected, actual
                )
            }

            LlmError::CacheConcatenationFailed { layer, reason } => {
                write!(
                    f,
                    "Failed to concatenate cache at layer {}: {}",
                    layer, reason
                )
            }

            LlmError::CacheEmpty { layer } => {
                write!(f, "Cache is empty at layer {}", layer)
            }

            LlmError::CacheSliceFailed { layer, reason } => {
                write!(
                    f,
                    "Failed to slice cache at layer {}: {}",
                    layer, reason
                )
            }

            LlmError::InvalidCacheState { reason } => {
                write!(f, "Invalid cache state: {}", reason)
            }

            LlmError::InvalidModelOutput { output_name, reason } => {
                write!(
                    f,
                    "Invalid model output '{}': {}",
                    output_name, reason
                )
            }

            LlmError::MissingInput { input_name } => {
                write!(f, "Missing required model input: '{}'", input_name)
            }

            LlmError::MissingOutput { output_name } => {
                write!(f, "Missing expected model output: '{}'", output_name)
            }

            LlmError::InvalidOutputShape { output_name, expected, actual } => {
                write!(
                    f,
                    "Invalid output shape for '{}': expected {}, got {}",
                    output_name, expected, actual
                )
            }

            LlmError::InvalidConfiguration { field, reason } => {
                write!(
                    f,
                    "Invalid configuration for '{}': {}",
                    field, reason
                )
            }

            LlmError::ConfigurationMismatch { expected, actual } => {
                write!(
                    f,
                    "Configuration mismatch: expected {}, got {}",
                    expected, actual
                )
            }

            LlmError::TokenizationError(msg) => {
                write!(f, "Tokenization error: {}", msg)
            }

            LlmError::GenerationFailed { reason } => {
                write!(f, "Generation failed: {}", reason)
            }

            LlmError::SamplingFailed { reason } => {
                write!(f, "Token sampling failed: {}", reason)
            }

            LlmError::TensorCreationFailed { reason } => {
                write!(f, "Failed to create tensor: {}", reason)
            }

            LlmError::TensorExtractionFailed { tensor_name, reason } => {
                write!(
                    f,
                    "Failed to extract tensor '{}': {}",
                    tensor_name, reason
                )
            }

            LlmError::ShapeConversionFailed { reason } => {
                write!(f, "Failed to convert shape: {}", reason)
            }

            LlmError::LayerCountMismatch { expected, actual } => {
                write!(
                    f,
                    "Layer count mismatch: expected {} layers, got {}",
                    expected, actual
                )
            }

            LlmError::InvalidInput { reason } => {
                write!(f, "Invalid input: {}", reason)
            }

            LlmError::EndOfSequence => {
                write!(f, "End of sequence reached")
            }
        }
    }
}

impl std::error::Error for LlmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LlmError::OrtError(err) => Some(err),
            _ => None,
        }
    }
}

// Automatic conversion from ort::Error
impl From<ort::Error> for LlmError {
    fn from(err: ort::Error) -> Self {
        LlmError::OrtError(err)
    }
}

// Automatic conversion from tokenizers::Error (if using tokenizers crate)
// Uncomment if you're using the tokenizers crate
// impl From<tokenizers::Error> for LlmError {
//     fn from(err: tokenizers::Error) -> Self {
//         LlmError::TokenizationError(err.to_string())
//     }
// }

// Helper methods for common error constructions
impl LlmError {
    /// Create a cache shape mismatch error
    pub fn cache_shape_mismatch(
        layer: usize,
        expected: impl fmt::Display,
        actual: impl fmt::Display,
    ) -> Self {
        LlmError::CacheShapeMismatch {
            layer,
            expected: expected.to_string(),
            actual: actual.to_string(),
        }
    }

    /// Create an invalid output shape error
    pub fn invalid_output_shape(
        output_name: impl Into<String>,
        expected: impl fmt::Display,
        actual: impl fmt::Display,
    ) -> Self {
        LlmError::InvalidOutputShape {
            output_name: output_name.into(),
            expected: expected.to_string(),
            actual: actual.to_string(),
        }
    }

    /// Create a missing output error
    pub fn missing_output(output_name: impl Into<String>) -> Self {
        LlmError::MissingOutput {
            output_name: output_name.into(),
        }
    }

    /// Create a missing input error
    pub fn missing_input(input_name: impl Into<String>) -> Self {
        LlmError::MissingInput {
            input_name: input_name.into(),
        }
    }

    /// Create an invalid configuration error
    pub fn invalid_config(field: impl Into<String>, reason: impl Into<String>) -> Self {
        LlmError::InvalidConfiguration {
            field: field.into(),
            reason: reason.into(),
        }
    }

    /// Create a tensor extraction error
    pub fn tensor_extraction_failed(
        tensor_name: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        LlmError::TensorExtractionFailed {
            tensor_name: tensor_name.into(),
            reason: reason.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_error_display() {
        let err = LlmError::cache_shape_mismatch(0, "[1, 32, 10, 64]", "[1, 32, 11, 64]");
        let display = format!("{}", err);
        assert!(display.contains("layer 0"));
        assert!(display.contains("[1, 32, 10, 64]"));
    }

    #[test]
    fn test_missing_output_error() {
        let err = LlmError::missing_output("logits");
        let display = format!("{}", err);
        assert!(display.contains("logits"));
        assert!(display.contains("Missing expected model output"));
    }

    #[test]
    fn test_invalid_config_error() {
        let err = LlmError::invalid_config("max_sequence_length", "must be greater than 0");
        let display = format!("{}", err);
        assert!(display.contains("max_sequence_length"));
        assert!(display.contains("must be greater than 0"));
    }

    #[test]
    fn test_layer_count_mismatch() {
        let err = LlmError::LayerCountMismatch {
            expected: 24,
            actual: 12,
        };
        let display = format!("{}", err);
        assert!(display.contains("24"));
        assert!(display.contains("12"));
    }

    #[test]
    fn test_result_type_usage() {
        fn returns_result() -> Result<i32> {
            Ok(42)
        }

        fn returns_error() -> Result<i32> {
            Err(LlmError::EndOfSequence)
        }

        assert!(returns_result().is_ok());
        assert!(returns_error().is_err());
    }

    #[test]
    fn test_error_source() {
        // Test that we can access the source error for OrtError
        // This would require an actual ort::Error to test properly
        let err = LlmError::CacheEmpty { layer: 0 };
        assert!(err.source().is_none());
    }
}
