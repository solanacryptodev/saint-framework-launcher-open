//! Error types for LLM inference operations
//!
//! This module defines error types that can occur during model loading,
//! inference, and generation.

use std::fmt;

/// Result type alias for LLM operations
pub type Result<T> = std::result::Result<T, LlmError>;

/// Main error type for LLM operations
#[derive(Debug)]
pub enum LlmError {
    /// Generation errors
    GenerationFailed { reason: String },

    /// Invalid input
    InvalidInput { reason: String },

    /// Configuration errors
    InvalidConfiguration { field: String, reason: String },

    /// Generic error wrapper
    Other(String),
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmError::GenerationFailed { reason } => {
                write!(f, "Generation failed: {}", reason)
            }
            LlmError::InvalidInput { reason } => {
                write!(f, "Invalid input: {}", reason)
            }
            LlmError::InvalidConfiguration { field, reason } => {
                write!(f, "Invalid configuration for '{}': {}", field, reason)
            }
            LlmError::Other(msg) => {
                write!(f, "{}", msg)
            }
        }
    }
}

impl std::error::Error for LlmError {}

impl From<anyhow::Error> for LlmError {
    fn from(err: anyhow::Error) -> Self {
        LlmError::Other(err.to_string())
    }
}
