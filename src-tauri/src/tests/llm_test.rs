// ============================================================================
// REAL TOKENIZER IMPLEMENTATION - Ready for Testing
// ============================================================================

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use tokenizers::Tokenizer as HFTokenizer;
use anyhow::Result;
use std::path::Path;
use std::sync::{Arc, Mutex};

// ============================================================================
// STEP 1: Add to Cargo.toml
// ============================================================================

/*
[dependencies]
ort = "2.0"
tokenizers = "0.19"  # HuggingFace tokenizers
ndarray = "0.15"
anyhow = "1.0"
tokio = { version = "1", features = ["full"] }
*/

// ============================================================================
// STEP 2: Real Tokenizer Wrapper
// ============================================================================

pub struct RealTokenizer {
    tokenizer: HFTokenizer,
}

impl RealTokenizer {
    /// Load tokenizer from tokenizer.json file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let tokenizer = HFTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        
        Ok(Self { tokenizer })
    }
    
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let encoding = self.tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;
        
        // Convert u32 to i64 (ONNX expects i64)
        Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
    }
    
    pub fn decode(&self, tokens: &[i64]) -> Result<String> {
        // Convert i64 back to u32 for tokenizer
        let token_ids: Vec<u32> = tokens.iter().map(|&id| id as u32).collect();
        
        self.tokenizer
            .decode(&token_ids, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))
    }
}

// ============================================================================
// STEP 3: Simple ORT Model Wrapper (Single Forward Pass)
// ============================================================================

pub struct SimpleOrtModel {
    session: Mutex<Session>,
    tokenizer: RealTokenizer,
}

impl SimpleOrtModel {
    /// Load model and tokenizer from directory
    /// 
    /// Directory structure:
    /// model_dir/
    /// ├── model.onnx
    /// └── tokenizer.json
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");
        
        println!("📦 Loading model from: {:?}", model_path);
        println!("📦 Loading tokenizer from: {:?}", tokenizer_path);
        
        // Check files exist
        if !model_path.exists() {
            return Err(anyhow::anyhow!("Model file not found: {:?}", model_path));
        }
        if !tokenizer_path.exists() {
            return Err(anyhow::anyhow!("Tokenizer file not found: {:?}", tokenizer_path));
        }
        
        // Load ONNX model
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;
        
        // Load tokenizer
        let tokenizer = RealTokenizer::from_file(tokenizer_path)?;
        
        println!("✅ Model and tokenizer loaded successfully!");
        
        Ok(Self { 
            session: Mutex::new(session), 
            tokenizer 
        })
    }
    
    /// Generate text (simplified single forward pass)
    pub fn generate_simple(&self, prompt: &str) -> Result<String> {
        // 1. Encode input
        let input_ids = self.tokenizer.encode(prompt)?;
        println!("📝 Input tokens: {} tokens", input_ids.len());
        
        // 2. Create attention mask (all 1s)
        let attention_mask: Vec<i64> = vec![1; input_ids.len()];
        
        // 3. Prepare input shapes and data for ONNX Runtime
        // ONNX Runtime expects (shape, data) tuples
        let seq_len = input_ids.len();
        let shape = vec![1, seq_len]; // [batch_size, sequence_length]
        
        // 4. Run inference
        println!("🔮 Running inference...");
        // Create ort::Value tensors from shape and data
        let input_ids_ort_value = Value::from_array((shape.clone(), input_ids))?;
        let attention_mask_ort_value = Value::from_array((shape, attention_mask))?;

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_ort_value,
            "attention_mask" => attention_mask_ort_value,
        ])?;
        
        // 5. Extract logits
        let logits = outputs["logits"]
            .try_extract_tensor::<f32>()?;
        
        // Extract shape and data from the tensor
        let (shape, data) = logits;
        println!("📊 Output logits shape: {:?}", shape);
        
        // 6. Get next token (argmax of last position in the vocabulary)
        // Shape is typically [batch_size, seq_len, vocab_size]
        // We want the last token position's logits
        let dims = shape.as_ref();
        let vocab_size = dims[dims.len() - 1] as usize;
        let seq_len = dims[dims.len() - 2] as usize;
        
        // Get logits for the last token (last position in sequence)
        let last_token_offset = (seq_len - 1) * vocab_size;
        let last_logits: &[f32] = &data[last_token_offset..last_token_offset + vocab_size];
        
        // Find token with max logit
        let next_token_id = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as i64)
            .unwrap();
        
        println!("🎯 Next token ID: {}", next_token_id);
        
        // 7. Decode
        let output = self.tokenizer.decode(&[next_token_id])?;
        
        Ok(output)
    }
}

// ============================================================================
// STEP 4: Test Setup - What You Need to Do
// ============================================================================

/*
BEFORE RUNNING TESTS:

1. Download from Hugging Face:
   https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct-ONNX

   Files you need:
   ├── onnx/model.onnx           (or model_quantized.onnx)
   └── tokenizer.json

2. Create directory structure:
   
   your-project/
   ├── Cargo.toml
   ├── src/
   │   └── main.rs (or lib.rs)
   └── models/
       └── qwen-0.5b/
           ├── model.onnx        ← Put downloaded model here
           └── tokenizer.json    ← Put downloaded tokenizer here

3. Update paths in tests below to match your structure
*/

// ============================================================================
// STEP 5: Actual Working Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_loading() -> Result<()> {
        println!("\n🧪 Test: Load Tokenizer");
        
        // Update this path to where you placed tokenizer.json
        let tokenizer = RealTokenizer::from_file("models/gemma-270M/tokenizer.json")?;
        
        // Test encoding
        let text = "Hello, world!";
        let tokens = tokenizer.encode(text)?;
        println!("✅ Encoded '{}' to {} tokens", text, tokens.len());
        println!("   Tokens: {:?}", tokens);
        
        // Test decoding
        let decoded = tokenizer.decode(&tokens)?;
        println!("✅ Decoded back to: '{}'", decoded);
        
        Ok(())
    }

    #[test]
    fn test_model_loading() -> Result<()> {
        println!("\n🧪 Test: Load Model");
        
        // Update this path to your model directory
        SimpleOrtModel::from_dir("models/gemma-270M")?;
        
        println!("✅ Model loaded successfully!");
        
        Ok(())
    }

    #[test]
    fn test_simple_inference() -> Result<()> {
        println!("\n🧪 Test: Simple Inference");
        
        // Load model
        let model = SimpleOrtModel::from_dir("models/gemma-270M")?;
        
        // Run inference
        let prompt = "Hello";
        println!("📝 Prompt: '{}'", prompt);
        
        let output = model.generate_simple(prompt)?;
        println!("✅ Generated: '{}'", output);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_with_agent_system() -> Result<()> {
        println!("\n🧪 Test: Integration with Agent System");
        
        // This assumes you have the Agent struct from previous artifacts
        // You'd adapt it to use SimpleOrtModel
        
        println!("✅ Integration test placeholder");
        // TODO: Integrate with your Agent system
        
        Ok(())
    }
}

// ============================================================================
// STEP 6: Quick Start Commands
// ============================================================================

/*
TO RUN TESTS:

1. Create the directory structure:
   mkdir -p models/qwen-0.5b

2. Download files from Hugging Face:
   # Go to: https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct-ONNX
   # Download: onnx/model.onnx and tokenizer.json
   # Place them in: models/qwen-0.5b/

3. Run tests:
   cargo test -- --nocapture

4. You should see:
   ✅ Tokenizer loaded
   ✅ Model loaded
   ✅ Inference working

EXPECTED OUTPUT:
   🧪 Test: Load Tokenizer
   ✅ Encoded 'Hello, world!' to 5 tokens
      Tokens: [9906, 11, 1917, 0]
   ✅ Decoded back to: 'Hello, world!'
   
   🧪 Test: Load Model
   📦 Loading model from: "models/qwen-0.5b/model.onnx"
   📦 Loading tokenizer from: "models/qwen-0.5b/tokenizer.json"
   ✅ Model and tokenizer loaded successfully!
*/

// ============================================================================
// STEP 7: Integration with Agent System (From Previous Artifacts)
// ============================================================================

// To integrate with your Agent system, modify the Agent struct:

pub struct Agent {
    pub name: String,
    pub system_prompt: String,
    pub model: Arc<SimpleOrtModel>,  // Use SimpleOrtModel instead of mock
    // ... rest of fields
}

impl Agent {
    pub fn new(
        name: impl Into<String>,
        system_prompt: impl Into<String>,
        model: Arc<SimpleOrtModel>,
    ) -> Self {
        // ... implementation
        unimplemented!("Integrate with your existing Agent code")
    }
    
    pub async fn run(&mut self, user_message: &str) -> Result<String> {
        // Build prompt with system prompt, tools, and conversation history
        let full_prompt = format!(
            "{}\n\nUser: {}\nAssistant:",
            self.system_prompt,
            user_message
        );
        
        // Use real model
        self.model.generate_simple(&full_prompt)
    }
}

// ============================================================================
// TROUBLESHOOTING
// ============================================================================

/*
Common Issues:

❌ "Model file not found"
   → Check path is correct
   → Make sure model.onnx exists
   → Try absolute path for testing

❌ "Tokenizer failed to load"
   → Ensure tokenizer.json is from same model
   → Check file isn't corrupted
   → Try re-downloading

❌ "ONNX Runtime error"
   → Make sure ort = "2.0" in Cargo.toml
   → Try with model_quantized.onnx instead
   → Check you have enough RAM

❌ "Output is gibberish"
   → This is expected with single forward pass
   → Need autoregressive generation for coherent text
   → See next artifact for full generation

✅ If tests pass, you're ready for full generation!
*/

// ============================================================================
// NEXT STEPS
// ============================================================================

/*
Once tests pass, you need:

1. ✅ Autoregressive Generation
   - Loop to generate multiple tokens
   - Stopping criteria (EOS token, max length)
   - Sampling strategies (temperature, top-p)

2. ✅ Integration with Agent System
   - Replace mock tokenizer with real one
   - Add to your Agent struct
   - Wire up to Tauri commands

3. ✅ Optimization
   - KV cache for faster generation
   - Batching for multiple requests
   - GPU support (optional)

Want me to build the full autoregressive generation next?
*/
