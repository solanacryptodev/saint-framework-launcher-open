use anyhow::Result;
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::path::Path;

/// Utility to inspect ONNX model inputs and outputs
pub struct OrtModelInspector {
    session: Session,
}

impl OrtModelInspector {
    /// Load an ONNX model from the given path
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;
        
        Ok(Self { session })
    }

    /// Inspect and print all model inputs and outputs
    pub fn inspect(&self) -> Result<()> {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║              ONNX MODEL INSPECTION REPORT                      ║");
        println!("╚════════════════════════════════════════════════════════════════╝\n");

        // Print all inputs
        println!("📥 INPUT TENSORS:");
        println!("─────────────────────────────────────────────────────────────────");
        for (i, input) in self.session.inputs.iter().enumerate() {
            println!("  [{}] Name: {}", i, input.name);
            println!("      Type: {:?}", input.input_type);
            println!();
        }

        // Print all outputs
        println!("📤 OUTPUT TENSORS:");
        println!("─────────────────────────────────────────────────────────────────");
        for (i, output) in self.session.outputs.iter().enumerate() {
            println!("  [{}] Name: {}", i, output.name);
            println!("      Type: {:?}", output.output_type);
            println!();
        }

        // Analyze KV cache structure
        self.analyze_kv_cache_structure()?;

        Ok(())
    }

    /// Analyze and document the KV cache structure
    fn analyze_kv_cache_structure(&self) -> Result<()> {
        println!("🔍 KV CACHE ANALYSIS:");
        println!("─────────────────────────────────────────────────────────────────");

        // Find input token IDs
        let input_ids = self.session.inputs.iter()
            .find(|i| i.name.to_lowercase().contains("input_ids") || i.name == "input_ids");
        
        if let Some(input) = input_ids {
            println!("  ✓ Token IDs Input: '{}'", input.name);
        } else {
            println!("  ✗ Token IDs Input: NOT FOUND");
        }

        // Find attention mask
        let attention_mask = self.session.inputs.iter()
            .find(|i| i.name.to_lowercase().contains("attention_mask") || i.name == "attention_mask");
        
        if let Some(input) = attention_mask {
            println!("  ✓ Attention Mask Input: '{}'", input.name);
        } else {
            println!("  ✗ Attention Mask Input: NOT FOUND");
        }

        // Find position IDs
        let position_ids = self.session.inputs.iter()
            .find(|i| i.name.to_lowercase().contains("position_ids") || i.name == "position_ids");
        
        if let Some(input) = position_ids {
            println!("  ✓ Position IDs Input: '{}'", input.name);
        } else {
            println!("  ⚠ Position IDs Input: NOT FOUND (may be optional)");
        }

        // Find past KV cache inputs
        let past_kv_inputs: Vec<_> = self.session.inputs.iter()
            .filter(|i| i.name.contains("past") || i.name.contains("cache"))
            .collect();
        
        if !past_kv_inputs.is_empty() {
            println!("  ✓ Past KV Cache Inputs ({} found):", past_kv_inputs.len());
            for input in &past_kv_inputs {
                println!("      - '{}'", input.name);
            }
        } else {
            println!("  ⚠ Past KV Cache Inputs: NOT FOUND (model may not support KV caching)");
        }

        // Find logits output
        let logits = self.session.outputs.iter()
            .find(|o| o.name.to_lowercase().contains("logits") || o.name == "logits");
        
        if let Some(output) = logits {
            println!("  ✓ Logits Output: '{}'", output.name);
        } else {
            println!("  ✗ Logits Output: NOT FOUND");
        }

        // Find present KV cache outputs
        let present_kv_outputs: Vec<_> = self.session.outputs.iter()
            .filter(|o| o.name.contains("present") || o.name.contains("cache"))
            .collect();
        
        if !present_kv_outputs.is_empty() {
            println!("  ✓ Present KV Cache Outputs ({} found):", present_kv_outputs.len());
            for output in &present_kv_outputs {
                println!("      - '{}'", output.name);
            }
            
            // Estimate number of transformer layers
            let key_count = present_kv_outputs.iter()
                .filter(|o| o.name.contains("key"))
                .count();
            let value_count = present_kv_outputs.iter()
                .filter(|o| o.name.contains("value"))
                .count();
            
            if key_count == value_count && key_count > 0 {
                println!("  ✓ Estimated Transformer Layers: {} (based on {} key-value pairs)", 
                         key_count, key_count);
            }
        } else {
            println!("  ⚠ Present KV Cache Outputs: NOT FOUND (model may not support KV caching)");
        }

        println!("─────────────────────────────────────────────────────────────────\n");

        Ok(())
    }
}

/// Convenience function to inspect a model from a directory
pub fn inspect_model_from_dir(model_dir: impl AsRef<Path>) -> Result<()> {
    let model_dir = model_dir.as_ref();
    
    // Try different common model filenames
    let possible_names = vec![
        "model_fp16.onnx",
        "model_quantized.onnx",
        "model.onnx",
    ];
    
    for name in possible_names {
        let model_path = model_dir.join(name);
        if model_path.exists() {
            println!("📁 Model Directory: {}", model_dir.display());
            println!("📄 Model File: {}\n", name);
            
            let inspector = OrtModelInspector::new(model_path)?;
            return inspector.inspect();
        }
    }
    
    Err(anyhow::anyhow!("No ONNX model file found in directory: {}", model_dir.display()))
}
