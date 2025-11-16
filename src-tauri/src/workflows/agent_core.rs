use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer as HFTokenizer;
use std::path::Path;
use ndarray::Array1;

// Import shared types
use crate::shared_types::{Tool, ToolParameter, Message, MessageRole, ToolCall};

// Import KV cache and config
use crate::kv_cache::CacheManager;
use crate::config::ModelConfig;
use crate::error::{Result, LlmError};

// ============================================================================
// REAL TOKENIZER - Using HuggingFace tokenizers
// ============================================================================

pub struct RealTokenizer {
    tokenizer: HFTokenizer,
}

impl RealTokenizer {
    /// Load tokenizer from tokenizer.json file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let tokenizer = HFTokenizer::from_file(path)
            .map_err(|e| LlmError::TokenizationError(format!("Failed to load tokenizer: {}", e)))?;
        
        Ok(Self { tokenizer })
    }
    
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let encoding = self.tokenizer
            .encode(text, false)
            .map_err(|e| LlmError::TokenizationError(format!("Encoding failed: {}", e)))?;
        
        // Convert u32 to i64 (ONNX expects i64)
        Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
    }
    
    pub fn decode(&self, tokens: &[i64]) -> Result<String> {
        // Convert i64 back to u32 for tokenizer
        let token_ids: Vec<u32> = tokens.iter().map(|&id| id as u32).collect();
        
        self.tokenizer
            .decode(&token_ids, true)
            .map_err(|e| LlmError::TokenizationError(format!("Decoding failed: {}", e)))
    }
}

// ============================================================================
// ORT MODEL WRAPPER - Handles ONNX inference with KV cache
// ============================================================================

pub struct OrtModel {
    session: Mutex<Session>,
    tokenizer: Arc<RealTokenizer>,
    cache_manager: Arc<Mutex<CacheManager>>,
    config: ModelConfig,
}

impl OrtModel {
    /// Returns a reference to the internal `CacheManager` for testing.
    #[cfg(test)]
    pub fn get_cache_manager(&self) -> Arc<Mutex<CacheManager>> {
        Arc::clone(&self.cache_manager)
    }

    /// Returns a reference to the model config for testing.
    #[cfg(test)]
    pub fn get_config(&self) -> &ModelConfig {
        &self.config
    }

    /// Returns a reference to the tokenizer for testing.
    #[cfg(test)]
    pub fn get_tokenizer(&self) -> Arc<RealTokenizer> {
        Arc::clone(&self.tokenizer)
    }

    pub fn new(model_path: &str, tokenizer: Arc<RealTokenizer>, config: ModelConfig) -> Result<Self> {
        // For ort 2.0.0-rc.10
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;
        
        // Create cache manager
        let cache_manager = Arc::new(Mutex::new(CacheManager::new(&config)));
        
        Ok(Self { 
            session: Mutex::new(session),
            tokenizer,
            cache_manager,
            config,
        })
    }

    /// Load model from directory (expects model.onnx and tokenizer.json)
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");
        
        println!("📂 Model directory: {:?}", model_dir);
        println!("📂 Model path: {:?}", model_path);
        println!("📂 Model exists: {}", model_path.exists());
        
        // Check for external data file
        let external_data_path = model_dir.join("model.onnx_data");
        if !external_data_path.exists() {
            println!("⚠️  Warning: model.onnx_data not found. Model may have external data.");
            println!("   If loading fails, the model might need to be re-exported without external data.");
        } else {
            println!("✅ Found external data file: model.onnx_data");
        }
        
        let tokenizer = Arc::new(RealTokenizer::from_file(tokenizer_path)?);
        
        // Create default config for Gemma 270M
        let config = ModelConfig::default();
        config.validate_shapes();
        
        let model_path_str = model_path.to_str()
            .ok_or_else(|| LlmError::InvalidConfiguration {
                field: "model_path".to_string(),
                reason: "Path contains invalid UTF-8 characters".to_string(),
            })?;
        
        Self::new(model_path_str, tokenizer, config)
    }

    /// 4.3: Prepare inputs for first pass (no cache)
    fn prepare_first_pass_inputs(&self, input_ids: &[i64]) -> Result<(Value, Value, Value)> {
        let seq_len = input_ids.len();
        let shape = vec![1, seq_len];
        
        // Create input_ids tensor [1, seq_len]
        let input_ids_value = Value::from_array((shape.clone(), input_ids.to_vec()))?.into_dyn();
        
        // Create attention_mask: all ones [1, seq_len]
        let attention_mask_value = Value::from_array((shape.clone(), vec![1i64; seq_len]))?.into_dyn();
        
        // Create position_ids: [0, 1, 2, ..., seq_len-1] [1, seq_len]
        let position_ids: Vec<i64> = (0..seq_len as i64).collect();
        let position_ids_value = Value::from_array((shape, position_ids))?.into_dyn();
        
        Ok((input_ids_value, attention_mask_value, position_ids_value))
    }

    /// 4.4: Prepare inputs for subsequent pass (with cache)
    fn prepare_cached_pass_inputs(&self, new_token_id: i64, cache_length: usize) -> Result<(Value, Value, Value)> {
        // Create input_ids tensor [1, 1] with just the new token
        let input_ids_value = Value::from_array((vec![1, 1], vec![new_token_id]))?.into_dyn();
        
        // Create attention_mask: all ones [1, cache_length + 1]
        let attention_mask_value = Value::from_array((vec![1, cache_length + 1], vec![1i64; cache_length + 1]))?.into_dyn();
        
        // Create position_ids: [cache_length] [1, 1]
        let position_ids_value = Value::from_array((vec![1, 1], vec![cache_length as i64]))?.into_dyn();
        
        Ok((input_ids_value, attention_mask_value, position_ids_value))
    }

    /// 4.5: Build inputs map for model inference
    fn build_inputs_map(
        &self,
        input_ids_value: Value,
        attention_mask_value: Value,
        position_ids_value: Value,
        past_kv_cache: Option<Vec<Option<(&Value, &Value)>>>,
    ) -> Result<Vec<(String, Value)>> {
        let mut inputs = Vec::new();
        
        // Add basic inputs
        inputs.push((self.config.io_names.input_ids_name.clone(), input_ids_value));
        inputs.push((self.config.io_names.attention_mask_name.clone(), attention_mask_value));
        
        // Add position_ids if configured
        if let Some(pos_name) = &self.config.io_names.position_ids_name {
            inputs.push((pos_name.clone(), position_ids_value));
        }
        
        // Always add past KV caches - either from cache or empty placeholders
        for layer_idx in 0..self.config.num_layers {
            let key_name = self.config.input_name_for_past_kv(layer_idx, true);
            let value_name = self.config.input_name_for_past_kv(layer_idx, false);
            
            if let Some(ref caches) = past_kv_cache {
                // Cached pass: use existing cache
                if let Some((key, value)) = caches[layer_idx] {
                    // Extract f32 tensors and recreate as owned Values
                    let (key_shape, key_data) = key.try_extract_tensor::<f32>()?;
                    let (value_shape, value_data) = value.try_extract_tensor::<f32>()?;
                    
                    let key_owned = Value::from_array((key_shape.to_vec(), key_data.to_vec()))?.into_dyn();
                    let value_owned = Value::from_array((value_shape.to_vec(), value_data.to_vec()))?.into_dyn();
                    
                    inputs.push((key_name, key_owned));
                    inputs.push((value_name, value_owned));
                }
            } else {
                // First pass: provide empty cache tensors [1, num_heads, 1, head_dim] with zeros
                let empty_shape = vec![1i64, self.config.num_attention_heads as i64, 1i64, self.config.head_dim as i64];
                let total_elements = (self.config.num_attention_heads * self.config.head_dim) as usize;
                let empty_data: Vec<f32> = vec![0.0; total_elements];
                
                let empty_key = Value::from_array((empty_shape.clone(), empty_data.clone()))?.into_dyn();
                let empty_value = Value::from_array((empty_shape, empty_data))?.into_dyn();
                
                inputs.push((key_name, empty_key));
                inputs.push((value_name, empty_value));
            }
        }
        
        Ok(inputs)
    }

    /// 4.6-4.7: Run inference and extract outputs
    fn run_inference_and_extract(&self, inputs: Vec<(String, Value)>) -> Result<(Array1<f32>, Vec<(Value, Value)>)> {
        let mut session = self.session.lock().unwrap();
        let outputs = session.run(inputs)?;
        
        // 4.7a: Extract logits
        let logits_value = outputs.get(&self.config.io_names.logits_output_name)
            .ok_or_else(|| LlmError::missing_output(&self.config.io_names.logits_output_name))?;
        
        let (shape, data) = logits_value.try_extract_tensor::<f32>()
            .map_err(|e| LlmError::TensorExtractionFailed {
                tensor_name: self.config.io_names.logits_output_name.clone(),
                reason: e.to_string(),
            })?;
        
        if shape.len() < 3 {
            return Err(LlmError::invalid_output_shape(
                &self.config.io_names.logits_output_name,
                "at least 3 dimensions",
                format!("{} dimensions", shape.len()),
            ));
        }
        
        let seq_len = shape[1] as usize;
        let vocab_size = shape[2] as usize;
        let last_token_offset = (seq_len - 1) * vocab_size;
        let last_logits = &data[last_token_offset..last_token_offset + vocab_size];
        let logits = Array1::from_vec(last_logits.to_vec());
        
        // 4.7b: Extract present KV caches
        let mut kv_pairs = Vec::with_capacity(self.config.num_layers);
        
        for layer_idx in 0..self.config.num_layers {
            let key_name = self.config.output_name_for_present_kv(layer_idx, true);
            let value_name = self.config.output_name_for_present_kv(layer_idx, false);
            
            let key = outputs.get(&key_name)
                .ok_or_else(|| LlmError::missing_output(&key_name))?;
            let value = outputs.get(&value_name)
                .ok_or_else(|| LlmError::missing_output(&value_name))?;
            
            let (key_shape, key_data) = key.try_extract_tensor::<f32>()
                .map_err(|e| LlmError::tensor_extraction_failed(&key_name, e.to_string()))?;
            let (value_shape, value_data) = value.try_extract_tensor::<f32>()
                .map_err(|e| LlmError::tensor_extraction_failed(&value_name, e.to_string()))?;
            
            let key_owned = Value::from_array((key_shape.to_vec(), key_data.to_vec()))
                .map_err(|e| LlmError::TensorCreationFailed {
                    reason: format!("Failed to create key tensor for layer {}: {}", layer_idx, e),
                })?
                .into_dyn();
            let value_owned = Value::from_array((value_shape.to_vec(), value_data.to_vec()))
                .map_err(|e| LlmError::TensorCreationFailed {
                    reason: format!("Failed to create value tensor for layer {}: {}", layer_idx, e),
                })?
                .into_dyn();
            
            kv_pairs.push((key_owned, value_owned));
        }
        
        Ok((logits, kv_pairs))
    }

    /// 4.11: Sample token from logits (greedy sampling for now)
    fn sample_token(&self, logits: Array1<f32>, _temperature: f32) -> i64 {
        // For now, just use greedy sampling (argmax)
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as i64)
            .unwrap()
    }

    /// 4.8-4.10: Generate next token with cache management
    fn generate_next_token(&self, input_tokens: Option<&[i64]>) -> Result<i64> {
        let cache_manager = self.cache_manager.lock().unwrap();
        
        // Determine if this is first pass or cached pass
        let is_first_pass = cache_manager.is_empty();
        
        let (input_ids_value, attention_mask_value, position_ids_value) = if is_first_pass {
            // First pass: use full input_tokens
            let tokens = input_tokens.ok_or_else(|| LlmError::InvalidInput {
                reason: "First pass requires input tokens".to_string(),
            })?;
            self.prepare_first_pass_inputs(tokens)?
        } else {
            // Subsequent pass: use last generated token
            let last_token = input_tokens
                .and_then(|t| t.last().copied())
                .ok_or_else(|| LlmError::InvalidInput {
                    reason: "No token provided for cached pass".to_string(),
                })?;
            let cache_len = cache_manager.current_length();
            self.prepare_cached_pass_inputs(last_token, cache_len)?
        };
        
        // Get cached KV if available
        let past_kv = if is_first_pass {
            None
        } else {
            Some(cache_manager.get_all_caches())
        };
        
        // Build inputs map
        let inputs = self.build_inputs_map(
            input_ids_value,
            attention_mask_value,
            position_ids_value,
            past_kv,
        )?;
        
        // Release cache_manager lock before inference
        drop(cache_manager);
        
        // Run inference and extract outputs
        let (logits, present_kv) = self.run_inference_and_extract(inputs)?;
        
        // Update cache manager
        let mut cache_manager = self.cache_manager.lock().unwrap();
        cache_manager.update_all_layers(present_kv)?;
        cache_manager.enforce_window()?;
        drop(cache_manager);
        
        // Sample next token
        let next_token = self.sample_token(logits, 0.7);
        
        Ok(next_token)
    }

    /// 4.12: Generate sequence of tokens
    pub fn generate_sequence(
        &self,
        prompt_tokens: &[i64],
        max_new_tokens: usize,
        eos_token_id: i64,
    ) -> Result<Vec<i64>> {
        let mut generated_tokens = Vec::new();
        
        // First iteration: process full prompt
        println!("🔮 First pass: processing {} prompt tokens", prompt_tokens.len());
        let first_token = self.generate_next_token(Some(prompt_tokens))?;
        
        println!("🎯 First generated token: {}", first_token);
        
        // Check for EOS
        if first_token == eos_token_id {
            println!("⚠️  EOS token generated on first pass");
            return Ok(generated_tokens);
        }
        
        generated_tokens.push(first_token);
        
        // Continue generation
        for i in 0..max_new_tokens - 1 {
            let next_token = self.generate_next_token(Some(&[generated_tokens[generated_tokens.len() - 1]]))?;
            
            println!("🎯 Generated token {}: {}", i + 2, next_token);
            
            // Check for EOS
            if next_token == eos_token_id {
                println!("✅ EOS token reached");
                break;
            }
            
            generated_tokens.push(next_token);
        }
        
        Ok(generated_tokens)
    }

    /// 4.13: Reset conversation and cache
    pub fn reset_conversation(&self) {
        let mut cache_manager = self.cache_manager.lock().unwrap();
        cache_manager.reset_all();
        println!("🔄 Cache reset complete");
    }

    /// Autoregressive generation with KV cache support (updated to use new methods)
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        println!("🤖 Starting text generation (max_tokens: {})", max_tokens);
        
        // Reset cache before new generation
        self.reset_conversation();
        
        // Tokenize input
        let prompt_tokens = self.tokenizer.encode(prompt)?;
        println!("📝 Tokenized input: {} tokens", prompt_tokens.len());
        
        // EOS token for Gemma (you may need to adjust this)
        let eos_token_id = 1; // Common EOS token
        
        // Generate tokens
        let generated_tokens = self.generate_sequence(&prompt_tokens, max_tokens, eos_token_id)?;
        
        println!("✅ Generated {} tokens", generated_tokens.len());
        
        // Decode result
        let result = self.tokenizer.decode(&generated_tokens)?;
        println!("💡 Full decode of all generated tokens:");
        println!("   '{}'", result);
        
        Ok(result)
    }
}

// ============================================================================
// AGENT - The main agentic system
// ============================================================================

pub struct Agent {
    pub name: String,
    pub system_prompt: String,
    pub model: Arc<OrtModel>,
    pub tools: HashMap<String, Tool>,
    pub conversation: Vec<Message>,
    pub max_iterations: usize,
}

impl Agent {
    pub fn new(
        name: impl Into<String>,
        system_prompt: impl Into<String>,
        model: Arc<OrtModel>,
    ) -> Self {
        let system_prompt = system_prompt.into();
        let mut conversation = Vec::new();
        conversation.push(Message::system(system_prompt.clone()));

        Self {
            name: name.into(),
            system_prompt,
            model,
            tools: HashMap::new(),
            conversation,
            max_iterations: 10,
        }
    }

    /// Add a tool to the agent's toolbox
    pub fn add_tool(&mut self, tool: Tool) {
        self.tools.insert(tool.name.clone(), tool);
    }

    /// Build the full prompt with tools and conversation history
    fn build_prompt(&self, user_message: &str) -> String {
        let mut prompt = String::new();

        // System prompt
        prompt.push_str(&format!("# SYSTEM\n{}\n\n", self.system_prompt));

        // Available tools
        if !self.tools.is_empty() {
            prompt.push_str("# TOOLS\nYou have access to these tools:\n");
            for tool in self.tools.values() {
                prompt.push_str(&format!("- {}\n", tool.to_prompt_format()));
            }
            prompt.push_str("\nTo use a tool, respond with JSON:\n");
            prompt.push_str("{\"tool\": \"tool_name\", \"args\": {\"param\": \"value\"}}\n\n");
        }

        // Conversation history (skip system message, it's already included)
        prompt.push_str("# CONVERSATION\n");
        for msg in self.conversation.iter().skip(1) {
            match msg.role {
                MessageRole::User => prompt.push_str(&format!("User: {}\n", msg.content)),
                MessageRole::Assistant => prompt.push_str(&format!("Assistant: {}\n", msg.content)),
                MessageRole::Tool => prompt.push_str(&format!("Tool Result: {}\n", msg.content)),
                MessageRole::System => {}
            }
        }

        // Current user message
        prompt.push_str(&format!("User: {}\n", user_message));
        prompt.push_str("Assistant:");

        prompt
    }

    /// Parse model output for tool calls
    fn parse_tool_call(&self, response: &str) -> Option<ToolCall> {
        // Try to parse JSON tool call
        let trimmed = response.trim();
        if trimmed.starts_with('{') && trimmed.contains("\"tool\"") {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(trimmed) {
                if let (Some(tool_name), Some(args)) = (
                    parsed.get("tool").and_then(|v| v.as_str()),
                    parsed.get("args").and_then(|v| v.as_object()),
                ) {
                    let mut arguments = HashMap::new();
                    for (k, v) in args {
                        if let Some(val) = v.as_str() {
                            arguments.insert(k.clone(), val.to_string());
                        }
                    }
                    return Some(ToolCall {
                        id: uuid::Uuid::new_v4().to_string(),
                        tool_name: tool_name.to_string(),
                        arguments,
                    });
                }
            }
        }
        None
    }

    /// Execute a tool call
    fn execute_tool(&self, tool_call: &ToolCall) -> Result<String> {
        let tool = self.tools.get(&tool_call.tool_name)
            .ok_or_else(|| LlmError::InvalidInput {
                reason: format!("Tool not found: {}", tool_call.tool_name),
            })?;
        
        (tool.handler)(&tool_call.arguments)
    }

    /// Main agentic loop - run the agent with tool calling
    pub async fn run(&mut self, user_message: &str) -> Result<String> {
        // Add user message
        self.conversation.push(Message::user(user_message));

        let mut iterations = 0;
        loop {
            if iterations >= self.max_iterations {
                return Err(LlmError::GenerationFailed {
                    reason: "Max iterations reached".to_string(),
                });
            }
            iterations += 1;

            // Build prompt with full context
            let prompt = self.build_prompt(user_message);

            // Get model response
            let response = self.model.generate(&prompt, 256)?;

            // Check if this is a tool call
            if let Some(tool_call) = self.parse_tool_call(&response) {
                println!("🔧 Agent calling tool: {}", tool_call.tool_name);
                
                // Execute the tool
                let tool_result = self.execute_tool(&tool_call)?;
                
                // Add tool call and result to conversation
                let mut assistant_msg = Message::assistant(response);
                assistant_msg.tool_calls = Some(vec![tool_call.clone()]);
                self.conversation.push(assistant_msg);
                
                self.conversation.push(Message::tool_result(
                    &tool_call.id,
                    &tool_result,
                ));

                // Continue loop to let agent see tool result
                continue;
            }

            // No tool call - this is the final response
            self.conversation.push(Message::assistant(&response));
            return Ok(response);
        }
    }

    /// Simple non-agentic query (no tool calling)
    pub fn query(&mut self, user_message: &str) -> Result<String> {
        self.conversation.push(Message::user(user_message));
        let prompt = self.build_prompt(user_message);
        let response = self.model.generate(&prompt, 256)?;
        self.conversation.push(Message::assistant(&response));
        Ok(response)
    }

    /// Reset conversation but keep system prompt
    pub fn reset(&mut self) {
        self.conversation.clear();
        self.conversation.push(Message::system(&self.system_prompt));
        self.model.reset_conversation();
    }
}

// ============================================================================
// EXAMPLE GAME TOOLS
// ============================================================================

pub fn create_game_tools() -> Vec<Tool> {
    vec![
        Tool::new(
            "examine",
            "Examine an object in the current location",
            vec![ToolParameter {
                name: "object".to_string(),
                description: "The object to examine".to_string(),
                param_type: "string".to_string(),
                required: true,
            }],
            |args| {
                let object = args.get("object").ok_or_else(|| LlmError::InvalidInput {
                    reason: "Missing object parameter".to_string(),
                })?;
                Ok(format!("You examine the {}. It appears to be...", object))
            },
        ),
        Tool::new(
            "move",
            "Move to a different location",
            vec![ToolParameter {
                name: "direction".to_string(),
                description: "Direction to move (north, south, east, west)".to_string(),
                param_type: "string".to_string(),
                required: true,
            }],
            |args| {
                let direction = args.get("direction").ok_or_else(|| LlmError::InvalidInput {
                    reason: "Missing direction parameter".to_string(),
                })?;
                Ok(format!("You move {}...", direction))
            },
        ),
        Tool::new(
            "take",
            "Pick up an item",
            vec![ToolParameter {
                name: "item".to_string(),
                description: "The item to pick up".to_string(),
                param_type: "string".to_string(),
                required: true,
            }],
            |args| {
                let item = args.get("item").ok_or_else(|| LlmError::InvalidInput {
                    reason: "Missing item parameter".to_string(),
                })?;
                Ok(format!("You pick up the {}", item))
            },
        ),
        Tool::new(
            "check_inventory",
            "Check your current inventory",
            vec![],
            |_| {
                Ok("Your inventory contains: medkit, translator_pda".to_string())
            },
        ),
        Tool::new(
            "get_xenon_level",
            "Check the xenon contamination level in current area",
            vec![],
            |_| {
                Ok("Xenon level: 4.2 (Caution: Approaching hazardous levels)".to_string())
            },
        ),
    ]
}
