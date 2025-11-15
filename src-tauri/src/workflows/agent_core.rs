use ort::{Environment, Session, SessionBuilder, Value};
use std::collections::HashMap;
use std::sync::Arc;
use anyhow::Result;

// Import shared types
use crate::shared_types::{Tool, ToolParameter, Message, MessageRole, ToolCall};

// ============================================================================
// ORT MODEL WRAPPER - Handles ONNX inference
// ============================================================================

pub struct OrtModel {
    session: Session,
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
}

// Simple trait for tokenization (you'd implement this for your specific model)
pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<i64>>;
    fn decode(&self, tokens: &[i64]) -> Result<String>;
}

impl OrtModel {
    pub fn new(model_path: &str, tokenizer: Arc<dyn Tokenizer + Send + Sync>) -> Result<Self> {
        // For ort 1.16.3 - create environment then session
        let environment = Arc::new(Environment::builder().build()?);
        let session = SessionBuilder::new(&environment)?.with_model_from_file(model_path)?;
        
        Ok(Self { session, tokenizer })
    }

    pub fn generate(&self, prompt: &str, _max_tokens: usize) -> Result<String> {
        // Tokenize input
        let input_ids = self.tokenizer.encode(prompt)?;
        
        // Prepare input tensor - create proper CowArray for ort 1.16.3
        let input_array = ndarray::Array::from_shape_vec((1, input_ids.len()), input_ids)?;
        let dyn_array = input_array.into_dyn();
        let cow_array: ndarray::CowArray<i64, ndarray::IxDyn> = ndarray::CowArray::from(dyn_array);
        let input_tensor = Value::from_array(self.session.allocator(), &cow_array)?;
        
        // Run inference (simplified - real implementation needs generation loop)
        let outputs = self.session.run(vec![input_tensor])?;
        
        // Extract and decode output tokens (access first output)
        let output_tensor = outputs[0].try_extract::<i64>()?;
        let output_ids: Vec<i64> = output_tensor.view().iter().copied().collect();
        
        self.tokenizer.decode(&output_ids)
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
            .ok_or_else(|| anyhow::anyhow!("Tool not found: {}", tool_call.tool_name))?;
        
        (tool.handler)(&tool_call.arguments)
    }

    /// Main agentic loop - run the agent with tool calling
    pub async fn run(&mut self, user_message: &str) -> Result<String> {
        // Add user message
        self.conversation.push(Message::user(user_message));

        let mut iterations = 0;
        loop {
            if iterations >= self.max_iterations {
                return Err(anyhow::anyhow!("Max iterations reached"));
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
                let object = args.get("object").ok_or_else(|| anyhow::anyhow!("Missing object"))?;
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
                let direction = args.get("direction").ok_or_else(|| anyhow::anyhow!("Missing direction"))?;
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
                let item = args.get("item").ok_or_else(|| anyhow::anyhow!("Missing item"))?;
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

// ============================================================================
// USAGE EXAMPLE
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Mock tokenizer for testing
    struct MockTokenizer;
    impl Tokenizer for MockTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<i64>> {
            Ok(text.chars().map(|c| c as i64).collect())
        }
        fn decode(&self, tokens: &[i64]) -> Result<String> {
            Ok(tokens.iter().map(|&t| char::from_u32(t as u32).unwrap_or('?')).collect())
        }
    }

    #[tokio::test]
    async fn example_game_agent() -> Result<()> {
        // Create model (you'd load real ONNX model here)
        let model = Arc::new(OrtModel::new(
            "path/to/gemma-270m.onnx",
            Arc::new(MockTokenizer),
        )?);

        // Create game master agent
        let mut game_master = Agent::new(
            "GameMaster",
            "You are the game master for a sci-fi horror game. \
             The player is exploring an abandoned research station. \
             Use tools to check game state and provide immersive responses.",
            model,
        );

        // Add game tools
        for tool in create_game_tools() {
            game_master.add_tool(tool);
        }

        // Run agent
        let response = game_master.run(
            "I want to examine the mirror on the wall"
        ).await?;

        println!("Game Master: {}", response);

        Ok(())
    }
}
