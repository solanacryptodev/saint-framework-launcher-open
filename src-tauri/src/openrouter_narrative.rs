use anyhow::Result;
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessage,
        ChatCompletionRequestSystemMessageContent,
        ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent,
        CreateChatCompletionRequest,
    },
};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;

/// Reads the API key from environment variable or from a file
/// Priority:
/// 1. OPENROUTER_API_KEY environment variable
/// 2. OPENROUTER_API_KEY_FILE environment variable (path to file containing the key)
/// 3. .openrouter_api_key file in the current directory
/// 4. .env file containing OPENROUTER_API_KEY=...
fn get_api_key() -> Result<String> {
    // First, try the environment variable directly
    if let Ok(api_key) = env::var("OPENROUTER_API_KEY") {
        println!("🔑 Using API key from OPENROUTER_API_KEY environment variable");
        return Ok(api_key);
    }

    // Second, try reading from a file specified by OPENROUTER_API_KEY_FILE
    if let Ok(file_path) = env::var("OPENROUTER_API_KEY_FILE") {
        if Path::new(&file_path).exists() {
            let content = fs::read_to_string(&file_path)
                .map_err(|e| anyhow::anyhow!("Failed to read API key file {}: {}", file_path, e))?;
            let key = content.trim().to_string();
            if !key.is_empty() {
                println!("🔑 Using API key from file: {}", file_path);
                return Ok(key);
            }
        }
    }

    // Third, try .openrouter_api_key file in current directory
    let key_file = Path::new(".openrouter_api_key");
    if key_file.exists() {
        let content = fs::read_to_string(key_file)
            .map_err(|e| anyhow::anyhow!("Failed to read .openrouter_api_key file: {}", e))?;
        let key = content.trim().to_string();
        if !key.is_empty() {
            println!("🔑 Using API key from .openrouter_api_key file");
            return Ok(key);
        }
    }

    // Fourth, try parsing .env file
    let env_file = Path::new(".env");
    if env_file.exists() {
        let content = fs::read_to_string(env_file)
            .map_err(|e| anyhow::anyhow!("Failed to read .env file: {}", e))?;
        for line in content.lines() {
            let line = line.trim();
            if line.starts_with("OPENROUTER_API_KEY=") {
                let key = line.strip_prefix("OPENROUTER_API_KEY=").unwrap().trim();
                // Remove quotes if present
                let key = key.trim_matches('"').trim_matches('\'');
                if !key.is_empty() {
                    println!("🔑 Using API key from .env file");
                    return Ok(key.to_string());
                }
            }
        }
    }

    Err(anyhow::anyhow!(
        "OPENROUTER_API_KEY not found. Set it via:\n\
         1. Environment variable: OPENROUTER_API_KEY=your_key\n\
         2. File path env var: OPENROUTER_API_KEY_FILE=/path/to/key_file\n\
         3. .openrouter_api_key file in current directory\n\
         4. .env file with OPENROUTER_API_KEY=your_key"
    ))
}

// ============================================================================
// NARRATIVE RESPONSE TYPES (Same as original)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeState {
    pub mission_briefing: String,
    pub command_options: Vec<String>,
}

// ============================================================================
// OPENROUTER CLIENT
// ============================================================================

pub struct OpenRouterClient {
    client: Client<OpenAIConfig>,
    model: String,
}

impl OpenRouterClient {
    /// Creates a new OpenRouter client configured for OpenRouter API
    pub fn new() -> Result<Self> {
        // Use the get_api_key function which checks multiple sources
        let api_key = get_api_key()?;

        let config = OpenAIConfig::new()
            .with_api_base("https://openrouter.ai/api/v1")
            .with_api_key(api_key);

        let client = Client::with_config(config);

        // Default model - can be changed via environment variable
        let model = env::var("OPENROUTER_MODEL")
            .unwrap_or_else(|_| "openrouter/pony-alpha".to_string());

        println!("🔗 OpenRouter client initialized with model: {}", model);

        Ok(Self { client, model })
    }

    /// Generate text using the OpenRouter API
    pub async fn generate(&self, system_prompt: &str, user_prompt: &str) -> Result<String> {
        println!("🔧 OpenRouter: Generating text...");

        let request = CreateChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![
                ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                    content: ChatCompletionRequestSystemMessageContent::Text(system_prompt.to_string()),
                    ..Default::default()
                }),
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text(user_prompt.to_string()),
                    ..Default::default()
                }),
            ],
            ..Default::default()
        };

        let response = self.client.chat().create(request).await?;

        let content = response.choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .ok_or_else(|| anyhow::anyhow!("No content in response"))?;

        println!("✅ OpenRouter: Generation complete ({} chars)", content.len());

        Ok(content)
    }
}

// ============================================================================
// OPENROUTER NARRATIVE SYSTEM
// ============================================================================

pub struct OpenRouterNarrativeSystem {
    client: OpenRouterClient,
    world_generator_prompt: String,
    options_generator_prompt: String,
    response_generator_prompt: String,
}

impl OpenRouterNarrativeSystem {
    /// Creates a new OpenRouterNarrativeSystem with specialized prompts
    pub fn new() -> Result<Self> {
        let client = OpenRouterClient::new()?;

        // Same prompts as the original NarrativeSystem agents
        let world_generator_prompt = 
            "You are a crime noir narrative generator. Create morally ambiguous protagonists, \
             femme fatales, and dark and gritty urban settings. Be concise (2-3 paragraphs max). \
             Focus on tension, mystery, and sensory details as well as themes of fatalism and corruption."
            .to_string();

        let options_generator_prompt = 
            "You are a choice generator for a crime noir narrative game. Based on the current situation, \
             generate exactly 5 action options for the player. Focus on themes of fatalism and corporate corruption. \
             Each option should be a single sentence. \
             Make options distinct and interesting. Output ONLY the 5 options, one per line, no numbering."
            .to_string();

        let response_generator_prompt = 
            "You are a crime noir narrative consequence generator. Based on the player's choice, describe what happens next. \
             Be atmospheric and engaging. Keep it to 2-3 paragraphs. Create tension and forward momentum in the story."
            .to_string();

        Ok(Self {
            client,
            world_generator_prompt,
            options_generator_prompt,
            response_generator_prompt,
        })
    }

    /// Generate initial world text and options
    pub async fn generate_initial_mission(&self) -> Result<NarrativeState> {
        println!("🌍 OpenRouterNarrativeSystem: Generating initial mission...");

        // Generate world setting
        let world_prompt = 
            "Generate an initial mission briefing for a urban crime noir scenario set in Charlotte, North Carolina at midnight.";

        println!("🗣️  Calling OpenRouter for world generation...");
        let mission_briefing = self.client.generate(&self.world_generator_prompt, world_prompt).await?;
        println!("✅ World generation completed. Briefing length: {} chars", mission_briefing.len());

        // Generate initial options based on the briefing
        let options_prompt = format!(
            "Based on this situation:\n\n{}\n\nGenerate 5 action options for the player:",
            mission_briefing
        );

        println!("🗣️  Calling OpenRouter for options generation...");
        let options_text = self.client.generate(&self.options_generator_prompt, &options_prompt).await?;
        println!("✅ Options generation completed");

        // Parse options (split by newlines, filter empty)
        let command_options: Vec<String> = options_text
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .take(5)
            .collect();

        Ok(NarrativeState {
            mission_briefing,
            command_options,
        })
    }

    /// Process a selected command option and generate new narrative state
    pub async fn process_command_option(&self, selected_option: &str, current_briefing: &str) -> Result<NarrativeState> {
        println!("🎮 OpenRouterNarrativeSystem: Processing player choice: '{}'", selected_option);

        // Generate narrative response to the choice using ResponseGenerator
        let response_prompt = format!(
            "Current situation:\n{}\n\nPlayer choice: {}\n\nDo not repeat yourself. Do not repeat the current situation \
             or the player choice as part of your output. Only respond accurately based on a mixture of the current \
             situation and the player choice, but the player choice is paramount and requires adherence to progress \
             the overall story. For example: if the current situation is in a building and the player chooses to run out of \
             the building and chase someone, then your response should depict that chase with proper prose and sentence \
             structure. Describe what happens next in the story:",
            current_briefing, selected_option
        );

        println!("🗣️  Calling OpenRouter for response generation...");
        let mission_briefing = self.client.generate(&self.response_generator_prompt, &response_prompt).await?;
        println!("✅ Response generation completed. New briefing length: {} chars", mission_briefing.len());

        // Generate new options based on the new situation
        let options_prompt = format!(
            "Based on this new situation:\n\n{}\n\nGenerate 5 action options for the player:",
            mission_briefing
        );

        println!("🗣️  Calling OpenRouter for options generation...");
        let options_text = self.client.generate(&self.options_generator_prompt, &options_prompt).await?;
        println!("✅ Options generation completed");

        // Parse options
        let command_options: Vec<String> = options_text
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .take(5)
            .collect();

        Ok(NarrativeState {
            mission_briefing,
            command_options,
        })
    }
}