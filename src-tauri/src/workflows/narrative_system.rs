use std::sync::{Arc, Mutex};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use super::agent_core::{Agent, OrtModel};
use crate::config::SamplingConfig;

// ============================================================================
// NARRATIVE RESPONSE TYPES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeState {
    pub mission_briefing: String,
    pub command_options: Vec<String>,
}

// ============================================================================
// NARRATIVE AGENTS
// ============================================================================

pub struct NarrativeSystem {
    world_generator: Mutex<Agent>,
    options_generator: Mutex<Agent>,
    response_generator: Mutex<Agent>,
}

impl NarrativeSystem {
    /// Creates a new NarrativeSystem with three specialized agents
    /// 
    /// # Architecture Note: Shared Model with Cache Management
    /// 
    /// All three agents share the same `OrtModel` instance (via `Arc::clone`) for memory efficiency.
    /// This is safe because:
    /// 
    /// 1. Each agent uses `query_stateless()` which resets the KV cache before generation
    /// 2. The cache is managed at the model level, not the agent level
    /// 3. Each stateless query is independent and doesn't accumulate conversation history
    /// 
    /// **Alternative approach (not implemented):** Create separate `OrtModel` instances per agent
    /// for complete cache isolation, but this would triple memory usage (~800MB per model).
    /// 
    /// The current approach balances memory efficiency with proper cache hygiene.
    pub fn new(model: Arc<OrtModel>) -> Self {
        // World text generator - creates initial setting
        let world_generator = Agent::new(
            "WorldGenerator",
            "You are a crime noir narrative generator. Create morally ambigous protagonists, \
             femme fatales, and dark and gritty urban settings. Be concise (2-3 paragraphs max). \
             Focus on tension, mystery, and sensory details as well as themes of fatalism and corruption.",
            model.clone(),
        );

        // Options generator - creates player choices
        let options_generator = Agent::new(
            "OptionsGenerator",
            "You are a choice generator for a crime noir narrative game. Based on the current situation, \
             generate exactly 5 action options for the player. Focus on themes of fatalism and corporate corruption. \
             Each option should be a single sentence. \
             Make options distinct and interesting. Output ONLY the 5 options, one per line, no numbering.",
            model.clone(),
        );

        // Response generator - reacts to player choices
        let response_generator = Agent::new(
            "ResponseGenerator",
            "You are a crime noir narrative consequence generator. Based on the player's choice, describe what happens next. \
             Be atmospheric and engaging. Keep it to 2-3 paragraphs. Create tension and forward momentum in the story.",
            model.clone(),
        );

        Self {
            world_generator: Mutex::new(world_generator),
            options_generator: Mutex::new(options_generator),
            response_generator: Mutex::new(response_generator),
        }
    }

    /// Generate initial world text and options
    pub fn generate_initial_mission(&self) -> Result<NarrativeState> {
        println!("🌍 NarrativeSystem: Generating initial mission...");
        
        // Creative sampling for world/story generation
        let world_sampling = SamplingConfig {
            temperature: 0.8,          // High variety
            top_k: 50,                 // More options
            top_p: 0.90,               // Slightly more creative
            repetition_penalty: 0.95,   // Avoid repetitive phrases
        };
        
        // Generate world setting
        let world_prompt = 
            "Generate an initial mission briefing for a urban crime noir scenario set in Charlotte, North Carolina at midnight.";
        
        println!("🗣️  Calling WorldGenerator agent...");
        let mut world_gen = self.world_generator.lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock world generator"))?;
        
        // Use query_stateless_with_config with creative sampling
        let mission_briefing = world_gen.query_stateless_with_config(world_prompt, Some(&world_sampling))?;
        println!("✅ WorldGenerator completed. Briefing length: {} chars", mission_briefing.len());

        // Diverse but focused sampling for options generation
        let options_sampling = SamplingConfig {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,               // More permissive to avoid getting stuck
            repetition_penalty: 0.90,  // Gentle penalty - too high can cause loops
        };

        // Generate initial options based on the briefing
        let options_prompt = format!(
            "Based on this situation:\n\n{}\n\nGenerate 5 action options for the player:",
            mission_briefing
        );
        
        println!("🗣️  Calling OptionsGenerator agent...");
        let mut opts_gen = self.options_generator.lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock options generator"))?;
        
        // Use query_stateless_with_config with options sampling
        let options_text = opts_gen.query_stateless_with_config(&options_prompt, Some(&options_sampling))?;
        println!("✅ OptionsGenerator completed");
        
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
    pub fn process_command_option(&self, selected_option: &str, current_briefing: &str) -> Result<NarrativeState> {
        println!("🎮 NarrativeSystem: Processing player choice: '{}'", selected_option);
        
        // Generate narrative response to the choice using WorldGenerator
        let world_prompt = format!(
            "Current situation:\n{}\n\nPlayer choice: {}\n\nDo not repeat yourself. Do not repeat the current situation
             or the player choice as part of your ouput. Only respond with accurately based on a mixture of the current
             situation and the player choice, but the player choice is paramount and requires adherence so progress
             the overall story. For example: if the current situation is in a building and the player chooses to run out of
             the building and chase someone, then your response should depict that chase with proper prose and sentence
             structure. Describe what happens next in the story:",
            current_briefing, selected_option
        );
        
        println!("🗣️  Calling WorldGenerator agent...");
        let mut world_gen = self.world_generator.lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock world generator"))?;
        
        // Use query_stateless to prevent conversation history pollution
        let mission_briefing = world_gen.query_stateless(&world_prompt)?;
        println!("✅ WorldGenerator completed. New briefing length: {} chars", mission_briefing.len());

        // Generate new options based on the new situation
        let options_prompt = format!(
            "Based on this new situation:\n\n{}\n\nGenerate 5 action options for the player:",
            mission_briefing
        );
        
        println!("🗣️  Calling OptionsGenerator agent...");
        let mut opts_gen = self.options_generator.lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock options generator"))?;
        
        // Use query_stateless to prevent conversation history pollution
        let options_text = opts_gen.query_stateless(&options_prompt)?;
        println!("✅ OptionsGenerator completed");
        
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
