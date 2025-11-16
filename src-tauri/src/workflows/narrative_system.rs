use std::sync::{Arc, Mutex};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use super::agent_core::{Agent, OrtModel};

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
    pub fn new(model: Arc<OrtModel>) -> Self {
        // World text generator - creates initial setting
        let world_generator = Agent::new(
            "WorldGenerator",
            "You are a sci-fi horror narrative generator. Create atmospheric descriptions of space stations, \
             alien environments, and mysterious situations. Be concise (2-3 paragraphs max). \
             Focus on tension, mystery, and sensory details.",
            model.clone(),
        );

        // Options generator - creates player choices
        let options_generator = Agent::new(
            "OptionsGenerator",
            "You are a choice generator for a sci-fi narrative game. Based on the current situation, \
             generate exactly 5 action options for the player. Each option should be a single sentence. \
             Make options distinct and interesting. Output ONLY the 5 options, one per line, no numbering.",
            model.clone(),
        );

        // Response generator - reacts to player choices
        let response_generator = Agent::new(
            "ResponseGenerator",
            "You are a narrative consequence generator. Based on the player's choice, describe what happens next. \
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
        
        // Generate world setting
        let world_prompt = "Generate an initial mission briefing for a space station investigation scenario. \
                           The player is approaching an abandoned research station.";
        
        println!("🗣️  Calling WorldGenerator agent...");
        let mut world_gen = self.world_generator.lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock world generator"))?;
        
        let mission_briefing = world_gen.query(world_prompt)?;
        println!("✅ WorldGenerator completed. Briefing length: {} chars", mission_briefing.len());

        // Generate initial options based on the briefing
        let options_prompt = format!(
            "Based on this situation:\n\n{}\n\nGenerate 5 action options for the player:",
            mission_briefing
        );
        
        println!("🗣️  Calling OptionsGenerator agent...");
        let mut opts_gen = self.options_generator.lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock options generator"))?;
        
        let options_text = opts_gen.query(&options_prompt)?;
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
        
        // Generate narrative response to the choice
        let response_prompt = format!(
            "Current situation:\n{}\n\nPlayer chose: {}\n\nDescribe what happens next:",
            current_briefing, selected_option
        );
        
        println!("🗣️  Calling ResponseGenerator agent...");
        let mut resp_gen = self.response_generator.lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock response generator"))?;
        
        let mission_briefing = resp_gen.query(&response_prompt)?;
        println!("✅ ResponseGenerator completed. New briefing length: {} chars", mission_briefing.len());

        // Generate new options based on the new situation
        let options_prompt = format!(
            "Based on this new situation:\n\n{}\n\nGenerate 5 action options for the player:",
            mission_briefing
        );
        
        println!("🗣️  Calling OptionsGenerator agent...");
        let mut opts_gen = self.options_generator.lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock options generator"))?;
        
        let options_text = opts_gen.query(&options_prompt)?;
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
