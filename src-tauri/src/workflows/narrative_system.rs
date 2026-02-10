use std::sync::Arc;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use super::agent_core::{Agent, AgentModel};

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
    world_generator: Agent,
    options_generator: Agent,
    response_generator: Agent,
}

impl NarrativeSystem {
    /// Creates a new NarrativeSystem with three specialized agents
    /// Each agent needs its own model since Agent takes ownership
    pub fn new(
        world_model: Arc<AgentModel>,
        options_model: Arc<AgentModel>,
        response_model: Arc<AgentModel>,
    ) -> Self {
        // World text generator - creates initial setting
        let world_generator = Agent::new(
            "WorldGenerator",
            "You are a crime noir narrative generator. Create morally ambiguous protagonists, \
             femme fatales, and dark and gritty urban settings. Be concise (2-3 paragraphs max). \
             Focus on tension, mystery, and sensory details as well as themes of fatalism and corruption.",
            world_model,
        );

        // Options generator - creates player choices
        let options_generator = Agent::new(
            "OptionsGenerator",
            "You are a choice generator for a crime noir narrative game. Based on the current situation, \
             generate exactly 5 action options for the player. Focus on themes of fatalism and corporate corruption. \
             Each option should be a single sentence. \
             Make options distinct and interesting. Output ONLY the 5 options, one per line, no numbering.",
            options_model,
        );

        // Response generator - reacts to player choices
        let response_generator = Agent::new(
            "ResponseGenerator",
            "You are a crime noir narrative consequence generator. Based on the player's choice, describe what happens next. \
             Be atmospheric and engaging. Keep it to 2-3 paragraphs. Create tension and forward momentum in the story.",
            response_model,
        );

        Self {
            world_generator,
            options_generator,
            response_generator,
        }
    }

    /// Generate initial world text and options
    pub async fn generate_initial_mission(&self) -> Result<NarrativeState> {
        println!("🌍 NarrativeSystem: Generating initial mission...");
        
        // Generate world setting
        let world_prompt = 
            "Generate an initial mission briefing for a urban crime noir scenario set in Charlotte, North Carolina at midnight.";
        
        println!("🗣️  Calling WorldGenerator agent...");
        let mission_briefing = self.world_generator.generate_text(world_prompt).await?;
        println!("✅ WorldGenerator completed. Briefing length: {} chars", mission_briefing.len());

        // Generate initial options based on the briefing
        let options_prompt = format!(
            "Based on this situation:\n\n{}\n\nGenerate 5 action options for the player:",
            mission_briefing
        );
        
        println!("🗣️  Calling OptionsGenerator agent...");
        let options_text = self.options_generator.generate_text(&options_prompt).await?;
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
    pub async fn process_command_option(&self, selected_option: &str, current_briefing: &str) -> Result<NarrativeState> {
        println!("🎮 NarrativeSystem: Processing player choice: '{}'", selected_option);
        
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
        
        println!("🗣️  Calling ResponseGenerator agent...");
        let mission_briefing = self.response_generator.generate_text(&response_prompt).await?;
        println!("✅ ResponseGenerator completed. New briefing length: {} chars", mission_briefing.len());

        // Generate new options based on the new situation
        let options_prompt = format!(
            "Based on this new situation:\n\n{}\n\nGenerate 5 action options for the player:",
            mission_briefing
        );
        
        println!("🗣️  Calling OptionsGenerator agent...");
        let options_text = self.options_generator.generate_text(&options_prompt).await?;
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
