use std::collections::HashMap;
use std::sync::Arc;
use anyhow::Result;

// Import shared types
use crate::shared_types::{Blackboard, BlackboardSlotStatus};

// Import Agent from agent_core
use super::agent_core::Agent;

// ============================================================================
// AGENT PAIR - Writer and Reader for a specific slot
// ============================================================================

pub struct AgentPair {
    pub slot_number: usize,
    pub writer: WriterAgent,
    pub reader: ReaderAgent,
}

impl AgentPair {
    pub fn new(
        slot_number: usize,
        writer: WriterAgent,
        reader: ReaderAgent,
    ) -> Self {
        Self {
            slot_number,
            writer,
            reader,
        }
    }
}

// ============================================================================
// WRITER AGENT - Writes to blackboard when flag is false
// ============================================================================

pub struct WriterAgent {
    pub name: String,
    pub agent: Agent,
    pub slot_number: usize,
}

impl WriterAgent {
    pub fn new(name: impl Into<String>, agent: Agent, slot_number: usize) -> Self {
        Self {
            name: name.into(),
            agent,
            slot_number,
        }
    }

    /// Writer attempts to generate and write data
    pub async fn try_write(
        &mut self,
        blackboard: &Blackboard,
        context: &str,
    ) -> Result<bool> {
        // Check if slot is ready for writing
        let status = blackboard.get_slot_status(self.slot_number)?;
        
        if !status.ready_to_read {
            println!("✍️  [Slot {}] {} generating data...", self.slot_number, self.name);
            
            // Generate data using agent
            let prompt = format!(
                "Context: {}\nGenerate information for slot {}:",
                context, self.slot_number
            );
            let data = self.agent.run(&prompt).await?;
            
            // Write to blackboard
            blackboard.write_to_slot(self.slot_number, data.clone())?;
            println!("✅ [Slot {}] {} wrote data (flag → true)", self.slot_number, self.name);
            
            Ok(true)
        } else {
            println!("⏸️  [Slot {}] {} waiting (reader hasn't consumed yet)", self.slot_number, self.name);
            Ok(false)
        }
    }
}

// ============================================================================
// READER AGENT - Reads from blackboard when flag is true
// ============================================================================

pub struct ReaderAgent {
    pub name: String,
    pub agent: Agent,
    pub slot_number: usize,
}

impl ReaderAgent {
    pub fn new(name: impl Into<String>, agent: Agent, slot_number: usize) -> Self {
        Self {
            name: name.into(),
            agent,
            slot_number,
        }
    }

    /// Reader attempts to read and process data
    pub async fn try_read(&mut self, blackboard: &Blackboard) -> Result<Option<String>> {
        // Attempt to read from blackboard
        if let Some(data) = blackboard.read_from_slot(self.slot_number)? {
            println!("📖 [Slot {}] {} read data (flag → false)", self.slot_number, self.name);
            
            // Process the data using agent
            let prompt = format!(
                "Process this data from slot {}:\n{}",
                self.slot_number, data
            );
            let result = self.agent.run(&prompt).await?;
            
            println!("✅ [Slot {}] {} processed data", self.slot_number, self.name);
            
            Ok(Some(result))
        } else {
            println!("⏸️  [Slot {}] {} waiting (no data ready)", self.slot_number, self.name);
            Ok(None)
        }
    }
}

// ============================================================================
// NUMBERED BLACKBOARD WORKFLOW - Orchestrates all agent pairs
// ============================================================================

pub struct NumberedBlackboardWorkflow {
    blackboard: Arc<Blackboard>,
    agent_pairs: HashMap<usize, AgentPair>,
    max_iterations: usize,
}

impl NumberedBlackboardWorkflow {
    pub fn new(max_iterations: usize) -> Self {
        Self {
            blackboard: Arc::new(Blackboard::new()),
            agent_pairs: HashMap::new(),
            max_iterations,
        }
    }

    /// Register an agent pair to a numbered slot
    pub fn register_pair(
        &mut self,
        slot_number: usize,
        writer: WriterAgent,
        reader: ReaderAgent,
    ) -> Result<()> {
        // Create slot on blackboard
        self.blackboard.create_slot(slot_number)?;
        
        // Register agent pair
        self.agent_pairs.insert(
            slot_number,
            AgentPair::new(slot_number, writer, reader),
        );
        
        println!("📋 Registered slot {}", slot_number);
        Ok(())
    }

    /// Run the workflow - all agents work asynchronously
    pub async fn run(&mut self, initial_context: &str) -> Result<WorkflowResults> {
        let mut results = WorkflowResults::new();
        let mut iteration = 0;

        println!("\n🚀 Starting Numbered Blackboard Workflow\n");

        while iteration < self.max_iterations {
            iteration += 1;
            println!("━━━ Iteration {} ━━━", iteration);

            let mut any_activity = false;

            // Phase 1: All writers try to write
            for (_slot_number, pair) in &mut self.agent_pairs {
                if pair.writer.try_write(&self.blackboard, initial_context).await? {
                    any_activity = true;
                }
            }

            // Phase 2: All readers try to read
            for (slot_number, pair) in &mut self.agent_pairs {
                if let Some(result) = pair.reader.try_read(&self.blackboard).await? {
                    results.add_result(*slot_number, result);
                    any_activity = true;
                }
            }

            // Check if all slots are idle (no activity)
            if !any_activity {
                println!("\n⚠️  All slots idle - workflow may be complete or deadlocked");
            }

            println!();
        }

        println!("🏁 Workflow complete after {} iterations\n", iteration);
        Ok(results)
    }

    /// Run a single cycle (useful for step-by-step execution)
    pub async fn run_single_cycle(&mut self, context: &str) -> Result<()> {
        println!("━━━ Single Cycle ━━━");

        // Writers
        for (_, pair) in &mut self.agent_pairs {
            pair.writer.try_write(&self.blackboard, context).await?;
        }

        // Readers
        for (_, pair) in &mut self.agent_pairs {
            pair.reader.try_read(&self.blackboard).await?;
        }

        Ok(())
    }

    /// Get current blackboard state
    pub fn get_state(&self) -> Vec<BlackboardSlotStatus> {
        self.blackboard.get_all_statuses()
    }
}

#[derive(Debug, Default)]
pub struct WorkflowResults {
    results_by_slot: HashMap<usize, Vec<String>>,
}

impl WorkflowResults {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_result(&mut self, slot_number: usize, result: String) {
        self.results_by_slot
            .entry(slot_number)
            .or_insert_with(Vec::new)
            .push(result);
    }

    pub fn get_slot_results(&self, slot_number: usize) -> Option<&Vec<String>> {
        self.results_by_slot.get(&slot_number)
    }

    pub fn all_results(&self) -> &HashMap<usize, Vec<String>> {
        &self.results_by_slot
    }
}

// ============================================================================
// EXAMPLE USAGE - Game Systems Using Numbered Blackboard
// ============================================================================

#[cfg(test)]
mod examples {
    use super::*;
    use super::super::agent_core::OrtModel;
    use std::sync::Arc;

    #[tokio::test]
    async fn example_game_systems() -> Result<()> {
        // Mock model (replace with real OrtModel)
        let model = Arc::new(create_mock_model()?);

        let mut workflow = NumberedBlackboardWorkflow::new(5);

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // SLOT 0: World State Manager
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        let writer_0 = WriterAgent::new(
            "WorldStateGenerator",
            Agent::new(
                "WorldStateGen",
                "Generate current world state: locations, NPCs, items",
                model.clone(),
            ),
            0,
        );

        let reader_0 = ReaderAgent::new(
            "WorldStateConsumer",
            Agent::new(
                "WorldStateProcessor",
                "Process world state and update game database",
                model.clone(),
            ),
            0,
        );

        workflow.register_pair(0, writer_0, reader_0)?;

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // SLOT 1: NPC Behavior System
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        let writer_1 = WriterAgent::new(
            "NPCBehaviorGenerator",
            Agent::new(
                "NPCBehaviorGen",
                "Generate NPC intentions and actions based on world state",
                model.clone(),
            ),
            1,
        );

        let reader_1 = ReaderAgent::new(
            "NPCBehaviorExecutor",
            Agent::new(
                "NPCBehaviorExec",
                "Execute NPC behaviors and update their states",
                model.clone(),
            ),
            1,
        );

        workflow.register_pair(1, writer_1, reader_1)?;

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // SLOT 2: Event System
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        let writer_2 = WriterAgent::new(
            "EventGenerator",
            Agent::new(
                "EventGen",
                "Generate dynamic events: xenon spikes, system failures, echoes",
                model.clone(),
            ),
            2,
        );

        let reader_2 = ReaderAgent::new(
            "EventHandler",
            Agent::new(
                "EventHandler",
                "Handle events and trigger consequences",
                model.clone(),
            ),
            2,
        );

        workflow.register_pair(2, writer_2, reader_2)?;

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // SLOT 3: Narrative Generator
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        let writer_3 = WriterAgent::new(
            "NarrativeWriter",
            Agent::new(
                "NarrativeGen",
                "Generate atmospheric narrative descriptions",
                model.clone(),
            ),
            3,
        );

        let reader_3 = ReaderAgent::new(
            "NarrativePresenter",
            Agent::new(
                "NarrativePresenter",
                "Format and present narrative to player",
                model.clone(),
            ),
            3,
        );

        workflow.register_pair(3, writer_3, reader_3)?;

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // SLOT 4: Puzzle State Manager
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        let writer_4 = WriterAgent::new(
            "PuzzleStateGenerator",
            Agent::new(
                "PuzzleGen",
                "Generate puzzle states and hints",
                model.clone(),
            ),
            4,
        );

        let reader_4 = ReaderAgent::new(
            "PuzzleStateEvaluator",
            Agent::new(
                "PuzzleEval",
                "Evaluate puzzle solutions and update progress",
                model.clone(),
            ),
            4,
        );

        workflow.register_pair(4, writer_4, reader_4)?;

        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // Run the workflow
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        let results = workflow.run("Player enters Laboratory Alpha").await?;

        println!("📊 Final Results:");
        for (slot, slot_results) in results.all_results() {
            println!("\nSlot {}: {} results", slot, slot_results.len());
            for result in slot_results {
                println!("  - {}", result);
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn example_step_by_step_execution() -> Result<()> {
        let model = Arc::new(create_mock_model()?);
        let mut workflow = NumberedBlackboardWorkflow::new(10);

        // Register one simple pair
        workflow.register_pair(
            0,
            WriterAgent::new(
                "DataProducer",
                Agent::new("Producer", "Produce data", model.clone()),
                0,
            ),
            ReaderAgent::new(
                "DataConsumer",
                Agent::new("Consumer", "Consume data", model.clone()),
                0,
            ),
        )?;

        // Run step by step
        for i in 0..3 {
            println!("\n━━━ Manual Step {} ━━━", i + 1);
            workflow.run_single_cycle("Step context").await?;
            
            // Check state
            let state = workflow.get_state();
            println!("State: {:?}", state);
        }

        Ok(())
    }

    fn create_mock_model() -> Result<OrtModel> {
        // Mock implementation
        unimplemented!("Replace with real OrtModel")
    }
}

// ============================================================================
// BUILDER PATTERN - Easy workflow construction
// ============================================================================

pub struct BlackboardWorkflowBuilder {
    workflow: NumberedBlackboardWorkflow,
}

impl BlackboardWorkflowBuilder {
    pub fn new(max_iterations: usize) -> Self {
        Self {
            workflow: NumberedBlackboardWorkflow::new(max_iterations),
        }
    }

    pub fn with_pair(
        mut self,
        slot_number: usize,
        writer: WriterAgent,
        reader: ReaderAgent,
    ) -> Result<Self> {
        self.workflow.register_pair(slot_number, writer, reader)?;
        Ok(self)
    }

    pub fn build(self) -> NumberedBlackboardWorkflow {
        self.workflow
    }
}
