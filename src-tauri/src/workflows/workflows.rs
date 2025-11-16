use std::collections::HashMap;
use crate::error::{Result, LlmError};

// Assuming we have the Agent struct from previous artifact
use super::agent_core::Agent;

// ============================================================================
// 1. SEQUENTIAL WORKFLOW - Chain agents in sequence
// ============================================================================

pub struct SequentialWorkflow {
    agents: Vec<Agent>,
}

impl SequentialWorkflow {
    pub fn new(agents: Vec<Agent>) -> Self {
        Self { agents }
    }

    /// Each agent processes the output of the previous agent
    pub async fn run(&mut self, initial_input: &str) -> Result<String> {
        let mut current_input = initial_input.to_string();
        
        for (i, agent) in self.agents.iter_mut().enumerate() {
            println!("🔄 Stage {}: {}", i + 1, agent.name);
            current_input = agent.run(&current_input).await?;
        }
        
        Ok(current_input)
    }
}

// Example: Game narrative pipeline
// Note: Commented out - requires tokenizer implementation
// pub async fn example_sequential_narrative() -> Result<()> {
//     let tokenizer = Arc::new(YourTokenizer::new()?);
//     let model = Arc::new(OrtModel::new("gemma.onnx", tokenizer)?);
//     
//     // Stage 1: Analyze player intent
//     let intent_analyzer = Agent::new(
//         "IntentAnalyzer",
//         "Analyze player input and extract: action, target, context",
//         model.clone(),
//     );
//     
//     // Stage 2: Execute game logic
//     let mut action_executor = Agent::new(
//         "ActionExecutor",
//         "Execute game actions based on analyzed intent",
//         model.clone(),
//     );
//     action_executor.add_tool(/* game tools */);
//     
//     // Stage 3: Generate narrative response
//     let narrator = Agent::new(
//         "Narrator",
//         "Generate atmospheric sci-fi horror narrative based on game events",
//         model.clone(),
//     );
//     
//     let mut workflow = SequentialWorkflow::new(vec![
//         intent_analyzer,
//         action_executor,
//         narrator,
//     ]);
//     
//     let result = workflow.run("I examine the mirror").await?;
//     println!("Final: {}", result);
//     
//     Ok(())
// }

// ============================================================================
// 2. PARALLEL WORKFLOW - Run multiple agents concurrently
// ============================================================================

pub struct ParallelWorkflow {
    agents: Vec<Agent>,
}

impl ParallelWorkflow {
    pub fn new(agents: Vec<Agent>) -> Self {
        Self { agents }
    }

    /// Run all agents in parallel and aggregate results
    pub async fn run(&mut self, input: &str) -> Result<Vec<String>> {
        let mut handles = vec![];
        
        for agent in &mut self.agents {
            let input = input.to_string();
            // In real implementation, use tokio::spawn with Arc<Mutex<Agent>>
            let response = agent.run(&input).await?;
            handles.push(response);
        }
        
        Ok(handles)
    }
    
    /// Aggregate multiple agent responses
    pub fn aggregate(responses: Vec<String>) -> String {
        responses.join("\n\n---\n\n")
    }
}

// Example: Multiple NPCs respond simultaneously
// Note: Commented out - requires tokenizer implementation
// pub async fn example_parallel_npc_reactions() -> Result<()> {
//     let tokenizer = Arc::new(YourTokenizer::new()?);
//     let model = Arc::new(OrtModel::new("qwen.onnx", tokenizer)?);
//     
//     let npc1 = Agent::new(
//         "Dr. Chen",
//         "You are Dr. Chen, a cautious scientist. React to events.",
//         model.clone(),
//     );
//     
//     let npc2 = Agent::new(
//         "Echo",
//         "You are Echo, a mysterious entity. React cryptically.",
//         model.clone(),
//     );
//     
//     let npc3 = Agent::new(
//         "Station AI",
//         "You are the damaged station AI. Provide system warnings.",
//         model.clone(),
//     );
//     
//     let mut workflow = ParallelWorkflow::new(vec![npc1, npc2, npc3]);
//     
//     let responses = workflow.run("The xenon levels are spiking!").await?;
//     for (i, response) in responses.iter().enumerate() {
//         println!("NPC {}: {}", i + 1, response);
//     }
//     
//     Ok(())
// }

// ============================================================================
// 3. ROUTING WORKFLOW - Route to different agents based on conditions
// ============================================================================

pub struct RoutingWorkflow {
    router: Agent,
    specialists: HashMap<String, Agent>,
}

impl RoutingWorkflow {
    pub fn new(router: Agent, specialists: HashMap<String, Agent>) -> Self {
        Self { router, specialists }
    }

    pub async fn run(&mut self, input: &str) -> Result<String> {
        // Router decides which specialist to use
        let route_decision = self.router.run(&format!(
            "Route this query to the appropriate specialist: {}",
            input
        )).await?;
        
        // Extract specialist name from router response
        let specialist_name = self.extract_specialist_name(&route_decision)?;
        
        // Route to specialist
        if let Some(specialist) = self.specialists.get_mut(&specialist_name) {
            println!("📍 Routing to: {}", specialist_name);
            specialist.run(input).await
        } else {
            Err(LlmError::InvalidInput {
                reason: format!("Unknown specialist: {}", specialist_name),
            })
        }
    }
    
    fn extract_specialist_name(&self, response: &str) -> Result<String> {
        // Parse router's decision (could be JSON or simple text)
        if let Some(name) = response.split(':').nth(1) {
            Ok(name.trim().to_string())
        } else {
            Err(LlmError::InvalidInput {
                reason: "Could not parse routing decision".to_string(),
            })
        }
    }
}

// Example: Route player queries to specialized game systems
// Note: Commented out - requires tokenizer implementation
// pub async fn example_routing_game_systems() -> Result<()> {
//     let tokenizer = Arc::new(YourTokenizer::new()?);
//     let model = Arc::new(OrtModel::new("gemma.onnx", tokenizer)?);
//     
//     let router = Agent::new(
//         "Router",
//         "Route queries to: combat, inventory, dialogue, exploration, puzzle",
//         model.clone(),
//     );
//     
//     let mut specialists = HashMap::new();
//     
//     specialists.insert(
//         "combat".to_string(),
//         Agent::new("CombatSystem", "Handle combat actions", model.clone()),
//     );
//     
//     specialists.insert(
//         "inventory".to_string(),
//         Agent::new("InventorySystem", "Manage player inventory", model.clone()),
//     );
//     
//     specialists.insert(
//         "dialogue".to_string(),
//         Agent::new("DialogueSystem", "Handle NPC conversations", model.clone()),
//     );
//     
//     let mut workflow = RoutingWorkflow::new(router, specialists);
//     
//     let result = workflow.run("I want to talk to Dr. Chen").await?;
//     println!("Result: {}", result);
//     
//     Ok(())
// }

// ============================================================================
// 4. CONSENSUS WORKFLOW - Multiple agents vote on decisions
// ============================================================================

pub struct ConsensusWorkflow {
    agents: Vec<Agent>,
    voting_strategy: VotingStrategy,
}

#[derive(Clone)]
pub enum VotingStrategy {
    Majority,
    Unanimous,
    Weighted(HashMap<String, f32>),
}

impl ConsensusWorkflow {
    pub fn new(agents: Vec<Agent>, strategy: VotingStrategy) -> Self {
        Self {
            agents,
            voting_strategy: strategy,
        }
    }

    pub async fn run(&mut self, input: &str) -> Result<String> {
        let mut votes: HashMap<String, usize> = HashMap::new();
        
        // Collect votes from all agents
        for agent in &mut self.agents {
            let response = agent.run(input).await?;
            *votes.entry(response.clone()).or_insert(0) += 1;
        }
        
        // Apply voting strategy
        self.apply_voting_strategy(votes)
    }
    
    fn apply_voting_strategy(&self, votes: HashMap<String, usize>) -> Result<String> {
        match &self.voting_strategy {
            VotingStrategy::Majority => {
                votes
                    .into_iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(response, _)| response)
                    .ok_or_else(|| LlmError::InvalidInput {
                        reason: "No votes received".to_string(),
                    })
            }
            VotingStrategy::Unanimous => {
                if votes.len() == 1 {
                    Ok(votes.into_keys().next().unwrap())
                } else {
                    Err(LlmError::GenerationFailed {
                        reason: "No consensus reached - agents gave different responses".to_string(),
                    })
                }
            }
            VotingStrategy::Weighted(_weights) => {
                // Implement weighted voting
                unimplemented!("Weighted voting")
            }
        }
    }
}

// Example: Multiple safety checks before executing dangerous action
// Note: Commented out - requires tokenizer implementation
// pub async fn example_consensus_safety_check() -> Result<()> {
//     let tokenizer = Arc::new(YourTokenizer::new()?);
//     let model = Arc::new(OrtModel::new("gemma.onnx", tokenizer)?);
//     
//     let safety_agent_1 = Agent::new(
//         "PhysicsSafety",
//         "Check if action violates physics/reality constraints",
//         model.clone(),
//     );
//     
//     let safety_agent_2 = Agent::new(
//         "NarrativeSafety",
//         "Check if action breaks narrative consistency",
//         model.clone(),
//     );
//     
//     let safety_agent_3 = Agent::new(
//         "GameRulesSafety",
//         "Check if action violates game rules",
//         model.clone(),
//     );
//     
//     let mut workflow = ConsensusWorkflow::new(
//         vec![safety_agent_1, safety_agent_2, safety_agent_3],
//         VotingStrategy::Unanimous,
//     );
//     
//     let result = workflow.run("Player wants to breathe in xenon gas").await?;
//     println!("Safety check: {}", result);
//     
//     Ok(())
// }

// ============================================================================
// 5. HIERARCHICAL WORKFLOW - Manager delegates to workers
// ============================================================================

pub struct HierarchicalWorkflow {
    manager: Agent,
    workers: HashMap<String, Agent>,
}

impl HierarchicalWorkflow {
    pub fn new(manager: Agent, workers: HashMap<String, Agent>) -> Self {
        Self { manager, workers }
    }

    pub async fn run(&mut self, input: &str) -> Result<String> {
        // Manager breaks down task
        let plan = self.manager.run(&format!(
            "Break down this task into subtasks: {}",
            input
        )).await?;
        
        // Parse subtasks
        let subtasks = self.parse_subtasks(&plan)?;
        
        // Execute subtasks
        let mut results = vec![];
        for (worker_name, subtask) in subtasks {
            if let Some(worker) = self.workers.get_mut(&worker_name) {
                println!("👷 {} executing: {}", worker_name, subtask);
                let result = worker.run(&subtask).await?;
                results.push(result);
            }
        }
        
        // Manager synthesizes results
        let synthesis_prompt = format!(
            "Original task: {}\nResults: {:?}\nSynthesize final answer:",
            input, results
        );
        
        self.manager.run(&synthesis_prompt).await
    }
    
    fn parse_subtasks(&self, plan: &str) -> Result<Vec<(String, String)>> {
        // Parse manager's plan (simplified)
        let mut subtasks = vec![];
        for line in plan.lines() {
            if let Some((worker, task)) = line.split_once(':') {
                subtasks.push((worker.trim().to_string(), task.trim().to_string()));
            }
        }
        Ok(subtasks)
    }
}

// Example: Game master delegates to specialized systems
// Note: Commented out - requires tokenizer implementation
// pub async fn example_hierarchical_game_master() -> Result<()> {
//     let tokenizer = Arc::new(YourTokenizer::new()?);
//     let model = Arc::new(OrtModel::new("gemma.onnx", tokenizer)?);
//     
//     let manager = Agent::new(
//         "GameMaster",
//         "You are the game master. Delegate tasks to: WorldBuilder, NPCManager, PuzzleGenerator",
//         model.clone(),
//     );
//     
//     let mut workers = HashMap::new();
//     
//     workers.insert(
//         "WorldBuilder".to_string(),
//         Agent::new("WorldBuilder", "Generate location descriptions", model.clone()),
//     );
//     
//     workers.insert(
//         "NPCManager".to_string(),
//         Agent::new("NPCManager", "Manage NPC behaviors and dialogue", model.clone()),
//     );
//     
//     workers.insert(
//         "PuzzleGenerator".to_string(),
//         Agent::new("PuzzleGenerator", "Create xenolinguistics puzzles", model.clone()),
//     );
//     
//     let mut workflow = HierarchicalWorkflow::new(manager, workers);
//     
//     let result = workflow.run("Generate a new chapter in the space station").await?;
//     println!("Generated chapter: {}", result);
//     
//     Ok(())
// }

// ============================================================================
// 6. ITERATIVE REFINEMENT WORKFLOW - Agent improves its own output
// ============================================================================

pub struct IterativeRefinementWorkflow {
    generator: Agent,
    critic: Agent,
    max_iterations: usize,
}

impl IterativeRefinementWorkflow {
    pub fn new(generator: Agent, critic: Agent, max_iterations: usize) -> Self {
        Self {
            generator,
            critic,
            max_iterations,
        }
    }

    pub async fn run(&mut self, input: &str) -> Result<String> {
        let mut current_output = self.generator.run(input).await?;
        
        for i in 0..self.max_iterations {
            println!("🔄 Refinement iteration {}", i + 1);
            
            // Critic evaluates current output
            let critique = self.critic.run(&format!(
                "Evaluate this output and suggest improvements:\n{}",
                current_output
            )).await?;
            
            // Check if critique is satisfied
            if critique.contains("approved") || critique.contains("satisfactory") {
                break;
            }
            
            // Generator refines based on critique
            current_output = self.generator.run(&format!(
                "Original task: {}\nPrevious attempt: {}\nCritique: {}\nGenerate improved version:",
                input, current_output, critique
            )).await?;
        }
        
        Ok(current_output)
    }
}

// Example: Refine narrative descriptions until atmospheric
// Note: Commented out - requires tokenizer implementation
// pub async fn example_iterative_narrative_polish() -> Result<()> {
//     let tokenizer = Arc::new(YourTokenizer::new()?);
//     let model = Arc::new(OrtModel::new("gemma.onnx", tokenizer)?);
//     
//     let generator = Agent::new(
//         "NarrativeWriter",
//         "Write atmospheric sci-fi horror descriptions",
//         model.clone(),
//     );
//     
//     let critic = Agent::new(
//         "NarrativeCritic",
//         "Critique descriptions for atmosphere, pacing, and tension",
//         model.clone(),
//     );
//     
//     let mut workflow = IterativeRefinementWorkflow::new(generator, critic, 3);
//     
//     let result = workflow.run("Describe the player's first sight of the mirror").await?;
//     println!("Polished description: {}", result);
//     
//     Ok(())
// }

// ============================================================================
// 7. REFLECTION WORKFLOW - Agent plans, acts, then reflects
// ============================================================================

pub struct ReflectionWorkflow {
    planner: Agent,
    actor: Agent,
    reflector: Agent,
}

impl ReflectionWorkflow {
    pub fn new(planner: Agent, actor: Agent, reflector: Agent) -> Self {
        Self { planner, actor, reflector }
    }

    pub async fn run(&mut self, input: &str) -> Result<ReflectionResult> {
        // 1. Plan
        let plan = self.planner.run(&format!(
            "Create a plan to accomplish: {}",
            input
        )).await?;
        
        // 2. Execute
        let execution_result = self.actor.run(&format!(
            "Execute this plan: {}",
            plan
        )).await?;
        
        // 3. Reflect
        let reflection = self.reflector.run(&format!(
            "Task: {}\nPlan: {}\nResult: {}\nReflect on what worked and what to improve:",
            input, plan, execution_result
        )).await?;
        
        Ok(ReflectionResult {
            plan,
            execution: execution_result,
            reflection,
        })
    }
}

pub struct ReflectionResult {
    pub plan: String,
    pub execution: String,
    pub reflection: String,
}

// Example: NPC learns from interactions
// Note: Commented out - requires tokenizer implementation
// pub async fn example_reflection_npc_learning() -> Result<()> {
//     let tokenizer = Arc::new(YourTokenizer::new()?);
//     let model = Arc::new(OrtModel::new("qwen.onnx", tokenizer)?);
//     
//     let planner = Agent::new(
//         "NPCPlanner",
//         "Plan how NPC should respond to player",
//         model.clone(),
//     );
//     
//     let actor = Agent::new(
//         "NPCDialogue",
//         "Generate NPC dialogue",
//         model.clone(),
//     );
//     
//     let reflector = Agent::new(
//         "NPCReflector",
//         "Analyze NPC interaction and learn",
//         model.clone(),
//     );
//     
//     let mut workflow = ReflectionWorkflow::new(planner, actor, reflector);
//     
//     let result = workflow.run("Player asks about the Resonance Echoes").await?;
//     println!("Plan: {}", result.plan);
//     println!("Dialogue: {}", result.execution);
//     println!("Learned: {}", result.reflection);
//     
//     Ok(())
// }

// ============================================================================
// 8. MULTI-AGENT DEBATE WORKFLOW - Agents debate to reach best answer
// ============================================================================

pub struct DebateWorkflow {
    agents: Vec<Agent>,
    moderator: Agent,
    rounds: usize,
}

impl DebateWorkflow {
    pub fn new(agents: Vec<Agent>, moderator: Agent, rounds: usize) -> Self {
        Self { agents, moderator, rounds }
    }

    pub async fn run(&mut self, topic: &str) -> Result<String> {
        let mut debate_history = vec![];
        
        for round in 0..self.rounds {
            println!("🎤 Debate round {}", round + 1);
            
            for agent in &mut self.agents {
                let context = debate_history.join("\n");
                let argument = agent.run(&format!(
                    "Topic: {}\nPrevious arguments: {}\nYour argument:",
                    topic, context
                )).await?;
                
                debate_history.push(format!("{}: {}", agent.name, argument));
            }
        }
        
        // Moderator synthesizes final answer
        self.moderator.run(&format!(
            "Topic: {}\nDebate: {}\nSynthesize the best answer:",
            topic,
            debate_history.join("\n")
        )).await
    }
}

// Example: Multiple perspectives on game puzzle solution
// Note: Commented out - requires tokenizer implementation
// pub async fn example_debate_puzzle_solution() -> Result<()> {
//     let tokenizer = Arc::new(YourTokenizer::new()?);
//     let model = Arc::new(OrtModel::new("gemma.onnx", tokenizer)?);
//     
//     let agent1 = Agent::new(
//         "ScientificApproach",
//         "Solve puzzles using scientific method",
//         model.clone(),
//     );
//     
//     let agent2 = Agent::new(
//         "IntuitiveApproach",
//         "Solve puzzles using intuition and pattern recognition",
//         model.clone(),
//     );
//     
//     let agent3 = Agent::new(
//         "XenolinguisticApproach",
//         "Solve puzzles using language and symbolism",
//         model.clone(),
//     );
//     
//     let moderator = Agent::new(
//         "PuzzleModerator",
//         "Synthesize different approaches into best solution",
//         model.clone(),
//     );
//     
//     let mut workflow = DebateWorkflow::new(
//         vec![agent1, agent2, agent3],
//         moderator,
//         2,
//     );
//     
//     let solution = workflow.run("How to activate the resonance mirror?").await?;
//     println!("Solution: {}", solution);
//     
//     Ok(())
// }

// ============================================================================
// WORKFLOW BUILDER - Composable workflow construction
// ============================================================================

pub struct WorkflowBuilder {
    steps: Vec<WorkflowStep>,
}

pub enum WorkflowStep {
    Agent(Agent),
    Sequential(SequentialWorkflow),
    Parallel(ParallelWorkflow),
    Routing(RoutingWorkflow),
    Consensus(ConsensusWorkflow),
}

impl WorkflowBuilder {
    pub fn new() -> Self {
        Self { steps: vec![] }
    }
    
    pub fn add_agent(mut self, agent: Agent) -> Self {
        self.steps.push(WorkflowStep::Agent(agent));
        self
    }
    
    pub fn add_parallel(mut self, workflow: ParallelWorkflow) -> Self {
        self.steps.push(WorkflowStep::Parallel(workflow));
        self
    }
    
    pub fn build(self) -> CompositeWorkflow {
        CompositeWorkflow { steps: self.steps }
    }
}

pub struct CompositeWorkflow {
    steps: Vec<WorkflowStep>,
}

impl CompositeWorkflow {
    pub async fn run(&mut self, input: &str) -> Result<String> {
        let mut current_input = input.to_string();
        
        for step in &mut self.steps {
            match step {
                WorkflowStep::Agent(agent) => {
                    current_input = agent.run(&current_input).await?;
                }
                WorkflowStep::Sequential(workflow) => {
                    current_input = workflow.run(&current_input).await?;
                }
                WorkflowStep::Parallel(workflow) => {
                    let results = workflow.run(&current_input).await?;
                    current_input = ParallelWorkflow::aggregate(results);
                }
                WorkflowStep::Routing(workflow) => {
                    current_input = workflow.run(&current_input).await?;
                }
                WorkflowStep::Consensus(workflow) => {
                    current_input = workflow.run(&current_input).await?;
                }
            }
        }
        
        Ok(current_input)
    }
}
