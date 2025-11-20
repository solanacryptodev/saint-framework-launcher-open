use crate::workflows::agent_core::{Agent, OrtModel};
use std::sync::Arc;

#[test]
fn test_sci_fi_storyteller_agent() {
    // Load the actual ONNX model
    let model_dir = "models/gemma-270M";
    let model = OrtModel::from_dir(model_dir).expect("Failed to load model");
    let model_arc = Arc::new(model);
    
    // Create an agent with a sci-fi storyteller system prompt
    let system_prompt = "You are a storyteller for a sci-fi adventure set on a mysterious space station \
                        called Station Xenon-7. The station was researching alien artifacts when all \
                        communications went dark. Strange energy readings are emanating from the lower decks. \
                        You narrate events with atmospheric detail and build tension.";
    
    let mut agent = Agent::new(
        "SciFiStoryteller",
        system_prompt,
        model_arc,
    );
    
    println!("🤖 Created SciFi Storyteller Agent");
    
    // Ask a question about the sci-fi world
    let user_question = "How many people were onboard Station Xenon-7?";
    
    println!("👤 User asks: {}", user_question);
    
    // Get response from the agent
    let response = agent.query(user_question)
        .expect("Failed to get response from agent");
    
    println!("🎭 Agent responds: {}", response);
    
    // Verify we got a non-empty response
    assert!(!response.is_empty(), "Agent should return a non-empty response");
    
    // Verify the response has some reasonable length (at least a few words)
    assert!(response.len() > 10, "Response should be more than just a few characters");
    
    println!("✅ Sci-fi storyteller agent test passed!");
}
