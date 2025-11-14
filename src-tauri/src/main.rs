#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

// src/main.rs
use tauri::{State, Builder, generate_handler, generate_context};
use std::sync::{Arc, Mutex};
use uuid::Uuid;
use serde_json::json;

mod graph_db;
use graph_db::{GraphDB};

#[derive(Default)]
pub struct AppState {
    pub db: Arc<Mutex<GraphDB>>,
}

#[tauri::command]
async fn initialize_new_game(app_state: State<'_, AppState>) -> Result<String, String> {
    let mut db = app_state.db.lock().map_err(|_| "Failed to lock database")?;
    
    // Clear existing data
    *db = GraphDB::new();
    
    // Create some example nodes with metadata
    let player_id = db.create_node(
        "character".to_string(),
        "Player".to_string(),
        json!({
            "role": "hero",
            "power_level": 1,
            "background": "ordinary person thrust into adventure"
        })
    );
    
    let tavern_id = db.create_node(
        "location".to_string(),
        "The Rusty Flask Tavern".to_string(),
        json!({
            "atmosphere": "warm",
            "patrons": 12,
            "has_secret_passage": true
        })
    );
    
    let npc_id = db.create_node(
        "character".to_string(),
        "Old Elara".to_string(),
        json!({
            "role": "mentor",
            "age": 78,
            "secrets": ["ancient prophecy", "lost treasure map"],
            "mood": "mysterious"
        })
    );
    
    let sword_id = db.create_node(
        "item".to_string(),
        "Rusty Sword".to_string(),
        json!({
            "rarity": "common",
            "condition": "poor",
            "hidden_power": "awakening"
        })
    );
    
    // Create edges with metadata
    db.create_edge(
        player_id, 
        tavern_id, 
        "LOCATED_AT".to_string(),
        json!({
            "since": "just arrived",
            "purpose": "seeking information"
        })
    );
    
    db.create_edge(
        npc_id, 
        tavern_id, 
        "RESIDES_IN".to_string(),
        json!({
            "duration": "20 years",
            "role": "bartender"
        })
    );
    
    db.create_edge(
        player_id, 
        npc_id, 
        "KNOWS".to_string(),
        json!({
            "trust_level": 0.3,
            "first_meeting": true,
            "conversation_topics": ["local gossip", "ancient legends"]
        })
    );
    
    db.create_edge(
        npc_id, 
        sword_id, 
        "OWNS".to_string(),
        json!({
            "willing_to_sell": false,
            "emotional_attachment": "high",
            "acquisition_story": "from fallen comrade"
        })
    );
    
    // Return both player_id and tavern_id (and other important IDs) for the frontend
    Ok(serde_json::json!({
        "playerId": player_id.to_string(),
        "tavernId": tavern_id.to_string(),
        "npcId": npc_id.to_string(),
        "swordId": sword_id.to_string()
    }).to_string())
}

#[tauri::command]
async fn get_node(app_state: State<'_, AppState>, node_id: String) -> Result<serde_json::Value, String> {
    let db = app_state.db.lock().map_err(|_| "Failed to lock database")?;
    
    let uuid = Uuid::parse_str(&node_id).map_err(|_| "Invalid node ID format")?;
    
    if let Some(node) = db.nodes.get(&uuid) {
        Ok(serde_json::json!({
            "id": node.id.to_string(),
            "type": node.node_type,
            "name": node.name,
            "metadata": node.metadata
        }))
    } else {
        Err("Node not found".to_string())
    }
}

#[tauri::command]
async fn get_edges_for_node(app_state: State<'_, AppState>, node_id: String) -> Result<serde_json::Value, String> {
    let db = app_state.db.lock().map_err(|_| "Failed to lock database")?;
    
    let uuid = Uuid::parse_str(&node_id).map_err(|_| "Invalid node ID format")?;
    
    let mut edges_result = Vec::new();
    
    // Get outgoing edges
    if let Some(edge_ids) = db.edges_by_source.get(&uuid) {
        for edge_id in edge_ids {
            if let Some(edge) = db.edges.get(edge_id) {
                edges_result.push(serde_json::json!({
                    "id": edge.id.to_string(),
                    "source": edge.source.to_string(),
                    "target": edge.target.to_string(),
                    "type": edge.edge_type,
                    "metadata": edge.metadata
                }));
            }
        }
    }
    
    // Get incoming edges
    if let Some(edge_ids) = db.edges_by_target.get(&uuid) {
        for edge_id in edge_ids {
            if let Some(edge) = db.edges.get(edge_id) {
                edges_result.push(serde_json::json!({
                    "id": edge.id.to_string(),
                    "source": edge.source.to_string(),
                    "target": edge.target.to_string(),
                    "type": edge.edge_type,
                    "metadata": edge.metadata
                }));
            }
        }
    }
    
    Ok(serde_json::json!(edges_result))
}

// src/main.rs (continued)
fn main() {
    Builder::default()
        .manage(AppState::default())
        .invoke_handler(generate_handler![
            initialize_new_game,
            get_node,
            get_edges_for_node
        ])
        .run(generate_context!())
        .expect("error while running tauri application");
}
