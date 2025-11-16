#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use tauri::{State, Manager};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

mod error;
mod config;
mod kv_cache;
mod graph_db;
mod graphs;
mod shared_types;
mod workflows;

use graph_db::GraphDB;
use graphs::world_graph::WorldGraph;
use graphs::lore_graph::LoreGraph;
use workflows::narrative_system::{NarrativeSystem, NarrativeState};
use workflows::agent_core::OrtModel;


#[derive(Clone)]
struct AppState {
    world_graph: Arc<Mutex<WorldGraph>>,
    lore_graph: Arc<Mutex<LoreGraph>>,
    narrative_system: Arc<Mutex<Option<NarrativeSystem>>>,
    current_briefing: Arc<Mutex<String>>,
}

#[tauri::command]
async fn initialize_new_game(state: State<'_, AppState>) -> Result<String, String> {
    // Reset world graph
    let mut world_graph = state.world_graph.lock().map_err(|_| "Failed to lock world graph")?;
    *world_graph = WorldGraph::new();
    
    // Reset lore graph
    let mut lore_graph = state.lore_graph.lock().map_err(|_| "Failed to lock lore graph")?;
    *lore_graph = LoreGraph::new();
    
    Ok("New game initialized successfully with demo graph data!".to_string())
}

#[tauri::command]
async fn get_world_snapshot(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let world_graph = state.world_graph.lock().map_err(|_| "Failed to lock world graph")?;
    
    // Get world snapshot
    let snapshot = world_graph.get_world_snapshot();
    
    Ok(snapshot)
}

#[tauri::command]
async fn move_player(state: State<'_, AppState>, target_location_id: String) -> Result<serde_json::Value, String> {
    let mut world_graph = state.world_graph.lock().map_err(|_| "Failed to lock world graph")?;
    
    // Parse location ID
    let location_uuid = Uuid::parse_str(&target_location_id).map_err(|_| "Invalid location ID format")?;
    
    // Move player
    if world_graph.move_player_to(location_uuid) {
        // Return updated snapshot
        Ok(world_graph.get_world_snapshot())
    } else {
        Err("Failed to move player to location".to_string())
    }
}

#[tauri::command]
async fn get_lore_context(state: State<'_, AppState>, location_id: String, player_state: serde_json::Value) -> Result<Vec<serde_json::Value>, String> {
    let lore_graph = state.lore_graph.lock().map_err(|_| "Failed to lock lore graph")?;
    
    // Get relevant lore
    let lore_context = lore_graph.get_relevant_lore(&location_id, player_state);
    
    Ok(lore_context)
}

#[tauri::command]
async fn generate_initial_mission(state: State<'_, AppState>) -> Result<NarrativeState, String> {
    println!("📡 Tauri Command: generate_initial_mission called");
    
    let narrative_system = state.narrative_system.lock()
        .map_err(|_| "Failed to lock narrative system")?;
    
    let narrative_system = narrative_system.as_ref()
        .ok_or_else(|| {
            eprintln!("❌ Narrative system not initialized");
            "Narrative system not initialized. Please load the AI model first.".to_string()
        })?;
    
    let narrative_state = narrative_system.generate_initial_mission()
        .map_err(|e| {
            let err_msg = format!("Failed to generate initial mission: {}", e);
            eprintln!("❌ {}", err_msg);
            err_msg
        })?;
    
    // Store current briefing
    let mut current_briefing = state.current_briefing.lock()
        .map_err(|_| "Failed to lock current briefing")?;
    *current_briefing = narrative_state.mission_briefing.clone();
    
    println!("✅ Initial mission generated successfully");
    Ok(narrative_state)
}

#[tauri::command]
async fn process_command_option(state: State<'_, AppState>, selected_option: String) -> Result<NarrativeState, String> {
    println!("📡 Tauri Command: process_command_option called");
    println!("   Selected option: '{}'", selected_option);
    
    let narrative_system = state.narrative_system.lock()
        .map_err(|_| "Failed to lock narrative system")?;
    
    let narrative_system = narrative_system.as_ref()
        .ok_or_else(|| {
            eprintln!("❌ Narrative system not initialized");
            "Narrative system not initialized. Please load the AI model first.".to_string()
        })?;
    
    // Get current briefing
    let current_briefing = state.current_briefing.lock()
        .map_err(|_| "Failed to lock current briefing")?
        .clone();
    
    let narrative_state = narrative_system.process_command_option(&selected_option, &current_briefing)
        .map_err(|e| {
            let err_msg = format!("Failed to process command option: {}", e);
            eprintln!("❌ {}", err_msg);
            err_msg
        })?;
    
    // Update current briefing
    let mut current_briefing = state.current_briefing.lock()
        .map_err(|_| "Failed to lock current briefing")?;
    *current_briefing = narrative_state.mission_briefing.clone();
    
    println!("✅ Command option processed successfully");
    Ok(narrative_state)
}

#[tauri::command]
async fn initialize_narrative_system(state: State<'_, AppState>) -> Result<String, String> {
    println!("📡 Tauri Command: initialize_narrative_system called");
    
    // Load the ONNX model
    println!("📦 Loading ONNX model from models/qwen3-0.6B/...");
    let model = OrtModel::from_dir("models/qwen3-0.6B")
        .map_err(|e| {
            let err_msg = format!("Failed to load model: {}", e);
            eprintln!("❌ {}", err_msg);
            err_msg
        })?;
    
    println!("✅ ONNX model loaded successfully");
    let model = Arc::new(model);
    
    // Create narrative system
    println!("🏗️  Creating NarrativeSystem with 3 agents...");
    let narrative_sys = NarrativeSystem::new(model);
    println!("✅ NarrativeSystem created");
    
    // Store in app state
    let mut narrative_system = state.narrative_system.lock()
        .map_err(|_| "Failed to lock narrative system")?;
    *narrative_system = Some(narrative_sys);
    
    println!("✅ Narrative system initialized successfully!");
    Ok("Narrative system initialized successfully!".to_string())
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            // Initialize graphs
            let world_graph = Arc::new(Mutex::new(WorldGraph::new()));
            let lore_graph = Arc::new(Mutex::new(LoreGraph::new()));
            
            // Initialize narrative system as None (will be loaded on demand)
            let narrative_system = Arc::new(Mutex::new(None));
            let current_briefing = Arc::new(Mutex::new(String::new()));
            
            // Create app state
            let app_state = AppState {
                world_graph,
                lore_graph,
                narrative_system,
                current_briefing,
            };
            
            // Manage state
            app.manage(app_state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            initialize_new_game,
            get_world_snapshot,
            move_player,
            get_lore_context,
            initialize_narrative_system,
            generate_initial_mission,
            process_command_option
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
