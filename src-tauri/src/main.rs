#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use tauri::{State, Manager};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

mod graph_db;
mod graphs;

use graph_db::GraphDB;
use graphs::world_graph::WorldGraph;
use graphs::lore_graph::LoreGraph;


#[derive(Clone)]
struct AppState {
    world_graph: Arc<Mutex<WorldGraph>>,    lore_graph: Arc<Mutex<LoreGraph>>,
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

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            // Initialize graphs
            let world_graph = Arc::new(Mutex::new(WorldGraph::new()));
            let lore_graph = Arc::new(Mutex::new(LoreGraph::new()));
            
            // Create app state
            let app_state = AppState {
                world_graph,
                lore_graph,
            };
            
            // Manage state
            app.manage(app_state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            initialize_new_game,
            get_world_snapshot,
            move_player,
            get_lore_context
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}