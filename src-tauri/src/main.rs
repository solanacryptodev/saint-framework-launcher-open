#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tauri::{Manager, State};
use tokio::sync::Mutex as TokioMutex;
use uuid::Uuid;

mod error;
mod graph_db;
mod graphs;
mod openrouter_narrative;
mod shared_types;
mod workflows;

use graph_db::GraphDB;
use graphs::lore_graph::LoreGraph;
use graphs::world_graph::WorldGraph;
use openrouter_narrative::{NarrativeState as OpenRouterNarrativeState, OpenRouterNarrativeSystem};
use workflows::agent_core::AgentModel;
use workflows::narrative_system::{NarrativeState, NarrativeSystem};

#[derive(Clone)]
struct AppState {
    world_graph: Arc<Mutex<WorldGraph>>,
    lore_graph: Arc<Mutex<LoreGraph>>,
    narrative_system: Arc<TokioMutex<Option<NarrativeSystem>>>,
    current_briefing: Arc<Mutex<String>>,
    // OpenRouter narrative system (temporary replacement for local model)
    openrouter_system: Arc<TokioMutex<Option<OpenRouterNarrativeSystem>>>,
    openrouter_briefing: Arc<Mutex<String>>,
}

#[tauri::command]
async fn initialize_new_game(state: State<'_, AppState>) -> Result<String, String> {
    // Reset world graph
    let mut world_graph = state
        .world_graph
        .lock()
        .map_err(|_| "Failed to lock world graph")?;
    *world_graph = WorldGraph::new();

    // Reset lore graph
    let mut lore_graph = state
        .lore_graph
        .lock()
        .map_err(|_| "Failed to lock lore graph")?;
    *lore_graph = LoreGraph::new();

    // Get the player node and starting location
    let player_node = world_graph
        .db
        .get_node_by_name("Player")
        .ok_or_else(|| "Player node not found".to_string())?;

    let starting_location = world_graph
        .get_player_location()
        .ok_or_else(|| "Starting location not found".to_string())?;

    // For now, we'll use the starting location as the "tavern" since that's what the frontend expects
    let node_ids = serde_json::json!({
        "playerId": player_node.id.to_string(),
        "tavernId": starting_location.to_string(),
        "startLocationId": starting_location.to_string()
    });

    Ok(node_ids.to_string())
}

#[tauri::command]
async fn get_node(
    state: State<'_, AppState>,
    node_id: String,
) -> Result<serde_json::Value, String> {
    let world_graph = state
        .world_graph
        .lock()
        .map_err(|_| "Failed to lock world graph")?;

    // Parse the node ID
    let uuid = Uuid::parse_str(&node_id).map_err(|_| "Invalid node ID format".to_string())?;

    // Get the node from the graph
    let node = world_graph
        .db
        .get_node(&uuid)
        .ok_or_else(|| format!("Node not found: {}", node_id))?;

    // Return node data as JSON
    Ok(serde_json::json!({
        "id": node.id.to_string(),
        "node_type": node.node_type,
        "name": node.name,
        "meta": node.meta
    }))
}

#[tauri::command]
async fn get_world_snapshot(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let world_graph = state
        .world_graph
        .lock()
        .map_err(|_| "Failed to lock world graph")?;

    // Get world snapshot
    let snapshot = world_graph.get_world_snapshot();

    Ok(snapshot)
}

#[tauri::command]
async fn move_player(
    state: State<'_, AppState>,
    target_location_id: String,
) -> Result<serde_json::Value, String> {
    let mut world_graph = state
        .world_graph
        .lock()
        .map_err(|_| "Failed to lock world graph")?;

    // Parse location ID
    let location_uuid =
        Uuid::parse_str(&target_location_id).map_err(|_| "Invalid location ID format")?;

    // Move player
    if world_graph.move_player_to(location_uuid) {
        // Return updated snapshot
        Ok(world_graph.get_world_snapshot())
    } else {
        Err("Failed to move player to location".to_string())
    }
}

#[tauri::command]
async fn get_lore_context(
    state: State<'_, AppState>,
    location_id: String,
    player_state: serde_json::Value,
) -> Result<Vec<serde_json::Value>, String> {
    let lore_graph = state
        .lore_graph
        .lock()
        .map_err(|_| "Failed to lock lore graph")?;

    // Get relevant lore
    let lore_context = lore_graph.get_relevant_lore(&location_id, player_state);

    Ok(lore_context)
}

#[tauri::command]
async fn load_model(app: tauri::AppHandle) -> Result<String, String> {
    // Construct path to the model directory (native weights, not GGUF)
    let model_path: PathBuf = std::env::current_dir()
        .map(|d| d.join("models").join("phi-4-mini-instruct"))
        .map_err(|e| format!("Failed to get current directory: {}", e))?;

    // Validate model directory exists
    if !Path::new(&model_path).exists() {
        return Err(format!("Model directory not found at: {:?}", model_path));
    }

    println!("📦 Loading model from directory: {:?}", model_path);

    // Load with mistralrs
    let model = AgentModel::new(model_path).await.map_err(|e| {
        let err_msg = format!("Failed to load model: {}", e);
        eprintln!("❌ {}", err_msg);
        err_msg
    })?;

    println!("✅ Phi-4 model loaded successfully");

    // Store model in app state for later use
    app.manage(Arc::new(model));

    Ok("Model loaded successfully".to_string())
}

#[tauri::command]
async fn initialize_narrative_system(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<String, String> {
    println!("📡 Tauri Command: initialize_narrative_system called");

    // Check if model is already loaded in state
    let shared_model = if let Some(state_model) = app.try_state::<Arc<AgentModel>>() {
        println!("✅ Using existing loaded model from Tauri state");
        state_model.inner().clone()
    } else {
        println!("⚠️ Model not found in state, loading new instance...");
        // Load the model from directory (native weights)
        let model_path: PathBuf = std::env::current_dir()
            .map(|d| d.join("models").join("phi-4-mini-instruct"))
            .map_err(|e| format!("Failed to get current directory: {}", e))?;

        Arc::new(
            AgentModel::new(model_path)
                .await
                .map_err(|e| format!("Failed to load shared model: {}", e))?,
        )
    };

    println!("✅ Shared model loaded");

    // Create agents sharing the same model
    // The Agent::new implementation now accepts Arc<AgentModel> and keeps it as Arc
    let world_model = shared_model.clone();
    let options_model = shared_model.clone();
    let response_model = shared_model.clone();

    println!("✅ All 3 models loaded");

    // Create narrative system with 3 models
    println!("🏗️  Creating NarrativeSystem with 3 agents...");
    let narrative_sys = NarrativeSystem::new(world_model, options_model, response_model);
    println!("✅ NarrativeSystem created");

    // Store in app state (TokioMutex uses .lock().await)
    let mut narrative_system = state.narrative_system.lock().await;
    *narrative_system = Some(narrative_sys);

    println!("✅ Narrative system initialized successfully!");
    Ok("Narrative system initialized successfully!".to_string())
}

#[tauri::command]
async fn generate_initial_mission(state: State<'_, AppState>) -> Result<NarrativeState, String> {
    println!("📡 Tauri Command: generate_initial_mission called");

    // Get a reference to the narrative system and hold the lock during the await
    let mut guard = state.narrative_system.lock().await;

    let narrative_system = guard.as_mut().ok_or_else(|| {
        eprintln!("❌ Narrative system not initialized");
        "Narrative system not initialized. Please load the AI model first.".to_string()
    })?;

    let narrative_state = narrative_system
        .generate_initial_mission()
        .await
        .map_err(|e| {
            let err_msg = format!("Failed to generate initial mission: {}", e);
            eprintln!("❌ {}", err_msg);
            err_msg
        })?;

    // Drop the guard before locking current_briefing
    drop(guard);

    // Store current briefing
    let mut current_briefing = state
        .current_briefing
        .lock()
        .map_err(|_| "Failed to lock current briefing")?;
    *current_briefing = narrative_state.mission_briefing.clone();

    println!("✅ Initial mission generated successfully");
    Ok(narrative_state)
}

#[tauri::command]
async fn process_command_option(
    state: State<'_, AppState>,
    selected_option: String,
) -> Result<NarrativeState, String> {
    println!("📡 Tauri Command: process_command_option called");
    println!("   Selected option: '{}'", selected_option);

    // Get current briefing first
    let current_briefing = state
        .current_briefing
        .lock()
        .map_err(|_| "Failed to lock current briefing")?
        .clone();

    // Get a reference to the narrative system and hold the lock during the await
    let mut guard = state.narrative_system.lock().await;

    let narrative_system = guard.as_mut().ok_or_else(|| {
        eprintln!("❌ Narrative system not initialized");
        "Narrative system not initialized. Please load the AI model first.".to_string()
    })?;

    let narrative_state = narrative_system
        .process_command_option(&selected_option, &current_briefing)
        .await
        .map_err(|e| {
            let err_msg = format!("Failed to process command option: {}", e);
            eprintln!("❌ {}", err_msg);
            err_msg
        })?;

    // Drop the guard before locking current_briefing
    drop(guard);

    // Update current briefing
    let mut current_briefing = state
        .current_briefing
        .lock()
        .map_err(|_| "Failed to lock current briefing")?;
    *current_briefing = narrative_state.mission_briefing.clone();

    println!("✅ Command option processed successfully");
    Ok(narrative_state)
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            // Initialize graphs
            let world_graph = Arc::new(Mutex::new(WorldGraph::new()));
            let lore_graph = Arc::new(Mutex::new(LoreGraph::new()));

            // Initialize narrative system as None (will be loaded on demand)
            // Use TokioMutex for narrative_system since it's used across await points
            let narrative_system = Arc::new(TokioMutex::new(None));
            let current_briefing = Arc::new(Mutex::new(String::new()));

            // Initialize OpenRouter narrative system
            let openrouter_system = Arc::new(TokioMutex::new(None));
            let openrouter_briefing = Arc::new(Mutex::new(String::new()));

            // Create app state
            let app_state = AppState {
                world_graph,
                lore_graph,
                narrative_system,
                current_briefing,
                openrouter_system,
                openrouter_briefing,
            };

            // Manage state
            app.manage(app_state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            initialize_new_game,
            get_node,
            get_world_snapshot,
            move_player,
            get_lore_context,
            load_model,
            initialize_narrative_system,
            generate_initial_mission,
            process_command_option,
            // OpenRouter commands (temporary replacement)
            initialize_openrouter_narrative,
            generate_openrouter_mission,
            process_openrouter_command_option
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

// ============================================================================
// OPENROUTER COMMANDS (Temporary replacement for local model)
// ============================================================================

#[tauri::command]
async fn initialize_openrouter_narrative(state: State<'_, AppState>) -> Result<String, String> {
    println!("📡 Tauri Command: initialize_openrouter_narrative called");

    // Create OpenRouter narrative system
    let openrouter_sys = OpenRouterNarrativeSystem::new().map_err(|e| {
        let err_msg = format!("Failed to initialize OpenRouter system: {}", e);
        eprintln!("❌ {}", err_msg);
        err_msg
    })?;

    // Store in app state
    let mut openrouter_system = state.openrouter_system.lock().await;
    *openrouter_system = Some(openrouter_sys);

    println!("✅ OpenRouter narrative system initialized successfully!");
    Ok("OpenRouter narrative system initialized successfully!".to_string())
}

#[tauri::command]
async fn generate_openrouter_mission(
    state: State<'_, AppState>,
) -> Result<OpenRouterNarrativeState, String> {
    println!("📡 Tauri Command: generate_openrouter_mission called");

    // Get a reference to the OpenRouter narrative system
    let mut guard = state.openrouter_system.lock().await;

    let openrouter_system = guard.as_mut().ok_or_else(|| {
        eprintln!("❌ OpenRouter narrative system not initialized");
        "OpenRouter narrative system not initialized. Please initialize first.".to_string()
    })?;

    let narrative_state = openrouter_system
        .generate_initial_mission()
        .await
        .map_err(|e| {
            let err_msg = format!("Failed to generate initial mission: {}", e);
            eprintln!("❌ {}", err_msg);
            err_msg
        })?;

    // Drop the guard before locking openrouter_briefing
    drop(guard);

    // Store current briefing
    let mut current_briefing = state
        .openrouter_briefing
        .lock()
        .map_err(|_| "Failed to lock openrouter_briefing")?;
    *current_briefing = narrative_state.mission_briefing.clone();

    println!("✅ OpenRouter initial mission generated successfully");
    Ok(narrative_state)
}

#[tauri::command]
async fn process_openrouter_command_option(
    state: State<'_, AppState>,
    selected_option: String,
) -> Result<OpenRouterNarrativeState, String> {
    println!("📡 Tauri Command: process_openrouter_command_option called");
    println!("   Selected option: '{}'", selected_option);

    // Get current briefing first
    let current_briefing = state
        .openrouter_briefing
        .lock()
        .map_err(|_| "Failed to lock openrouter_briefing")?
        .clone();

    // Get a reference to the OpenRouter narrative system
    let mut guard = state.openrouter_system.lock().await;

    let openrouter_system = guard.as_mut().ok_or_else(|| {
        eprintln!("❌ OpenRouter narrative system not initialized");
        "OpenRouter narrative system not initialized. Please initialize first.".to_string()
    })?;

    let narrative_state = openrouter_system
        .process_command_option(&selected_option, &current_briefing)
        .await
        .map_err(|e| {
            let err_msg = format!("Failed to process command option: {}", e);
            eprintln!("❌ {}", err_msg);
            err_msg
        })?;

    // Drop the guard before locking openrouter_briefing
    drop(guard);

    // Update current briefing
    let mut current_briefing = state
        .openrouter_briefing
        .lock()
        .map_err(|_| "Failed to lock openrouter_briefing")?;
    *current_briefing = narrative_state.mission_briefing.clone();

    println!("✅ OpenRouter command option processed successfully");
    Ok(narrative_state)
}
