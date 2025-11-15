use crate::graph_db::{GraphDB, Node};
use serde_json::json;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct WorldGraph {
    pub db: GraphDB,
}

impl WorldGraph {
    pub fn new() -> Self {
        let mut db = GraphDB::new();
        
        // Create initial world structure
        Self::initialize_world(&mut db);
        
        Self { db }
    }
    
    fn initialize_world(db: &mut GraphDB) {
        // Create hierarchical location structure
        let earth_id = db.create_node(
            "location".to_string(),
            "Earth".to_string(),
            json!({
                "description": "Derelict research outpost orbiting a gas giant"
            })
        );
        
        let aethelstan_id = db.create_node(
            "location".to_string(),
            "Aethelstan Research Station".to_string(),
            json!({
                "description": "Once cutting-edge xenolinguistics facility, now silent",
                "xenon_level": 2.0
            })
        );
        
        let airlock_id = db.create_node(
            "room".to_string(),
            "Airlock Alpha".to_string(),
            json!({
                "description": "Entrance chamber with emergency lighting",
                "xenon_level": 2.0,
                "inventory": ["translator_pda", "medkit"]
            })
        );
        
        // Create containment relationships
        db.create_edge(
            earth_id,
            aethelstan_id,
            "contains".to_string(),
            json!({})
        );
        
        db.create_edge(
            aethelstan_id,
            airlock_id,
            "contains".to_string(),
            json!({})
        );
        
        // Create player agent
        let player_id = db.create_node(
            "agent".to_string(),
            "Player".to_string(),
            json!({
                "role": "hero",
                "inventory": ["translator_pda", "medkit"],
                "status": ["conscious"]
            })
        );
        
        // Place player in airlock
        db.create_edge(
            player_id,
            airlock_id,
            "occupies".to_string(),
            json!({})
        );
    }
    
    pub fn get_player_location(&self) -> Option<Uuid> {
        if let Some(player) = self.db.get_node_by_name("Player") {
            // Get edges from player
            let edges = self.db.get_edges_for_node(&player.id);
            for edge in edges {
                if edge.edge_type == "occupies" {
                    return Some(edge.target);
                }
            }
        }
        None
    }
    
    pub fn move_player_to(&mut self, location_id: Uuid) -> bool {
        if let Some(player) = self.db.get_node_by_name("Player") {
            // Remove existing location edges
            let edges = self.db.get_edges_for_node(&player.id);
            for edge in edges.iter() {
                if edge.edge_type == "occupies" {
                    // In a real implementation, we'd remove the edge
                    // For demo, we'll just create a new one
                }
            }
            
            // Create new location edge
            self.db.create_edge(
                player.id,
                location_id,
                "occupies".to_string(),
                json!({})
            );
            return true;
        }
        false
    }
    
    pub fn get_nearby_locations(&self, location_id: &Uuid) -> Vec<&Node> {
        self.db.get_descendants(location_id, "contains")
    }
    
    pub fn get_world_snapshot(&self) -> serde_json::Value {
        if let Some(player) = self.db.get_node_by_name("Player") {
            if let Some(current_location_id) = self.get_player_location() {
                if let Some(current_location) = self.db.get_node(&current_location_id) {
                    let nearby_locations = self.get_nearby_locations(&current_location_id);
                    
                    return json!({
                        "current_location": {
                            "id": current_location.id.to_string(),
                            "name": current_location.name,
                            "description": current_location.meta.get("description").and_then(|v| v.as_str()).unwrap_or(""),
                            "xenon_level": current_location.meta.get("xenon_level").and_then(|v| v.as_f64()).unwrap_or(0.0)
                        },
                        "nearby_locations": nearby_locations.iter().map(|loc| {
                            json!({
                                "id": loc.id.to_string(),
                                "name": loc.name,
                                "description": loc.meta.get("description").and_then(|v| v.as_str()).unwrap_or("")
                            })
                        }).collect::<Vec<_>>(),
                        "inventory": player.meta.get("inventory")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect::<Vec<String>>())
                            .unwrap_or_default()
                    });
                }
            }
        }
        
        json!({
            "current_location": {
                "id": "",
                "name": "Unknown",
                "description": "Location not found",
                "xenon_level": 0.0
            },
            "nearby_locations": [],
            "inventory": []
        })
    }
}