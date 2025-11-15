use crate::GraphDB;
use serde_json::json;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct LoreGraph {
    pub db: GraphDB,
}

impl LoreGraph {
    pub fn new() -> Self {
        let mut db = GraphDB::new();
        
        // Create initial lore entries
        Self::initialize_lore(&mut db);
        
        Self { db }
    }
    
    fn initialize_lore(db: &mut GraphDB) {
        // Mission brief lore
        let mission_brief_id = db.create_node(
            "lore".to_string(),
            "Mission Brief".to_string(),
            json!({
                "content": "The Aethelstan was studying Resonance Echoes—structured energy patterns from the Kuiper Belt anomaly.",
                "section": ["missions", "background"],
                "metadata": {
                    "helpfulness": 10,
                    "harmfulness": 0,
                    "deprecated": false,
                    "temporal_scope": {
                        "from": "beginning",
                        "to": "∞"
                    }
                }
            })
        );
        
        // Xenon hazard lore
        let xenon_hazard_id = db.create_node(
            "lore".to_string(),
            "Xenon Hazard".to_string(),
            json!({
                "content": "Prolonged xenon-133 exposure causes auditory hallucinations and temporal disorientation.",
                "section": ["hazards", "science"],
                "metadata": {
                    "helpfulness": 8,
                    "harmfulness": 0,
                    "deprecated": false
                }
            })
        );
        
        // Echo nature lore
        let echo_nature_id = db.create_node(
            "lore".to_string(),
            "Echo Nature".to_string(),
            json!({
                "content": "The Echo isn't an entity—it's a voice trying to find form in our reality.",
                "section": ["echoes", "truth"],
                "metadata": {
                    "helpfulness": 7,
                    "harmfulness": 0,
                    "deprecated": false
                }
            })
        );
        
        // Create relationships between lore entries
        db.create_edge(
            xenon_hazard_id,
            echo_nature_id,
            "causes".to_string(),
            json!({})
        );
    }
    
    pub fn get_relevant_lore(&self, location_id: &str, player_state: serde_json::Value) -> Vec<serde_json::Value> {
        // Get all lore entries
        let all_lore = self.db.get_nodes_by_type("lore");
        
        // Filter and sort by helpfulness
        let mut filtered_lore: Vec<serde_json::Value> = all_lore.iter()
            .filter_map(|node| {
                if let Some(metadata) = node.meta.get("metadata") {
                    if let Some(deprecated) = metadata.get("deprecated").and_then(|v| v.as_bool()) {
                        if deprecated {
                            return None;
                        }
                    }
                    if let Some(helpfulness) = metadata.get("helpfulness").and_then(|v| v.as_i64()) {
                        if helpfulness >= 3 {
                            return Some(json!({
                                "id": node.id.to_string(),
                                "content": node.meta.get("content").and_then(|v| v.as_str()).unwrap_or(""),
                                "section": node.meta.get("section")
                                    .and_then(|v| v.as_array())
                                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect::<Vec<String>>())
                                    .unwrap_or_default(),
                                "metadata": metadata.clone()
                            }));
                        }
                    }
                }
                None
            })
            .collect();
        
        // Sort by helpfulness descending
        filtered_lore.sort_by(|a, b| {
            let a_help = a.get("metadata").and_then(|m| m.get("helpfulness")).and_then(|v| v.as_i64()).unwrap_or(0);
            let b_help = b.get("metadata").and_then(|m| m.get("helpfulness")).and_then(|v| v.as_i64()).unwrap_or(0);
            b_help.cmp(&a_help)
        });
        
        // Limit to 5 results
        filtered_lore.truncate(5);
        
        filtered_lore
    }
    
    pub fn add_lore_entry(&mut self, content: String, section: Vec<String>, metadata: serde_json::Value) -> Uuid {
        self.db.create_node(
            "lore".to_string(),
            format!("Lore Entry {}", self.db.get_nodes_by_type("lore").len() + 1),
            json!({
                "content": content,
                "section": section,
                "metadata": metadata
            })
        )
    }
}