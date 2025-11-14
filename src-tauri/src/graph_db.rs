// src/graph_db.rs
use std::collections::{HashMap, HashSet};
use uuid::Uuid;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: Uuid,
    pub node_type: String,  // "character", "location", "item", etc.
    pub name: String,
    pub metadata: serde_json::Value,  // Flexible metadata storage
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: Uuid,
    pub source: Uuid,
    pub target: Uuid,
    pub edge_type: String,  // "KNOWS", "OWNS", "LOCATED_AT", etc.
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Default)]
pub struct GraphDB {
    pub nodes: HashMap<Uuid, Node>,
    pub edges: HashMap<Uuid, Edge>,
    pub edges_by_source: HashMap<Uuid, HashSet<Uuid>>,
    pub edges_by_target: HashMap<Uuid, HashSet<Uuid>>,
}

impl GraphDB {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_node(&mut self, node_type: String, name: String, metadata: serde_json::Value) -> Uuid {
        let id = Uuid::new_v4();
        let node = Node {
            id,
            node_type,
            name,
            metadata,
        };
        self.nodes.insert(id, node);
        id
    }

    pub fn create_edge(&mut self, source: Uuid, target: Uuid, edge_type: String, metadata: serde_json::Value) -> Uuid {
        let id = Uuid::new_v4();
        let edge = Edge {
            id,
            source,
            target,
            edge_type,
            metadata,
        };
        
        self.edges.insert(id, edge);
        
        // Update indexes
        self.edges_by_source.entry(source).or_default().insert(id);
        self.edges_by_target.entry(target).or_default().insert(id);
        
        id
    }
}