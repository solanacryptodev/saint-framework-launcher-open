use std::collections::{HashMap, HashSet};
use uuid::Uuid;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: Uuid,
    pub node_type: String,    // "location", "agent", "item", "lore", etc.
    pub name: String,
    pub meta: serde_json::Value,  // Flexible metadata storage
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: Uuid,
    pub source: Uuid,
    pub target: Uuid,
    pub edge_type: String,    // "contains", "knows", "supports", etc.
    pub meta: serde_json::Value,
}

#[derive(Debug, Default, Clone)]
pub struct GraphDB {
    nodes: HashMap<Uuid, Node>,
    edges: HashMap<Uuid, Edge>,
    // Indexes for fast lookups
    edges_by_source: HashMap<Uuid, HashSet<Uuid>>,
    edges_by_target: HashMap<Uuid, HashSet<Uuid>>,
    nodes_by_type: HashMap<String, HashSet<Uuid>>,
}

impl GraphDB {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_node(&mut self, node_type: String, name: String, meta: serde_json::Value) -> Uuid {
        let id = Uuid::new_v4();
        let node = Node { id, node_type: node_type.clone(), name, meta };
        self.nodes.insert(id, node);
        self.nodes_by_type.entry(node_type).or_default().insert(id);
        id
    }

    pub fn create_edge(&mut self, source: Uuid, target: Uuid, edge_type: String, meta: serde_json::Value) -> Uuid {
        let id = Uuid::new_v4();
        let edge = Edge { id, source, target, edge_type: edge_type.clone(), meta };
        
        self.edges.insert(id, edge.clone());
        
        // Update indexes
        self.edges_by_source.entry(source).or_default().insert(id);
        self.edges_by_target.entry(target).or_default().insert(id);
        
        id
    }

    pub fn get_node(&self, id: &Uuid) -> Option<&Node> {
        self.nodes.get(id)
    }

    pub fn get_node_by_name(&self, name: &str) -> Option<&Node> {
        self.nodes.values().find(|node| node.name == name)
    }

    pub fn get_edges_for_node(&self, id: &Uuid) -> Vec<&Edge> {
        let mut results = Vec::new();
        
        // Get outgoing edges
        if let Some(edge_ids) = self.edges_by_source.get(id) {
            for &edge_id in edge_ids {
                if let Some(edge) = self.edges.get(&edge_id) {
                    results.push(edge);
                }
            }
        }
        
        // Get incoming edges
        if let Some(edge_ids) = self.edges_by_target.get(id) {
            for &edge_id in edge_ids {
                if let Some(edge) = self.edges.get(&edge_id) {
                    results.push(edge);
                }
            }
        }
        
        results
    }

    pub fn get_nodes_by_type(&self, node_type: &str) -> Vec<&Node> {
        if let Some(node_ids) = self.nodes_by_type.get(node_type) {
            node_ids.iter()
                .filter_map(|id| self.nodes.get(id))
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn get_descendants(&self, node_id: &Uuid, edge_type: &str) -> Vec<&Node> {
        let mut results = Vec::new();
        let mut to_visit = vec![node_id];
        let mut visited = HashSet::new();
        
        while let Some(current_id) = to_visit.pop() {
            if visited.contains(current_id) {
                continue;
            }
            visited.insert(*current_id);
            
            // Get all edges from this node
            if let Some(edge_ids) = self.edges_by_source.get(current_id) {
                for &edge_id in edge_ids {
                    if let Some(edge) = self.edges.get(&edge_id) {
                        if edge.edge_type == edge_type {
                            if let Some(node) = self.nodes.get(&edge.target) {
                                results.push(node);
                                to_visit.push(&edge.target);
                            }
                        }
                    }
                }
            }
        }
        
        results
    }
}