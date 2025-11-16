use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::error::{Result, LlmError};

// ============================================================================
// TOOL SYSTEM - Define tools that agents can use
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameter {
    pub name: String,
    pub description: String,
    pub param_type: String,
    pub required: bool,
}

#[derive(Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParameter>,
    pub handler: Arc<dyn Fn(&HashMap<String, String>) -> Result<String> + Send + Sync>,
}

impl std::fmt::Debug for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .field("handler", &"<function>")
            .finish()
    }
}

impl Tool {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Vec<ToolParameter>,
        handler: impl Fn(&HashMap<String, String>) -> Result<String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            handler: Arc::new(handler),
        }
    }

    pub fn to_prompt_format(&self) -> String {
        let params = self.parameters
            .iter()
            .map(|p| format!("{}: {} ({})", p.name, p.param_type, p.description))
            .collect::<Vec<_>>()
            .join(", ");
        
        format!("{}({}): {}", self.name, params, self.description)
    }
}

// ============================================================================
// MESSAGE SYSTEM - Conversation history
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: content.into(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub tool_name: String,
    pub arguments: HashMap<String, String>,
}

// ============================================================================
// BLACKBOARD SLOT - Core data structure for each numbered slot
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackboardSlot {
    pub slot_number: usize,
    pub data: Option<String>,
    pub ready_to_read: bool,  // Flag: true = ready for reader, false = ready for writer
    pub metadata: HashMap<String, String>,
}

impl BlackboardSlot {
    pub fn new(slot_number: usize) -> Self {
        Self {
            slot_number,
            data: None,
            ready_to_read: false,
            metadata: HashMap::new(),
        }
    }

    /// Writer writes data and sets flag to true
    pub fn write(&mut self, data: String) {
        self.data = Some(data);
        self.ready_to_read = true;
    }

    /// Reader consumes data and sets flag to false
    pub fn read(&mut self) -> Option<String> {
        if self.ready_to_read {
            let data = self.data.clone();
            self.ready_to_read = false;
            data
        } else {
            None
        }
    }

    /// Check if slot is ready for writing (flag is false)
    pub fn can_write(&self) -> bool {
        !self.ready_to_read
    }

    /// Check if slot is ready for reading (flag is true)
    pub fn can_read(&self) -> bool {
        self.ready_to_read
    }
}

// ============================================================================
// BLACKBOARD - Manages all numbered slots
// ============================================================================

pub struct Blackboard {
    slots: Arc<Mutex<HashMap<usize, BlackboardSlot>>>,
}

impl Blackboard {
    pub fn new() -> Self {
        Self {
            slots: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Initialize a new slot
    pub fn create_slot(&self, slot_number: usize) -> Result<()> {
        let mut slots = self.slots.lock().unwrap();
        slots.insert(slot_number, BlackboardSlot::new(slot_number));
        Ok(())
    }

    /// Writer attempts to write to slot (only if flag is false)
    pub fn write_to_slot(&self, slot_number: usize, data: String) -> Result<bool> {
        let mut slots = self.slots.lock().unwrap();
        
        if let Some(slot) = slots.get_mut(&slot_number) {
            if slot.can_write() {
                slot.write(data);
                Ok(true)
            } else {
                Ok(false) // Slot not ready for writing yet
            }
        } else {
            Err(LlmError::InvalidInput {
                reason: format!("Slot {} does not exist", slot_number),
            })
        }
    }

    /// Reader attempts to read from slot (only if flag is true)
    pub fn read_from_slot(&self, slot_number: usize) -> Result<Option<String>> {
        let mut slots = self.slots.lock().unwrap();
        
        if let Some(slot) = slots.get_mut(&slot_number) {
            Ok(slot.read())
        } else {
            Err(LlmError::InvalidInput {
                reason: format!("Slot {} does not exist", slot_number),
            })
        }
    }

    /// Check slot status
    pub fn get_slot_status(&self, slot_number: usize) -> Result<BlackboardSlotStatus> {
        let slots = self.slots.lock().unwrap();
        
        if let Some(slot) = slots.get(&slot_number) {
            Ok(BlackboardSlotStatus {
                slot_number,
                ready_to_read: slot.ready_to_read,
                has_data: slot.data.is_some(),
            })
        } else {
            Err(LlmError::InvalidInput {
                reason: format!("Slot {} does not exist", slot_number),
            })
        }
    }

    /// Get all slot statuses
    pub fn get_all_statuses(&self) -> Vec<BlackboardSlotStatus> {
        let slots = self.slots.lock().unwrap();
        slots
            .values()
            .map(|slot| BlackboardSlotStatus {
                slot_number: slot.slot_number,
                ready_to_read: slot.ready_to_read,
                has_data: slot.data.is_some(),
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct BlackboardSlotStatus {
    pub slot_number: usize,
    pub ready_to_read: bool,
    pub has_data: bool,
}
