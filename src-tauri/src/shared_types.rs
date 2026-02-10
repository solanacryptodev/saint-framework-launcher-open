use crate::error::{LlmError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ============================================================================
// BLACKBOARD SLOT - Core data structure for each numbered slot
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackboardSlot {
    pub slot_number: usize,
    pub data: Option<String>,
    pub ready_to_read: bool, // Flag: true = ready for reader, false = ready for writer
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
