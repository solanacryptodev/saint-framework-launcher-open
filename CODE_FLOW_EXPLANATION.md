# Code Flow Explanation - Where Agents Are Called

## Complete Call Chain

### 1. Frontend Initialization (`MissionScreen.tsx`)
```typescript
// Line 40-41: Component mounts
onMount(async () => {
  await invoke("initialize_narrative_system");  // ← Calls Tauri command
  const initialState = await invoke<NarrativeState>("generate_initial_mission"); // ← Calls Tauri command
});
```

### 2. Tauri Command Layer (`main.rs`)

**Command: `initialize_narrative_system`** (Lines 121-136)
```rust
#[tauri::command]
async fn initialize_narrative_system(state: State<'_, AppState>) -> Result<String, String> {
    let model = OrtModel::from_dir("models/qwen3-0.6B")?;  // ← Or any ONNX compatible model from Hugging Face. Uses agent_core.rs
    let model = Arc::new(model);
    let narrative_sys = NarrativeSystem::new(model);  // ← Creates agents here!
    *narrative_system = Some(narrative_sys);
    Ok("Narrative system initialized successfully!".to_string())
}
```

**Command: `generate_initial_mission`** (Lines 77-93)
```rust
#[tauri::command]
async fn generate_initial_mission(state: State<'_, AppState>) -> Result<NarrativeState, String> {
    let narrative_system = state.narrative_system.lock()?.as_ref()?;
    let narrative_state = narrative_system.generate_initial_mission()?;  // ← Calls narrative_system.rs
    Ok(narrative_state)
}
```

### 3. Narrative System Layer (`narrative_system.rs`)

**Function: `NarrativeSystem::new()`** (Lines 28-54)
```rust
pub fn new(model: Arc<OrtModel>) -> Self {
    // Creates 3 Agent instances from agent_core!
    let world_generator = Agent::new(
        "WorldGenerator",
        "system prompt...",
        model.clone(),  // ← Agent uses OrtModel from agent_core.rs
    );

    let options_generator = Agent::new(
        "OptionsGenerator", 
        "system prompt...",
        model.clone(),
    );

    let response_generator = Agent::new(
        "ResponseGenerator",
        "system prompt...",
        model.clone(),
    );

    Self {
        world_generator: Mutex::new(world_generator),  // ← Storing agents
        options_generator: Mutex::new(options_generator),
        response_generator: Mutex::new(response_generator),
    }
}
```

**Function: `generate_initial_mission()`** (Lines 58-93)
```rust
pub fn generate_initial_mission(&self) -> Result<NarrativeState> {
    let mut world_gen = self.world_generator.lock()?;
    let mission_briefing = world_gen.query(world_prompt)?;  // ← AGENT CALL #1 (agent_core.rs)

    let mut opts_gen = self.options_generator.lock()?;
    let options_text = opts_gen.query(&options_prompt)?;  // ← AGENT CALL #2 (agent_core.rs)
    
    Ok(NarrativeState { mission_briefing, command_options })
}
```

**Function: `process_command_option()`** (Lines 96-132)
```rust
pub fn process_command_option(&self, selected_option: &str, current_briefing: &str) -> Result<NarrativeState> {
    let mut resp_gen = self.response_generator.lock()?;
    let mission_briefing = resp_gen.query(&response_prompt)?;  // ← AGENT CALL #3 (agent_core.rs)

    let mut opts_gen = self.options_generator.lock()?;
    let options_text = opts_gen.query(&options_prompt)?;  // ← AGENT CALL #4 (agent_core.rs)
    
    Ok(NarrativeState { mission_briefing, command_options })
}
```

### 4. Agent Core Layer (`agent_core.rs`)

**Function: `Agent::query()`** (Lines 260-266)
```rust
pub fn query(&mut self, user_message: &str) -> Result<String> {
    self.conversation.push(Message::user(user_message));
    let prompt = self.build_prompt(user_message);
    let response = self.model.generate(&prompt, 256)?;  // ← Calls OrtModel.generate()
    self.conversation.push(Message::assistant(&response));
    Ok(response)
}
```

**Function: `OrtModel::generate()`** (Lines 82-135) - AUTOREGRESSIVE GENERATION
```rust
pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
    let mut input_ids = self.tokenizer.encode(prompt)?;
    
    // Autoregressive loop - generates multiple tokens
    for _ in 0..max_tokens {
        let attention_mask: Vec<i64> = vec![1; input_ids.len()];
        
        // Run ONNX inference
        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ])?;
        
        // Get next token from logits
        let next_token_id = /* argmax logic */;
        
        // Check for end of sequence
        if next_token_id == 151643 || next_token_id == 2 {
            break;
        }
        
        input_ids.push(next_token_id);  // ← Autoregressive: append token and continue
    }
    
    self.tokenizer.decode(&generated_ids)  // ← Return generated text
}
```

## Summary of Agent Usage

| File | Component | Purpose | Calls |
|------|-----------|---------|-------|
| `main.rs` | Tauri Commands | Entry point from frontend | Creates `NarrativeSystem` with `OrtModel` |
| `narrative_system.rs` | NarrativeSystem | Orchestrates 3 agents | Creates & calls `Agent.query()` 4 times |
| `agent_core.rs` | Agent | Manages conversation & prompts | Calls `OrtModel.generate()` |
| `agent_core.rs` | OrtModel | Autoregressive text gen | Calls ONNX model in loop |

## Where Agents Live

```
NarrativeSystem {
    world_generator: Agent {            ← agent_core::Agent
        model: OrtModel { ... }         ← agent_core::OrtModel  
        conversation: Vec<Message>
        system_prompt: "You are a sci-fi horror narrative generator..."
    },
    options_generator: Agent {          ← agent_core::Agent
        model: OrtModel { ... }         ← Same OrtModel instance (Arc)
        system_prompt: "You are a choice generator..."
    },
    response_generator: Agent {         ← agent_core::Agent
        model: OrtModel { ... }         ← Same OrtModel instance (Arc)
        system_prompt: "You are a narrative consequence generator..."
    }
}
```

## Proof Points

1. **Agents are created**: `narrative_system.rs` line 31, 39, 47
2. **Agents are stored**: `narrative_system.rs` line 50-52
3. **Agents are called**: 
   - `narrative_system.rs` line 70 (`world_gen.query()`)
   - `narrative_system.rs` line 79 (`opts_gen.query()`)
   - `narrative_system.rs` line 106 (`resp_gen.query()`)
   - `narrative_system.rs` line 116 (`opts_gen.query()`)
4. **Agent.query() uses model**: `agent_core.rs` line 263
5. **Model generates text**: `agent_core.rs` lines 82-135

The agents ARE being used - they're just wrapped in the `NarrativeSystem` abstraction!
