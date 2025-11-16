# Interactive Narrative System - Implementation Guide

## Overview

This system integrates AI-powered narrative generation with the frontend to create an interactive storytelling experience. The player can make choices, and the AI generates new narrative content and options in response.

## Architecture

### Backend (Rust/Tauri)

1. **OrtModel** (`src-tauri/src/workflows/agent_core.rs`)
   - Autoregressive text generation using ONNX Runtime
   - Real tokenizer implementation using HuggingFace tokenizers
   - Supports loading models from directory (model.onnx + tokenizer.json)

2. **NarrativeSystem** (`src-tauri/src/workflows/narrative_system.rs`)
   - Three specialized agents:
     - `WorldGenerator`: Creates initial mission briefings
     - `OptionsGenerator`: Generates 5 player choice options
     - `ResponseGenerator`: Creates narrative responses to player choices

3. **Tauri Commands** (`src-tauri/src/main.rs`)
   - `initialize_narrative_system`: Loads the AI model
   - `generate_initial_mission`: Creates starting narrative and options
   - `process_command_option`: Processes player choice and generates new content

### Frontend (SolidJS)

**MissionScreen.tsx** now includes:
- Dynamic mission briefing with typewriter effect
- Dynamic command options from AI
- Click handlers to process player choices
- Loading states and error handling

## Features Implemented

✅ **Autoregressive Text Generation**
- Multiple token generation in a loop
- EOS token detection
- Proper attention mask handling

✅ **Typewriter Effect**
- Character-by-character text display
- Configurable speed (20ms per character)
- Automatic cleanup

✅ **Interactive Narrative Loop**
1. User clicks a command option
2. Backend processes the choice
3. AI generates new narrative text
4. AI generates 5 new options
5. Frontend displays with typewriter effect

✅ **Error Handling**
- Fallback to static content if AI fails
- Loading states during generation
- Disabled buttons during processing

## Testing the System

### Prerequisites

1. Ensure you have the model files:
   ```
   src-tauri/models/qwen3-0.6B/
   ├── model.onnx
   └── tokenizer.json
   ```

2. Dependencies are installed (should already be in Cargo.toml):
   - `ort = "2.0.0-rc.10"`
   - `tokenizers = "0.19"`
   - `serde`/`serde_json`
   - `anyhow`

### Running the Application

1. Build and run:
   ```bash
   npm run tauri dev
   ```

2. The app will:
   - Load the ONNX model on startup
   - Generate initial mission briefing
   - Display 5 command options

3. Click any command option to:
   - See the narrative update
   - Watch typewriter effect
   - Get new options to continue the story

### Expected Behavior

- **Initial Load**: 5-10 seconds (loading model + generating initial content)
- **After Click**: 3-5 seconds (generating response + new options)
- **Text Display**: Smooth typewriter animation at ~20ms/char

### Troubleshooting

**Model won't load:**
- Check `models/qwen3-0.6B/` directory exists
- Verify `model.onnx` and `tokenizer.json` are present
- Check console for error messages

**Generation is slow:**
- Normal for CPU inference
- Each generation creates ~50-100 tokens
- Consider reducing `max_tokens` in `generate()` calls if needed

**Static fallback appears:**
- AI model failed to initialize
- Check browser console for errors
- Verify model files aren't corrupted

## Customization

### Adjust Typewriter Speed
In `MissionScreen.tsx`, line ~30:
```typescript
}, 20); // Change this value (milliseconds per character)
```

### Change System Prompts
In `narrative_system.rs`, modify agent creation:
```rust
let world_generator = Agent::new(
    "WorldGenerator",
    "Your custom system prompt here...",
    model.clone(),
);
```

### Modify Generation Length
In `agent_core.rs`, `generate()` method:
```rust
pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String>
```
Change `max_tokens` parameter when calling from narrative system.

## API Reference

### Tauri Commands

```typescript
// Initialize the AI system
await invoke("initialize_narrative_system");

// Get initial narrative
const state = await invoke<NarrativeState>("generate_initial_mission");

// Process player choice
const newState = await invoke<NarrativeState>("process_command_option", {
  selectedOption: "Dock immediately"
});
```

### NarrativeState Interface

```typescript
interface NarrativeState {
  mission_briefing: string;      // The narrative text
  command_options: string[];     // Array of 5 action choices
}
```

## Performance Notes

- Model loading: ~5 seconds (one-time on init)
- Text generation: ~3-5 seconds per response
- Typewriter effect: ~2-4 seconds for typical response
- Memory usage: ~500MB-1GB for the loaded model

## Future Enhancements

Potential improvements:
- Add conversation history to agents for better context
- Implement save/load narrative state
- Add multiple narrative branches/paths
- Integrate with world graph for location-aware narratives
- Add voice synthesis for audio narration
- Implement streaming token generation
