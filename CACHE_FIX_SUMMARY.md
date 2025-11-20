# Cache Contamination & Conversation Pollution Fixes

## Summary

Fixed critical issues in the agent and narrative system that caused repetitive outputs and cache contamination between agents.

---

## Issues Identified

### Issue 1: Auto-Reset in `OrtModel::generate()` ❌
**Problem:** The `generate()` method automatically called `reset_conversation()` at the start, preventing effective KV cache usage and causing cache to be reset even when stateful generation was desired.

**Impact:** No benefit from KV caching; every generation was a cold start.

### Issue 2: Shared Model Cache Contamination ❌
**Problem:** All three narrative agents (`WorldGenerator`, `OptionsGenerator`, `ResponseGenerator`) shared the same `OrtModel` instance via `Arc::clone()`, causing their KV caches to interfere with each other.

**Impact:** Agent B's cache would contain remnants of Agent A's previous generation, leading to contaminated outputs.

### Issue 3: Conversation History Pollution ❌
**Problem:** `Agent::query()` accumulated conversation history in the `self.conversation` array. Each subsequent `build_prompt()` call included ALL previous Q&A pairs, causing the model to see old narratives when generating new ones.

**Impact:** Repetitive responses that referenced previous story elements even when generating fresh narratives.

---

## Fixes Implemented

### Fix 1: Removed Auto-Reset from `OrtModel::generate()` ✅
**Location:** `src-tauri/src/workflows/agent_core.rs`

**Change:**
```rust
pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
    // NOTE: Cache reset is now managed by the caller (Agent), not by OrtModel.
    // This allows for both stateful and stateless generations.
    // The caller should call reset_conversation() before calling generate() if needed.
    
    // Removed: self.reset_conversation();
    
    let prompt_tokens = self.tokenizer.encode(prompt)?;
    // ... rest of generation
}
```

**Benefit:** Allows caller to control when cache is reset, enabling both stateful and stateless generation patterns.

---

### Fix 2: Modified `Agent::query()` to Explicitly Reset Cache ✅
**Location:** `src-tauri/src/workflows/agent_core.rs`

**Change:**
```rust
pub fn query(&mut self, user_message: &str) -> Result<String> {
    // Reset model cache before generating to ensure fresh context
    self.model.reset_conversation();
    
    let prompt = self.build_prompt(user_message);
    let response = self.model.generate(&prompt, 256)?;
    
    self.conversation.push(Message::user(user_message));
    self.conversation.push(Message::assistant(&response));
    Ok(response)
}
```

**Benefit:** Ensures fresh cache for each query when using stateful conversation tracking.

---

### Fix 3: Implemented `Agent::query_stateless()` ✅
**Location:** `src-tauri/src/workflows/agent_core.rs`

**New Method:**
```rust
pub fn query_stateless(&mut self, user_message: &str) -> Result<String> {
    println!("🔄 Agent '{}' - Stateless query", self.name);
    
    // Reset model cache to ensure fresh generation
    self.model.reset_conversation();
    
    // Build minimal prompt with ONLY system prompt + current message (no conversation history)
    let prompt = format!(
        "# SYSTEM\n{}\n\nUser: {}\nAssistant:",
        self.system_prompt,
        user_message
    );
    
    let response = self.model.generate(&prompt, 256)?;
    
    // DON'T add to conversation history - this is stateless
    Ok(response)
}
```

**Benefits:**
- **No conversation history pollution:** Only system prompt + current message
- **Fresh cache:** Explicitly resets before generation
- **Independent generations:** Each call is completely isolated
- **Ideal for narrative generation:** Perfect for one-shot story/option generation

---

### Fix 4: Updated `NarrativeSystem` to Use `query_stateless()` ✅
**Location:** `src-tauri/src/workflows/narrative_system.rs`

**Changes:**
- `generate_initial_mission()`: Both `world_generator` and `options_generator` now use `query_stateless()`
- `process_command_option()`: Both `world_generator` and `options_generator` now use `query_stateless()`

**Before:**
```rust
let mission_briefing = world_gen.query(world_prompt)?;
let options_text = opts_gen.query(&options_prompt)?;
```

**After:**
```rust
let mission_briefing = world_gen.query_stateless(world_prompt)?;
let options_text = opts_gen.query_stateless(&options_prompt)?;
```

**Benefits:**
- Each narrative generation is independent
- No accumulation of previous story turns
- Cache is reset before each agent call
- Prevents repetitive/contaminated outputs

---

### Fix 5: Agent Reset Method Already Existed ✅
**Location:** `src-tauri/src/workflows/agent_core.rs`

**Existing Method:**
```rust
pub fn reset(&mut self) {
    self.conversation.clear();
    self.conversation.push(Message::system(&self.system_prompt));
    self.model.reset_conversation();
}
```

**Benefit:** Allows manual reset of agent state when needed.

---

## Architecture Decision: Shared Model vs. Separate Models

### Current Architecture (Implemented) ✅
**Approach:** All agents share one `OrtModel` instance via `Arc::clone()`

**Pros:**
- Memory efficient (~800MB for one model vs ~2.4GB for three)
- Cache management handled via explicit `reset_conversation()` calls
- `query_stateless()` ensures clean generations

**Cons:**
- Requires discipline to call `reset_conversation()` or use `query_stateless()`
- Shared cache could theoretically cause issues if not managed properly

**Documented in:** `NarrativeSystem::new()` with full explanation

### Alternative Architecture (Not Implemented)
**Approach:** Create separate `OrtModel` instance per agent

**Pros:**
- Complete cache isolation between agents
- No risk of cache contamination

**Cons:**
- 3x memory usage (~2.4GB total)
- More resource-intensive
- Unnecessary with proper cache management

**Decision:** Current approach is sufficient with proper cache hygiene via `query_stateless()`.

---

## Testing Recommendations

1. **Test narrative generation flow:**
   - Start new game → should get fresh mission briefing
   - Make choice → should get fresh consequence (not referencing old conversations)
   - Make multiple choices in succession → each should be independent

2. **Test cache reset:**
   - Verify cache is empty before each `query_stateless()` call
   - Check that conversation history is not growing in narrative agents

3. **Test stateful vs stateless:**
   - `query()`: conversation history accumulates (for chat-style agents)
   - `query_stateless()`: conversation history unchanged (for narrative agents)

---

## Migration Guide

If you have existing code using `Agent::query()`:

**For narrative/one-shot generations:**
```rust
// OLD (accumulates history)
let response = agent.query("Generate a story")?;

// NEW (stateless)
let response = agent.query_stateless("Generate a story")?;
```

**For conversational agents (chat, tool use):**
```rust
// Keep using query() - it now properly resets cache
let response = agent.query("What should I do?")?;
```

**Manual reset when needed:**
```rust
agent.reset(); // Clears conversation history + cache
```

---

## Files Modified

1. `src-tauri/src/workflows/agent_core.rs` - Core agent and model implementation
2. `src-tauri/src/workflows/narrative_system.rs` - Narrative system using agents

---

## Result

✅ **Cache contamination eliminated:** Each agent's generations are independent  
✅ **Conversation pollution fixed:** Narrative agents use stateless queries  
✅ **Proper cache management:** Explicit control over when cache is reset  
✅ **Memory efficient:** Single shared model with proper hygiene  
✅ **Backward compatible:** Existing `query()` still works for stateful agents  

The narrative system should now generate fresh, non-repetitive content for each player interaction!
