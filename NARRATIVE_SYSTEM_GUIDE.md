## SAINT Framework: ACE & Narrative System Architecture

### Overview

SAINT (Solomonic Autonomous Intelligent Narrative Technology) is a local-first narrative engine that generates coherent emergent storytelling through swarm intelligence rather than central authorship. The system creates living worlds where player actions ripple through 10,000+ agents with persistent memory, emotional continuity, and self-healing narrative coherence—all running 100% locally with zero cloud dependencies.

Core Innovation: Intelligence emerges from structured context (ACE engine) rather than model size. A small/small-ish language model (Phi-4 Mini, etc) becomes narratively sophisticated when scaffolded by gravitational physics, concept propagation, and swarm dynamics.

### Architecture

#### Dual Graph Foundation

| Graph    | Purpose | Content | Persistence |
| -------- | ------- | ------- | ------- |
| World Graph  | Current narrative state    | Agents, locations, relationships, emotional,charges, gravitational signatures    | Volatile (updated frequently)
| Lore Graph | Immutable narrative history    | Events, concept propagation chains, causal relationships, phase progression    | Append-only (never rewritten)     

Critical Separation: World Graph = where agents are now. Lore Graph = what has happened. This prevents narrative paradoxes while enabling rich historical context for AI generation.

### ACE Agent Roles

| Role    | Responsibility | Access Pattern | Example Agents |
| -------- | ------- | ------- | ------- |
| Reflectors  | Quantify event significance → update World Graph    | World Graph: R/W Lore Graph: R | The Gravity Judge
| Curators | Maintain narrative coherence → update Lore Graph | World Graph: R Lore Graph: R/W | The Lore Keeper, The Liability Adjuster
| Generators | Transform context into prose/dialogue  | World Graph: R Lore Graph: R | The Griot, The Prose Stylist

Architectural Guardrail: Generators never modify world state directly. They propose text → Curators validate → Reflectors apply physics-based consequences. This prevents AI hallucinations from corrupting narrative integrity.

### ANIS: Artificial Narrative Intelligence Swarm

Hybrid swarm intelligence implementing three algorithms for emergent narrative behavior:

| Algorithm | Narrative Function | SAINT Implementation |
| -------- | ------- | ------- |
| AFSA (Artificial Fish Swarm Algorithm) | Bounded awareness "Agents only perceive narratively relevant events" | Visual scope radius = relationship hops × emotional resonance |
| ABC (Artificial Bee Colony) | Role specialization "Not all agents need equal agency" | Scouts (explore concepts) → Onlookers (spread gossip) → Experienced (maintain coherence) |
| GWO (Gray Wolf Optimization) | Leadership hierarchy "Factions self-organize without god-mode GM" | Alpha (narrative_weight) + Beta (moral_polarity) + Delta (method_intensity) |

### Features Implemented

✅ Narrative Physics Engine

1. Gravitational mass propagation ([trauma, hope, mystery])
2. Concept diffusion through relationship networks
3. Phase-locked emergence (Hero's Journey progression gates)

✅ Coherence Immune System

1. Automatic contradiction detection against Lore Graph
2. Plausibility membranes for retroactive continuity
3. Emotional continuity enforcement across agent lifetimes

✅ True Emergence at Scale

1. 10,000+ agents with persistent state
2. Tiered awareness preventing simulation overload
3. Faction formation via swarm dynamics (no scripting)

✅ Narrative Sovereignty

1. 100% local execution (no cloud APIs)
2. Zero IP claims on user-created worlds
3. World seeds fully portable (no platform lock-in)

✅ Blackboard Coordination

1. Ephemeral per-generation coordination surface
2. Sequential agent dependencies without tight coupling
3. Full observability for narrative debugging
