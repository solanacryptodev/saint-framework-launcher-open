# ONNX Model Specifications

## Gemma 270M Model

**Model File:** `models/gemma-270M/model.onnx`

### Input Tensors

| Index | Name | Type | Shape | Description |
|-------|------|------|-------|-------------|
| 0 | `input_ids` | Int64 | `[batch_size, sequence_length]` | Token IDs for the current input |
| 1 | `attention_mask` | Int64 | `[batch_size, total_sequence_length]` | Attention mask covering all tokens |
| 2 | `position_ids` | Int64 | `[batch_size, sequence_length]` | Position IDs for the current input |
| 3-38 | `past_key_values.{0-17}.{key\|value}` | Float32 | `[batch_size, 1, past_sequence_length, 256]` | Past KV cache (36 tensors total) |

### Output Tensors

| Index | Name | Type | Shape | Description |
|-------|------|------|-------|-------------|
| 0 | `logits` | Float32 | `[batch_size, sequence_length, 262144]` | Logits for next token prediction |
| 1-36 | `present.{0-17}.{key\|value}` | Float32 | `[batch_size, 1, total_sequence_length, 256]` | Present KV cache (36 tensors total) |

### Model Architecture Details

- **Transformer Layers:** 18
- **KV Heads:** 1 (per layer)
- **Head Dimension:** 256
- **Vocabulary Size:** 262,144
- **KV Cache Data Type:** Float32
- **Logits Data Type:** Float32

### KV Cache Structure

**Past KV Cache Inputs (for incremental generation):**
```
past_key_values.0.key, past_key_values.0.value
past_key_values.1.key, past_key_values.1.value
...
past_key_values.17.key, past_key_values.17.value
```

**Present KV Cache Outputs (to be fed back as past):**
```
present.0.key, present.0.value
present.1.key, present.1.value
...
present.17.key, present.17.value
```

### Implementation Notes

1. **First Pass (Prompt Processing):**
   - Input: `input_ids`, `attention_mask`, `position_ids`
   - Past KV caches can be optional or empty (all zeros with shape `[1, 1, 0, 256]`)
   - Output: `logits` + all 36 `present` KV cache tensors

2. **Subsequent Passes (Token-by-Token Generation):**
   - Input: Single new token in `input_ids` (shape: `[1, 1]`)
   - Input: Updated `attention_mask` (length = past_length + 1)
   - Input: Updated `position_ids` for the new token
   - Input: All 36 `past_key_values` from previous pass's `present` outputs
   - Output: `logits` for next token + updated `present` KV caches

3. **Dimension Symbols:**
   - `batch_size`: Typically 1 for single-sequence generation
   - `sequence_length`: Number of new tokens being processed (typically 1 after prompt)
   - `past_sequence_length`: Length of previously cached sequence
   - `total_sequence_length`: `past_sequence_length + sequence_length`

4. **Memory Requirements (per layer):**
   - Key cache: `[batch, 1, seq_len, 256]` × Float32 = `seq_len × 1024` bytes
   - Value cache: `[batch, 1, seq_len, 256]` × Float32 = `seq_len × 1024` bytes
   - Total per layer: `seq_len × 2048` bytes
   - Total for 18 layers: `seq_len × 36,864` bytes
   - Example: 2048 token context = ~73.7 MB

### Next Steps for KV Cache Implementation

1. **Create `config.rs`:**
   - Define model configuration struct with these parameters
   - Hard dimension: 18 layers, 1 KV head per layer, 256 head dimension

2. **Create `kv_cache.rs`:**
   - Implement KV cache storage for all 18 layers
   - Support initial allocation and incremental updates
   - Provide methods to get past caches and store present caches

3. **Update `agent_core.rs`:**
   - Integrate KV cache system into generation loop
   - First pass: process full prompt, initialize caches
   - Subsequent passes: use cached KV values, process one token at a time

### CLI Command to run Model Inspection
```
cargo run --bin inspect_model
```
