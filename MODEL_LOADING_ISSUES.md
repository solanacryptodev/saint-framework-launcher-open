# Model Loading Issues - Troubleshooting Guide

## Issue: Missing model.onnx_data

### Problem
The error "Failed to load model: ...model.onnx_data" indicates that your ONNX model has **external data** that is stored separately from the main model file.

### What's Happening
When ONNX models are exported, large weight tensors can be stored in an external file (`model.onnx_data`) to keep the main `.onnx` file smaller. Your `model.onnx` file contains references to this external data file.

**Current directory structure:**
```
models/qwen3-0.6B/
├── model.onnx          ✅ (present)
├── tokenizer.json      ✅ (present)
└── model.onnx_data     ❌ (MISSING - this is the problem!)
```

### Solution Options

#### Option 1: Download the External Data File (Recommended)
If you downloaded the model from Hugging Face, the `model.onnx_data` file should be available:

1. Go to the model page on Hugging Face
2. Look for `model.onnx_data` in the files list
3. Download it to `src-tauri/models/qwen3-0.6B/`

**Expected final structure:**
```
models/qwen3-0.6B/
├── model.onnx
├── model.onnx_data     ← Add this file
└── tokenizer.json
```

#### Option 2: Use a Different Model File
Some model repos provide multiple versions:
- `model.onnx` (with external data)
- `model_quantized.onnx` (might be self-contained)
- Look for files without external data requirements

#### Option 3: Re-export the Model Without External Data
If you have access to the original model, re-export it with all data embedded:

```python
import onnx
from onnx import external_data_helper

# Load model with external data
model = onnx.load("model.onnx", load_external_data=True)

# Save with all data embedded
onnx.save(
    model,
    "model_embedded.onnx",
    save_as_external_data=False  # Embed all data
)
```

Then use `model_embedded.onnx` instead.

### Verifying the Fix

After adding the external data file, you should see:

```
📂 Model directory: "models/qwen3-0.6B"
📂 Model path: "models/qwen3-0.6B/model.onnx"
📂 Model exists: true
✅ Found external data file: model.onnx_data
📦 Loading ONNX model from models/qwen3-0.6B/...
✅ ONNX model loaded successfully
```

### Alternative: Use a Self-Contained Model

If you can't find the external data file, consider using a smaller model that's fully embedded:
- Qwen2.5-0.5B-Instruct
- Phi-2
- TinyLlama

These smaller models often have all weights embedded in the `.onnx` file.

## Quick Check Commands

Run the app with `npm run tauri dev` and look for these log messages:

✅ **Success:**
```
✅ Found external data file: model.onnx_data
✅ ONNX model loaded successfully
```

❌ **Problem:**
```
⚠️  Warning: model.onnx_data not found
❌ Failed to load model: [error about external data]
```

## Need Help?

If you're still stuck:
1. Check the exact error message in the console
2. Verify all files are in `src-tauri/models/qwen3-0.6B/`
3. Try downloading a fresh copy of the model files
4. Consider using a different model variant
