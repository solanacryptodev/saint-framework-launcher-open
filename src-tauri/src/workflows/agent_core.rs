use crate::error::LlmError;
use mistralrs::core::{
    DeviceMapSetting, MistralRsBuilder, NormalLoaderBuilder, NormalSpecificConfig, TokenSource,
};
use mistralrs::{DefaultSchedulerMethod, Device, IsqType, ModelDType, SchedulerConfig};
use std::path::PathBuf;
use std::sync::Arc;

pub struct AgentModel {
    pub model: Arc<mistralrs::MistralRs>,
}

impl AgentModel {
    /// model_path should be a DIRECTORY containing native weights (not a .gguf file)
    pub async fn new(model_path: PathBuf) -> Result<Self, LlmError> {
        println!("📂 Loading native weights from: {:?}", model_path);

        // ✅ v0.6.x: NormalLoaderBuilder::new() still requires 6 params
        let loader = NormalLoaderBuilder::new(
            NormalSpecificConfig::default(),                // config
            None,                                           // chat_template (auto-detect)
            None,                                           // tokenizer_json (auto-detect)
            Some(model_path.to_string_lossy().to_string()), // model_id (directory path)
            false,                                          // no_kv_cache
            None,                                           // jinja_explicit
        )
        .build(None)
        .map_err(|e| LlmError::Other(format!("Loader build failed: {}", e)))?;

        // ✅ v0.6.x: Quantization moved to load_model_from_hf() params
        let device = Device::Cpu; // Use Device::cuda_if_available(0) for GPU

        println!("⏳ Loading model weights with ISQ Q4K quantization...");
        let pipeline = loader
            .load_model_from_hf(
                None,                      // revision
                TokenSource::CacheToken,   // HF token (none needed for Phi-4)
                &ModelDType::Auto,         // dtype
                &device,                   // device
                false,                     // silent (false = show progress)
                DeviceMapSetting::dummy(), // device mapping
                Some(IsqType::Q4K),        // ⭐ ISQ quantization HERE (replaces .with_isq())
                None,                      // paged attention config
            )
            .map_err(|e| LlmError::Other(format!("Model load failed: {}", e)))?;

        // Create scheduler config
        let scheduler_config = SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(1.try_into().unwrap()),
        };

        // ✅ v0.6.x: MistralRsBuilder::new() requires 4 params
        let model = MistralRsBuilder::new(
            pipeline,         // pipeline
            scheduler_config, // scheduler config
            false,            // throughput_logging
            None,             // search_embedding_model
        )
        .build()
        .await;

        println!("✅ Model loaded successfully (ISQ Q4K, ~3.1GB RAM for Phi-4-mini)");
        Ok(Self { model })
    }
}

pub struct Agent {
    pub name: String,
    pub system_prompt: String,
    model: Arc<AgentModel>,
}

impl Agent {
    pub fn new(name: &str, system_prompt: &str, model: Arc<AgentModel>) -> Self {
        Self {
            name: name.to_string(),
            system_prompt: system_prompt.to_string(),
            model,
        }
    }

    /// Alias for generate_text - used by workflow system
    pub async fn run(&self, prompt: &str) -> Result<String, LlmError> {
        self.generate_text(prompt).await
    }

    pub async fn generate_text(&self, prompt: &str) -> Result<String, LlmError> {
        println!("🔧 [{}] Starting text generation...", self.name);

        // Clone data needed in the blocking task
        let model = self.model.clone();
        let name = self.name.clone();
        let system_prompt = self.system_prompt.clone();
        let prompt = prompt.to_string();

        // Spawn blocking task for all mistralrs operations
        tokio::task::spawn_blocking(move || {
            // Create a new runtime for this blocking context
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| LlmError::Other(format!("Runtime creation failed: {}", e)))?;

            rt.block_on(async move {
                use indexmap::IndexMap;
                use mistralrs::{MessageContent, RequestMessage, Response};
                use std::time::Instant;

                // Build messages in the format expected by RequestMessage::Chat
                let mut messages = Vec::new();

                // System message
                let mut system_msg = IndexMap::new();
                system_msg.insert(
                    "role".to_string(),
                    MessageContent::Left(system_prompt.clone()),
                );
                system_msg.insert(
                    "content".to_string(),
                    MessageContent::Left(system_prompt.clone()),
                );
                messages.push(system_msg);

                // User message
                let mut user_msg = IndexMap::new();
                user_msg.insert("role".to_string(), MessageContent::Left("user".to_string()));
                user_msg.insert("content".to_string(), MessageContent::Left(prompt.clone()));
                messages.push(user_msg);

                let (tx, mut rx) = tokio::sync::mpsc::channel(10_000);

                let request = mistralrs::Request::Normal(Box::new(mistralrs::NormalRequest {
                    messages: RequestMessage::Chat {
                        messages,
                        enable_thinking: None,
                        reasoning_effort: None,
                    },
                    sampling_params: mistralrs::SamplingParams::deterministic(),
                    response: tx,
                    return_logprobs: false,
                    is_streaming: true,
                    id: 0,
                    constraint: mistralrs::Constraint::None,
                    suffix: None,
                    tools: None,
                    tool_choice: None,
                    logits_processors: None,
                    return_raw_logits: false,
                    web_search_options: None,
                    model_id: None,
                    truncate_sequence: false,
                }));

                model.model.send_request(request).map_err(|e| {
                    eprintln!("❌ [{}] Model send_request failed: {}", name, e);
                    LlmError::Other(e.to_string())
                })?;

                println!("   🔄 Waiting for response...");
                let start = Instant::now();

                let mut full_response = String::new();
                let mut token_count = 0;

                while let Some(response) = rx.recv().await {
                    match response {
                        Response::Chunk(chunk) => {
                            token_count += 1;
                            if token_count == 1 {
                                println!(
                                    "   ⚡ First token after {:.1}s",
                                    start.elapsed().as_secs_f32()
                                );
                            }
                            for choice in chunk.choices {
                                if let Some(content) = choice.delta.content {
                                    full_response.push_str(&content);
                                    // Removed blocking print!() to prevent async runtime blocking
                                }
                            }
                        }
                        Response::Done(_) => break,
                        Response::InternalError(e) => {
                            eprintln!("❌ [{}] Internal error: {}", name, e);
                            return Err(LlmError::Other(e.to_string()));
                        }
                        _ => {}
                    }
                }
                println!();

                if full_response.is_empty() {
                    return Err(LlmError::GenerationFailed {
                        reason: "No content generated".to_string(),
                    });
                }

                println!(
                    "🎉 [{}] Generation complete ({} tokens, {:.1}s)",
                    name,
                    token_count,
                    start.elapsed().as_secs_f32()
                );
                Ok(full_response)
            })
        })
        .await
        .map_err(|e| LlmError::Other(format!("Task join error: {}", e)))?
    }
}
