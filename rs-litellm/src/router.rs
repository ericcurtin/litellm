use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::config::{parse_provider_model, Config, ModelEntry};
use crate::error::AppError;
use crate::providers::{resolve_provider, ByteStream, Provider};
use crate::types::{
    ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse,
    ModelListResponse, ModelObject,
};

/// A single deployment that can serve a model.
struct Deployment {
    entry: ModelEntry,
    provider: Box<dyn Provider>,
}

/// The router manages multiple deployments per model and handles load balancing,
/// retries, and fallbacks.
pub struct Router {
    /// Map from model alias -> list of deployments (for load balancing).
    deployments: HashMap<String, Vec<Deployment>>,
    /// Round-robin counters per model.
    counters: HashMap<String, Arc<AtomicUsize>>,
    /// Global number of retries on failure.
    num_retries: u32,
    /// Fallback chains: model -> list of fallback model names.
    fallbacks: HashMap<String, Vec<String>>,
}

impl Router {
    /// Build a router from the parsed config.
    pub fn from_config(config: &Config) -> Result<Self, AppError> {
        let mut deployments: HashMap<String, Vec<Deployment>> = HashMap::new();

        for entry in &config.model_list {
            let (provider_name, _) = parse_provider_model(&entry.litellm_params.model);
            let provider = resolve_provider(provider_name)?;

            deployments
                .entry(entry.model_name.clone())
                .or_default()
                .push(Deployment {
                    entry: entry.clone(),
                    provider,
                });
        }

        let counters: HashMap<String, Arc<AtomicUsize>> = deployments
            .keys()
            .map(|k| (k.clone(), Arc::new(AtomicUsize::new(0))))
            .collect();

        // Parse fallbacks from config
        let mut fallbacks: HashMap<String, Vec<String>> = HashMap::new();
        if let Some(ref fb_list) = config.general_settings.fallbacks {
            for fb_map in fb_list {
                for (model, targets) in fb_map {
                    fallbacks.insert(model.clone(), targets.clone());
                }
            }
        }

        let num_retries = config
            .general_settings
            .num_retries
            .or(config.litellm_settings.num_retries)
            .unwrap_or(0);

        Ok(Self {
            deployments,
            counters,
            num_retries,
            fallbacks,
        })
    }

    /// Get the list of configured model names.
    pub fn model_names(&self) -> Vec<String> {
        self.deployments.keys().cloned().collect()
    }

    /// List models in OpenAI format.
    pub fn list_models(&self) -> ModelListResponse {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut models: Vec<ModelObject> = self
            .deployments
            .keys()
            .map(|name| ModelObject {
                id: name.clone(),
                object: "model".to_string(),
                created: now,
                owned_by: "rs-litellm".to_string(),
            })
            .collect();
        models.sort_by(|a, b| a.id.cmp(&b.id));

        ModelListResponse {
            object: "list".to_string(),
            data: models,
        }
    }

    /// Pick the next deployment index for the given model using round-robin.
    fn next_index(&self, model: &str) -> Option<usize> {
        let deps = self.deployments.get(model)?;
        if deps.is_empty() {
            return None;
        }
        let counter = self.counters.get(model)?;
        Some(counter.fetch_add(1, Ordering::Relaxed) % deps.len())
    }

    /// Perform a chat completion with load balancing and retries.
    pub async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, AppError> {
        let model = &request.model;
        let max_attempts = (self.num_retries + 1) as usize;
        let mut last_error = None;

        let dep_count = self.deployments.get(model).map_or(0, |v| v.len());

        // Try the primary model
        for attempt in 0..max_attempts.max(dep_count) {
            if let Some(idx) = self.next_index(model) {
                let dep = &self.deployments[model][idx];
                match dep
                    .provider
                    .chat_completion(request, &dep.entry.litellm_params)
                    .await
                {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        tracing::warn!(
                            "Attempt {}/{} for model '{}' failed: {}",
                            attempt + 1,
                            max_attempts,
                            model,
                            e
                        );
                        last_error = Some(e);
                    }
                }
            } else {
                break;
            }
        }

        // Try fallbacks
        if let Some(fallback_models) = self.fallbacks.get(model) {
            for fb_model in fallback_models {
                tracing::info!("Trying fallback model: {}", fb_model);
                if let Some(idx) = self.next_index(fb_model) {
                    let dep = &self.deployments[fb_model][idx];
                    match dep
                        .provider
                        .chat_completion(request, &dep.entry.litellm_params)
                        .await
                    {
                        Ok(result) => return Ok(result),
                        Err(e) => {
                            tracing::warn!("Fallback {} failed: {}", fb_model, e);
                            last_error = Some(e);
                        }
                    }
                }
            }
        }

        if dep_count == 0 && last_error.is_none() {
            return Err(AppError::ModelNotFound(model.to_string()));
        }

        Err(last_error.unwrap_or_else(|| AppError::AllDeploymentsFailed(model.to_string())))
    }

    /// Perform a streaming chat completion with load balancing and retries.
    pub async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ByteStream, AppError> {
        let model = &request.model;
        let max_attempts = (self.num_retries + 1) as usize;
        let mut last_error = None;

        let dep_count = self.deployments.get(model).map_or(0, |v| v.len());

        for attempt in 0..max_attempts.max(dep_count) {
            if let Some(idx) = self.next_index(model) {
                let dep = &self.deployments[model][idx];
                match dep
                    .provider
                    .chat_completion_stream(request, &dep.entry.litellm_params)
                    .await
                {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        tracing::warn!(
                            "Stream attempt {}/{} for model '{}' failed: {}",
                            attempt + 1,
                            max_attempts,
                            model,
                            e
                        );
                        last_error = Some(e);
                    }
                }
            } else {
                break;
            }
        }

        // Try fallbacks
        if let Some(fallback_models) = self.fallbacks.get(model) {
            for fb_model in fallback_models {
                tracing::info!("Trying fallback model: {}", fb_model);
                if let Some(idx) = self.next_index(fb_model) {
                    let dep = &self.deployments[fb_model][idx];
                    match dep
                        .provider
                        .chat_completion_stream(request, &dep.entry.litellm_params)
                        .await
                    {
                        Ok(result) => return Ok(result),
                        Err(e) => {
                            tracing::warn!("Fallback {} failed: {}", fb_model, e);
                            last_error = Some(e);
                        }
                    }
                }
            }
        }

        if dep_count == 0 && last_error.is_none() {
            return Err(AppError::ModelNotFound(model.to_string()));
        }

        Err(last_error.unwrap_or_else(|| AppError::AllDeploymentsFailed(model.to_string())))
    }

    /// Perform an embedding request with load balancing and retries.
    pub async fn embedding(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<EmbeddingResponse, AppError> {
        let model = &request.model;
        let max_attempts = (self.num_retries + 1) as usize;
        let mut last_error = None;

        let dep_count = self.deployments.get(model).map_or(0, |v| v.len());

        for attempt in 0..max_attempts.max(dep_count) {
            if let Some(idx) = self.next_index(model) {
                let dep = &self.deployments[model][idx];
                match dep
                    .provider
                    .embedding(request, &dep.entry.litellm_params)
                    .await
                {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        tracing::warn!(
                            "Embedding attempt {}/{} for model '{}' failed: {}",
                            attempt + 1,
                            max_attempts,
                            model,
                            e
                        );
                        last_error = Some(e);
                    }
                }
            } else {
                break;
            }
        }

        // Try fallbacks
        if let Some(fallback_models) = self.fallbacks.get(model) {
            for fb_model in fallback_models {
                if let Some(idx) = self.next_index(fb_model) {
                    let dep = &self.deployments[fb_model][idx];
                    match dep
                        .provider
                        .embedding(request, &dep.entry.litellm_params)
                        .await
                    {
                        Ok(result) => return Ok(result),
                        Err(e) => {
                            tracing::warn!("Fallback {} failed: {}", fb_model, e);
                            last_error = Some(e);
                        }
                    }
                }
            }
        }

        if dep_count == 0 && last_error.is_none() {
            return Err(AppError::ModelNotFound(model.to_string()));
        }

        Err(last_error.unwrap_or_else(|| AppError::AllDeploymentsFailed(model.to_string())))
    }
}
