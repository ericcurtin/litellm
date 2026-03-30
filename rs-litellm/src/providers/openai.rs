use axum::http::StatusCode;
use futures::StreamExt;
use reqwest::Client;

use crate::config::LitellmParams;
use crate::error::AppError;
use crate::types::{
    ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse,
};

use super::{build_api_url, parse_upstream_error, resolve_api_key, ByteStream, Provider};

/// OpenAI-compatible provider.
/// Works for OpenAI and any provider that uses the same API format
/// (Together AI, Groq, Mistral, DeepSeek, Fireworks, OpenRouter, vLLM, Ollama, etc.)
pub struct OpenAIProvider {
    client: Client,
}

impl OpenAIProvider {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }

    fn provider_name_for_model<'a>(&self, params: &'a LitellmParams) -> &'a str {
        let (provider, _) = crate::config::parse_provider_model(&params.model);
        provider
    }
}

#[async_trait::async_trait]
impl Provider for OpenAIProvider {
    async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
        params: &LitellmParams,
    ) -> Result<ChatCompletionResponse, AppError> {
        let provider_name = self.provider_name_for_model(params);
        let url = build_api_url(provider_name, params, "/chat/completions");

        // Build the request body — use the actual model name (not the alias)
        let (_, actual_model) = crate::config::parse_provider_model(&params.model);
        let mut body = serde_json::to_value(request).map_err(|e| {
            AppError::Internal(format!("Failed to serialize request: {}", e))
        })?;
        body["model"] = serde_json::Value::String(actual_model.to_string());
        body["stream"] = serde_json::Value::Bool(false);

        let mut req = self.client.post(&url).json(&body);

        // Set auth header
        if let Some(api_key) = resolve_api_key(provider_name, params) {
            req = req.bearer_auth(&api_key);
        }

        // Azure-specific query parameters
        if provider_name == "azure" || provider_name == "azure_ai" {
            if let Some(ref version) = params.api_version {
                req = req.query(&[("api-version", version.as_str())]);
            }
        }

        let timeout_secs = params.timeout.unwrap_or(600.0);
        req = req.timeout(std::time::Duration::from_secs_f64(timeout_secs));

        let response = req.send().await.map_err(|e| {
            AppError::ProviderError {
                status: StatusCode::BAD_GATEWAY,
                message: format!("Failed to reach provider: {}", e),
            }
        })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(parse_upstream_error(
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
                &body,
            ));
        }

        let resp: ChatCompletionResponse = response.json().await.map_err(|e| {
            AppError::Internal(format!("Failed to parse provider response: {}", e))
        })?;

        Ok(resp)
    }

    async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
        params: &LitellmParams,
    ) -> Result<ByteStream, AppError> {
        let provider_name = self.provider_name_for_model(params);
        let url = build_api_url(provider_name, params, "/chat/completions");

        let (_, actual_model) = crate::config::parse_provider_model(&params.model);
        let mut body = serde_json::to_value(request).map_err(|e| {
            AppError::Internal(format!("Failed to serialize request: {}", e))
        })?;
        body["model"] = serde_json::Value::String(actual_model.to_string());
        body["stream"] = serde_json::Value::Bool(true);

        let mut req = self.client.post(&url).json(&body);

        if let Some(api_key) = resolve_api_key(provider_name, params) {
            req = req.bearer_auth(&api_key);
        }

        if provider_name == "azure" || provider_name == "azure_ai" {
            if let Some(ref version) = params.api_version {
                req = req.query(&[("api-version", version.as_str())]);
            }
        }

        let timeout_secs = params.timeout.unwrap_or(600.0);
        req = req.timeout(std::time::Duration::from_secs_f64(timeout_secs));

        let response = req.send().await.map_err(|e| {
            AppError::ProviderError {
                status: StatusCode::BAD_GATEWAY,
                message: format!("Failed to reach provider: {}", e),
            }
        })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(parse_upstream_error(
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
                &body,
            ));
        }

        // Stream the SSE response directly from the provider
        let byte_stream = response.bytes_stream();
        let stream = byte_stream.map(|chunk| {
            chunk.map_err(|e| {
                AppError::Internal(format!("Stream error: {}", e))
            })
        });

        Ok(Box::pin(stream))
    }

    async fn embedding(
        &self,
        request: &EmbeddingRequest,
        params: &LitellmParams,
    ) -> Result<EmbeddingResponse, AppError> {
        let provider_name = self.provider_name_for_model(params);
        let url = build_api_url(provider_name, params, "/embeddings");

        let (_, actual_model) = crate::config::parse_provider_model(&params.model);
        let mut body = serde_json::to_value(request).map_err(|e| {
            AppError::Internal(format!("Failed to serialize request: {}", e))
        })?;
        body["model"] = serde_json::Value::String(actual_model.to_string());

        let mut req = self.client.post(&url).json(&body);

        if let Some(api_key) = resolve_api_key(provider_name, params) {
            req = req.bearer_auth(&api_key);
        }

        let timeout_secs = params.timeout.unwrap_or(600.0);
        req = req.timeout(std::time::Duration::from_secs_f64(timeout_secs));

        let response = req.send().await.map_err(|e| {
            AppError::ProviderError {
                status: StatusCode::BAD_GATEWAY,
                message: format!("Failed to reach provider: {}", e),
            }
        })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(parse_upstream_error(
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
                &body,
            ));
        }

        let resp: EmbeddingResponse = response.json().await.map_err(|e| {
            AppError::Internal(format!("Failed to parse provider response: {}", e))
        })?;

        Ok(resp)
    }
}
