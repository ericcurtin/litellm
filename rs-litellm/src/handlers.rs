use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Json, Response};
use tokio_stream::StreamExt;

use crate::error::AppError;
use crate::router::Router;
use crate::types::{
    ChatCompletionRequest, EmbeddingRequest, HealthResponse,
};

/// Shared application state.
pub struct AppState {
    pub router: Router,
    pub master_key: Option<String>,
}

// ── Health check ──

pub async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        models: state.router.model_names(),
    })
}

// ── Model listing ──

pub async fn list_models_handler(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    Json(state.router.list_models())
}

// ── Chat completions ──

pub async fn chat_completion_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, AppError> {
    let is_stream = request.stream.unwrap_or(false);

    if is_stream {
        // Streaming response
        let byte_stream = state.router.chat_completion_stream(&request).await?;

        let body_stream = tokio_stream::wrappers::UnboundedReceiverStream::new({
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();
            tokio::spawn(async move {
                let mut stream = std::pin::pin!(byte_stream);
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(bytes) => {
                            if tx.send(Ok(bytes)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let err_msg = format!("data: {{\"error\": \"{}\"}}\n\n", e);
                            let _ = tx.send(Ok(bytes::Bytes::from(err_msg)));
                            break;
                        }
                    }
                }
            });
            rx
        });

        let body = Body::from_stream(body_stream);

        Ok(Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header(header::CONNECTION, "keep-alive")
            .header("x-litellm-version", env!("CARGO_PKG_VERSION"))
            .body(body)
            .unwrap())
    } else {
        // Non-streaming response
        let response = state.router.chat_completion(&request).await?;

        let mut resp = Json(response).into_response();
        resp.headers_mut().insert(
            "x-litellm-version",
            env!("CARGO_PKG_VERSION").parse().unwrap(),
        );
        Ok(resp)
    }
}

// ── Embeddings ──

pub async fn embeddings_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<impl IntoResponse, AppError> {
    let response = state.router.embedding(&request).await?;
    Ok(Json(response))
}
