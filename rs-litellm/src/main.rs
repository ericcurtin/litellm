mod auth;
mod config;
mod error;
mod handlers;
mod providers;
mod router;
mod types;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router as AxumRouter;
use clap::Parser;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

use handlers::AppState;

/// rs-litellm: A high-performance LLM proxy gateway written in Rust.
///
/// Provides an OpenAI-compatible API that routes requests to 30+ LLM providers
/// with load balancing, retries, fallbacks, and authentication.
#[derive(Parser, Debug)]
#[command(name = "rs-litellm", version, about)]
struct Cli {
    /// Path to the YAML configuration file.
    #[arg(short, long)]
    config: PathBuf,

    /// Host address to bind to.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to listen on.
    #[arg(short, long, default_value_t = 4000)]
    port: u16,

    /// Enable verbose logging (sets RUST_LOG=debug).
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Initialize logging
    let log_filter = if cli.verbose {
        "rs_litellm=debug,tower_http=debug"
    } else {
        "rs_litellm=info,tower_http=info"
    };

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_filter)),
        )
        .init();

    // Load configuration
    tracing::info!("Loading config from: {}", cli.config.display());
    let cfg = match config::Config::load(&cli.config) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Failed to load config: {}", e);
            std::process::exit(1);
        }
    };

    let model_count = cfg.model_list.len();
    let model_names: Vec<&str> = cfg.model_list.iter().map(|m| m.model_name.as_str()).collect();
    tracing::info!("Loaded {} model deployment(s): {:?}", model_count, model_names);

    let master_key = cfg.general_settings.master_key.clone();
    if master_key.is_some() {
        tracing::info!("Authentication enabled (master_key is set)");
    } else {
        tracing::warn!("No master_key configured — API is open to all requests");
    }

    // Build router
    let llm_router = match router::Router::from_config(&cfg) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Failed to build router: {}", e);
            std::process::exit(1);
        }
    };

    let state = Arc::new(AppState {
        router: llm_router,
        master_key: master_key.clone(),
    });

    // Build HTTP server
    let app = build_app(state);

    let addr: SocketAddr = format!("{}:{}", cli.host, cli.port)
        .parse()
        .expect("Invalid host:port");

    tracing::info!("🚀 rs-litellm v{} listening on {}", env!("CARGO_PKG_VERSION"), addr);
    tracing::info!("   OpenAI-compatible API: http://{}/v1/chat/completions", addr);
    tracing::info!("   Models endpoint:       http://{}/v1/models", addr);
    tracing::info!("   Health check:          http://{}/health", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

fn build_app(state: Arc<AppState>) -> AxumRouter {
    let master_key = state.master_key.clone();

    // Auth middleware that injects the master key into request extensions
    let auth_layer = axum::middleware::from_fn(move |mut request: axum::extract::Request, next: axum::middleware::Next| {
        let key = master_key.clone();
        async move {
            request.extensions_mut().insert(key);
            auth::auth_middleware(request, next).await
        }
    });

    // Routes that require authentication
    let protected_routes = AxumRouter::new()
        .route(
            "/v1/chat/completions",
            post(handlers::chat_completion_handler),
        )
        .route(
            "/chat/completions",
            post(handlers::chat_completion_handler),
        )
        .route("/v1/embeddings", post(handlers::embeddings_handler))
        .route("/embeddings", post(handlers::embeddings_handler))
        .route("/v1/models", get(handlers::list_models_handler))
        .route("/models", get(handlers::list_models_handler))
        .layer(auth_layer);

    // Routes that do NOT require authentication
    let public_routes = AxumRouter::new()
        .route("/health", get(handlers::health_handler))
        .route("/", get(handlers::health_handler));

    public_routes
        .merge(protected_routes)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
