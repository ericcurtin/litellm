use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use http::StatusCode;

use crate::error::AppError;

/// Extract and validate the API key from the Authorization header.
/// If no master_key is configured, all requests are allowed.
pub async fn auth_middleware(
    request: Request,
    next: Next,
) -> Result<Response, AppError> {
    // Retrieve the master key from request extensions (set during app state setup)
    let master_key = request
        .extensions()
        .get::<Option<String>>()
        .cloned()
        .flatten();

    if let Some(ref expected_key) = master_key {
        let auth_header = request
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok());

        let provided_key = match auth_header {
            Some(header) => {
                if let Some(token) = header.strip_prefix("Bearer ") {
                    token.to_string()
                } else {
                    header.to_string()
                }
            }
            None => {
                // Also check x-api-key header
                request
                    .headers()
                    .get("x-api-key")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("")
                    .to_string()
            }
        };

        if provided_key != *expected_key {
            return Err(AppError::ProviderError {
                status: StatusCode::UNAUTHORIZED,
                message: "Invalid API key".to_string(),
            });
        }
    }

    Ok(next.run(request).await)
}
