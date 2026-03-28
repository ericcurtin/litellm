package litellm

import (
	"errors"
	"fmt"
	"net/http"
)

// Standard error types for LiteLLM operations.
var (
	ErrNoProvider       = errors.New("litellm: no provider found for model")
	ErrNoDeployments    = errors.New("litellm: no healthy deployments available")
	ErrAllRetriesFailed = errors.New("litellm: all retries exhausted")
	ErrStreamClosed     = errors.New("litellm: stream closed")
	ErrInvalidRequest   = errors.New("litellm: invalid request")
)

// APIError represents an error returned by an LLM provider.
type APIError struct {
	StatusCode int    `json:"status_code"`
	Message    string `json:"message"`
	Type       string `json:"type,omitempty"`
	Provider   string `json:"provider,omitempty"`
	Model      string `json:"model,omitempty"`
}

func (e *APIError) Error() string {
	if e.Provider != "" {
		return fmt.Sprintf("litellm.%s: %s (status %d)", e.Provider, e.Message, e.StatusCode)
	}
	return fmt.Sprintf("litellm: %s (status %d)", e.Message, e.StatusCode)
}

// IsRetryable returns true if the error should be retried.
func (e *APIError) IsRetryable() bool {
	switch e.StatusCode {
	case http.StatusTooManyRequests,
		http.StatusInternalServerError,
		http.StatusBadGateway,
		http.StatusServiceUnavailable,
		http.StatusGatewayTimeout:
		return true
	default:
		return false
	}
}

// IsContextWindowExceeded returns true if the error is due to context length.
func (e *APIError) IsContextWindowExceeded() bool {
	return e.StatusCode == http.StatusBadRequest && e.Type == "context_length_exceeded"
}

// IsContentPolicyViolation returns true if the error is a content policy violation.
func (e *APIError) IsContentPolicyViolation() bool {
	return e.StatusCode == http.StatusBadRequest && e.Type == "content_policy_violation"
}

// IsAuthError returns true if the error is an authentication error.
func (e *APIError) IsAuthError() bool {
	return e.StatusCode == http.StatusUnauthorized || e.StatusCode == http.StatusForbidden
}
