package litellm

import (
	"context"
)

// Provider defines the interface that all LLM providers must implement.
type Provider interface {
	// Name returns the provider identifier.
	Name() string

	// Complete sends a chat completion request and returns the response.
	Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error)

	// CompleteStream sends a streaming chat completion request.
	// The returned channel receives chunks until the stream ends or an error occurs.
	CompleteStream(ctx context.Context, req CompletionRequest) (*Stream, error)

	// Embed sends an embedding request and returns the response.
	Embed(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error)

	// Models returns the list of available models for this provider.
	Models(ctx context.Context) ([]ModelInfo, error)
}
