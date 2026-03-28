// Package litellm provides a unified interface for multiple LLM providers.
//
// It translates inputs to provider-specific endpoints while providing
// consistent OpenAI-format output across all providers. Features include:
//   - Chat completions with streaming support
//   - Embeddings
//   - Router with load balancing, fallback, and retry logic
//   - OpenAI-compatible proxy server
//
// Basic usage:
//
//	client := litellm.NewClient()
//	resp, err := client.Complete(ctx, litellm.CompletionRequest{
//	    Model:    "openai/gpt-4",
//	    Messages: []litellm.Message{{Role: "user", Content: litellm.StringContent("Hello")}},
//	})
package litellm
