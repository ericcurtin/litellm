package litellm

import (
	"context"
	"fmt"
	"sync"
)

// Client is the main entry point for LiteLLM operations.
// It manages provider instances and routes requests to the appropriate provider.
type Client struct {
	config    Config
	providers map[string]Provider
	mu        sync.RWMutex
}

// NewClient creates a new LiteLLM client with the default configuration.
func NewClient() *Client {
	return &Client{
		config:    DefaultConfig(),
		providers: make(map[string]Provider),
	}
}

// NewClientWithConfig creates a new LiteLLM client with a custom configuration.
func NewClientWithConfig(cfg Config) *Client {
	return &Client{
		config:    cfg,
		providers: make(map[string]Provider),
	}
}

// RegisterProvider registers a provider instance for a given provider name.
func (c *Client) RegisterProvider(name string, provider Provider) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.providers[name] = provider
}

// GetProvider returns the provider for a given name, or nil if not registered.
func (c *Client) GetProvider(name string) Provider {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.providers[name]
}

// Complete sends a chat completion request. The model string is parsed to
// determine the provider (e.g., "openai/gpt-4" or "anthropic/claude-3-sonnet").
func (c *Client) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	provider, model, err := c.resolveProvider(req.Model)
	if err != nil {
		return nil, err
	}

	req.Model = model
	req.APIKey = ResolveAPIKey(req.APIKey, provider.Name(), c.config)

	return provider.Complete(ctx, req)
}

// CompleteStream sends a streaming chat completion request.
func (c *Client) CompleteStream(ctx context.Context, req CompletionRequest) (*Stream, error) {
	provider, model, err := c.resolveProvider(req.Model)
	if err != nil {
		return nil, err
	}

	req.Model = model
	req.APIKey = ResolveAPIKey(req.APIKey, provider.Name(), c.config)
	req.Stream = true

	return provider.CompleteStream(ctx, req)
}

// Embed sends an embedding request.
func (c *Client) Embed(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
	provider, model, err := c.resolveProvider(req.Model)
	if err != nil {
		return nil, err
	}

	req.Model = model
	req.APIKey = ResolveAPIKey(req.APIKey, provider.Name(), c.config)

	return provider.Embed(ctx, req)
}

// Models returns the list of all available models across all providers.
func (c *Client) Models(ctx context.Context) ([]ModelInfo, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var allModels []ModelInfo
	for _, p := range c.providers {
		models, err := p.Models(ctx)
		if err != nil {
			continue // Skip providers that fail
		}
		allModels = append(allModels, models...)
	}
	return allModels, nil
}

// resolveProvider determines which provider to use based on the model string.
func (c *Client) resolveProvider(model string) (Provider, string, error) {
	providerName, modelName := ParseModelProvider(model)

	c.mu.RLock()
	provider, ok := c.providers[providerName]
	c.mu.RUnlock()

	if !ok {
		return nil, "", fmt.Errorf("%w: provider %q not registered (model: %q)", ErrNoProvider, providerName, model)
	}

	return provider, modelName, nil
}
