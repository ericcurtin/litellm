package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	litellm "github.com/ericcurtin/litellm/go-litellm"
)

// AzureProvider implements the litellm.Provider interface for Azure OpenAI.
type AzureProvider struct {
	apiKey     string
	baseURL    string // e.g., https://your-resource.openai.azure.com
	apiVersion string
	httpClient *http.Client
}

// NewAzureProvider creates a new Azure OpenAI provider.
func NewAzureProvider(apiKey, baseURL, apiVersion string) *AzureProvider {
	if apiVersion == "" {
		apiVersion = "2024-02-15-preview"
	}
	return &AzureProvider{
		apiKey:     apiKey,
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiVersion: apiVersion,
		httpClient: &http.Client{
			Timeout: 5 * time.Minute,
		},
	}
}

func (p *AzureProvider) Name() string { return "azure" }

func (p *AzureProvider) Complete(ctx context.Context, req litellm.CompletionRequest) (*litellm.CompletionResponse, error) {
	apiKey := p.resolveAPIKey(req.APIKey)
	if apiKey == "" {
		return nil, &litellm.APIError{StatusCode: 401, Message: "Azure API key is required", Provider: "azure"}
	}

	baseURL := p.resolveBaseURL(req.BaseURL)
	if baseURL == "" {
		return nil, &litellm.APIError{StatusCode: 400, Message: "Azure API base URL is required", Provider: "azure"}
	}

	body := buildOpenAIRequestBody(req)
	body["stream"] = false

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := p.buildURL(baseURL, req.Model, "chat/completions")
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("api-key", apiKey)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, parseOpenAIError(respBody, resp.StatusCode, "azure")
	}

	var result litellm.CompletionResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &result, nil
}

func (p *AzureProvider) CompleteStream(ctx context.Context, req litellm.CompletionRequest) (*litellm.Stream, error) {
	apiKey := p.resolveAPIKey(req.APIKey)
	if apiKey == "" {
		return nil, &litellm.APIError{StatusCode: 401, Message: "Azure API key is required", Provider: "azure"}
	}

	baseURL := p.resolveBaseURL(req.BaseURL)
	if baseURL == "" {
		return nil, &litellm.APIError{StatusCode: 400, Message: "Azure API base URL is required", Provider: "azure"}
	}

	body := buildOpenAIRequestBody(req)
	body["stream"] = true

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := p.buildURL(baseURL, req.Model, "chat/completions")
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("api-key", apiKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	return litellm.DoStreamRequest(ctx, p.httpClient, httpReq, req.Model)
}

func (p *AzureProvider) Embed(ctx context.Context, req litellm.EmbeddingRequest) (*litellm.EmbeddingResponse, error) {
	apiKey := p.resolveAPIKey(req.APIKey)
	if apiKey == "" {
		return nil, &litellm.APIError{StatusCode: 401, Message: "Azure API key is required", Provider: "azure"}
	}

	baseURL := p.resolveBaseURL(req.BaseURL)
	if baseURL == "" {
		return nil, &litellm.APIError{StatusCode: 400, Message: "Azure API base URL is required", Provider: "azure"}
	}

	body := map[string]interface{}{
		"input": req.Input,
	}
	if req.EncodingFormat != "" {
		body["encoding_format"] = req.EncodingFormat
	}
	if req.Dimensions != nil {
		body["dimensions"] = *req.Dimensions
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := p.buildURL(baseURL, req.Model, "embeddings")
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("api-key", apiKey)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, parseOpenAIError(respBody, resp.StatusCode, "azure")
	}

	var result litellm.EmbeddingResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &result, nil
}

func (p *AzureProvider) Models(_ context.Context) ([]litellm.ModelInfo, error) {
	return []litellm.ModelInfo{
		{ID: "gpt-4", Object: "model", OwnedBy: "azure"},
		{ID: "gpt-4-turbo", Object: "model", OwnedBy: "azure"},
		{ID: "gpt-35-turbo", Object: "model", OwnedBy: "azure"},
		{ID: "text-embedding-ada-002", Object: "model", OwnedBy: "azure"},
		{ID: "text-embedding-3-small", Object: "model", OwnedBy: "azure"},
		{ID: "text-embedding-3-large", Object: "model", OwnedBy: "azure"},
	}, nil
}

func (p *AzureProvider) buildURL(baseURL, deploymentName, endpoint string) string {
	return fmt.Sprintf("%s/openai/deployments/%s/%s?api-version=%s",
		baseURL, deploymentName, endpoint, p.apiVersion)
}

func (p *AzureProvider) resolveAPIKey(explicit string) string {
	if explicit != "" {
		return explicit
	}
	return p.apiKey
}

func (p *AzureProvider) resolveBaseURL(explicit string) string {
	if explicit != "" {
		return explicit
	}
	return p.baseURL
}

// buildOpenAIRequestBody creates the request body shared between OpenAI and Azure.
func buildOpenAIRequestBody(req litellm.CompletionRequest) map[string]interface{} {
	body := map[string]interface{}{
		"messages": req.Messages,
	}
	// Note: Azure doesn't use "model" in body, it's in the URL path

	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}
	if req.N != nil {
		body["n"] = *req.N
	}
	if req.MaxTokens != nil {
		body["max_tokens"] = *req.MaxTokens
	}
	if req.PresencePenalty != nil {
		body["presence_penalty"] = *req.PresencePenalty
	}
	if req.FrequencyPenalty != nil {
		body["frequency_penalty"] = *req.FrequencyPenalty
	}
	if len(req.Stop) > 0 {
		body["stop"] = req.Stop
	}
	if len(req.Tools) > 0 {
		body["tools"] = req.Tools
	}
	if req.ToolChoice != nil {
		body["tool_choice"] = req.ToolChoice
	}
	if req.ResponseFormat != nil {
		body["response_format"] = req.ResponseFormat
	}
	if req.Seed != nil {
		body["seed"] = *req.Seed
	}
	if req.User != "" {
		body["user"] = req.User
	}
	if len(req.LogitBias) > 0 {
		body["logit_bias"] = req.LogitBias
	}

	return body
}
