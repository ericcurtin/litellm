package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	litellm "github.com/ericcurtin/litellm/go-litellm/internal/litellm"
)

const openAIDefaultBase = "https://api.openai.com/v1"

// OpenAIProvider implements the litellm.Provider interface for OpenAI.
type OpenAIProvider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// NewOpenAIProvider creates a new OpenAI provider.
func NewOpenAIProvider(apiKey, baseURL string) *OpenAIProvider {
	if baseURL == "" {
		baseURL = openAIDefaultBase
	}
	return &OpenAIProvider{
		apiKey:  apiKey,
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 5 * time.Minute,
		},
	}
}

func (p *OpenAIProvider) Name() string { return "openai" }

func (p *OpenAIProvider) Complete(ctx context.Context, req litellm.CompletionRequest) (*litellm.CompletionResponse, error) {
	apiKey := p.resolveAPIKey(req.APIKey)
	if apiKey == "" {
		return nil, &litellm.APIError{StatusCode: 401, Message: "OpenAI API key is required", Provider: "openai"}
	}

	body := p.buildRequestBody(req)
	body["stream"] = false

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.resolveBaseURL(req.BaseURL)+"/chat/completions", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)

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
		return nil, parseOpenAIError(respBody, resp.StatusCode, "openai")
	}

	var result litellm.CompletionResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &result, nil
}

func (p *OpenAIProvider) CompleteStream(ctx context.Context, req litellm.CompletionRequest) (*litellm.Stream, error) {
	apiKey := p.resolveAPIKey(req.APIKey)
	if apiKey == "" {
		return nil, &litellm.APIError{StatusCode: 401, Message: "OpenAI API key is required", Provider: "openai"}
	}

	body := p.buildRequestBody(req)
	body["stream"] = true

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.resolveBaseURL(req.BaseURL)+"/chat/completions", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	return litellm.DoStreamRequest(ctx, p.httpClient, httpReq, req.Model)
}

func (p *OpenAIProvider) Embed(ctx context.Context, req litellm.EmbeddingRequest) (*litellm.EmbeddingResponse, error) {
	apiKey := p.resolveAPIKey(req.APIKey)
	if apiKey == "" {
		return nil, &litellm.APIError{StatusCode: 401, Message: "OpenAI API key is required", Provider: "openai"}
	}

	body := map[string]interface{}{
		"model": req.Model,
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

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.resolveBaseURL(req.BaseURL)+"/embeddings", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)

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
		return nil, parseOpenAIError(respBody, resp.StatusCode, "openai")
	}

	var result litellm.EmbeddingResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &result, nil
}

func (p *OpenAIProvider) Models(ctx context.Context) ([]litellm.ModelInfo, error) {
	apiKey := p.apiKey
	if apiKey == "" {
		return nil, &litellm.APIError{StatusCode: 401, Message: "OpenAI API key is required", Provider: "openai"}
	}

	httpReq, err := http.NewRequestWithContext(ctx, "GET", p.baseURL+"/models", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Authorization", "Bearer "+apiKey)

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
		return nil, parseOpenAIError(respBody, resp.StatusCode, "openai")
	}

	var result litellm.ModelListResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return result.Data, nil
}

func (p *OpenAIProvider) resolveAPIKey(explicit string) string {
	if explicit != "" {
		return explicit
	}
	return p.apiKey
}

func (p *OpenAIProvider) resolveBaseURL(explicit string) string {
	if explicit != "" {
		return explicit
	}
	return p.baseURL
}

func (p *OpenAIProvider) buildRequestBody(req litellm.CompletionRequest) map[string]interface{} {
	body := map[string]interface{}{
		"model":    req.Model,
		"messages": req.Messages,
	}

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

// parseOpenAIError parses an error response from the OpenAI API.
func parseOpenAIError(body []byte, statusCode int, provider string) error {
	var errResp struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error"`
	}

	if err := json.Unmarshal(body, &errResp); err != nil {
		return &litellm.APIError{
			StatusCode: statusCode,
			Message:    string(body),
			Provider:   provider,
		}
	}

	return &litellm.APIError{
		StatusCode: statusCode,
		Message:    errResp.Error.Message,
		Type:       errResp.Error.Type,
		Provider:   provider,
	}
}
