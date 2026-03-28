package providers

import (
	"bufio"
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

const anthropicDefaultBase = "https://api.anthropic.com"
const anthropicAPIVersion = "2023-06-01"

// AnthropicProvider implements the litellm.Provider interface for Anthropic.
type AnthropicProvider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// NewAnthropicProvider creates a new Anthropic provider.
func NewAnthropicProvider(apiKey, baseURL string) *AnthropicProvider {
	if baseURL == "" {
		baseURL = anthropicDefaultBase
	}
	return &AnthropicProvider{
		apiKey:  apiKey,
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 5 * time.Minute,
		},
	}
}

func (p *AnthropicProvider) Name() string { return "anthropic" }

// Anthropic-specific request/response types for API transformation.
type anthropicRequest struct {
	Model         string                 `json:"model"`
	Messages      []anthropicMessage     `json:"messages"`
	System        string                 `json:"system,omitempty"`
	MaxTokens     int                    `json:"max_tokens"`
	Temperature   *float64               `json:"temperature,omitempty"`
	TopP          *float64               `json:"top_p,omitempty"`
	StopSequences []string               `json:"stop_sequences,omitempty"`
	Stream        bool                   `json:"stream,omitempty"`
	Tools         []anthropicTool        `json:"tools,omitempty"`
	ToolChoice    *anthropicToolChoice   `json:"tool_choice,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

type anthropicMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // string or []anthropicContentBlock
}

type anthropicContentBlock struct {
	Type      string `json:"type"`
	Text      string `json:"text,omitempty"`
	ID        string `json:"id,omitempty"`
	Name      string `json:"name,omitempty"`
	Input     interface{} `json:"input,omitempty"`
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   string `json:"content,omitempty"`
}

type anthropicTool struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	InputSchema interface{} `json:"input_schema"`
}

type anthropicToolChoice struct {
	Type string `json:"type"` // "auto", "any", "tool"
	Name string `json:"name,omitempty"`
}

type anthropicResponse struct {
	ID           string                  `json:"id"`
	Type         string                  `json:"type"`
	Role         string                  `json:"role"`
	Content      []anthropicContentBlock `json:"content"`
	Model        string                  `json:"model"`
	StopReason   string                  `json:"stop_reason"`
	StopSequence *string                 `json:"stop_sequence"`
	Usage        anthropicUsage          `json:"usage"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type anthropicStreamEvent struct {
	Type    string          `json:"type"`
	Message json.RawMessage `json:"message,omitempty"`
	Index   int             `json:"index,omitempty"`
	Delta   json.RawMessage `json:"delta,omitempty"`
	Usage   json.RawMessage `json:"usage,omitempty"`
}

func (p *AnthropicProvider) Complete(ctx context.Context, req litellm.CompletionRequest) (*litellm.CompletionResponse, error) {
	apiKey := p.resolveAPIKey(req.APIKey)
	if apiKey == "" {
		return nil, &litellm.APIError{StatusCode: 401, Message: "Anthropic API key is required", Provider: "anthropic"}
	}

	anthReq := p.transformRequest(req)
	anthReq.Stream = false

	jsonBody, err := json.Marshal(anthReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.resolveBaseURL(req.BaseURL)+"/v1/messages", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	p.setHeaders(httpReq, apiKey)

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
		return nil, p.parseError(respBody, resp.StatusCode)
	}

	var anthResp anthropicResponse
	if err := json.Unmarshal(respBody, &anthResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return p.transformResponse(anthResp), nil
}

func (p *AnthropicProvider) CompleteStream(ctx context.Context, req litellm.CompletionRequest) (*litellm.Stream, error) {
	apiKey := p.resolveAPIKey(req.APIKey)
	if apiKey == "" {
		return nil, &litellm.APIError{StatusCode: 401, Message: "Anthropic API key is required", Provider: "anthropic"}
	}

	anthReq := p.transformRequest(req)
	anthReq.Stream = true

	jsonBody, err := json.Marshal(anthReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.resolveBaseURL(req.BaseURL)+"/v1/messages", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	p.setHeaders(httpReq, apiKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("stream request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		return nil, p.parseError(body, resp.StatusCode)
	}

	_, cancel := context.WithCancel(ctx)
	stream := litellm.NewStreamForProvider(cancel)

	go p.parseAnthropicStream(resp.Body, stream, req.Model)

	return stream, nil
}

func (p *AnthropicProvider) parseAnthropicStream(body io.ReadCloser, stream *litellm.Stream, model string) {
	defer body.Close()
	defer stream.CloseStream(nil)

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var chunkID string
	var created int64

	for scanner.Scan() {
		line := scanner.Text()

		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		var evt anthropicStreamEvent
		if err := json.Unmarshal([]byte(data), &evt); err != nil {
			stream.SendEvent(litellm.StreamEvent{Err: fmt.Errorf("failed to parse event: %w", err)})
			return
		}

		switch evt.Type {
		case "message_start":
			var msg struct {
				ID    string `json:"id"`
				Model string `json:"model"`
				Usage struct {
					InputTokens int `json:"input_tokens"`
				} `json:"usage"`
			}
			if err := json.Unmarshal(evt.Message, &msg); err == nil {
				chunkID = msg.ID
				if msg.Model != "" {
					model = msg.Model
				}
				created = time.Now().Unix()
			}

		case "content_block_delta":
			var delta struct {
				Type string `json:"type"`
				Text string `json:"text,omitempty"`
			}
			if err := json.Unmarshal(evt.Delta, &delta); err == nil {
				chunk := &litellm.StreamChunk{
					ID:      chunkID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   model,
					Choices: []litellm.StreamingChoice{{
						Index: 0,
						Delta: litellm.Delta{
							Content: delta.Text,
						},
					}},
				}
				stream.SendEvent(litellm.StreamEvent{Chunk: chunk})
			}

		case "message_delta":
			var delta struct {
				StopReason string `json:"stop_reason"`
			}
			var usage struct {
				OutputTokens int `json:"output_tokens"`
			}
			if err := json.Unmarshal(evt.Delta, &delta); err != nil {
				stream.SendEvent(litellm.StreamEvent{Err: fmt.Errorf("failed to parse message_delta: %w", err)})
				return
			}
			if evt.Usage != nil {
				// Usage is optional on message_delta; ignore unmarshal errors for it
				json.Unmarshal(evt.Usage, &usage)
			}

			finishReason := mapAnthropicStopReason(delta.StopReason)
			chunk := &litellm.StreamChunk{
				ID:      chunkID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []litellm.StreamingChoice{{
					Index:        0,
					FinishReason: finishReason,
				}},
			}
			stream.SendEvent(litellm.StreamEvent{Chunk: chunk})

		case "message_stop":
			return
		}
	}

	if err := scanner.Err(); err != nil {
		stream.SendEvent(litellm.StreamEvent{Err: fmt.Errorf("stream read error: %w", err)})
	}
}

func (p *AnthropicProvider) Embed(_ context.Context, _ litellm.EmbeddingRequest) (*litellm.EmbeddingResponse, error) {
	return nil, &litellm.APIError{
		StatusCode: 400,
		Message:    "Anthropic does not support embeddings",
		Provider:   "anthropic",
	}
}

func (p *AnthropicProvider) Models(_ context.Context) ([]litellm.ModelInfo, error) {
	models := []litellm.ModelInfo{
		{ID: "claude-3-5-sonnet-20241022", Object: "model", OwnedBy: "anthropic"},
		{ID: "claude-3-5-haiku-20241022", Object: "model", OwnedBy: "anthropic"},
		{ID: "claude-3-opus-20240229", Object: "model", OwnedBy: "anthropic"},
		{ID: "claude-3-sonnet-20240229", Object: "model", OwnedBy: "anthropic"},
		{ID: "claude-3-haiku-20240307", Object: "model", OwnedBy: "anthropic"},
	}
	return models, nil
}

func (p *AnthropicProvider) transformRequest(req litellm.CompletionRequest) anthropicRequest {
	anthReq := anthropicRequest{
		Model:     req.Model,
		MaxTokens: 4096,
	}

	if req.MaxTokens != nil {
		anthReq.MaxTokens = *req.MaxTokens
	}
	if req.Temperature != nil {
		anthReq.Temperature = req.Temperature
	}
	if req.TopP != nil {
		anthReq.TopP = req.TopP
	}
	if len(req.Stop) > 0 {
		anthReq.StopSequences = req.Stop
	}

	// Convert messages, extract system message
	for _, msg := range req.Messages {
		if msg.Role == "system" {
			anthReq.System = msg.Content.Text
			continue
		}

		anthMsg := anthropicMessage{
			Role: msg.Role,
		}

		// Handle tool results
		if msg.Role == "tool" {
			anthMsg.Role = "user"
			anthMsg.Content = []anthropicContentBlock{{
				Type:      "tool_result",
				ToolUseID: msg.ToolCallID,
				Content:   msg.Content.Text,
			}}
		} else if len(msg.ToolCalls) > 0 {
			// Assistant with tool calls
			blocks := make([]anthropicContentBlock, 0)
			if msg.Content.Text != "" {
				blocks = append(blocks, anthropicContentBlock{
					Type: "text",
					Text: msg.Content.Text,
				})
			}
			for _, tc := range msg.ToolCalls {
				var input interface{}
				if err := json.Unmarshal([]byte(tc.Function.Arguments), &input); err != nil {
					// If arguments can't be parsed as JSON, use them as a raw string
					input = tc.Function.Arguments
				}
				blocks = append(blocks, anthropicContentBlock{
					Type:  "tool_use",
					ID:    tc.ID,
					Name:  tc.Function.Name,
					Input: input,
				})
			}
			anthMsg.Content = blocks
		} else if len(msg.Content.Parts) > 0 {
			// Multimodal content
			blocks := make([]anthropicContentBlock, 0, len(msg.Content.Parts))
			for _, part := range msg.Content.Parts {
				blocks = append(blocks, anthropicContentBlock{
					Type: part.Type,
					Text: part.Text,
				})
			}
			anthMsg.Content = blocks
		} else {
			anthMsg.Content = msg.Content.Text
		}

		anthReq.Messages = append(anthReq.Messages, anthMsg)
	}

	// Convert tools
	for _, tool := range req.Tools {
		anthReq.Tools = append(anthReq.Tools, anthropicTool{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			InputSchema: tool.Function.Parameters,
		})
	}

	// Convert tool choice
	if req.ToolChoice != nil {
		switch v := req.ToolChoice.(type) {
		case string:
			switch v {
			case "auto":
				anthReq.ToolChoice = &anthropicToolChoice{Type: "auto"}
			case "required":
				anthReq.ToolChoice = &anthropicToolChoice{Type: "any"}
			case "none":
				// Don't set tool choice, Anthropic doesn't have "none"
			}
		case map[string]interface{}:
			if fn, ok := v["function"].(map[string]interface{}); ok {
				if name, ok := fn["name"].(string); ok {
					anthReq.ToolChoice = &anthropicToolChoice{Type: "tool", Name: name}
				}
			}
		}
	}

	return anthReq
}

func (p *AnthropicProvider) transformResponse(resp anthropicResponse) *litellm.CompletionResponse {
	var content string
	var toolCalls []litellm.ToolCall

	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			content += block.Text
		case "tool_use":
			args, _ := json.Marshal(block.Input)
			toolCalls = append(toolCalls, litellm.ToolCall{
				ID:   block.ID,
				Type: "function",
				Function: litellm.FunctionCall{
					Name:      block.Name,
					Arguments: string(args),
				},
			})
		}
	}

	finishReason := mapAnthropicStopReason(resp.StopReason)

	return &litellm.CompletionResponse{
		ID:      resp.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   resp.Model,
		Choices: []litellm.Choice{{
			Index: 0,
			Message: litellm.Message{
				Role:      "assistant",
				Content:   litellm.StringContent(content),
				ToolCalls: toolCalls,
			},
			FinishReason: finishReason,
		}},
		Usage: &litellm.Usage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
	}
}

func mapAnthropicStopReason(reason string) string {
	switch reason {
	case "end_turn":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	case "stop_sequence":
		return "stop"
	default:
		if reason == "" {
			return ""
		}
		return reason
	}
}

func (p *AnthropicProvider) setHeaders(req *http.Request, apiKey string) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", anthropicAPIVersion)
}

func (p *AnthropicProvider) resolveAPIKey(explicit string) string {
	if explicit != "" {
		return explicit
	}
	return p.apiKey
}

func (p *AnthropicProvider) resolveBaseURL(explicit string) string {
	if explicit != "" {
		return explicit
	}
	return p.baseURL
}

func (p *AnthropicProvider) parseError(body []byte, statusCode int) error {
	var errResp struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}

	if err := json.Unmarshal(body, &errResp); err != nil {
		return &litellm.APIError{
			StatusCode: statusCode,
			Message:    string(body),
			Provider:   "anthropic",
		}
	}

	return &litellm.APIError{
		StatusCode: statusCode,
		Message:    errResp.Error.Message,
		Type:       errResp.Error.Type,
		Provider:   "anthropic",
	}
}
