package litellm

import (
	"encoding/json"
	"testing"
)

func TestStringContent(t *testing.T) {
	c := StringContent("hello")
	if c.Text != "hello" {
		t.Errorf("expected text 'hello', got %q", c.Text)
	}
	if len(c.Parts) != 0 {
		t.Errorf("expected no parts, got %d", len(c.Parts))
	}
}

func TestMultiContent(t *testing.T) {
	c := MultiContent(
		ContentPart{Type: "text", Text: "describe this"},
		ContentPart{Type: "image_url", ImageURL: &ImageURL{URL: "http://example.com/img.png"}},
	)
	if len(c.Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(c.Parts))
	}
	if c.Parts[0].Type != "text" {
		t.Errorf("expected first part type 'text', got %q", c.Parts[0].Type)
	}
	if c.Parts[1].ImageURL.URL != "http://example.com/img.png" {
		t.Errorf("expected image URL, got %q", c.Parts[1].ImageURL.URL)
	}
}

func TestContentMarshalJSON_String(t *testing.T) {
	c := StringContent("hello world")
	data, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `"hello world"` {
		t.Errorf("expected string JSON, got %s", data)
	}
}

func TestContentMarshalJSON_Parts(t *testing.T) {
	c := MultiContent(ContentPart{Type: "text", Text: "hi"})
	data, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) == `"hi"` {
		t.Error("expected array JSON, got string")
	}
	var parts []ContentPart
	if err := json.Unmarshal(data, &parts); err != nil {
		t.Fatal(err)
	}
	if len(parts) != 1 || parts[0].Text != "hi" {
		t.Errorf("unexpected parts: %+v", parts)
	}
}

func TestContentUnmarshalJSON_String(t *testing.T) {
	var c Content
	if err := json.Unmarshal([]byte(`"hello"`), &c); err != nil {
		t.Fatal(err)
	}
	if c.Text != "hello" {
		t.Errorf("expected 'hello', got %q", c.Text)
	}
}

func TestContentUnmarshalJSON_Array(t *testing.T) {
	var c Content
	data := `[{"type":"text","text":"hello"},{"type":"image_url","image_url":{"url":"http://x.com/i.png"}}]`
	if err := json.Unmarshal([]byte(data), &c); err != nil {
		t.Fatal(err)
	}
	if len(c.Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(c.Parts))
	}
}

func TestMessageJSON(t *testing.T) {
	msg := Message{
		Role:    "user",
		Content: StringContent("hello"),
	}
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}

	var decoded Message
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Role != "user" {
		t.Errorf("expected role 'user', got %q", decoded.Role)
	}
	if decoded.Content.Text != "hello" {
		t.Errorf("expected content 'hello', got %q", decoded.Content.Text)
	}
}

func TestCompletionResponseJSON(t *testing.T) {
	resp := CompletionResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "gpt-4",
		Choices: []Choice{{
			Index: 0,
			Message: Message{
				Role:    "assistant",
				Content: StringContent("Hello!"),
			},
			FinishReason: "stop",
		}},
		Usage: &Usage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}

	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatal(err)
	}

	var decoded CompletionResponse
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}

	if decoded.ID != "chatcmpl-123" {
		t.Errorf("expected ID 'chatcmpl-123', got %q", decoded.ID)
	}
	if len(decoded.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(decoded.Choices))
	}
	if decoded.Choices[0].FinishReason != "stop" {
		t.Errorf("expected finish_reason 'stop', got %q", decoded.Choices[0].FinishReason)
	}
	if decoded.Usage.TotalTokens != 15 {
		t.Errorf("expected 15 total tokens, got %d", decoded.Usage.TotalTokens)
	}
}

func TestToolCallJSON(t *testing.T) {
	msg := Message{
		Role:    "assistant",
		Content: StringContent(""),
		ToolCalls: []ToolCall{{
			ID:   "call_abc123",
			Type: "function",
			Function: FunctionCall{
				Name:      "get_weather",
				Arguments: `{"location":"SF"}`,
			},
		}},
	}

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}

	var decoded Message
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if len(decoded.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(decoded.ToolCalls))
	}
	if decoded.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("expected function name 'get_weather', got %q", decoded.ToolCalls[0].Function.Name)
	}
}

func TestStreamChunkJSON(t *testing.T) {
	chunk := StreamChunk{
		ID:      "chatcmpl-123",
		Object:  "chat.completion.chunk",
		Created: 1234567890,
		Model:   "gpt-4",
		Choices: []StreamingChoice{{
			Index: 0,
			Delta: Delta{
				Content: "Hello",
			},
		}},
	}

	data, err := json.Marshal(chunk)
	if err != nil {
		t.Fatal(err)
	}

	var decoded StreamChunk
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Object != "chat.completion.chunk" {
		t.Errorf("expected object 'chat.completion.chunk', got %q", decoded.Object)
	}
	if decoded.Choices[0].Delta.Content != "Hello" {
		t.Errorf("expected delta content 'Hello', got %q", decoded.Choices[0].Delta.Content)
	}
}
