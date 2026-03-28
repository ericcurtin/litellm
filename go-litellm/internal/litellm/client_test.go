package litellm

import (
	"context"
	"errors"
	"testing"
)

func TestClient_RegisterAndGetProvider(t *testing.T) {
	client := NewClient()
	mock := &mockProvider{name: "test", response: &CompletionResponse{Model: "test-model"}}

	client.RegisterProvider("test", mock)

	got := client.GetProvider("test")
	if got == nil {
		t.Fatal("expected provider, got nil")
	}
	if got.Name() != "test" {
		t.Errorf("expected provider name 'test', got %q", got.Name())
	}
}

func TestClient_Complete(t *testing.T) {
	client := NewClient()
	mock := &mockProvider{name: "openai", response: &CompletionResponse{
		ID:     "test-id",
		Model:  "gpt-4",
		Object: "chat.completion",
		Choices: []Choice{{
			Index:        0,
			Message:      Message{Role: "assistant", Content: StringContent("Hello!")},
			FinishReason: "stop",
		}},
	}}
	client.RegisterProvider("openai", mock)

	resp, err := client.Complete(context.Background(), CompletionRequest{
		Model:    "openai/gpt-4",
		Messages: []Message{{Role: "user", Content: StringContent("Hi")}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.ID != "test-id" {
		t.Errorf("expected ID 'test-id', got %q", resp.ID)
	}
	if resp.Choices[0].Message.Content.Text != "Hello!" {
		t.Errorf("expected 'Hello!', got %q", resp.Choices[0].Message.Content.Text)
	}
}

func TestClient_Complete_NoProvider(t *testing.T) {
	client := NewClient()

	_, err := client.Complete(context.Background(), CompletionRequest{
		Model:    "unknown/model",
		Messages: []Message{{Role: "user", Content: StringContent("Hi")}},
	})
	if err == nil {
		t.Fatal("expected error for unregistered provider")
	}
	if !errors.Is(err, ErrNoProvider) {
		t.Errorf("expected ErrNoProvider, got %v", err)
	}
}

func TestClient_Embed(t *testing.T) {
	client := NewClient()
	mock := &mockProvider{name: "openai"}
	client.RegisterProvider("openai", mock)

	// mockProvider.Embed returns nil, nil - just testing routing
	_, err := client.Embed(context.Background(), EmbeddingRequest{
		Model: "openai/text-embedding-3-small",
		Input: "test text",
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestClient_Models(t *testing.T) {
	client := NewClient()
	mock := &mockProvider{name: "openai"}
	client.RegisterProvider("openai", mock)

	models, err := client.Models(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}
	if models[0].ID != "test-model" {
		t.Errorf("expected 'test-model', got %q", models[0].ID)
	}
}

func TestClient_WithConfig(t *testing.T) {
	cfg := Config{
		OpenAIKey:      "test-key",
		DefaultTimeout: 60,
	}
	client := NewClientWithConfig(cfg)
	if client == nil {
		t.Fatal("expected non-nil client")
	}
}
