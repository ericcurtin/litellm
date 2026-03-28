package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	litellm "github.com/ericcurtin/litellm/go-litellm/internal/litellm"
)

// mockProvider implements litellm.Provider for testing.
type mockProvider struct {
	response  *litellm.CompletionResponse
	embedResp *litellm.EmbeddingResponse
	err       error
}

func (m *mockProvider) Name() string { return "mock" }
func (m *mockProvider) Complete(_ context.Context, _ litellm.CompletionRequest) (*litellm.CompletionResponse, error) {
	return m.response, m.err
}
func (m *mockProvider) CompleteStream(_ context.Context, _ litellm.CompletionRequest) (*litellm.Stream, error) {
	return nil, m.err
}
func (m *mockProvider) Embed(_ context.Context, _ litellm.EmbeddingRequest) (*litellm.EmbeddingResponse, error) {
	return m.embedResp, m.err
}
func (m *mockProvider) Models(_ context.Context) ([]litellm.ModelInfo, error) {
	return []litellm.ModelInfo{{ID: "test-model", Object: "model", OwnedBy: "mock"}}, nil
}

func setupTestServer(masterKey string) (*Server, *mockProvider) {
	mock := &mockProvider{
		response: &litellm.CompletionResponse{
			ID:      "test-id",
			Object:  "chat.completion",
			Created: 1234567890,
			Model:   "test-model",
			Choices: []litellm.Choice{{
				Index: 0,
				Message: litellm.Message{
					Role:    "assistant",
					Content: litellm.StringContent("Hello!"),
				},
				FinishReason: "stop",
			}},
			Usage: &litellm.Usage{
				PromptTokens:     10,
				CompletionTokens: 5,
				TotalTokens:      15,
			},
		},
	}

	client := litellm.NewClient()
	client.RegisterProvider("openai", mock)

	server := NewServer(ServerConfig{
		Client:    client,
		MasterKey: masterKey,
	})

	return server, mock
}

func TestHealthEndpoint(t *testing.T) {
	server, _ := setupTestServer("")

	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()
	server.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var resp map[string]string
	json.NewDecoder(w.Body).Decode(&resp)
	if resp["status"] != "healthy" {
		t.Errorf("expected status 'healthy', got %q", resp["status"])
	}
}

func TestChatCompletions(t *testing.T) {
	server, _ := setupTestServer("")

	body := map[string]interface{}{
		"model": "openai/test-model",
		"messages": []map[string]interface{}{
			{"role": "user", "content": "Hello"},
		},
	}
	jsonBody, _ := json.Marshal(body)

	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(jsonBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp litellm.CompletionResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.ID != "test-id" {
		t.Errorf("expected ID 'test-id', got %q", resp.ID)
	}
	if resp.Choices[0].Message.Content.Text != "Hello!" {
		t.Errorf("expected 'Hello!', got %q", resp.Choices[0].Message.Content.Text)
	}
}

func TestChatCompletions_MissingModel(t *testing.T) {
	server, _ := setupTestServer("")

	body := map[string]interface{}{
		"messages": []map[string]interface{}{
			{"role": "user", "content": "Hello"},
		},
	}
	jsonBody, _ := json.Marshal(body)

	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(jsonBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

func TestAuthentication(t *testing.T) {
	server, _ := setupTestServer("sk-test-key")

	body := map[string]interface{}{
		"model": "openai/test-model",
		"messages": []map[string]interface{}{
			{"role": "user", "content": "Hello"},
		},
	}
	jsonBody, _ := json.Marshal(body)

	// Without auth
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(jsonBody))
	w := httptest.NewRecorder()
	server.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 without auth, got %d", w.Code)
	}

	// With wrong key
	req = httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(jsonBody))
	req.Header.Set("Authorization", "Bearer wrong-key")
	w = httptest.NewRecorder()
	server.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 with wrong key, got %d", w.Code)
	}

	// With correct key
	req = httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(jsonBody))
	req.Header.Set("Authorization", "Bearer sk-test-key")
	w = httptest.NewRecorder()
	server.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("expected 200 with correct key, got %d: %s", w.Code, w.Body.String())
	}
}

func TestModelsEndpoint(t *testing.T) {
	server, _ := setupTestServer("")

	req := httptest.NewRequest("GET", "/v1/models", nil)
	w := httptest.NewRecorder()
	server.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var resp litellm.ModelListResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Object != "list" {
		t.Errorf("expected object 'list', got %q", resp.Object)
	}
	if len(resp.Data) != 1 {
		t.Errorf("expected 1 model, got %d", len(resp.Data))
	}
}

func TestMethodNotAllowed(t *testing.T) {
	server, _ := setupTestServer("")

	req := httptest.NewRequest("GET", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	server.ServeHTTP(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", w.Code)
	}
}

func TestProviderError(t *testing.T) {
	server, mock := setupTestServer("")
	mock.response = nil
	mock.err = &litellm.APIError{
		StatusCode: 429,
		Message:    "rate limit exceeded",
		Type:       "rate_limit_error",
		Provider:   "openai",
	}

	body := map[string]interface{}{
		"model": "openai/test-model",
		"messages": []map[string]interface{}{
			{"role": "user", "content": "Hello"},
		},
	}
	jsonBody, _ := json.Marshal(body)

	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(jsonBody))
	w := httptest.NewRecorder()
	server.ServeHTTP(w, req)

	if w.Code != 429 {
		t.Errorf("expected 429, got %d", w.Code)
	}

	respBody, _ := io.ReadAll(w.Body)
	var errResp map[string]interface{}
	json.Unmarshal(respBody, &errResp)
	errObj := errResp["error"].(map[string]interface{})
	if errObj["message"] != "rate limit exceeded" {
		t.Errorf("expected 'rate limit exceeded', got %q", errObj["message"])
	}
}

func TestHealthNoAuth(t *testing.T) {
	server, _ := setupTestServer("sk-test-key")

	// Health endpoint should not require auth
	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()
	server.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
}
