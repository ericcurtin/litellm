package litellm

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

// mockProvider is a test provider that returns configurable responses.
type mockProvider struct {
	name      string
	response  *CompletionResponse
	err       error
	callCount int64
	delay     time.Duration
}

func (m *mockProvider) Name() string { return m.name }

func (m *mockProvider) Complete(_ context.Context, _ CompletionRequest) (*CompletionResponse, error) {
	atomic.AddInt64(&m.callCount, 1)
	if m.delay > 0 {
		time.Sleep(m.delay)
	}
	if m.err != nil {
		return nil, m.err
	}
	return m.response, nil
}

func (m *mockProvider) CompleteStream(_ context.Context, _ CompletionRequest) (*Stream, error) {
	return nil, nil
}

func (m *mockProvider) Embed(_ context.Context, _ EmbeddingRequest) (*EmbeddingResponse, error) {
	return nil, nil
}

func (m *mockProvider) Models(_ context.Context) ([]ModelInfo, error) {
	return []ModelInfo{{ID: "test-model", Object: "model", OwnedBy: m.name}}, nil
}

func TestRouterShuffle(t *testing.T) {
	mock1 := &mockProvider{name: "mock1", response: &CompletionResponse{Model: "model-1"}}
	mock2 := &mockProvider{name: "mock2", response: &CompletionResponse{Model: "model-2"}}

	deployments := []Deployment{
		{ModelName: "test", LiteLLMModel: "model-1", Provider: mock1, Weight: 1},
		{ModelName: "test", LiteLLMModel: "model-2", Provider: mock2, Weight: 1},
	}

	cfg := DefaultRouterConfig()
	cfg.Strategy = StrategyShuffle
	router := NewRouter(cfg, deployments)

	// Run multiple times to verify both deployments are used
	seen := map[string]bool{}
	for i := 0; i < 100; i++ {
		resp, err := router.Complete(context.Background(), CompletionRequest{Model: "test"})
		if err != nil {
			t.Fatal(err)
		}
		seen[resp.Model] = true
	}

	if !seen["model-1"] || !seen["model-2"] {
		t.Errorf("expected both models to be used, got %v", seen)
	}
}

func TestRouterRoundRobin(t *testing.T) {
	mock1 := &mockProvider{name: "mock1", response: &CompletionResponse{Model: "model-1"}}
	mock2 := &mockProvider{name: "mock2", response: &CompletionResponse{Model: "model-2"}}

	deployments := []Deployment{
		{ModelName: "test", LiteLLMModel: "model-1", Provider: mock1},
		{ModelName: "test", LiteLLMModel: "model-2", Provider: mock2},
	}

	cfg := DefaultRouterConfig()
	cfg.Strategy = StrategyRoundRobin
	router := NewRouter(cfg, deployments)

	ctx := context.Background()

	resp1, _ := router.Complete(ctx, CompletionRequest{Model: "test"})
	resp2, _ := router.Complete(ctx, CompletionRequest{Model: "test"})

	if resp1.Model == resp2.Model {
		t.Errorf("round-robin should alternate, got same model: %s", resp1.Model)
	}
}

func TestRouterRetries(t *testing.T) {
	failProvider := &mockProvider{
		name: "fail",
		err: &APIError{
			StatusCode: 500,
			Message:    "server error",
			Provider:   "test",
		},
	}
	successProvider := &mockProvider{
		name:     "success",
		response: &CompletionResponse{Model: "success-model"},
	}

	deployments := []Deployment{
		{ModelName: "test", LiteLLMModel: "fail-model", Provider: failProvider},
		{ModelName: "test", LiteLLMModel: "success-model", Provider: successProvider},
	}

	cfg := DefaultRouterConfig()
	cfg.NumRetries = 3
	router := NewRouter(cfg, deployments)

	resp, err := router.Complete(context.Background(), CompletionRequest{Model: "test"})
	if err != nil {
		t.Fatalf("expected success after retries, got: %v", err)
	}
	if resp.Model != "success-model" {
		t.Errorf("expected success-model, got %q", resp.Model)
	}
}

func TestRouterFallbacks(t *testing.T) {
	failProvider := &mockProvider{
		name: "fail",
		err: &APIError{
			StatusCode: 500,
			Message:    "all fail",
			Provider:   "test",
		},
	}
	successProvider := &mockProvider{
		name:     "success",
		response: &CompletionResponse{Model: "fallback-model"},
	}

	deployments := []Deployment{
		{ModelName: "primary", LiteLLMModel: "fail-model", Provider: failProvider},
		{ModelName: "fallback", LiteLLMModel: "fallback-model", Provider: successProvider},
	}

	cfg := DefaultRouterConfig()
	cfg.NumRetries = 0
	cfg.Fallbacks = map[string][]string{
		"primary": {"fallback"},
	}
	router := NewRouter(cfg, deployments)

	resp, err := router.Complete(context.Background(), CompletionRequest{Model: "primary"})
	if err != nil {
		t.Fatalf("expected fallback success, got: %v", err)
	}
	if resp.Model != "fallback-model" {
		t.Errorf("expected fallback-model, got %q", resp.Model)
	}
}

func TestRouterCooldown(t *testing.T) {
	failProvider := &mockProvider{
		name: "fail",
		err: &APIError{
			StatusCode: 500,
			Message:    "error",
			Provider:   "test",
		},
	}

	deployments := []Deployment{
		{ModelName: "test", LiteLLMModel: "model", Provider: failProvider},
	}

	cfg := DefaultRouterConfig()
	cfg.AllowedFails = 2
	cfg.CooldownTime = 100 * time.Millisecond
	cfg.NumRetries = 5
	router := NewRouter(cfg, deployments)

	// Should fail and enter cooldown
	_, err := router.Complete(context.Background(), CompletionRequest{Model: "test"})
	if err == nil {
		t.Fatal("expected error")
	}

	// Check cooldown state
	count := router.GetHealthyDeployments("test")
	if count != 0 {
		t.Errorf("expected 0 healthy deployments during cooldown, got %d", count)
	}

	// Wait for cooldown to expire
	time.Sleep(150 * time.Millisecond)
	count = router.GetHealthyDeployments("test")
	if count != 1 {
		t.Errorf("expected 1 healthy deployment after cooldown, got %d", count)
	}
}

func TestRouterNoDeployments(t *testing.T) {
	cfg := DefaultRouterConfig()
	router := NewRouter(cfg, nil)

	_, err := router.Complete(context.Background(), CompletionRequest{Model: "nonexistent"})
	if err == nil {
		t.Fatal("expected error for missing model group")
	}
}

func TestRouterLeastBusy(t *testing.T) {
	slow := &mockProvider{name: "slow", response: &CompletionResponse{Model: "slow"}, delay: 50 * time.Millisecond}
	fast := &mockProvider{name: "fast", response: &CompletionResponse{Model: "fast"}}

	deployments := []Deployment{
		{ModelName: "test", LiteLLMModel: "slow", Provider: slow},
		{ModelName: "test", LiteLLMModel: "fast", Provider: fast},
	}

	cfg := DefaultRouterConfig()
	cfg.Strategy = StrategyLeastBusy
	router := NewRouter(cfg, deployments)

	// Both start with 0 active requests, so either could be picked
	resp, err := router.Complete(context.Background(), CompletionRequest{Model: "test"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Model != "slow" && resp.Model != "fast" {
		t.Errorf("unexpected model: %s", resp.Model)
	}
}

func TestRouterDefaultFallbacks(t *testing.T) {
	failProvider := &mockProvider{
		name: "fail",
		err:  &APIError{StatusCode: 500, Message: "fail", Provider: "test"},
	}
	successProvider := &mockProvider{
		name:     "success",
		response: &CompletionResponse{Model: "default-fallback"},
	}

	deployments := []Deployment{
		{ModelName: "primary", LiteLLMModel: "fail", Provider: failProvider},
		{ModelName: "default-fb", LiteLLMModel: "default-fallback", Provider: successProvider},
	}

	cfg := DefaultRouterConfig()
	cfg.NumRetries = 0
	cfg.DefaultFallbacks = []string{"default-fb"}
	router := NewRouter(cfg, deployments)

	resp, err := router.Complete(context.Background(), CompletionRequest{Model: "primary"})
	if err != nil {
		t.Fatalf("expected default fallback, got: %v", err)
	}
	if resp.Model != "default-fallback" {
		t.Errorf("expected default-fallback, got %q", resp.Model)
	}
}
