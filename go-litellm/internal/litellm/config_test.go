package litellm

import (
	"os"
	"testing"
)

func TestParseModelProvider(t *testing.T) {
	tests := []struct {
		input         string
		wantProvider  string
		wantModel     string
	}{
		{"openai/gpt-4", "openai", "gpt-4"},
		{"anthropic/claude-3-sonnet", "anthropic", "claude-3-sonnet"},
		{"azure/my-deployment", "azure", "my-deployment"},
		{"ollama/llama2", "ollama", "llama2"},
		{"groq/mixtral-8x7b", "groq", "mixtral-8x7b"},
		{"gpt-4", "openai", "gpt-4"},
		{"gpt-3.5-turbo", "openai", "gpt-3.5-turbo"},
		{"claude-3-sonnet", "anthropic", "claude-3-sonnet"},
		{"text-embedding-3-small", "openai", "text-embedding-3-small"},
		{"o1-preview", "openai", "o1-preview"},
		{"unknown-model", "openai", "unknown-model"}, // defaults to openai
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			provider, model := ParseModelProvider(tt.input)
			if provider != tt.wantProvider {
				t.Errorf("ParseModelProvider(%q) provider = %q, want %q", tt.input, provider, tt.wantProvider)
			}
			if model != tt.wantModel {
				t.Errorf("ParseModelProvider(%q) model = %q, want %q", tt.input, model, tt.wantModel)
			}
		})
	}
}

func TestResolveAPIKey(t *testing.T) {
	cfg := Config{
		OpenAIKey:    "config-key",
		AnthropicKey: "anth-config-key",
	}

	// Explicit key takes priority
	got := ResolveAPIKey("explicit-key", "openai", cfg)
	if got != "explicit-key" {
		t.Errorf("expected explicit-key, got %q", got)
	}

	// Config key is second priority
	got = ResolveAPIKey("", "openai", cfg)
	if got != "config-key" {
		t.Errorf("expected config-key, got %q", got)
	}

	// Environment variable is third priority
	os.Setenv("ANTHROPIC_API_KEY", "env-key")
	defer os.Unsetenv("ANTHROPIC_API_KEY")

	got = ResolveAPIKey("", "anthropic", Config{})
	if got != "env-key" {
		t.Errorf("expected env-key, got %q", got)
	}

	// Config overrides env
	got = ResolveAPIKey("", "anthropic", cfg)
	if got != "anth-config-key" {
		t.Errorf("expected anth-config-key, got %q", got)
	}
}

func TestResolveBaseURL(t *testing.T) {
	cfg := Config{
		AzureAPIBase: "https://my-resource.openai.azure.com",
	}

	// Explicit URL takes priority
	got := ResolveBaseURL("https://custom.com", "azure", cfg)
	if got != "https://custom.com" {
		t.Errorf("expected custom URL, got %q", got)
	}

	// Config is second
	got = ResolveBaseURL("", "azure", cfg)
	if got != "https://my-resource.openai.azure.com" {
		t.Errorf("expected azure config URL, got %q", got)
	}

	// Env variable for OpenAI
	os.Setenv("OPENAI_API_BASE", "https://env.openai.com")
	defer os.Unsetenv("OPENAI_API_BASE")

	got = ResolveBaseURL("", "openai", Config{})
	if got != "https://env.openai.com" {
		t.Errorf("expected env URL, got %q", got)
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.DefaultTimeout != 300 {
		t.Errorf("expected timeout 300, got %d", cfg.DefaultTimeout)
	}
	if cfg.DefaultMaxRetry != 2 {
		t.Errorf("expected max retry 2, got %d", cfg.DefaultMaxRetry)
	}
}
