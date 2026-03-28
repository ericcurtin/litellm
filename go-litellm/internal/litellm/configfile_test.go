package litellm

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadConfigFile_Simple(t *testing.T) {
	content := `model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: sk-test-123
  - model_name: claude-3
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: sk-ant-test

general_settings:
  master_key: sk-proxy-key

litellm_settings:
  drop_params: True
`
	path := writeTestConfig(t, content)

	cfg, err := LoadConfigFile(path)
	if err != nil {
		t.Fatalf("LoadConfigFile() error: %v", err)
	}

	if len(cfg.ModelList) != 2 {
		t.Fatalf("expected 2 models, got %d", len(cfg.ModelList))
	}

	// First model
	if cfg.ModelList[0].ModelName != "gpt-4" {
		t.Errorf("model[0].ModelName = %q, want %q", cfg.ModelList[0].ModelName, "gpt-4")
	}
	if cfg.ModelList[0].LiteLLMParams.Model != "openai/gpt-4" {
		t.Errorf("model[0].LiteLLMParams.Model = %q, want %q", cfg.ModelList[0].LiteLLMParams.Model, "openai/gpt-4")
	}
	if cfg.ModelList[0].LiteLLMParams.APIKey != "sk-test-123" {
		t.Errorf("model[0].LiteLLMParams.APIKey = %q, want %q", cfg.ModelList[0].LiteLLMParams.APIKey, "sk-test-123")
	}

	// Second model
	if cfg.ModelList[1].ModelName != "claude-3" {
		t.Errorf("model[1].ModelName = %q, want %q", cfg.ModelList[1].ModelName, "claude-3")
	}
	if cfg.ModelList[1].LiteLLMParams.Model != "anthropic/claude-3-5-sonnet-20241022" {
		t.Errorf("model[1].LiteLLMParams.Model = %q, want %q", cfg.ModelList[1].LiteLLMParams.Model, "anthropic/claude-3-5-sonnet-20241022")
	}

	// General settings
	if cfg.GeneralSettings.MasterKey != "sk-proxy-key" {
		t.Errorf("GeneralSettings.MasterKey = %q, want %q", cfg.GeneralSettings.MasterKey, "sk-proxy-key")
	}

	// LiteLLM settings
	if !cfg.LiteLLMSettings.DropParams {
		t.Error("LiteLLMSettings.DropParams = false, want true")
	}
}

func TestLoadConfigFile_WithRouterSettings(t *testing.T) {
	content := `model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
    tpm: 50000
    rpm: 100

router_settings:
  routing_strategy: least-busy
  num_retries: 5
`
	path := writeTestConfig(t, content)

	cfg, err := LoadConfigFile(path)
	if err != nil {
		t.Fatalf("LoadConfigFile() error: %v", err)
	}

	if cfg.ModelList[0].TPM != 50000 {
		t.Errorf("model.TPM = %d, want %d", cfg.ModelList[0].TPM, 50000)
	}
	if cfg.ModelList[0].RPM != 100 {
		t.Errorf("model.RPM = %d, want %d", cfg.ModelList[0].RPM, 100)
	}
	if cfg.RouterSettings.RoutingStrategy != "least-busy" {
		t.Errorf("RouterSettings.RoutingStrategy = %q, want %q", cfg.RouterSettings.RoutingStrategy, "least-busy")
	}
	if cfg.RouterSettings.NumRetries != 5 {
		t.Errorf("RouterSettings.NumRetries = %d, want %d", cfg.RouterSettings.NumRetries, 5)
	}
}

func TestLoadConfigFile_WithEnvironmentVars(t *testing.T) {
	content := `model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4

environment_variables:
  OPENAI_API_KEY: sk-from-config
  ANTHROPIC_API_KEY: sk-ant-from-config
`
	path := writeTestConfig(t, content)

	cfg, err := LoadConfigFile(path)
	if err != nil {
		t.Fatalf("LoadConfigFile() error: %v", err)
	}

	if cfg.EnvironmentVars["OPENAI_API_KEY"] != "sk-from-config" {
		t.Errorf("env OPENAI_API_KEY = %q, want %q", cfg.EnvironmentVars["OPENAI_API_KEY"], "sk-from-config")
	}
	if cfg.EnvironmentVars["ANTHROPIC_API_KEY"] != "sk-ant-from-config" {
		t.Errorf("env ANTHROPIC_API_KEY = %q, want %q", cfg.EnvironmentVars["ANTHROPIC_API_KEY"], "sk-ant-from-config")
	}

	// Test ApplyEnvironmentVars
	cfg.ApplyEnvironmentVars()
	defer os.Unsetenv("OPENAI_API_KEY")
	defer os.Unsetenv("ANTHROPIC_API_KEY")

	if got := os.Getenv("OPENAI_API_KEY"); got != "sk-from-config" {
		t.Errorf("os.Getenv(OPENAI_API_KEY) = %q, want %q", got, "sk-from-config")
	}
}

func TestLoadConfigFile_WithComments(t *testing.T) {
	content := `# This is a config file
model_list:
  # OpenAI models
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      # api_key: this-is-commented-out
`
	path := writeTestConfig(t, content)

	cfg, err := LoadConfigFile(path)
	if err != nil {
		t.Fatalf("LoadConfigFile() error: %v", err)
	}

	if len(cfg.ModelList) != 1 {
		t.Fatalf("expected 1 model, got %d", len(cfg.ModelList))
	}
	if cfg.ModelList[0].LiteLLMParams.APIKey != "" {
		t.Errorf("commented api_key should not be parsed, got %q", cfg.ModelList[0].LiteLLMParams.APIKey)
	}
}

func TestLoadConfigFile_AzureModel(t *testing.T) {
	content := `model_list:
  - model_name: gpt-4-azure
    litellm_params:
      model: azure/my-gpt4-deployment
      api_key: azure-key-123
      api_base: https://my-resource.openai.azure.com
      api_version: 2024-02-15-preview
`
	path := writeTestConfig(t, content)

	cfg, err := LoadConfigFile(path)
	if err != nil {
		t.Fatalf("LoadConfigFile() error: %v", err)
	}

	m := cfg.ModelList[0]
	if m.LiteLLMParams.APIBase != "https://my-resource.openai.azure.com" {
		t.Errorf("api_base = %q, want azure URL", m.LiteLLMParams.APIBase)
	}
	if m.LiteLLMParams.APIVersion != "2024-02-15-preview" {
		t.Errorf("api_version = %q, want %q", m.LiteLLMParams.APIVersion, "2024-02-15-preview")
	}
}

func TestLoadConfigFile_FileNotFound(t *testing.T) {
	_, err := LoadConfigFile("/nonexistent/config.yaml")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestLoadConfigFile_LoadBalancer(t *testing.T) {
	content := `model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
      api_key: sk-key1
    tpm: 20000
    rpm: 3
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
      api_key: sk-key2
    tpm: 20000
    rpm: 3

router_settings:
  routing_strategy: simple-shuffle
  num_retries: 3
`
	path := writeTestConfig(t, content)

	cfg, err := LoadConfigFile(path)
	if err != nil {
		t.Fatalf("LoadConfigFile() error: %v", err)
	}

	if len(cfg.ModelList) != 2 {
		t.Fatalf("expected 2 models, got %d", len(cfg.ModelList))
	}

	// Both should have same model_name (for load balancing)
	if cfg.ModelList[0].ModelName != cfg.ModelList[1].ModelName {
		t.Error("load-balanced models should have same model_name")
	}
	if cfg.ModelList[0].LiteLLMParams.APIKey == cfg.ModelList[1].LiteLLMParams.APIKey {
		t.Error("load-balanced models should have different api_keys")
	}
}

func writeTestConfig(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write test config: %v", err)
	}
	return path
}
