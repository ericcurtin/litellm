package litellm

import (
	"os"
	"strings"
)

// Config holds global configuration for the LiteLLM client.
type Config struct {
	// Default API keys per provider (override with env vars or per-request)
	OpenAIKey    string
	AnthropicKey string
	AzureKey     string

	// Azure-specific
	AzureAPIBase    string
	AzureAPIVersion string

	// Default settings
	DefaultTimeout  int     // seconds
	DefaultMaxRetry int
	Temperature     float64

	// Behavior
	DropParams bool // Drop unsupported params instead of erroring
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		DefaultTimeout:  300,
		DefaultMaxRetry: 2,
	}
}

// envKeyMap maps provider names to their environment variable names.
var envKeyMap = map[string]string{
	"openai":    "OPENAI_API_KEY",
	"anthropic": "ANTHROPIC_API_KEY",
	"azure":     "AZURE_API_KEY",
	"cohere":    "COHERE_API_KEY",
	"groq":      "GROQ_API_KEY",
	"mistral":   "MISTRAL_API_KEY",
	"together":  "TOGETHER_API_KEY",
	"perplexity": "PERPLEXITYAI_API_KEY",
	"deepseek":  "DEEPSEEK_API_KEY",
	"ollama":    "",
}

// envBaseURLMap maps provider names to their base URL environment variables.
var envBaseURLMap = map[string]string{
	"openai":    "OPENAI_API_BASE",
	"azure":     "AZURE_API_BASE",
	"ollama":    "OLLAMA_API_BASE",
}

// ResolveAPIKey resolves the API key for a provider using the priority:
// 1. Explicit parameter
// 2. Config value
// 3. Environment variable
func ResolveAPIKey(explicit string, provider string, cfg Config) string {
	if explicit != "" {
		return explicit
	}

	// Check config
	switch provider {
	case "openai":
		if cfg.OpenAIKey != "" {
			return cfg.OpenAIKey
		}
	case "anthropic":
		if cfg.AnthropicKey != "" {
			return cfg.AnthropicKey
		}
	case "azure":
		if cfg.AzureKey != "" {
			return cfg.AzureKey
		}
	}

	// Check environment variable
	if envKey, ok := envKeyMap[provider]; ok && envKey != "" {
		if val := os.Getenv(envKey); val != "" {
			return val
		}
	}

	return ""
}

// ResolveBaseURL resolves the base URL for a provider.
func ResolveBaseURL(explicit string, provider string, cfg Config) string {
	if explicit != "" {
		return explicit
	}

	// Check config for Azure
	if provider == "azure" && cfg.AzureAPIBase != "" {
		return cfg.AzureAPIBase
	}

	// Check environment variable
	if envKey, ok := envBaseURLMap[provider]; ok && envKey != "" {
		if val := os.Getenv(envKey); val != "" {
			return val
		}
	}

	return ""
}

// ParseModelProvider extracts the provider and model name from a combined
// model string. Format: "provider/model-name" or just "model-name".
// If no provider prefix is given, the provider is inferred from the model name.
func ParseModelProvider(model string) (provider, modelName string) {
	// Check for explicit provider prefix
	if idx := strings.Index(model, "/"); idx > 0 {
		prefix := model[:idx]
		rest := model[idx+1:]

		// Known provider prefixes
		switch prefix {
		case "openai", "anthropic", "azure", "ollama", "groq",
			"mistral", "together", "perplexity", "cohere", "deepseek":
			return prefix, rest
		}

		// Azure format: azure/deployment-name
		if prefix == "azure" {
			return "azure", rest
		}
	}

	// Infer provider from model name
	lower := strings.ToLower(model)
	switch {
	case strings.HasPrefix(lower, "gpt-") ||
		strings.HasPrefix(lower, "o1-") ||
		strings.HasPrefix(lower, "o3-") ||
		strings.HasPrefix(lower, "chatgpt") ||
		strings.HasPrefix(lower, "text-embedding") ||
		strings.HasPrefix(lower, "text-davinci"):
		return "openai", model
	case strings.HasPrefix(lower, "claude"):
		return "anthropic", model
	case strings.Contains(lower, "llama") && !strings.Contains(lower, "together"):
		return "ollama", model
	case strings.HasPrefix(lower, "mistral") || strings.HasPrefix(lower, "mixtral"):
		return "mistral", model
	default:
		return "openai", model // Default to OpenAI
	}
}
