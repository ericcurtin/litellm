// Command go-litellm runs an OpenAI-compatible LLM proxy server.
//
// Usage:
//
//	go run . --port 8080 --master-key sk-my-key
//	go run . --config config.yaml
//
// Environment variables:
//
//	OPENAI_API_KEY    - OpenAI API key
//	ANTHROPIC_API_KEY - Anthropic API key
//	AZURE_API_KEY     - Azure OpenAI API key
//	AZURE_API_BASE    - Azure OpenAI base URL
//	LITELLM_MASTER_KEY - Proxy master key for authentication
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	litellm "github.com/ericcurtin/litellm/go-litellm/internal/litellm"
	"github.com/ericcurtin/litellm/go-litellm/internal/providers"
	"github.com/ericcurtin/litellm/go-litellm/internal/proxy"
)

func main() {
	port := flag.Int("port", 4000, "Port to listen on")
	masterKey := flag.String("master-key", "", "API key for proxy authentication")
	enableRouter := flag.Bool("router", false, "Enable router mode with multiple deployments")
	configFile := flag.String("config", "", "Path to YAML configuration file")
	flag.Parse()

	if *masterKey == "" {
		*masterKey = os.Getenv("LITELLM_MASTER_KEY")
	}

	// If a config file is provided, load and apply it
	if *configFile != "" {
		runFromConfig(*configFile, *port, *masterKey)
		return
	}

	// Create client and register providers from environment variables
	client := litellm.NewClient()
	registerProvidersFromEnv(client)

	serverCfg := proxy.ServerConfig{
		Client:    client,
		MasterKey: *masterKey,
	}

	// Optionally set up router with load balancing
	if *enableRouter {
		deployments := buildDeployments()
		if len(deployments) > 0 {
			routerCfg := litellm.DefaultRouterConfig()
			routerCfg.EnableLogging = true
			router := litellm.NewRouter(routerCfg, deployments)
			serverCfg.Router = router
			log.Printf("[litellm-proxy] Router enabled with %d deployments", len(deployments))
		}
	}

	startServer(serverCfg, *port)
}

// runFromConfig loads the given YAML config file and starts the server.
func runFromConfig(path string, port int, masterKey string) {
	fileCfg, err := litellm.LoadConfigFile(path)
	if err != nil {
		log.Fatalf("Failed to load config file: %v", err)
	}
	log.Printf("[litellm-proxy] Loaded config from %s", path)

	// Apply environment variables from config file first
	fileCfg.ApplyEnvironmentVars()

	// Master key: flag > config file > env
	if masterKey == "" {
		masterKey = fileCfg.GeneralSettings.MasterKey
	}
	if masterKey == "" {
		masterKey = os.Getenv("LITELLM_MASTER_KEY")
	}

	client := litellm.NewClient()

	// Build deployments from model_list
	deployments := buildDeploymentsFromConfig(fileCfg, client)

	serverCfg := proxy.ServerConfig{
		Client:    client,
		MasterKey: masterKey,
	}

	// Set up router if there are multiple deployments for any model group
	if len(deployments) > 0 {
		routerCfg := litellm.DefaultRouterConfig()
		routerCfg.EnableLogging = true

		// Apply router settings from config
		if s := fileCfg.RouterSettings.RoutingStrategy; s != "" {
			routerCfg.Strategy = litellm.RoutingStrategy(s)
		}
		if n := fileCfg.RouterSettings.NumRetries; n > 0 {
			routerCfg.NumRetries = n
		}

		router := litellm.NewRouter(routerCfg, deployments)
		serverCfg.Router = router
		log.Printf("[litellm-proxy] Router enabled with %d deployments from config", len(deployments))
	}

	startServer(serverCfg, port)
}

// buildDeploymentsFromConfig creates deployments and registers providers from a config file.
func buildDeploymentsFromConfig(fileCfg *litellm.FileConfig, client *litellm.Client) []litellm.Deployment {
	var deployments []litellm.Deployment
	registeredProviders := make(map[string]bool)

	for _, entry := range fileCfg.ModelList {
		providerName, modelName := litellm.ParseModelProvider(entry.LiteLLMParams.Model)

		// Resolve the API key: entry-level > environment variable
		apiKey := entry.LiteLLMParams.APIKey
		if apiKey == "" {
			apiKey = litellm.ResolveAPIKey("", providerName, litellm.DefaultConfig())
		}

		// Create and register provider if not already registered
		if !registeredProviders[providerName] {
			provider := createProvider(providerName, apiKey, entry.LiteLLMParams.APIBase, entry.LiteLLMParams.APIVersion)
			if provider != nil {
				client.RegisterProvider(providerName, provider)
				registeredProviders[providerName] = true
				log.Printf("[litellm-proxy] Registered %s provider from config", providerName)
			}
		}

		provider := client.GetProvider(providerName)
		if provider == nil {
			log.Printf("[litellm-proxy] Warning: could not create provider %q for model %q", providerName, entry.ModelName)
			continue
		}

		dep := litellm.Deployment{
			ModelName:    entry.ModelName,
			LiteLLMModel: modelName,
			Provider:     provider,
			APIKey:       apiKey,
			BaseURL:      entry.LiteLLMParams.APIBase,
			TPMLimit:     entry.TPM,
			RPMLimit:     entry.RPM,
			Weight:       1,
		}
		deployments = append(deployments, dep)
	}

	return deployments
}

// createProvider creates a Provider instance for the given provider name.
func createProvider(name, apiKey, baseURL, apiVersion string) litellm.Provider {
	switch name {
	case "openai":
		return providers.NewOpenAIProvider(apiKey, baseURL)
	case "anthropic":
		return providers.NewAnthropicProvider(apiKey, baseURL)
	case "azure":
		return providers.NewAzureProvider(apiKey, baseURL, apiVersion)
	default:
		// For unknown providers, try OpenAI-compatible
		if baseURL != "" {
			return providers.NewOpenAIProvider(apiKey, strings.TrimSuffix(baseURL, "/"))
		}
		return nil
	}
}

// registerProvidersFromEnv registers providers based on environment variables.
func registerProvidersFromEnv(client *litellm.Client) {
	if key := os.Getenv("OPENAI_API_KEY"); key != "" {
		client.RegisterProvider("openai", providers.NewOpenAIProvider(key, ""))
		log.Println("[litellm-proxy] Registered OpenAI provider")
	}

	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" {
		client.RegisterProvider("anthropic", providers.NewAnthropicProvider(key, ""))
		log.Println("[litellm-proxy] Registered Anthropic provider")
	}

	if key := os.Getenv("AZURE_API_KEY"); key != "" {
		base := os.Getenv("AZURE_API_BASE")
		version := os.Getenv("AZURE_API_VERSION")
		client.RegisterProvider("azure", providers.NewAzureProvider(key, base, version))
		log.Println("[litellm-proxy] Registered Azure provider")
	}
}

// startServer starts the proxy HTTP server.
func startServer(serverCfg proxy.ServerConfig, port int) {
	server := proxy.NewServer(serverCfg)

	addr := fmt.Sprintf(":%d", port)
	log.Printf("[litellm-proxy] Server starting on http://0.0.0.0%s", addr)
	log.Printf("[litellm-proxy] Endpoints:")
	log.Printf("[litellm-proxy]   POST /v1/chat/completions")
	log.Printf("[litellm-proxy]   POST /v1/embeddings")
	log.Printf("[litellm-proxy]   GET  /v1/models")
	log.Printf("[litellm-proxy]   GET  /health")

	if err := server.ListenAndServe(addr); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

func buildDeployments() []litellm.Deployment {
	var deployments []litellm.Deployment

	if key := os.Getenv("OPENAI_API_KEY"); key != "" {
		openai := providers.NewOpenAIProvider(key, "")

		deployments = append(deployments, litellm.Deployment{
			ModelName:    "gpt-4",
			LiteLLMModel: "gpt-4",
			Provider:     openai,
			APIKey:       key,
			Weight:       1,
		})
		deployments = append(deployments, litellm.Deployment{
			ModelName:    "gpt-3.5-turbo",
			LiteLLMModel: "gpt-3.5-turbo",
			Provider:     openai,
			APIKey:       key,
			Weight:       1,
		})
	}

	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" {
		anthropic := providers.NewAnthropicProvider(key, "")

		deployments = append(deployments, litellm.Deployment{
			ModelName:    "claude-3-sonnet",
			LiteLLMModel: "claude-3-5-sonnet-20241022",
			Provider:     anthropic,
			APIKey:       key,
			Weight:       1,
		})
	}

	return deployments
}
