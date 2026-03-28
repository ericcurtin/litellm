// Command go-litellm runs an OpenAI-compatible LLM proxy server.
//
// Usage:
//
//	go run . --port 8080 --master-key sk-my-key
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

	litellm "github.com/ericcurtin/litellm/go-litellm/internal/litellm"
	"github.com/ericcurtin/litellm/go-litellm/internal/providers"
	"github.com/ericcurtin/litellm/go-litellm/internal/proxy"
)

func main() {
	port := flag.Int("port", 4000, "Port to listen on")
	masterKey := flag.String("master-key", "", "API key for proxy authentication")
	enableRouter := flag.Bool("router", false, "Enable router mode with multiple deployments")
	flag.Parse()

	if *masterKey == "" {
		*masterKey = os.Getenv("LITELLM_MASTER_KEY")
	}

	// Create client and register providers
	client := litellm.NewClient()

	// Register OpenAI provider
	if key := os.Getenv("OPENAI_API_KEY"); key != "" {
		client.RegisterProvider("openai", providers.NewOpenAIProvider(key, ""))
		log.Println("[litellm-proxy] Registered OpenAI provider")
	}

	// Register Anthropic provider
	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" {
		client.RegisterProvider("anthropic", providers.NewAnthropicProvider(key, ""))
		log.Println("[litellm-proxy] Registered Anthropic provider")
	}

	// Register Azure provider
	if key := os.Getenv("AZURE_API_KEY"); key != "" {
		base := os.Getenv("AZURE_API_BASE")
		version := os.Getenv("AZURE_API_VERSION")
		client.RegisterProvider("azure", providers.NewAzureProvider(key, base, version))
		log.Println("[litellm-proxy] Registered Azure provider")
	}

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

	server := proxy.NewServer(serverCfg)

	addr := fmt.Sprintf(":%d", *port)
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
