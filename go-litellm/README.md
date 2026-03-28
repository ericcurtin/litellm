# go-litellm

A Go implementation of the core [LiteLLM](https://github.com/BerriAI/litellm) features — a unified interface for multiple LLM providers with consistent OpenAI-format output.

## Features

- **Unified API** — Single interface for OpenAI, Anthropic, and Azure OpenAI
- **Streaming** — Full SSE streaming support with automatic format translation
- **Router** — Load balancing across multiple deployments with retry and fallback
- **Proxy Server** — OpenAI-compatible HTTP proxy with authentication
- **Tool Calling** — Function/tool calling support across providers
- **Type Safety** — Strongly typed request/response models

## Installation

```bash
go get github.com/ericcurtin/litellm/go-litellm
```

## Quick Start

### Basic Completion

```go
package main

import (
    "context"
    "fmt"
    "os"

    litellm "github.com/ericcurtin/litellm/go-litellm"
    "github.com/ericcurtin/litellm/go-litellm/providers"
)

func main() {
    client := litellm.NewClient()
    client.RegisterProvider("openai", providers.NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), ""))

    resp, err := client.Complete(context.Background(), litellm.CompletionRequest{
        Model:    "openai/gpt-4",
        Messages: []litellm.Message{
            {Role: "user", Content: litellm.StringContent("Hello!")},
        },
    })
    if err != nil {
        panic(err)
    }
    fmt.Println(resp.Choices[0].Message.Content.Text)
}
```

### Streaming

```go
stream, err := client.CompleteStream(ctx, litellm.CompletionRequest{
    Model:    "openai/gpt-4",
    Messages: []litellm.Message{
        {Role: "user", Content: litellm.StringContent("Tell me a story")},
    },
})
if err != nil {
    panic(err)
}
defer stream.Close()

for {
    chunk, err := stream.Recv()
    if err == io.EOF {
        break
    }
    if err != nil {
        panic(err)
    }
    fmt.Print(chunk.Choices[0].Delta.Content)
}
```

### Multiple Providers

```go
client := litellm.NewClient()

// Register providers
client.RegisterProvider("openai", providers.NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), ""))
client.RegisterProvider("anthropic", providers.NewAnthropicProvider(os.Getenv("ANTHROPIC_API_KEY"), ""))
client.RegisterProvider("azure", providers.NewAzureProvider(
    os.Getenv("AZURE_API_KEY"),
    os.Getenv("AZURE_API_BASE"),
    "2024-02-15-preview",
))

// Use any provider with the same API
resp, _ := client.Complete(ctx, litellm.CompletionRequest{
    Model:    "anthropic/claude-3-5-sonnet-20241022",
    Messages: []litellm.Message{
        {Role: "user", Content: litellm.StringContent("Hello!")},
    },
})
```

### Router with Load Balancing

```go
openai := providers.NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), "")

deployments := []litellm.Deployment{
    {
        ModelName:    "gpt",
        LiteLLMModel: "gpt-4",
        Provider:     openai,
        Weight:       2, // Gets 2x traffic
    },
    {
        ModelName:    "gpt",
        LiteLLMModel: "gpt-3.5-turbo",
        Provider:     openai,
        Weight:       1,
    },
}

cfg := litellm.DefaultRouterConfig()
cfg.Strategy = litellm.StrategyShuffle        // or StrategyLeastBusy, StrategyRoundRobin, StrategyLatencyBased
cfg.NumRetries = 3
cfg.Fallbacks = map[string][]string{
    "gpt": {"claude"},                         // Fallback to Claude if all GPT deployments fail
}

router := litellm.NewRouter(cfg, deployments)

resp, err := router.Complete(ctx, litellm.CompletionRequest{
    Model:    "gpt",  // Routes to one of the gpt deployments
    Messages: messages,
})
```

### Proxy Server

```go
package main

import (
    "log"

    litellm "github.com/ericcurtin/litellm/go-litellm"
    "github.com/ericcurtin/litellm/go-litellm/providers"
    "github.com/ericcurtin/litellm/go-litellm/proxy"
)

func main() {
    client := litellm.NewClient()
    client.RegisterProvider("openai", providers.NewOpenAIProvider("sk-...", ""))

    server := proxy.NewServer(proxy.ServerConfig{
        Client:    client,
        MasterKey: "sk-proxy-key",  // Optional auth
    })

    log.Fatal(server.ListenAndServe(":4000"))
}
```

Then use it like any OpenAI-compatible API:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-proxy-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Or run the built-in proxy CLI:

```bash
OPENAI_API_KEY=sk-... go run ./cmd/proxy --port 4000 --master-key sk-proxy-key
```

### Tool Calling

```go
resp, err := client.Complete(ctx, litellm.CompletionRequest{
    Model: "openai/gpt-4",
    Messages: []litellm.Message{
        {Role: "user", Content: litellm.StringContent("What's the weather in SF?")},
    },
    Tools: []litellm.Tool{{
        Type: "function",
        Function: litellm.ToolFunction{
            Name:        "get_weather",
            Description: "Get current weather",
            Parameters: map[string]interface{}{
                "type": "object",
                "properties": map[string]interface{}{
                    "location": map[string]interface{}{
                        "type": "string",
                    },
                },
            },
        },
    }},
})

if len(resp.Choices[0].Message.ToolCalls) > 0 {
    tc := resp.Choices[0].Message.ToolCalls[0]
    fmt.Printf("Function: %s, Args: %s\n", tc.Function.Name, tc.Function.Arguments)
}
```

## Proxy Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming supported) |
| `/v1/embeddings` | POST | Text embeddings |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

## Routing Strategies

| Strategy | Description |
|----------|-------------|
| `simple-shuffle` | Weighted random selection (default) |
| `least-busy` | Route to deployment with fewest active requests |
| `round-robin` | Cycle through deployments sequentially |
| `latency-based` | Route to deployment with lowest average latency |

## Architecture

```
go-litellm/
├── client.go          # Main LiteLLM client
├── types.go           # OpenAI-compatible types
├── provider.go        # Provider interface
├── router.go          # Load balancing router
├── streaming.go       # SSE streaming support
├── config.go          # Configuration management
├── errors.go          # Error types
├── providers/
│   ├── openai.go      # OpenAI provider
│   ├── anthropic.go   # Anthropic (Claude) provider
│   └── azure.go       # Azure OpenAI provider
├── proxy/
│   └── server.go      # HTTP proxy server
└── cmd/
    ├── proxy/         # Proxy CLI
    └── example/       # Usage examples
```

## Running Tests

```bash
cd go-litellm
go test ./...
```

## Building

```bash
# Build the proxy server
go build -o litellm-proxy ./cmd/proxy

# Run it
OPENAI_API_KEY=sk-... ./litellm-proxy --port 4000
```
