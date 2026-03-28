# go-litellm

A Go implementation of the core [LiteLLM](https://github.com/BerriAI/litellm) features — an OpenAI-compatible proxy server that routes requests to multiple LLM providers.

## Features

- **Unified Proxy** — Single OpenAI-compatible API for OpenAI, Anthropic, and Azure OpenAI
- **Streaming** — Full SSE streaming support with automatic format translation
- **Router** — Load balancing across multiple deployments with retry and fallback
- **Authentication** — API key–based proxy authentication
- **Tool Calling** — Function/tool calling support across providers

## Building

```bash
cd go-litellm
go build -o go-litellm .
```

## Usage

```bash
# Set provider API keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Run the proxy
./go-litellm --port 4000 --master-key sk-proxy-key
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `4000` | Port to listen on |
| `--master-key` | (none) | API key for proxy authentication (also `LITELLM_MASTER_KEY` env) |
| `--router` | `false` | Enable router mode with load-balanced deployments |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `AZURE_API_KEY` | Azure OpenAI API key |
| `AZURE_API_BASE` | Azure OpenAI base URL |
| `AZURE_API_VERSION` | Azure OpenAI API version |
| `LITELLM_MASTER_KEY` | Proxy master key (alternative to `--master-key`) |

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming supported) |
| `/v1/embeddings` | POST | Text embeddings |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

## Example Requests

### Chat Completion

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-proxy-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-proxy-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### Embeddings

```bash
curl http://localhost:4000/v1/embeddings \
  -H "Authorization: Bearer sk-proxy-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/text-embedding-3-small",
    "input": "Hello world"
  }'
```

### List Models

```bash
curl http://localhost:4000/v1/models \
  -H "Authorization: Bearer sk-proxy-key"
```

## Routing Strategies

When running with `--router`, the proxy load-balances across multiple deployments:

| Strategy | Description |
|----------|-------------|
| `simple-shuffle` | Weighted random selection (default) |
| `least-busy` | Route to deployment with fewest active requests |
| `round-robin` | Cycle through deployments sequentially |
| `latency-based` | Route to deployment with lowest average latency |

## Architecture

```
go-litellm/
├── main.go                        # Binary entry point (proxy server CLI)
├── internal/
│   ├── litellm/
│   │   ├── client.go              # Core LiteLLM client
│   │   ├── types.go               # OpenAI-compatible types
│   │   ├── provider.go            # Provider interface
│   │   ├── router.go              # Load balancing router
│   │   ├── streaming.go           # SSE streaming support
│   │   ├── config.go              # Configuration management
│   │   └── errors.go              # Error types
│   ├── providers/
│   │   ├── openai.go              # OpenAI provider
│   │   ├── anthropic.go           # Anthropic (Claude) provider
│   │   └── azure.go               # Azure OpenAI provider
│   └── proxy/
│       └── server.go              # HTTP proxy server
└── go.mod
```

## Running Tests

```bash
cd go-litellm
go test ./...
```
