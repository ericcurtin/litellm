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

### Quick Start with Environment Variables

```bash
# Set provider API keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Run the proxy
./go-litellm --port 4000 --master-key sk-proxy-key
```

### Quick Start with a Configuration File

```bash
# Run the proxy using a YAML config file
./go-litellm --config config.yaml
```

See [Configuration File](#configuration-file) below for details.

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `4000` | Port to listen on |
| `--master-key` | (none) | API key for proxy authentication (also `LITELLM_MASTER_KEY` env) |
| `--router` | `false` | Enable router mode with load-balanced deployments |
| `--config` | (none) | Path to a YAML configuration file |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `AZURE_API_KEY` | Azure OpenAI API key |
| `AZURE_API_BASE` | Azure OpenAI base URL |
| `AZURE_API_VERSION` | Azure OpenAI API version |
| `LITELLM_MASTER_KEY` | Proxy master key (alternative to `--master-key`) |

## Configuration File

Instead of using environment variables and CLI flags, you can configure go-litellm with a YAML file. The format is compatible with the Python [LiteLLM proxy config](https://docs.litellm.ai/docs/proxy/configs).

Pass the file path with the `--config` flag:

```bash
./go-litellm --config config.yaml
```

### Config File Structure

A config file has five optional top-level sections:

```yaml
model_list:           # Required — defines the models the proxy serves
general_settings:     # Proxy-level settings (master key, etc.)
litellm_settings:     # Global LiteLLM behaviour
router_settings:      # Load-balancing / retry settings
environment_variables: # Env vars applied on startup
```

### Minimal Example

The simplest config just lists models. API keys come from environment variables:

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
  - model_name: claude-3-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
```

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
./go-litellm --config config.yaml
```

### Full Example

```yaml
# Models available through the proxy
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: sk-openai-key-here
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: sk-openai-key-here
  - model_name: claude-3-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: sk-ant-key-here
  - model_name: gpt-4-azure
    litellm_params:
      model: azure/my-gpt4-deployment
      api_key: azure-key-here
      api_base: https://my-resource.openai.azure.com
      api_version: 2024-02-15-preview

# Proxy authentication
general_settings:
  master_key: sk-my-proxy-key

# Global behaviour
litellm_settings:
  drop_params: True

# Router / load-balancing
router_settings:
  routing_strategy: simple-shuffle
  num_retries: 3

# Environment variables (applied on startup as defaults)
environment_variables:
  OPENAI_API_KEY: sk-...
  ANTHROPIC_API_KEY: sk-ant-...
```

### Load Balancing with Config Files

To load-balance across multiple deployments of the same model, give them the **same `model_name`**. The router is enabled automatically when a config file contains multiple deployments per model name:

```yaml
model_list:
  # Two OpenAI keys for gpt-3.5-turbo, load-balanced
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: sk-key-1
    tpm: 20000
    rpm: 3
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: sk-key-2
    tpm: 20000
    rpm: 3
  # Fallback to a different provider
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: sk-ant-key

router_settings:
  routing_strategy: least-busy
  num_retries: 3
```

### Config File Reference

#### `model_list` entries

| Field | Description |
|-------|-------------|
| `model_name` | The name clients use in the `model` request field |
| `litellm_params.model` | Provider/model string (e.g. `openai/gpt-4`, `anthropic/claude-3-5-sonnet-20241022`) |
| `litellm_params.api_key` | API key for this deployment (overrides env var) |
| `litellm_params.api_base` | Custom base URL (required for Azure, optional for others) |
| `litellm_params.api_version` | API version (Azure only) |
| `tpm` | Tokens-per-minute limit (optional, for router metrics) |
| `rpm` | Requests-per-minute limit (optional, for router metrics) |

#### `general_settings`

| Field | Description |
|-------|-------------|
| `master_key` | API key required to call the proxy |

#### `litellm_settings`

| Field | Description |
|-------|-------------|
| `drop_params` | Drop unsupported params instead of erroring (`True`/`False`) |

#### `router_settings`

| Field | Description |
|-------|-------------|
| `routing_strategy` | One of `simple-shuffle`, `least-busy`, `round-robin`, `latency-based` |
| `num_retries` | Number of retries on failure (default: 2) |

#### `environment_variables`

Key-value pairs that are set as environment variables on startup. Useful for keeping API keys out of the config file itself (set them in the shell instead) or for providing defaults:

```yaml
environment_variables:
  OPENAI_API_KEY: sk-...
  ANTHROPIC_API_KEY: sk-ant-...
```

### Priority Order

When the same setting can be specified in multiple places, the priority is:

1. **CLI flag** (e.g. `--master-key`)
2. **Config file** (`general_settings.master_key`)
3. **Environment variable** (`LITELLM_MASTER_KEY`)

For API keys within `model_list` entries:

1. **`litellm_params.api_key`** in the config file
2. **Environment variable** (e.g. `OPENAI_API_KEY`)

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
├── config.yaml                    # Example configuration file
├── internal/
│   ├── litellm/
│   │   ├── client.go              # Core LiteLLM client
│   │   ├── types.go               # OpenAI-compatible types
│   │   ├── provider.go            # Provider interface
│   │   ├── router.go              # Load balancing router
│   │   ├── streaming.go           # SSE streaming support
│   │   ├── config.go              # Configuration management
│   │   ├── configfile.go          # YAML config file parser
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
