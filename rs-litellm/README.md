# rs-litellm

A high-performance LLM proxy gateway written in Rust. Provides an OpenAI-compatible API that routes requests to 30+ LLM providers with load balancing, retries, fallbacks, and authentication.

## Features

- **OpenAI-compatible API** — Drop-in replacement for the OpenAI API (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`)
- **Multi-provider support** — OpenAI, Anthropic, Azure, Together AI, Groq, Mistral, DeepSeek, Fireworks AI, OpenRouter, Perplexity, xAI, NVIDIA NIM, Ollama, vLLM, LM Studio, and more
- **Load balancing** — Round-robin across multiple deployments of the same model
- **Retries & fallbacks** — Automatic retries across deployments and fallback to alternative models
- **Streaming** — Full SSE streaming support for chat completions (including Anthropic-to-OpenAI stream translation)
- **Authentication** — API key-based authentication via `master_key`
- **YAML configuration** — Compatible with LiteLLM Python config format
- **Environment variable resolution** — Supports `os.environ/VAR_NAME` and `${VAR_NAME}` syntax in configs
- **Lightweight** — Single static binary, ~5MB compiled, minimal memory footprint

## Quick Start

### Build

```bash
cd rs-litellm
cargo build --release
```

The binary will be at `target/release/rs-litellm`.

### Configure

Create a config file (e.g., `config.yaml`):

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: llama-3
    litellm_params:
      model: together_ai/meta-llama/Llama-3-70b-chat-hf
      api_key: os.environ/TOGETHER_API_KEY

general_settings:
  master_key: sk-my-secret-key
  num_retries: 2

litellm_settings:
  drop_params: true
```

### Run

```bash
# Set API keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Start the proxy
./target/release/rs-litellm --config config.yaml --port 4000
```

### Use

```bash
# Chat completion
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet",
    "messages": [{"role": "user", "content": "Write a haiku about Rust"}],
    "stream": true
  }'

# List models
curl http://localhost:4000/v1/models \
  -H "Authorization: Bearer sk-my-secret-key"

# Health check (no auth required)
curl http://localhost:4000/health
```

## Configuration

### Model List

Each entry in `model_list` defines a deployment:

```yaml
model_list:
  - model_name: gpt-3.5-turbo        # The name clients use
    litellm_params:
      model: openai/gpt-3.5-turbo    # provider/actual_model_name
      api_key: sk-...                 # API key (or use env vars)
      api_base: https://...           # Custom API base URL (optional)
      api_version: "2024-02-01"       # API version (Azure)
      timeout: 30.0                   # Request timeout in seconds
    tpm: 20000                        # Tokens per minute limit (informational)
    rpm: 3                            # Requests per minute limit (informational)
```

### Load Balancing

Add multiple entries with the same `model_name` for load balancing:

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: sk-key-1
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: sk-key-2
```

### Fallbacks

Configure fallback models:

```yaml
general_settings:
  fallbacks:
    - gpt-4o: [claude-3-5-sonnet, gpt-3.5-turbo]
```

### Authentication

Set `master_key` to require API key authentication:

```yaml
general_settings:
  master_key: sk-my-secret-key
```

Clients must include `Authorization: Bearer sk-my-secret-key` or `x-api-key: sk-my-secret-key`.

### Provider Reference

| Provider | Model prefix | Default API base | Env var |
|----------|-------------|------------------|---------|
| OpenAI | `openai/` | `api.openai.com` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/` | `api.anthropic.com` | `ANTHROPIC_API_KEY` |
| Azure OpenAI | `azure/` | (requires `api_base`) | `AZURE_API_KEY` |
| Together AI | `together_ai/` | `api.together.xyz` | `TOGETHER_API_KEY` |
| Groq | `groq/` | `api.groq.com` | `GROQ_API_KEY` |
| DeepSeek | `deepseek/` | `api.deepseek.com` | `DEEPSEEK_API_KEY` |
| Fireworks AI | `fireworks_ai/` | `api.fireworks.ai` | `FIREWORKS_AI_API_KEY` |
| Mistral | `mistral/` | `api.mistral.ai` | `MISTRAL_API_KEY` |
| OpenRouter | `openrouter/` | `openrouter.ai` | `OPENROUTER_API_KEY` |
| Perplexity | `perplexity/` | `api.perplexity.ai` | `PERPLEXITY_API_KEY` |
| xAI | `xai/` | `api.x.ai` | `XAI_API_KEY` |
| NVIDIA NIM | `nvidia_nim/` | `integrate.api.nvidia.com` | `NVIDIA_API_KEY` |
| Cerebras | `cerebras/` | `api.cerebras.ai` | `CEREBRAS_API_KEY` |
| SambaNova | `sambanova/` | `api.sambanova.ai` | `SAMBANOVA_API_KEY` |
| DeepInfra | `deepinfra/` | `api.deepinfra.com` | `DEEPINFRA_API_KEY` |
| Ollama | `ollama/` | `localhost:11434` | — |
| vLLM | `vllm/` | `localhost:8000` | — |
| LM Studio | `lm_studio/` | `localhost:1234` | — |

### CLI Options

```
rs-litellm --config <config.yaml> [OPTIONS]

Options:
  -c, --config <PATH>    Path to YAML config file (required)
      --host <HOST>      Host to bind to [default: 0.0.0.0]
  -p, --port <PORT>      Port to listen on [default: 4000]
  -v, --verbose          Enable debug logging
  -h, --help             Print help
  -V, --version          Print version
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion (streaming + non-streaming) |
| `POST` | `/chat/completions` | Chat completion (alias) |
| `POST` | `/v1/embeddings` | Create embeddings |
| `POST` | `/embeddings` | Create embeddings (alias) |
| `GET` | `/v1/models` | List available models |
| `GET` | `/models` | List models (alias) |
| `GET` | `/health` | Health check (no auth) |
| `GET` | `/` | Health check (alias) |

## OpenAI SDK Compatibility

Use any OpenAI SDK by pointing it at rs-litellm:

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-my-secret-key",
    base_url="http://localhost:4000/v1"
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
    apiKey: 'sk-my-secret-key',
    baseURL: 'http://localhost:4000/v1',
});

const response = await client.chat.completions.create({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: 'Hello!' }],
});
```

## License

MIT
