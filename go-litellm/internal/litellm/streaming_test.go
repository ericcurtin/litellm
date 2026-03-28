package litellm

import (
	"bytes"
	"context"
	"io"
	"strings"
	"testing"
)

func TestParseSSEStream(t *testing.T) {
	sseData := `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1234,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1234,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1234,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1234,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`

	body := io.NopCloser(bytes.NewBufferString(sseData))
	_, cancel := context.WithCancel(context.Background())
	stream := NewStreamForProvider(cancel)

	go ParseSSEStream(body, stream, "gpt-4")

	var chunks []*StreamChunk
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 4 {
		t.Fatalf("expected 4 chunks, got %d", len(chunks))
	}

	// First chunk has role
	if chunks[0].Choices[0].Delta.Role != "assistant" {
		t.Errorf("expected role 'assistant' in first chunk")
	}

	// Content chunks
	if chunks[1].Choices[0].Delta.Content != "Hello" {
		t.Errorf("expected 'Hello', got %q", chunks[1].Choices[0].Delta.Content)
	}
	if chunks[2].Choices[0].Delta.Content != " world" {
		t.Errorf("expected ' world', got %q", chunks[2].Choices[0].Delta.Content)
	}

	// Final chunk has finish_reason
	if chunks[3].Choices[0].FinishReason != "stop" {
		t.Errorf("expected finish_reason 'stop', got %q", chunks[3].Choices[0].FinishReason)
	}
}

func TestParseSSEStream_EmptyLinesAndComments(t *testing.T) {
	sseData := `: this is a comment
data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}

: another comment

data: [DONE]

`

	body := io.NopCloser(strings.NewReader(sseData))
	_, cancel := context.WithCancel(context.Background())
	stream := NewStreamForProvider(cancel)

	go ParseSSEStream(body, stream, "m")

	chunk, err := stream.Recv()
	if err != nil {
		t.Fatal(err)
	}
	if chunk.Choices[0].Delta.Content != "ok" {
		t.Errorf("expected 'ok', got %q", chunk.Choices[0].Delta.Content)
	}

	// Should get EOF after [DONE]
	_, err = stream.Recv()
	if err != io.EOF {
		t.Errorf("expected EOF, got %v", err)
	}
}

func TestStream_Close(t *testing.T) {
	_, cancel := context.WithCancel(context.Background())
	stream := NewStreamForProvider(cancel)

	// Close an empty stream
	go stream.CloseStream(nil)

	_, err := stream.Recv()
	if err != io.EOF {
		t.Errorf("expected EOF after close, got %v", err)
	}
}
