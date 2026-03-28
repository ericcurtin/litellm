package litellm

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
)

// Stream wraps a streaming response from an LLM provider.
// It is safe to read from a single goroutine.
type Stream struct {
	ch     chan StreamEvent
	cancel context.CancelFunc
	done   chan struct{}
	err    error
	mu     sync.Mutex
}

// StreamEvent carries either a chunk or an error.
type StreamEvent struct {
	Chunk *StreamChunk
	Err   error
}

// newStream creates a new Stream.
func newStream(cancel context.CancelFunc) *Stream {
	return &Stream{
		ch:     make(chan StreamEvent, 64),
		cancel: cancel,
		done:   make(chan struct{}),
	}
}

// NewStreamForProvider creates a new Stream (exported for provider use).
func NewStreamForProvider(cancel context.CancelFunc) *Stream {
	return newStream(cancel)
}

// Recv receives the next streaming chunk. Returns nil, io.EOF when the stream ends.
func (s *Stream) Recv() (*StreamChunk, error) {
	evt, ok := <-s.ch
	if !ok {
		s.mu.Lock()
		err := s.err
		s.mu.Unlock()
		if err != nil {
			return nil, err
		}
		return nil, io.EOF
	}
	if evt.Err != nil {
		return nil, evt.Err
	}
	return evt.Chunk, nil
}

// Close closes the stream and releases resources.
func (s *Stream) Close() {
	s.cancel()
	// Drain remaining events with a safety check
	for {
		select {
		case _, ok := <-s.ch:
			if !ok {
				return
			}
		case <-s.done:
			return
		}
	}
}

// send sends an event to the stream channel.
func (s *Stream) send(evt StreamEvent) {
	s.ch <- evt
}

// SendEvent sends an event to the stream channel (exported for provider use).
func (s *Stream) SendEvent(evt StreamEvent) {
	s.send(evt)
}

// close closes the channel and records any final error.
func (s *Stream) close(err error) {
	s.mu.Lock()
	s.err = err
	s.mu.Unlock()
	close(s.ch)
	close(s.done)
}

// CloseStream closes the channel (exported for provider use).
func (s *Stream) CloseStream(err error) {
	s.close(err)
}

// ParseSSEStream reads an SSE stream from an HTTP response body and parses
// OpenAI-format chunks. It sends parsed chunks to the Stream.
func ParseSSEStream(body io.ReadCloser, stream *Stream, model string) {
	defer body.Close()
	defer stream.close(nil)

	scanner := bufio.NewScanner(body)
	// Increase buffer size for large responses
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()

		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}

		// Parse SSE data field
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		// Check for stream end
		if data == "[DONE]" {
			return
		}

		var chunk StreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			stream.send(StreamEvent{Err: fmt.Errorf("failed to parse chunk: %w", err)})
			return
		}

		if chunk.Model == "" {
			chunk.Model = model
		}

		stream.send(StreamEvent{Chunk: &chunk})
	}

	if err := scanner.Err(); err != nil {
		stream.send(StreamEvent{Err: fmt.Errorf("stream read error: %w", err)})
	}
}

// DoStreamRequest performs an HTTP request and returns a Stream that parses SSE events.
func DoStreamRequest(ctx context.Context, httpClient *http.Client, req *http.Request, model string) (*Stream, error) {
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("stream request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		return nil, &APIError{
			StatusCode: resp.StatusCode,
			Message:    string(body),
		}
	}

	streamCtx, cancel := context.WithCancel(ctx)
	stream := newStream(cancel)

	go func() {
		// Close body if context is cancelled
		go func() {
			<-streamCtx.Done()
			resp.Body.Close()
		}()
		ParseSSEStream(resp.Body, stream, model)
	}()

	return stream, nil
}
