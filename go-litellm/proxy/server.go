package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	litellm "github.com/ericcurtin/litellm/go-litellm"
)

// Server is an OpenAI-compatible proxy server that routes requests through LiteLLM.
type Server struct {
	client    *litellm.Client
	router    *litellm.Router
	masterKey string
	mux       *http.ServeMux
}

// ServerConfig configures the proxy server.
type ServerConfig struct {
	// Client is the LiteLLM client (used if Router is nil).
	Client *litellm.Client

	// Router is the LiteLLM router for load balancing (takes precedence over Client).
	Router *litellm.Router

	// MasterKey is the API key required for authentication (empty = no auth).
	MasterKey string
}

// NewServer creates a new proxy server.
func NewServer(cfg ServerConfig) *Server {
	s := &Server{
		client:    cfg.Client,
		router:    cfg.Router,
		masterKey: cfg.MasterKey,
		mux:       http.NewServeMux(),
	}

	s.registerRoutes()
	return s
}

func (s *Server) registerRoutes() {
	s.mux.HandleFunc("/v1/chat/completions", s.authMiddleware(s.handleChatCompletions))
	s.mux.HandleFunc("/v1/completions", s.authMiddleware(s.handleChatCompletions))
	s.mux.HandleFunc("/v1/embeddings", s.authMiddleware(s.handleEmbeddings))
	s.mux.HandleFunc("/v1/models", s.authMiddleware(s.handleModels))
	s.mux.HandleFunc("/health", s.handleHealth)
	s.mux.HandleFunc("/health/readiness", s.handleHealth)
	s.mux.HandleFunc("/", s.handleRoot)
}

// ServeHTTP implements http.Handler.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

// ListenAndServe starts the proxy server on the given address.
func (s *Server) ListenAndServe(addr string) error {
	log.Printf("[litellm-proxy] Starting on %s", addr)
	server := &http.Server{
		Addr:         addr,
		Handler:      s,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 5 * time.Minute,
		IdleTimeout:  120 * time.Second,
	}
	return server.ListenAndServe()
}

// authMiddleware validates the API key if MasterKey is set.
func (s *Server) authMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.masterKey != "" {
			auth := r.Header.Get("Authorization")
			key := strings.TrimPrefix(auth, "Bearer ")
			if key != s.masterKey {
				writeError(w, http.StatusUnauthorized, "invalid_api_key", "Invalid API key")
				return
			}
		}
		next(w, r)
	}
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only POST is allowed")
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "Failed to read request body")
		return
	}
	defer r.Body.Close()

	var req litellm.CompletionRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "Invalid JSON: "+err.Error())
		return
	}

	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "invalid_request", "model is required")
		return
	}

	ctx := r.Context()

	if req.Stream {
		s.handleStreamingCompletion(ctx, w, req)
		return
	}

	s.handleNonStreamingCompletion(ctx, w, req)
}

func (s *Server) handleNonStreamingCompletion(ctx context.Context, w http.ResponseWriter, req litellm.CompletionRequest) {
	var resp *litellm.CompletionResponse
	var err error

	if s.router != nil {
		resp, err = s.router.Complete(ctx, req)
	} else if s.client != nil {
		resp, err = s.client.Complete(ctx, req)
	} else {
		writeError(w, http.StatusInternalServerError, "server_error", "No client or router configured")
		return
	}

	if err != nil {
		handleProviderError(w, err)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleStreamingCompletion(ctx context.Context, w http.ResponseWriter, req litellm.CompletionRequest) {
	var stream *litellm.Stream
	var err error

	if s.router != nil {
		stream, err = s.router.CompleteStream(ctx, req)
	} else if s.client != nil {
		stream, err = s.client.CompleteStream(ctx, req)
	} else {
		writeError(w, http.StatusInternalServerError, "server_error", "No client or router configured")
		return
	}

	if err != nil {
		handleProviderError(w, err)
		return
	}
	defer stream.Close()

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "server_error", "Streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Transfer-Encoding", "chunked")

	for {
		chunk, err := stream.Recv()
		if err != nil {
			if err == io.EOF {
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
				return
			}
			// Write error as SSE event
			errData, _ := json.Marshal(map[string]interface{}{
				"error": map[string]interface{}{
					"message": err.Error(),
					"type":    "stream_error",
				},
			})
			fmt.Fprintf(w, "data: %s\n\n", errData)
			flusher.Flush()
			return
		}

		data, err := json.Marshal(chunk)
		if err != nil {
			continue
		}

		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
}

func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only POST is allowed")
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "Failed to read request body")
		return
	}
	defer r.Body.Close()

	var req litellm.EmbeddingRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "Invalid JSON: "+err.Error())
		return
	}

	ctx := r.Context()
	var resp *litellm.EmbeddingResponse

	if s.router != nil {
		resp, err = s.router.Embed(ctx, req)
	} else if s.client != nil {
		resp, err = s.client.Embed(ctx, req)
	} else {
		writeError(w, http.StatusInternalServerError, "server_error", "No client or router configured")
		return
	}

	if err != nil {
		handleProviderError(w, err)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only GET is allowed")
		return
	}

	ctx := r.Context()
	var models []litellm.ModelInfo

	if s.client != nil {
		var err error
		models, err = s.client.Models(ctx)
		if err != nil {
			handleProviderError(w, err)
			return
		}
	}

	resp := litellm.ModelListResponse{
		Object: "list",
		Data:   models,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "healthy",
	})
}

func (s *Server) handleRoot(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"message": "LiteLLM Proxy Server (Go)",
		"docs":    "/v1/models",
		"health":  "/health",
	})
}

// writeError sends a JSON error response in OpenAI format.
func writeError(w http.ResponseWriter, status int, errType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"error": map[string]interface{}{
			"message": message,
			"type":    errType,
			"code":    status,
		},
	})
}

// handleProviderError converts a provider error to an HTTP response.
func handleProviderError(w http.ResponseWriter, err error) {
	if apiErr, ok := err.(*litellm.APIError); ok {
		writeError(w, apiErr.StatusCode, apiErr.Type, apiErr.Message)
		return
	}
	writeError(w, http.StatusInternalServerError, "internal_error", err.Error())
}
