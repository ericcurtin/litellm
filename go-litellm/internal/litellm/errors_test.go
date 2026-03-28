package litellm

import (
	"testing"
)

func TestAPIError_Error(t *testing.T) {
	err := &APIError{
		StatusCode: 429,
		Message:    "rate limit exceeded",
		Provider:   "openai",
	}
	got := err.Error()
	want := "litellm.openai: rate limit exceeded (status 429)"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	// Without provider
	err2 := &APIError{
		StatusCode: 500,
		Message:    "internal error",
	}
	got2 := err2.Error()
	want2 := "litellm: internal error (status 500)"
	if got2 != want2 {
		t.Errorf("got %q, want %q", got2, want2)
	}
}

func TestAPIError_IsRetryable(t *testing.T) {
	tests := []struct {
		status int
		want   bool
	}{
		{429, true},
		{500, true},
		{502, true},
		{503, true},
		{504, true},
		{400, false},
		{401, false},
		{403, false},
		{404, false},
	}

	for _, tt := range tests {
		err := &APIError{StatusCode: tt.status}
		if got := err.IsRetryable(); got != tt.want {
			t.Errorf("APIError{StatusCode: %d}.IsRetryable() = %v, want %v", tt.status, got, tt.want)
		}
	}
}

func TestAPIError_IsContextWindowExceeded(t *testing.T) {
	err := &APIError{StatusCode: 400, Type: "context_length_exceeded"}
	if !err.IsContextWindowExceeded() {
		t.Error("expected true for context_length_exceeded")
	}

	err2 := &APIError{StatusCode: 400, Type: "invalid_request"}
	if err2.IsContextWindowExceeded() {
		t.Error("expected false for invalid_request")
	}
}

func TestAPIError_IsAuthError(t *testing.T) {
	if !(&APIError{StatusCode: 401}).IsAuthError() {
		t.Error("expected 401 to be auth error")
	}
	if !(&APIError{StatusCode: 403}).IsAuthError() {
		t.Error("expected 403 to be auth error")
	}
	if (&APIError{StatusCode: 400}).IsAuthError() {
		t.Error("expected 400 to not be auth error")
	}
}
