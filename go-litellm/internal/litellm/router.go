package litellm

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// RoutingStrategy defines how the router selects deployments.
type RoutingStrategy string

const (
	StrategyShuffle        RoutingStrategy = "simple-shuffle"
	StrategyLeastBusy      RoutingStrategy = "least-busy"
	StrategyRoundRobin     RoutingStrategy = "round-robin"
	StrategyLatencyBased   RoutingStrategy = "latency-based"
)

// Deployment represents a single model deployment in the router.
type Deployment struct {
	// ModelName is the user-facing model group name (e.g., "gpt-4").
	ModelName string

	// LiteLLMModel is the actual model string sent to the provider (e.g., "openai/gpt-4").
	LiteLLMModel string

	// Provider is the provider instance for this deployment.
	Provider Provider

	// APIKey overrides the provider's default API key.
	APIKey string

	// BaseURL overrides the provider's default base URL.
	BaseURL string

	// TPMLimit is the tokens-per-minute limit (0 = unlimited).
	TPMLimit int

	// RPMLimit is the requests-per-minute limit (0 = unlimited).
	RPMLimit int

	// MaxParallelRequests limits concurrent requests (0 = unlimited).
	MaxParallelRequests int

	// Weight for weighted routing (higher = more traffic).
	Weight int
}

// deploymentState tracks the runtime state of a deployment.
type deploymentState struct {
	deployment    *Deployment
	healthy       bool
	cooldownUntil time.Time
	failCount     int
	activeReqs    int64
	totalLatency  int64 // nanoseconds
	totalReqs     int64
}

// RouterConfig configures the router behavior.
type RouterConfig struct {
	Strategy    RoutingStrategy
	NumRetries  int
	Timeout     time.Duration
	CooldownTime time.Duration
	AllowedFails int

	// Fallbacks maps model group names to fallback model groups.
	// e.g., {"gpt-4": ["gpt-3.5-turbo", "claude-3-sonnet"]}
	Fallbacks map[string][]string

	// ContextWindowFallbacks maps models to larger-context alternatives.
	ContextWindowFallbacks map[string][]string

	// DefaultFallbacks are tried for any model.
	DefaultFallbacks []string

	// EnableLogging enables debug logging.
	EnableLogging bool
}

// DefaultRouterConfig returns a RouterConfig with sensible defaults.
func DefaultRouterConfig() RouterConfig {
	return RouterConfig{
		Strategy:     StrategyShuffle,
		NumRetries:   2,
		Timeout:      5 * time.Minute,
		CooldownTime: 5 * time.Second,
		AllowedFails: 3,
	}
}

// Router provides load balancing and failover across multiple LLM deployments.
type Router struct {
	config      RouterConfig
	deployments map[string][]*deploymentState // model group -> deployments
	mu          sync.RWMutex
	roundRobin  uint64
	rng         *rand.Rand
}

// NewRouter creates a new Router with the given configuration and deployments.
func NewRouter(cfg RouterConfig, deployments []Deployment) *Router {
	r := &Router{
		config:      cfg,
		deployments: make(map[string][]*deploymentState),
		rng:         rand.New(rand.NewSource(time.Now().UnixNano())),
	}

	for i := range deployments {
		d := &deployments[i]
		state := &deploymentState{
			deployment: d,
			healthy:    true,
		}
		r.deployments[d.ModelName] = append(r.deployments[d.ModelName], state)
	}

	return r
}

// Complete sends a completion request through the router with load balancing and retries.
func (r *Router) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	return r.completeWithFallbacks(ctx, req, nil)
}

func (r *Router) completeWithFallbacks(ctx context.Context, req CompletionRequest, tried map[string]bool) (*CompletionResponse, error) {
	if tried == nil {
		tried = make(map[string]bool)
	}

	modelGroup := req.Model
	tried[modelGroup] = true

	// Try the primary model group with retries
	resp, err := r.completeWithRetries(ctx, req)
	if err == nil {
		return resp, nil
	}

	// Check for context window fallbacks
	var apiErr *APIError
	if errors.As(err, &apiErr) && apiErr.IsContextWindowExceeded() {
		if fallbacks, ok := r.config.ContextWindowFallbacks[modelGroup]; ok {
			for _, fb := range fallbacks {
				if tried[fb] {
					continue
				}
				fbReq := req
				fbReq.Model = fb
				resp, fbErr := r.completeWithFallbacks(ctx, fbReq, tried)
				if fbErr == nil {
					return resp, nil
				}
			}
		}
	}

	// Try model-specific fallbacks
	if fallbacks, ok := r.config.Fallbacks[modelGroup]; ok {
		for _, fb := range fallbacks {
			if tried[fb] {
				continue
			}
			fbReq := req
			fbReq.Model = fb
			resp, fbErr := r.completeWithFallbacks(ctx, fbReq, tried)
			if fbErr == nil {
				return resp, nil
			}
		}
	}

	// Try default fallbacks
	for _, fb := range r.config.DefaultFallbacks {
		if tried[fb] {
			continue
		}
		fbReq := req
		fbReq.Model = fb
		resp, fbErr := r.completeWithFallbacks(ctx, fbReq, tried)
		if fbErr == nil {
			return resp, nil
		}
	}

	return nil, fmt.Errorf("%w: %v", ErrAllRetriesFailed, err)
}

func (r *Router) completeWithRetries(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	var lastErr error

	// Apply timeout to the overall retry loop
	if r.config.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, r.config.Timeout)
		defer cancel()
	}

	for attempt := 0; attempt <= r.config.NumRetries; attempt++ {
		state, err := r.selectDeployment(req.Model)
		if err != nil {
			lastErr = err
			continue
		}

		// Track active requests
		atomic.AddInt64(&state.activeReqs, 1)
		start := time.Now()

		deployReq := req
		deployReq.Model = state.deployment.LiteLLMModel
		if state.deployment.APIKey != "" {
			deployReq.APIKey = state.deployment.APIKey
		}
		if state.deployment.BaseURL != "" {
			deployReq.BaseURL = state.deployment.BaseURL
		}

		resp, err := state.deployment.Provider.Complete(ctx, deployReq)
		elapsed := time.Since(start)

		atomic.AddInt64(&state.activeReqs, -1)
		atomic.AddInt64(&state.totalLatency, int64(elapsed))
		atomic.AddInt64(&state.totalReqs, 1)

		if err != nil {
			lastErr = err
			r.recordFailure(state, err)

			var apiErr *APIError
			if errors.As(err, &apiErr) && !apiErr.IsRetryable() {
				return nil, err // Don't retry non-retryable errors
			}

			if r.config.EnableLogging {
				log.Printf("[litellm-router] attempt %d/%d failed for %s: %v",
					attempt+1, r.config.NumRetries+1, state.deployment.LiteLLMModel, err)
			}

			// Exponential backoff
			if attempt < r.config.NumRetries {
				backoff := time.Duration(math.Pow(2, float64(attempt))) * 100 * time.Millisecond
				time.Sleep(backoff)
			}
			continue
		}

		// Success - reset failure count
		r.mu.Lock()
		state.failCount = 0
		state.healthy = true
		r.mu.Unlock()

		return resp, nil
	}

	return nil, lastErr
}

// CompleteStream sends a streaming completion through the router.
func (r *Router) CompleteStream(ctx context.Context, req CompletionRequest) (*Stream, error) {
	state, err := r.selectDeployment(req.Model)
	if err != nil {
		return nil, err
	}

	deployReq := req
	deployReq.Model = state.deployment.LiteLLMModel
	if state.deployment.APIKey != "" {
		deployReq.APIKey = state.deployment.APIKey
	}
	if state.deployment.BaseURL != "" {
		deployReq.BaseURL = state.deployment.BaseURL
	}
	deployReq.Stream = true

	return state.deployment.Provider.CompleteStream(ctx, deployReq)
}

// Embed sends an embedding request through the router.
func (r *Router) Embed(ctx context.Context, req EmbeddingRequest) (*EmbeddingResponse, error) {
	state, err := r.selectDeployment(req.Model)
	if err != nil {
		return nil, err
	}

	deployReq := req
	deployReq.Model = state.deployment.LiteLLMModel
	if state.deployment.APIKey != "" {
		deployReq.APIKey = state.deployment.APIKey
	}
	if state.deployment.BaseURL != "" {
		deployReq.BaseURL = state.deployment.BaseURL
	}

	return state.deployment.Provider.Embed(ctx, deployReq)
}

// selectDeployment picks a deployment based on the routing strategy.
func (r *Router) selectDeployment(modelGroup string) (*deploymentState, error) {
	r.mu.RLock()
	states, ok := r.deployments[modelGroup]
	r.mu.RUnlock()

	if !ok || len(states) == 0 {
		return nil, fmt.Errorf("%w: no deployments for model group %q", ErrNoDeployments, modelGroup)
	}

	// Filter healthy deployments
	now := time.Now()
	healthy := make([]*deploymentState, 0, len(states))
	for _, s := range states {
		r.mu.RLock()
		isHealthy := s.healthy || now.After(s.cooldownUntil)
		r.mu.RUnlock()
		if isHealthy {
			healthy = append(healthy, s)
		}
	}

	if len(healthy) == 0 {
		// All in cooldown, try the one with shortest cooldown remaining
		var best *deploymentState
		for _, s := range states {
			if best == nil || s.cooldownUntil.Before(best.cooldownUntil) {
				best = s
			}
		}
		return best, nil
	}

	switch r.config.Strategy {
	case StrategyLeastBusy:
		return r.selectLeastBusy(healthy), nil
	case StrategyRoundRobin:
		return r.selectRoundRobin(healthy), nil
	case StrategyLatencyBased:
		return r.selectLatencyBased(healthy), nil
	default: // StrategyShuffle
		return r.selectShuffle(healthy), nil
	}
}

func (r *Router) selectShuffle(states []*deploymentState) *deploymentState {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Weighted random selection
	totalWeight := 0
	for _, s := range states {
		w := s.deployment.Weight
		if w <= 0 {
			w = 1
		}
		totalWeight += w
	}

	target := r.rng.Intn(totalWeight)

	cumulative := 0
	for _, s := range states {
		w := s.deployment.Weight
		if w <= 0 {
			w = 1
		}
		cumulative += w
		if target < cumulative {
			return s
		}
	}
	return states[0]
}

func (r *Router) selectLeastBusy(states []*deploymentState) *deploymentState {
	var best *deploymentState
	var bestActive int64 = math.MaxInt64

	for _, s := range states {
		active := atomic.LoadInt64(&s.activeReqs)
		if active < bestActive {
			bestActive = active
			best = s
		}
	}
	return best
}

func (r *Router) selectRoundRobin(states []*deploymentState) *deploymentState {
	idx := atomic.AddUint64(&r.roundRobin, 1) - 1
	return states[idx%uint64(len(states))]
}

func (r *Router) selectLatencyBased(states []*deploymentState) *deploymentState {
	var best *deploymentState
	var bestAvgLatency int64 = math.MaxInt64

	for _, s := range states {
		reqs := atomic.LoadInt64(&s.totalReqs)
		if reqs == 0 {
			return s // Prefer untested deployments
		}
		avgLatency := atomic.LoadInt64(&s.totalLatency) / reqs
		if avgLatency < bestAvgLatency {
			bestAvgLatency = avgLatency
			best = s
		}
	}
	return best
}

func (r *Router) recordFailure(state *deploymentState, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	state.failCount++
	if state.failCount >= r.config.AllowedFails {
		state.healthy = false
		state.cooldownUntil = time.Now().Add(r.config.CooldownTime)
		if r.config.EnableLogging {
			log.Printf("[litellm-router] deployment %s entering cooldown for %v after %d failures",
				state.deployment.LiteLLMModel, r.config.CooldownTime, state.failCount)
		}
	}
}

// GetHealthyDeployments returns the number of healthy deployments for a model group.
func (r *Router) GetHealthyDeployments(modelGroup string) int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	states, ok := r.deployments[modelGroup]
	if !ok {
		return 0
	}

	now := time.Now()
	count := 0
	for _, s := range states {
		if s.healthy || now.After(s.cooldownUntil) {
			count++
		}
	}
	return count
}
