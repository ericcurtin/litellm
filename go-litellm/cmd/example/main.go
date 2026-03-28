// Command example demonstrates basic usage of the go-litellm library.
package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"

	litellm "github.com/ericcurtin/litellm/go-litellm"
	"github.com/ericcurtin/litellm/go-litellm/providers"
)

func main() {
	// Create client
	client := litellm.NewClient()

	// Register providers from environment variables
	if key := os.Getenv("OPENAI_API_KEY"); key != "" {
		client.RegisterProvider("openai", providers.NewOpenAIProvider(key, ""))
	}
	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" {
		client.RegisterProvider("anthropic", providers.NewAnthropicProvider(key, ""))
	}

	ctx := context.Background()

	// Example 1: Simple completion (OpenAI)
	fmt.Println("=== Example 1: OpenAI Completion ===")
	resp, err := client.Complete(ctx, litellm.CompletionRequest{
		Model:    "openai/gpt-4",
		Messages: []litellm.Message{
			{Role: "user", Content: litellm.StringContent("Say hello in 10 words or less")},
		},
	})
	if err != nil {
		log.Printf("OpenAI error: %v", err)
	} else {
		fmt.Printf("Response: %s\n", resp.Choices[0].Message.Content.Text)
		fmt.Printf("Tokens: %d prompt + %d completion = %d total\n",
			resp.Usage.PromptTokens, resp.Usage.CompletionTokens, resp.Usage.TotalTokens)
	}

	// Example 2: Anthropic completion
	fmt.Println("\n=== Example 2: Anthropic Completion ===")
	resp, err = client.Complete(ctx, litellm.CompletionRequest{
		Model:    "anthropic/claude-3-5-sonnet-20241022",
		Messages: []litellm.Message{
			{Role: "user", Content: litellm.StringContent("What is Go programming language? One sentence.")},
		},
	})
	if err != nil {
		log.Printf("Anthropic error: %v", err)
	} else {
		fmt.Printf("Response: %s\n", resp.Choices[0].Message.Content.Text)
	}

	// Example 3: Streaming
	fmt.Println("\n=== Example 3: Streaming Completion ===")
	stream, err := client.CompleteStream(ctx, litellm.CompletionRequest{
		Model:    "openai/gpt-4",
		Messages: []litellm.Message{
			{Role: "user", Content: litellm.StringContent("Count from 1 to 5")},
		},
	})
	if err != nil {
		log.Printf("Stream error: %v", err)
	} else {
		for {
			chunk, err := stream.Recv()
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Printf("Stream recv error: %v", err)
				break
			}
			if len(chunk.Choices) > 0 {
				fmt.Print(chunk.Choices[0].Delta.Content)
			}
		}
		fmt.Println()
		stream.Close()
	}

	// Example 4: Router with load balancing
	fmt.Println("\n=== Example 4: Router with Fallbacks ===")
	if key := os.Getenv("OPENAI_API_KEY"); key != "" {
		openaiProvider := providers.NewOpenAIProvider(key, "")

		deployments := []litellm.Deployment{
			{
				ModelName:    "gpt",
				LiteLLMModel: "gpt-4",
				Provider:     openaiProvider,
				Weight:       2,
			},
			{
				ModelName:    "gpt",
				LiteLLMModel: "gpt-3.5-turbo",
				Provider:     openaiProvider,
				Weight:       1,
			},
		}

		routerCfg := litellm.DefaultRouterConfig()
		routerCfg.Strategy = litellm.StrategyShuffle
		routerCfg.EnableLogging = true
		router := litellm.NewRouter(routerCfg, deployments)

		resp, err := router.Complete(ctx, litellm.CompletionRequest{
			Model:    "gpt",
			Messages: []litellm.Message{
				{Role: "user", Content: litellm.StringContent("Hello!")},
			},
		})
		if err != nil {
			log.Printf("Router error: %v", err)
		} else {
			fmt.Printf("Model used: %s\n", resp.Model)
			fmt.Printf("Response: %s\n", resp.Choices[0].Message.Content.Text)
		}
	} else {
		fmt.Println("Skipped (OPENAI_API_KEY not set)")
	}

	// Example 5: Tool calling
	fmt.Println("\n=== Example 5: Tool Calling ===")
	resp, err = client.Complete(ctx, litellm.CompletionRequest{
		Model: "openai/gpt-4",
		Messages: []litellm.Message{
			{Role: "user", Content: litellm.StringContent("What's the weather in San Francisco?")},
		},
		Tools: []litellm.Tool{{
			Type: "function",
			Function: litellm.ToolFunction{
				Name:        "get_weather",
				Description: "Get the weather for a location",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type":        "string",
							"description": "City name",
						},
					},
					"required": []string{"location"},
				},
			},
		}},
	})
	if err != nil {
		log.Printf("Tool calling error: %v", err)
	} else {
		if len(resp.Choices[0].Message.ToolCalls) > 0 {
			tc := resp.Choices[0].Message.ToolCalls[0]
			fmt.Printf("Tool call: %s(%s)\n", tc.Function.Name, tc.Function.Arguments)
		} else {
			fmt.Printf("Response: %s\n", resp.Choices[0].Message.Content.Text)
		}
	}
}
