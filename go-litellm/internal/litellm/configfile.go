package litellm

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// FileConfig represents the top-level structure of a LiteLLM YAML config file.
type FileConfig struct {
	ModelList       []ModelEntry          `json:"model_list"`
	LiteLLMSettings LiteLLMSettings      `json:"litellm_settings"`
	GeneralSettings GeneralSettings      `json:"general_settings"`
	EnvironmentVars map[string]string    `json:"environment_variables"`
	RouterSettings  RouterSettingsConfig `json:"router_settings"`
}

// ModelEntry represents a single model in the model_list.
type ModelEntry struct {
	ModelName    string       `json:"model_name"`
	LiteLLMParams LiteLLMParams `json:"litellm_params"`
	TPM          int          `json:"tpm"`
	RPM          int          `json:"rpm"`
}

// LiteLLMParams contains the provider-specific parameters for a model.
type LiteLLMParams struct {
	Model   string `json:"model"`
	APIKey  string `json:"api_key"`
	APIBase string `json:"api_base"`
	APIVersion string `json:"api_version"`
}

// LiteLLMSettings contains global LiteLLM settings.
type LiteLLMSettings struct {
	DropParams bool `json:"drop_params"`
}

// GeneralSettings contains general proxy settings.
type GeneralSettings struct {
	MasterKey string `json:"master_key"`
}

// RouterSettingsConfig contains router configuration from the config file.
type RouterSettingsConfig struct {
	RoutingStrategy string `json:"routing_strategy"`
	NumRetries      int    `json:"num_retries"`
}

// LoadConfigFile reads and parses a LiteLLM YAML configuration file.
// It supports the subset of YAML used by LiteLLM config files:
// key-value pairs, lists with "- " prefix, and nested maps with indentation.
func LoadConfigFile(path string) (*FileConfig, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open config file: %w", err)
	}
	defer f.Close()

	cfg := &FileConfig{
		EnvironmentVars: make(map[string]string),
	}

	scanner := bufio.NewScanner(f)
	var currentSection string
	var currentModel *ModelEntry
	var currentSubSection string

	for scanner.Scan() {
		line := scanner.Text()

		// Skip empty lines and comments
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}

		// Determine indentation level
		indent := len(line) - len(strings.TrimLeft(line, " "))

		// Top-level keys (no indentation)
		if indent == 0 {
			key := strings.TrimSuffix(trimmed, ":")
			key = strings.TrimSpace(key)
			currentSection = key
			currentModel = nil
			currentSubSection = ""

			// Handle inline key: value at top level
			if strings.Contains(trimmed, ":") && !strings.HasSuffix(trimmed, ":") {
				k, v := splitKeyValue(trimmed)
				switch k {
				case "model_list", "litellm_settings", "general_settings",
					"environment_variables", "router_settings":
					currentSection = k
				}
				_ = v // top-level sections typically don't have inline values
			}
			continue
		}

		// Section content
		switch currentSection {
		case "model_list":
			if strings.HasPrefix(trimmed, "- ") {
				// New list item
				entry := ModelEntry{}
				rest := strings.TrimPrefix(trimmed, "- ")
				if strings.Contains(rest, ":") {
					k, v := splitKeyValue(rest)
					if k == "model_name" {
						entry.ModelName = v
					}
				}
				cfg.ModelList = append(cfg.ModelList, entry)
				currentModel = &cfg.ModelList[len(cfg.ModelList)-1]
				currentSubSection = ""
			} else if currentModel != nil {
				k, v := splitKeyValue(trimmed)
				switch {
				case k == "model_name":
					currentModel.ModelName = v
					currentSubSection = ""
				case k == "litellm_params" && v == "":
					currentSubSection = "litellm_params"
				case k == "tpm":
					currentModel.TPM, _ = strconv.Atoi(v)
					currentSubSection = ""
				case k == "rpm":
					currentModel.RPM, _ = strconv.Atoi(v)
					currentSubSection = ""
				case currentSubSection == "litellm_params":
					switch k {
					case "model":
						currentModel.LiteLLMParams.Model = v
					case "api_key":
						currentModel.LiteLLMParams.APIKey = v
					case "api_base":
						currentModel.LiteLLMParams.APIBase = v
					case "api_version":
						currentModel.LiteLLMParams.APIVersion = v
					}
				}
			}

		case "litellm_settings":
			k, v := splitKeyValue(trimmed)
			switch k {
			case "drop_params":
				cfg.LiteLLMSettings.DropParams = parseBool(v)
			}

		case "general_settings":
			k, v := splitKeyValue(trimmed)
			switch k {
			case "master_key":
				cfg.GeneralSettings.MasterKey = v
			}

		case "environment_variables":
			k, v := splitKeyValue(trimmed)
			if k != "" {
				cfg.EnvironmentVars[k] = v
			}

		case "router_settings":
			k, v := splitKeyValue(trimmed)
			switch k {
			case "routing_strategy":
				cfg.RouterSettings.RoutingStrategy = v
			case "num_retries":
				cfg.RouterSettings.NumRetries, _ = strconv.Atoi(v)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading config file: %w", err)
	}

	return cfg, nil
}

// ApplyEnvironmentVars sets environment variables from the config file.
func (fc *FileConfig) ApplyEnvironmentVars() {
	for k, v := range fc.EnvironmentVars {
		if v != "" {
			os.Setenv(k, v)
		}
	}
}

// splitKeyValue splits a "key: value" string into its key and value parts.
func splitKeyValue(s string) (string, string) {
	idx := strings.Index(s, ":")
	if idx < 0 {
		return strings.TrimSpace(s), ""
	}
	key := strings.TrimSpace(s[:idx])
	val := strings.TrimSpace(s[idx+1:])
	// Remove surrounding quotes
	val = strings.Trim(val, `"'`)
	return key, val
}

// parseBool parses a YAML boolean value (True/False/true/false/yes/no).
func parseBool(s string) bool {
	lower := strings.ToLower(strings.TrimSpace(s))
	return lower == "true" || lower == "yes" || lower == "1"
}
