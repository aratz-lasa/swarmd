package main

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func TestAgentRunFlagUsageOmitsAPIKeyDefaults(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-openai-secret")
	t.Setenv("ANTHROPIC_API_KEY", "sk-anthropic-secret")

	var stderr bytes.Buffer
	err := runAgentRun(context.Background(), []string{"-bogus"}, commandIO{stdout: &bytes.Buffer{}, stderr: &stderr})
	if err == nil {
		t.Fatal("expected parse error")
	}
	out := stderr.String()
	if strings.Contains(out, "sk-openai-secret") || strings.Contains(out, "sk-anthropic-secret") {
		t.Fatalf("usage leaked API key material:\n%s", out)
	}
}

func TestServerFlagUsageOmitsSecretDefaults(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-openai-secret")
	t.Setenv("ANTHROPIC_API_KEY", "sk-anthropic-secret")
	t.Setenv("SLACK_USER_TOKEN", "xoxp-slack-secret")

	var stderr bytes.Buffer
	_, err := parseServerConfig([]string{"-bogus"}, &stderr)
	if err == nil {
		t.Fatal("expected parse error")
	}
	out := stderr.String()
	for _, secret := range []string{"sk-openai-secret", "sk-anthropic-secret", "xoxp-slack-secret"} {
		if strings.Contains(out, secret) {
			t.Fatalf("usage leaked %q:\n%s", secret, out)
		}
	}
}

func TestParseServerConfigFallsBackToEnvSecrets(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-openai-from-env")
	t.Setenv("ANTHROPIC_API_KEY", "sk-anthropic-from-env")
	t.Setenv("SLACK_USER_TOKEN", "xoxp-from-env")

	cfg, err := parseServerConfig([]string{"-data-dir", t.TempDir()}, &bytes.Buffer{})
	if err != nil {
		t.Fatalf("parseServerConfig() error = %v", err)
	}
	if cfg.openAIAPIKey != "sk-openai-from-env" {
		t.Fatalf("openAIAPIKey = %q, want env fallback", cfg.openAIAPIKey)
	}
	if cfg.anthropicAPIKey != "sk-anthropic-from-env" {
		t.Fatalf("anthropicAPIKey = %q, want env fallback", cfg.anthropicAPIKey)
	}
	if cfg.slackUserToken != "xoxp-from-env" {
		t.Fatalf("slackUserToken = %q, want env fallback", cfg.slackUserToken)
	}
}
