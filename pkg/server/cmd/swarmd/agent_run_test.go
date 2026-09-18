package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/richardartoul/swarmd/pkg/agent"
	"github.com/richardartoul/swarmd/pkg/server"
	cpstore "github.com/richardartoul/swarmd/pkg/server/store"
)

type testDriverFactory func(context.Context, cpstore.RunnableAgent) (agent.Driver, error)

func (f testDriverFactory) NewWorkerDriver(ctx context.Context, record cpstore.RunnableAgent) (agent.Driver, error) {
	return f(ctx, record)
}

func TestRunAgentOnceCompletesWithScriptedDriver(t *testing.T) {
	t.Parallel()

	configDir := t.TempDir()
	specPath := filepath.Join(configDir, "hello.yaml")
	specYAML := "" +
		"version: 1\n" +
		"agent_id: hello-once\n" +
		"name: Hello Once\n" +
		"model:\n" +
		"  provider: openai\n" +
		"  name: gpt-test\n" +
		"prompt: |\n" +
		"  Call server_log once, then finish.\n" +
		"tools:\n" +
		"  - server_log\n" +
		"runtime:\n" +
		"  max_steps: 4\n" +
		"  step_timeout: 10s\n" +
		"  max_attempts: 1\n" +
		"schedules:\n" +
		"  - id: once\n" +
		"    cron: \"0 * * * *\"\n" +
		"    payload:\n" +
		"      kind: scheduled_run\n"
	if err := os.WriteFile(specPath, []byte(specYAML), 0o644); err != nil {
		t.Fatalf("write spec: %v", err)
	}

	spec, err := server.LoadAgentSpecFile(specPath)
	if err != nil {
		t.Fatalf("LoadAgentSpecFile() error = %v", err)
	}

	root := filepath.Join(t.TempDir(), "root")
	dataDir := t.TempDir()
	var stdout, stderr bytes.Buffer
	err = runAgentOnce(context.Background(), spec, agentOnceOptions{
		dataDir:      dataDir,
		root:         root,
		timeout:      10 * time.Second,
		liveOutput:   false,
		pollInterval: 20 * time.Millisecond,
		driverFactory: testDriverFactory(func(_ context.Context, _ cpstore.RunnableAgent) (agent.Driver, error) {
			return agent.DriverFunc(func(_ context.Context, req agent.Request) (agent.Decision, error) {
				if req.Step == 1 {
					return agent.Decision{
						Tool: &agent.ToolAction{
							Name:  "server_log",
							Kind:  agent.ToolKindFunction,
							Input: `{"level":"info","message":"hello from agent run"}`,
						},
					}, nil
				}
				return agent.Decision{Finish: &agent.FinishAction{Value: "done"}}, nil
			}), nil
		}),
	}, commandIO{stdout: &stdout, stderr: &stderr})
	if err != nil {
		t.Fatalf("runAgentOnce() error = %v\nstdout=%s\nstderr=%s", err, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "agent-run> completed") {
		t.Fatalf("stdout = %q, want completion line", stdout.String())
	}
	if _, err := os.Stat(filepath.Join(dataDir, defaultSQLiteFilename)); err != nil {
		t.Fatalf("expected sqlite db under data dir: %v", err)
	}
}

func TestRunAgentOnceRejectsPausedAgent(t *testing.T) {
	t.Parallel()

	configDir := t.TempDir()
	specPath := filepath.Join(configDir, "paused.yaml")
	specYAML := "" +
		"version: 1\n" +
		"agent_id: paused-agent\n" +
		"model:\n" +
		"  name: gpt-test\n" +
		"prompt: stay quiet\n" +
		"tools:\n" +
		"  - server_log\n" +
		"runtime:\n" +
		"  desired_state: paused\n"
	if err := os.WriteFile(specPath, []byte(specYAML), 0o644); err != nil {
		t.Fatalf("write spec: %v", err)
	}
	spec, err := server.LoadAgentSpecFile(specPath)
	if err != nil {
		t.Fatalf("LoadAgentSpecFile() error = %v", err)
	}
	err = runAgentOnce(context.Background(), spec, agentOnceOptions{
		dataDir: t.TempDir(),
		driverFactory: testDriverFactory(func(_ context.Context, _ cpstore.RunnableAgent) (agent.Driver, error) {
			return agent.DriverFunc(func(context.Context, agent.Request) (agent.Decision, error) {
				return agent.Decision{Finish: &agent.FinishAction{Value: "nope"}}, nil
			}), nil
		}),
	}, commandIO{stdout: &bytes.Buffer{}, stderr: &bytes.Buffer{}})
	if err == nil {
		t.Fatal("runAgentOnce() error = nil, want paused rejection")
	}
	if !strings.Contains(err.Error(), "paused") {
		t.Fatalf("runAgentOnce() error = %v, want paused mention", err)
	}
}

func TestRunAgentOnceDisablesTools(t *testing.T) {
	t.Parallel()

	configDir := t.TempDir()
	specPath := filepath.Join(configDir, "tools.yaml")
	specYAML := "" +
		"version: 1\n" +
		"agent_id: tool-agent\n" +
		"model:\n" +
		"  name: gpt-test\n" +
		"prompt: finish immediately\n" +
		"tools:\n" +
		"  - server_log\n" +
		"  - slack_post\n" +
		"runtime:\n" +
		"  max_steps: 2\n" +
		"  max_attempts: 1\n"
	if err := os.WriteFile(specPath, []byte(specYAML), 0o644); err != nil {
		t.Fatalf("write spec: %v", err)
	}
	spec, err := server.LoadAgentSpecFile(specPath)
	if err != nil {
		t.Fatalf("LoadAgentSpecFile() error = %v", err)
	}

	var sawSlack bool
	err = runAgentOnce(context.Background(), spec, agentOnceOptions{
		dataDir:      t.TempDir(),
		root:         filepath.Join(t.TempDir(), "root"),
		disableTools: []string{"slack_post"},
		timeout:      10 * time.Second,
		pollInterval: 20 * time.Millisecond,
		driverFactory: testDriverFactory(func(_ context.Context, record cpstore.RunnableAgent) (agent.Driver, error) {
			return agent.DriverFunc(func(_ context.Context, req agent.Request) (agent.Decision, error) {
				for _, tool := range req.Tools {
					if tool.Name == "slack_post" {
						sawSlack = true
					}
				}
				return agent.Decision{Finish: &agent.FinishAction{Value: "done"}}, nil
			}), nil
		}),
	}, commandIO{stdout: &bytes.Buffer{}, stderr: &bytes.Buffer{}})
	if err != nil {
		t.Fatalf("runAgentOnce() error = %v", err)
	}
	if sawSlack {
		t.Fatal("disabled slack_post still appeared in request tools")
	}
}
