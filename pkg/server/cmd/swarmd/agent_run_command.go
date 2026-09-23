package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/richardartoul/swarmd/pkg/server"
	cpstore "github.com/richardartoul/swarmd/pkg/server/store"
)

type stringList []string

func (s *stringList) String() string { return strings.Join(*s, ",") }
func (s *stringList) Set(value string) error {
	value = strings.TrimSpace(value)
	if value == "" {
		return fmt.Errorf("value must not be empty")
	}
	*s = append(*s, value)
	return nil
}

type agentOnceOptions struct {
	dataDir       string
	root          string
	disableTools  []string
	timeout       time.Duration
	liveOutput    bool
	pollInterval  time.Duration
	driverFactory server.WorkerDriverFactory
	openAIAPIKey  string
	anthropicKey  string
}

func runAgentRun(ctx context.Context, args []string, streams commandIO) error {
	fs := flag.NewFlagSet("agent run", flag.ContinueOnError)
	fs.SetOutput(streams.stderr)
	configPath := fs.String("config", "", "path to a single agent YAML spec (required)")
	dataDir := fs.String("data-dir", envOr("SWARMD_SERVER_DATA_DIR", defaultDataDir), "base directory for SQLite and default agent roots")
	rootDir := fs.String("root", "", "sandbox root; sets root_path before sync (relative paths are resolved)")
	timeout := fs.Duration("timeout", 45*time.Minute, "maximum time to wait for the run to finish")
	liveOutput := fs.Bool("live-output", true, "mirror worker stdout/stderr to this process")
	openAIAPIKey := fs.String("api-key", strings.TrimSpace(os.Getenv("OPENAI_API_KEY")), "OpenAI API key for worker agents")
	anthropicAPIKey := fs.String("anthropic-api-key", strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY")), "Anthropic API key for worker agents")
	var disableTools stringList
	fs.Var(&disableTools, "disable-tool", "disable a tool by id before sync (repeatable)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*configPath) == "" {
		return fmt.Errorf("agent run requires -config <agent.yaml>")
	}

	spec, err := server.LoadAgentSpecFile(strings.TrimSpace(*configPath))
	if err != nil {
		return err
	}
	return runAgentOnce(ctx, spec, agentOnceOptions{
		dataDir:      strings.TrimSpace(*dataDir),
		root:         strings.TrimSpace(*rootDir),
		disableTools: append([]string(nil), disableTools...),
		timeout:      *timeout,
		liveOutput:   *liveOutput,
		openAIAPIKey: strings.TrimSpace(*openAIAPIKey),
		anthropicKey: strings.TrimSpace(*anthropicAPIKey),
	}, streams)
}

func runAgentOnce(ctx context.Context, spec server.AgentSpec, opts agentOnceOptions, streams commandIO) error {
	desired := strings.TrimSpace(spec.Runtime.DesiredState)
	if desired == string(cpstore.AgentDesiredStatePaused) || desired == string(cpstore.AgentDesiredStateStopped) {
		return fmt.Errorf("agent %q/%q has desired_state %q and cannot be run", spec.NamespaceID, spec.AgentID, desired)
	}

	if opts.root != "" {
		absRoot, err := filepath.Abs(opts.root)
		if err != nil {
			return fmt.Errorf("resolve -root: %w", err)
		}
		if err := os.MkdirAll(absRoot, 0o755); err != nil {
			return fmt.Errorf("create -root %q: %w", absRoot, err)
		}
		spec.RootPath = absRoot
	}

	disableSet := make(map[string]struct{}, len(opts.disableTools))
	for _, name := range opts.disableTools {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		disableSet[name] = struct{}{}
	}
	if len(disableSet) > 0 {
		disabled := false
		matched := make(map[string]struct{}, len(disableSet))
		for i := range spec.Tools {
			id := strings.TrimSpace(spec.Tools[i].ID)
			if _, ok := disableSet[id]; ok {
				spec.Tools[i].Enabled = &disabled
				matched[id] = struct{}{}
			}
		}
		var unknown []string
		for id := range disableSet {
			if _, ok := matched[id]; !ok {
				unknown = append(unknown, id)
			}
		}
		if len(unknown) > 0 {
			sort.Strings(unknown)
			return fmt.Errorf("unknown -disable-tool id(s): %s", strings.Join(unknown, ", "))
		}
	}

	if err := validateAgentRunAPIKeys(spec, opts); err != nil {
		return err
	}
	lookupEnv := func(key string) string {
		return strings.TrimSpace(os.Getenv(strings.TrimSpace(key)))
	}
	if err := server.ValidateReferencedToolEnv([]server.AgentSpec{spec}, lookupEnv); err != nil {
		return err
	}
	if err := server.ValidateReferencedAgentConfigEnv([]server.AgentSpec{spec}, lookupEnv); err != nil {
		return err
	}

	dataDir := strings.TrimSpace(opts.dataDir)
	if dataDir == "" {
		return fmt.Errorf("agent run requires -data-dir or SWARMD_SERVER_DATA_DIR")
	}
	dbPath := defaultSQLitePathForDataDir(dataDir)
	rootBase := defaultRootBaseForDataDir(dataDir)
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return fmt.Errorf("ensure data dir %q: %w", dataDir, err)
	}
	if err := os.MkdirAll(rootBase, 0o755); err != nil {
		return fmt.Errorf("ensure root base %q: %w", rootBase, err)
	}

	store, err := cpstore.Open(ctx, dbPath)
	if err != nil {
		return err
	}
	defer store.Close()

	configRoot := filepath.Dir(spec.SourcePath)
	summary, err := server.SyncSpecsUpsert(ctx, store, []server.AgentSpec{spec}, configRoot, rootBase)
	if err != nil {
		return err
	}
	fmt.Fprintf(
		streams.stdout,
		"agent-run> agent=%s/%s data_dir=%s root=%s\n",
		spec.NamespaceID,
		spec.AgentID,
		dataDir,
		firstNonEmpty(spec.RootPath, filepath.Join(rootBase, spec.NamespaceID, spec.AgentID)),
	)
	fmt.Fprintf(
		streams.stdout,
		"sync> namespaces(created=%d updated=%d) agents(created=%d updated=%d deleted=%d) schedules(created=%d deleted=%d)\n",
		summary.NamespacesCreated,
		summary.NamespacesUpdated,
		summary.AgentsCreated,
		summary.AgentsUpdated,
		summary.AgentsDeleted,
		summary.SchedulesCreated,
		summary.SchedulesDeleted,
	)

	agentRecord, err := store.GetAgent(ctx, spec.NamespaceID, spec.AgentID)
	if err != nil {
		return fmt.Errorf("load synced agent %q/%q: %w", spec.NamespaceID, spec.AgentID, err)
	}

	payload := agentRunTriggerPayload(spec)
	params := cpstore.CreateMailboxMessageParams{
		NamespaceID:      spec.NamespaceID,
		RecipientAgentID: spec.AgentID,
		Kind:             "schedule.fire",
		Payload:          payload,
		Metadata: map[string]any{
			"source": "agent.run",
		},
	}
	if agentRecord.MaxAttempts > 0 {
		params.MaxAttempts = agentRecord.MaxAttempts
	}
	message, err := store.EnqueueMessage(ctx, params)
	if err != nil {
		return fmt.Errorf("enqueue one-shot trigger for %q/%q: %w", spec.NamespaceID, spec.AgentID, err)
	}
	fmt.Fprintf(streams.stdout, "queued> message=%s thread=%s\n", message.ID, message.ThreadID)

	driverFactory := opts.driverFactory
	if driverFactory == nil {
		driverFactory = server.MultiProviderWorkerDriverFactory{
			OpenAIAPIKey:    opts.openAIAPIKey,
			AnthropicAPIKey: opts.anthropicKey,
		}
	}
	pollInterval := opts.pollInterval
	if pollInterval <= 0 {
		pollInterval = 250 * time.Millisecond
	}
	var liveStdout, liveStderr = streams.stdout, streams.stderr
	if !opts.liveOutput {
		liveStdout, liveStderr = nil, nil
	}
	manager := &server.RuntimeManager{
		Store:         store,
		DriverFactory: driverFactory,
		PollInterval:  pollInterval,
		Stdout:        liveStdout,
		Stderr:        liveStderr,
		Logger:        server.NewRuntimeLogger(streams.stdout, streams.stderr),
		EnvLookup:     lookupEnv,
	}

	timeout := opts.timeout
	if timeout <= 0 {
		timeout = 45 * time.Minute
	}
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	errCh := make(chan error, 1)
	go func() {
		errCh <- manager.Run(runCtx)
	}()

	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()
	for {
		select {
		case <-runCtx.Done():
			cancel()
			runtimeErr := <-errCh
			if errors.Is(runCtx.Err(), context.DeadlineExceeded) {
				return fmt.Errorf("agent run timed out after %s waiting for message %s", timeout, message.ID)
			}
			if runtimeErr != nil && !errors.Is(runtimeErr, context.Canceled) && !errors.Is(runtimeErr, context.DeadlineExceeded) {
				return runtimeErr
			}
			return runCtx.Err()
		case runtimeErr := <-errCh:
			if runtimeErr != nil && !errors.Is(runtimeErr, context.Canceled) {
				return runtimeErr
			}
			return fmt.Errorf("runtime stopped before message %s finished", message.ID)
		case <-ticker.C:
			current, err := store.GetMailboxMessage(ctx, spec.NamespaceID, message.ID)
			if err != nil {
				cancel()
				<-errCh
				return fmt.Errorf("poll mailbox message %s: %w", message.ID, err)
			}
			switch current.Status {
			case cpstore.MailboxMessageStatusCompleted:
				cancel()
				runtimeErr := <-errCh
				if runtimeErr != nil && !errors.Is(runtimeErr, context.Canceled) && !errors.Is(runtimeErr, context.DeadlineExceeded) {
					return runtimeErr
				}
				fmt.Fprintf(streams.stdout, "agent-run> completed message=%s\n", message.ID)
				return nil
			case cpstore.MailboxMessageStatusDeadLetter:
				cancel()
				<-errCh
				reason := strings.TrimSpace(current.DeadLetterReason)
				if reason == "" {
					reason = strings.TrimSpace(current.LastError)
				}
				if reason == "" {
					reason = "dead_letter"
				}
				return fmt.Errorf("agent run failed: message=%s status=dead_letter reason=%s", message.ID, reason)
			}
		}
	}
}

func validateAgentRunAPIKeys(spec server.AgentSpec, opts agentOnceOptions) error {
	if opts.driverFactory != nil {
		return nil
	}
	switch strings.TrimSpace(spec.Model.Provider) {
	case "anthropic":
		if opts.anthropicKey == "" {
			return fmt.Errorf("agent run requires ANTHROPIC_API_KEY for anthropic agents")
		}
	case "", "openai":
		if opts.openAIAPIKey == "" {
			return fmt.Errorf("agent run requires OPENAI_API_KEY for openai agents")
		}
	}
	return nil
}

func agentRunTriggerPayload(spec server.AgentSpec) any {
	if len(spec.Schedules) > 0 && spec.Schedules[0].Payload != nil {
		return spec.Schedules[0].Payload
	}
	scheduleID := spec.AgentID + "-schedule-1"
	if len(spec.Schedules) > 0 && strings.TrimSpace(spec.Schedules[0].ID) != "" {
		scheduleID = strings.TrimSpace(spec.Schedules[0].ID)
	}
	return map[string]any{
		"kind":     "scheduled_run",
		"agent_id": spec.AgentID,
		"schedule": scheduleID,
		"source":   filepath.Base(spec.SourcePath),
	}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
