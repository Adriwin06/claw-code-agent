const repoMetrics = [
  { value: "Python", label: "Library-first runtime", detail: "Packaged with setuptools and exposed through src.main." },
  { value: "LiteLLM", label: "Model transport", detail: "Targets OpenAI-compatible local and hosted APIs." },
  { value: "CLI", label: "Operational surface", detail: "One-shot, chat, TUI, background, daemon, and utility commands." },
  { value: "Tools", label: "Agent actions", detail: "Workspace, shell, search, MCP, LSP, task, team, workflow, and subagent tools." },
  { value: "State", label: "Local persistence", detail: "Sessions, scratchpads, plans, tasks, config, account, and runtime history." },
  { value: "Static", label: "Docs site", detail: "Build-free HTML, CSS, and JavaScript in docs/." },
];

const architectureRows = [
  {
    label: "Entrypoints",
    cells: [
      ["src/main.py", "Loads workspace environment values, builds the parser, and dispatches commands.", "green"],
      ["src/cli/parser_build.py", "Defines CLI commands, utility commands, agent modes, and shared flags.", "blue"],
      ["src/cli/command_dispatch.py", "Routes parsed commands to runtime handlers and feature facades.", "red"],
      ["src/textual_ui.py", "Optional Textual application for sessions, transcript, prompt composer, and side panels.", "amber"],
    ],
  },
  {
    label: "Agent core",
    cells: [
      ["src/agent/runtime/", "Owns LocalCodingAgent, prompt loops, tool execution, delegation, refresh, and persistence.", "green"],
      ["src/agent/context/", "Builds context snapshots, prompt sections, usage accounting, and pressure passes.", "blue"],
      ["src/agent/tools/", "Defines tool schemas, execution handlers, shell security, and effective registries.", "red"],
      ["src/agent/models/", "Carries message, run result, budget, permission, model, and usage dataclasses.", "amber"],
    ],
  },
  {
    label: "Runtimes",
    cells: [
      ["src/features/integration/", "Search, MCP, remote profiles, remote triggers, plugins, and LSP-style intelligence.", "green"],
      ["src/features/orchestration/", "Plans, tasks, workflows, and managed git worktrees.", "blue"],
      ["src/features/collaboration/", "Ask-user queues and local team/message collaboration state.", "red"],
      ["src/features/system/", "Account, config, doctor, env, hook policy, tokenizer, and background sessions.", "amber"],
    ],
  },
  {
    label: "Support",
    cells: [
      ["src/llm/", "LiteLLM-backed chat client factory and parser/error helpers.", "green"],
      ["src/session/", "Session store, transcript handling, compaction, microcompaction, and persistence helpers.", "blue"],
      ["src/ui/", "Conversation state, slash suggestions, formatting, workspace files, and Textual event bridge.", "red"],
      ["tests/ and benchmarks/", "Runtime unit tests plus benchmark adapters for code, math, instruction, and general suites.", "amber"],
    ],
  },
];

const capabilities = [
  {
    name: "Agent loop",
    summary: "Prompt handling, local slash preprocessing, model calls, tool loops, continuation handling, and final run assembly.",
    items: ["LocalCodingAgent", "streaming", "tool loop"],
  },
  {
    name: "Context engine",
    summary: "Workspace, git, memory, scratchpad, runtime state, and system prompt assembly.",
    items: ["CLAUDE.md", "git snapshot", "PromptContext"],
  },
  {
    name: "Tooling",
    summary: "Core workspace tools plus integrations, delegation, skills, and background task helpers.",
    items: ["files", "shell", "Agent tool"],
  },
  {
    name: "Integrations",
    summary: "Local search, MCP, remote profiles, remote triggers, plugins, and LSP-style code intelligence.",
    items: ["search", "MCP", "LSP"],
  },
  {
    name: "Governance",
    summary: "Permission tiers, shell safety checks, hook policy, token budgets, and cost tracking.",
    items: ["permissions", "budgets", "hooks"],
  },
  {
    name: "Persistence",
    summary: "Saved sessions, transcripts, file history replay, plugin state, compaction, and scratchpads.",
    items: ["resume", "transcript", "compaction"],
  },
  {
    name: "Interfaces",
    summary: "Argparse CLI, interactive chat, Textual TUI, local background sessions, and daemon wrappers.",
    items: ["agent", "agent-chat", "agent-tui"],
  },
  {
    name: "Verification",
    summary: "Doctor checks, unit tests, and benchmark adapters cover the runtime and its integrations.",
    items: ["doctor", "unittest", "benchmarks"],
  },
];

const commandItems = [
  cli("doctor", "Core agent", "Check workspace, backend, and optional UI readiness."),
  cli("dev", "Developer", "Namespace for porting, catalog, and mirrored runtime utilities."),
  cli("remote-status", "Integration", "Show local remote runtime status."),
  cli("remote-profiles", "Integration", "List configured local remote profiles."),
  cli("remote-disconnect", "Integration", "Disconnect the active local remote target."),
  cli("worktree-status", "Orchestration", "Show managed git worktree status."),
  cli("worktree-enter", "Orchestration", "Create and enter a managed git worktree."),
  cli("worktree-exit", "Orchestration", "Exit the active managed git worktree."),
  cli("account-status", "System", "Show local account runtime status."),
  cli("account-profiles", "System", "List configured local account profiles."),
  cli("account-login", "System", "Activate a local account profile or ephemeral identity."),
  cli("account-logout", "System", "Clear the active local account session."),
  cli("ask-status", "Collaboration", "Show local ask-user runtime status."),
  cli("ask-history", "Collaboration", "Show local ask-user interaction history."),
  cli("search-status", "Integration", "Show local search runtime status."),
  cli("search-providers", "Integration", "List configured local search providers."),
  cli("search-activate", "Integration", "Set the active local search provider."),
  cli("search", "Integration", "Run web search against the configured local search runtime."),
  cli("mcp-status", "Integration", "Show local MCP runtime status."),
  cli("mcp-resources", "Integration", "List MCP resources from manifests and transport-backed servers."),
  cli("mcp-resource", "Integration", "Read an MCP resource by URI."),
  cli("mcp-tools", "Integration", "List MCP tools exposed by configured MCP servers."),
  cli("mcp-call-tool", "Integration", "Call an MCP tool exposed by a configured server."),
  cli("config-status", "System", "Show local workspace config runtime summary."),
  cli("config-effective", "System", "Render merged effective local workspace config."),
  cli("config-source", "System", "Render a specific local config source."),
  cli("config-get", "System", "Read a local config value by dotted key path."),
  cli("config-set", "System", "Write a local config value by dotted key path."),
  cli("lsp-status", "LSP", "Show local LSP runtime summary."),
  cli("lsp-symbols", "LSP", "Show document symbols for one file."),
  cli("lsp-workspace-symbols", "LSP", "Search workspace symbols through the local LSP runtime."),
  cli("lsp-definition", "LSP", "Run a local definition query."),
  cli("lsp-references", "LSP", "Run a local references query."),
  cli("lsp-hover", "LSP", "Run a local hover query."),
  cli("lsp-diagnostics", "LSP", "Show local diagnostics."),
  cli("lsp-call-hierarchy", "LSP", "Show call hierarchy at a position."),
  cli("lsp-incoming-calls", "LSP", "Show incoming calls at a position."),
  cli("lsp-outgoing-calls", "LSP", "Show outgoing calls at a position."),
  cli("workflow-list", "Orchestration", "List local workflow definitions."),
  cli("workflow-get", "Orchestration", "Show one local workflow definition."),
  cli("workflow-run", "Orchestration", "Record and render a local workflow run."),
  cli("trigger-list", "Integration", "List local remote triggers."),
  cli("trigger-get", "Integration", "Show one local remote trigger."),
  cli("trigger-create", "Integration", "Create a local remote trigger."),
  cli("trigger-update", "Integration", "Update a local remote trigger."),
  cli("trigger-run", "Integration", "Run a local remote trigger."),
  cli("team-status", "Collaboration", "Show local collaboration team runtime summary."),
  cli("team-list", "Collaboration", "List local collaboration teams."),
  cli("team-get", "Collaboration", "Show one local collaboration team."),
  cli("team-create", "Collaboration", "Create a local collaboration team."),
  cli("team-delete", "Collaboration", "Delete a local collaboration team."),
  cli("team-messages", "Collaboration", "Show local team messages."),
  cli("agent", "Core agent", "Run the Python local-model agent one-shot."),
  cli("agent-bg", "Background", "Run the Python local-model agent as a background session."),
  cli("agent-bg-worker", "Background", "Internal worker entrypoint used by local background sessions."),
  cli("agent-ps", "Background", "List local background agent sessions."),
  cli("agent-logs", "Background", "Show logs for a local background agent session."),
  cli("agent-attach", "Background", "Show the current background output snapshot."),
  cli("agent-kill", "Background", "Stop a background session."),
  cli("daemon", "Background", "Daemon-style wrapper over local background sessions."),
  cli("daemon start", "Background", "Launch a local daemon-style background agent session."),
  cli("daemon worker", "Background", "Internal daemon worker entrypoint."),
  cli("daemon ps", "Background", "List local daemon-style background sessions."),
  cli("daemon logs", "Background", "Show logs for a daemon-style background session."),
  cli("daemon attach", "Background", "Show the current daemon output snapshot."),
  cli("daemon kill", "Background", "Stop a daemon-style background session."),
  cli("agent-chat", "Core agent", "Run an interactive Python local-model chat loop."),
  cli("agent-tui", "Core agent", "Run the Textual terminal UI."),
  cli("agent-resume", "Core agent", "Resume a saved Python local-model agent session."),
  cli("agent-prompt", "Core agent", "Render the Python agent system prompt."),
  cli("agent-context", "Core agent", "Render Python context usage accounting."),
  cli("agent-context-raw", "Core agent", "Render the raw Python context snapshot."),
  cli("token-budget", "Core agent", "Render token budget and prompt-length limits."),
  cli("agents", "Core agent", "List active local agent configs or show one definition."),
];

const devCommandItems = [
  "summary", "manifest", "parity-audit", "setup-report", "command-graph",
  "tool-pool", "bootstrap-graph", "subsystems", "commands", "tools",
  "route", "bootstrap", "turn-loop", "flush-transcript", "load-session",
  "remote-mode", "ssh-mode", "teleport-mode", "direct-connect-mode",
  "deep-link-mode", "show-command", "show-tool", "exec-command", "exec-tool",
].map((name) => cli(`dev ${name}`, "Developer", "Porting, catalog, mirrored runtime, or diagnostics utility."));

const slashCommandItems = [
  slash("help", "commands", "Show built-in Python slash commands."),
  slash("context", "usage", "Show estimated session context usage."),
  slash("context-raw", "env", "Show the raw environment and context snapshot."),
  slash("token-budget", "budget", "Show token-budget window, reserves, and prompt limits."),
  slash("mcp", "", "Show discovered local MCP manifests, tools, resources, or one tool."),
  slash("search", "", "Search status, toggles, context size, providers, activation, or query."),
  slash("remote", "", "Show remote runtime status or activate a target/profile."),
  slash("worktree", "", "Show managed worktree status or enter/exit the active worktree."),
  slash("account", "", "Show account runtime status or profiles."),
  slash("ask", "", "Show ask-user runtime status or history."),
  slash("login", "", "Activate a local account profile or identity."),
  slash("logout", "", "Clear the active local account session."),
  slash("config", "settings", "Show config state, effective config, sources, or one value."),
  slash("lsp", "", "Run LSP status, symbols, definitions, references, hover, hierarchy, or diagnostics."),
  slash("remotes", "", "List configured local remote profiles."),
  slash("ssh", "", "Activate a local SSH remote target/profile."),
  slash("teleport", "", "Activate a teleport remote target/profile."),
  slash("direct-connect", "", "Activate a direct-connect remote target/profile."),
  slash("deep-link", "", "Activate a deep-link remote target/profile."),
  slash("disconnect", "remote-disconnect", "Disconnect the active local remote runtime target."),
  slash("resources", "", "List local MCP resources."),
  slash("resource", "", "Render a local MCP resource by URI."),
  slash("tasks", "todo", "Show the local runtime task list."),
  slash("workflows", "", "List local workflows."),
  slash("workflow", "", "Show or run one local workflow."),
  slash("triggers", "", "List local remote triggers."),
  slash("trigger", "", "Show or run one local remote trigger."),
  slash("teams", "", "List configured collaboration teams."),
  slash("team", "", "Show one collaboration team."),
  slash("messages", "", "Show recorded collaboration messages."),
  slash("task-next", "next-task", "Show the next actionable tasks."),
  slash("plan", "planner", "Show the current local runtime plan."),
  slash("task", "", "Show a local runtime task by id."),
  slash("prompt", "system-prompt", "Render the effective Python system prompt."),
  slash("permissions", "", "Show active tool permission mode."),
  slash("hooks", "policy", "Show local hook and policy manifests."),
  slash("trust", "", "Show trust mode, managed settings, and safe environment values."),
  slash("model", "", "Show or update the active model."),
  slash("tools", "", "List registered tools and permission status."),
  slash("agents", "", "List active local agent configs or show one definition."),
  slash("memory", "", "Show loaded CLAUDE.md memory bundle and files."),
  slash("status", "session", "Show runtime/session status summary."),
  slash("clear", "", "Clear ephemeral runtime state."),
  slash("compact", "", "Summarise and compact the conversation."),
  slash("cost", "", "Show total cost and duration of the current session."),
  slash("exit", "quit", "Exit the REPL."),
  slash("diff", "", "View uncommitted working directory changes."),
  slash("files", "", "List files currently loaded in session context."),
  slash("copy", "", "Copy the last assistant response to a temp file."),
  slash("export", "", "Export the conversation to a text file."),
  slash("stats", "", "Show session usage statistics."),
  slash("tag", "", "Add or remove a searchable tag on the current session."),
  slash("rename", "", "Rename the current conversation."),
  slash("branch", "", "Create a branch of the current conversation."),
  slash("effort", "", "Show or set model effort level."),
  slash("doctor", "", "Diagnose and verify installation and settings."),
  slash("commit", "", "Create a git commit through a prompt-type command."),
  slash("pr-comments", "pr_comments", "Get comments from a GitHub pull request."),
  slash("resume", "continue", "Resume a previous conversation."),
  slash("add-dir", "", "Add a new working directory."),
  slash("skills", "", "List available skills."),
  slash("fast", "", "Toggle fast mode."),
  slash("vim", "", "Toggle between Vim and normal editing modes."),
  slash("rewind", "checkpoint", "Restore the conversation to a previous point."),
  slash("init", "", "Analyze the project and generate a CLAUDE.md file."),
];

const allCommandItems = [...commandItems, ...devCommandItems, ...slashCommandItems];

const toolItems = [
  tool("list_dir", "Workspace read", "List files and directories under a workspace path.", "read"),
  tool("read_file", "Workspace read", "Read a file with optional line ranges and numbered output.", "read"),
  tool("write_file", "Workspace write", "Write or create a complete file.", "write"),
  tool("edit_file", "Workspace write", "Replace an exact string in a file.", "write"),
  tool("multi_edit", "Workspace write", "Apply ordered exact-string edits to one file.", "write"),
  tool("notebook_edit", "Workspace write", "Edit or append Jupyter notebook cells.", "write"),
  tool("glob_search", "Workspace read", "Find files by glob pattern.", "read"),
  tool("grep_search", "Workspace read", "Search file contents by regex or literal string.", "read"),
  tool("bash", "Shell", "Run shell commands subject to permission and destructive-command checks.", "shell"),
  tool("LSP", "Code intelligence", "Definitions, references, hover, symbols, implementation, and hierarchy.", "read"),
  tool("web_fetch", "Integration", "Fetch text from http, https, or file URLs.", "read"),
  tool("search_status", "Search", "Show search runtime summary or provider details.", "read"),
  tool("search_list_providers", "Search", "List configured local search providers.", "read"),
  tool("search_activate_provider", "Search", "Set the active local search provider.", "write"),
  tool("web_search", "Search", "Run a search through the active provider.", "read"),
  tool("tool_search", "Tooling", "Search the active tool registry by name or description.", "read"),
  tool("list_available_tools", "Tooling", "List tool names available in the session.", "read"),
  tool("sleep", "Tooling", "Bounded local wait for short polling flows.", "read"),
  tool("ask_user_question", "Collaboration", "Request an answer through ask-user runtime queues or interaction.", "agent"),
  tool("account_status", "Account", "Show account runtime summary or one profile.", "read"),
  tool("account_list_profiles", "Account", "List local account profiles.", "read"),
  tool("account_login", "Account", "Persist an active account profile or ephemeral identity.", "write"),
  tool("account_logout", "Account", "Clear active account session state.", "write"),
  tool("config_list", "Config", "List merged or source-specific workspace config keys.", "read"),
  tool("config_get", "Config", "Read a config value by dotted key path.", "read"),
  tool("config_set", "Config", "Write a config value into a selected source.", "write"),
  tool("mcp_list_resources", "MCP", "List MCP resources discovered from manifests and servers.", "read"),
  tool("mcp_read_resource", "MCP", "Read an MCP resource by URI.", "read"),
  tool("mcp_list_tools", "MCP", "List MCP tools exposed by configured servers.", "read"),
  tool("mcp_call_tool", "MCP", "Call a configured MCP server tool.", "agent"),
  tool("remote_status", "Remote", "Show local remote runtime summary.", "read"),
  tool("remote_list_profiles", "Remote", "List local remote profiles.", "read"),
  tool("remote_connect", "Remote", "Persist an active remote target/profile.", "write"),
  tool("remote_disconnect", "Remote", "Disconnect the active remote target.", "write"),
  tool("worktree_status", "Worktree", "Show managed git worktree state.", "read"),
  tool("worktree_enter", "Worktree", "Create and enter a managed git worktree.", "write"),
  tool("worktree_exit", "Worktree", "Exit the active managed worktree session.", "write"),
  tool("workflow_list", "Workflow", "List local workflow definitions.", "read"),
  tool("workflow_get", "Workflow", "Show one workflow definition.", "read"),
  tool("workflow_run", "Workflow", "Record and render a workflow run.", "agent"),
  tool("remote_trigger", "Remote trigger", "Create, update, inspect, or run local remote triggers.", "agent"),
  tool("plan_get", "Plan", "Inspect current plan runtime state.", "read"),
  tool("update_plan", "Plan", "Replace the current local runtime plan.", "write"),
  tool("plan_clear", "Plan", "Clear the current local plan.", "write"),
  tool("task_next", "Task", "Show next actionable tasks.", "read"),
  tool("team_list", "Team", "List local collaboration teams.", "read"),
  tool("team_get", "Team", "Show one collaboration team.", "read"),
  tool("team_create", "Team", "Create a locally stored collaboration team.", "write"),
  tool("team_delete", "Team", "Delete a team and its recorded messages.", "write"),
  tool("send_message", "Team", "Persist a local collaboration message.", "write"),
  tool("team_messages", "Team", "Show recorded team messages.", "read"),
  tool("task_list", "Task", "List locally stored runtime tasks.", "read"),
  tool("task_get", "Task", "Show a task by id.", "read"),
  tool("task_create", "Task", "Create a locally stored runtime task.", "write"),
  tool("task_update", "Task", "Update a runtime task by id.", "write"),
  tool("task_start", "Task", "Mark a task in progress or blocked by dependencies.", "write"),
  tool("task_complete", "Task", "Mark a task completed and release dependents.", "write"),
  tool("task_block", "Task", "Mark a task blocked with optional dependencies.", "write"),
  tool("task_cancel", "Task", "Mark a task cancelled.", "write"),
  tool("EnterPlanMode", "Planning mode", "Switch into read-oriented planning mode.", "agent"),
  tool("ExitPlanMode", "Planning mode", "Exit planning mode.", "agent"),
  tool("TaskOutput", "Background task", "Read or wait for a background task result.", "read"),
  tool("TaskStop", "Background task", "Stop a running background task.", "write"),
  tool("todo_write", "Task", "Replace the task list with a structured todo list.", "write"),
  tool("Agent", "Subagent", "Launch specialized child agents with optional batching and dependencies.", "agent"),
  tool("delegate_agent", "Subagent", "Legacy child-agent delegation tool.", "agent"),
  tool("Skill", "Skill", "Execute a registered skill in the main conversation.", "agent"),
];

const promptModes = [
  { id: "new", label: "New prompt", note: "Fresh run() path with a new session id and scratchpad." },
  { id: "resume", label: "Resume prompt", note: "resume() hydrates persisted state before _run_prompt()." },
  { id: "slash", label: "Slash command", note: "Local command preprocessing may return before any model call." },
  { id: "tool", label: "Tool follow-up", note: "Tool results become the next model context." },
];

const phases = ["prompt", "context", "budget", "model", "tools"];

const steps = [
  step("prompt-entry", "prompt", "Prompt enters LocalCodingAgent", "src/agent/runtime/agent.py::run", ["new", "resume", "slash", "tool"], "run() creates a session id and scratchpad. resume() rebuilds a persisted AgentSessionState before entering _run_prompt().", ["user prompt", "runtime_config", "model_config"], ["session_id", "scratchpad_directory", "base_session or None"], ["scratchpadDirectory", "session id"], ["runtime", "scratchpad"]),
  step("slash-command", "prompt", "Slash command preprocessing", "src/agent/commands/slash.py::preprocess_slash_command", ["new", "resume", "slash"], "Local slash commands can return immediately without contacting the model when should_query is false.", ["raw prompt"], ["slash_result.prompt", "slash_result.output", "should_query"], ["local command transcript"], ["runtime"]),
  step("before-hooks", "prompt", "Before-prompt hooks rewrite the prompt", "src/agent/runtime/agent.py::_apply_*_before_prompt_hooks", ["new", "resume", "tool"], "Hook policy messages, plugin before-prompt messages, and resume hooks can wrap the prompt in system-reminder blocks.", ["raw or slash-rewritten prompt", "hook policy runtime", "plugin runtime"], ["effective_prompt"], ["hook policy reminders", "plugin before-prompt hooks"], ["plugin", "user-context"]),
  step("managed-agent", "prompt", "Managed agent run is recorded", "src/agent/runtime/manager.py::AgentManager.start_agent", ["new", "resume", "tool"], "AgentManager records prompt, parent/child metadata, group id, label, and resume source for later reporting.", ["effective_prompt", "parent_agent_id", "managed labels"], ["managed_agent_id"], ["agent manager summary"], ["runtime"]),
  step("build-session", "context", "Session and prompt context are built", "src/agent/runtime/agent.py::build_session", ["new"], "build_session() creates PromptContext, system prompt parts, and AgentSessionState with captured user and system context dictionaries.", ["runtime_config", "model_config", "scratchpad_directory"], ["AgentSessionState", "PromptContext", "system_prompt_parts"], ["user_context", "system_context", "system prompt parts"], ["runtime", "model", "prompt-context", "system-prompt", "session"]),
  step("resume-session", "context", "Persisted session is hydrated", "src/agent/runtime/agent.py::resume", ["resume"], "resume() reconstructs AgentSessionState from stored prompt parts, context dictionaries, messages, file history, and compaction replay.", ["StoredAgentSession", "stored plugin_state", "stored file_history"], ["base_session", "resume_source_session_id"], ["stored context", "file history replay"], ["session", "persisted"]),
  step("context-snapshot", "context", "AgentContextSnapshot captures environment", "src/agent/context/snapshot.py::build_context_snapshot", ["new"], "The snapshot resolves cwd, shell, platform, current date, git flags, scratchpad, extra directories, user context, and system context.", ["AgentRuntimeConfig", "scratchpad_directory", "AgentRuntimeDependencies"], ["AgentContextSnapshot"], ["cwd", "shell", "platform", "currentDate", "git flags"], ["runtime", "scratchpad", "system-context", "user-context"]),
  step("user-context", "context", "User context sources are gathered", "src/agent/context/snapshot.py::_build_user_context", ["new"], "User context can include current date, scratchpad guidance, CLAUDE.md memory, plugins, hook policy, MCP, remote, search, account, config, LSP, plan, tasks, team, workflow, and worktree summaries.", ["cwd", "additional directories", "runtime dependencies"], ["user_context dictionary"], ["currentDate", "claudeMd", "pluginRuntime", "mcpRuntime", "searchRuntime", "taskRuntime"], ["memory", "plugin", "mcp", "workspace-state", "user-context"]),
  step("system-context", "context", "System context is gathered", "src/agent/context/snapshot.py::get_system_context", ["new"], "System context adds git status, optional cache breaker, and scratchpad directory. Git status is a start-of-run snapshot.", ["cwd", "scratchpad_directory", "system prompt injection"], ["system_context dictionary"], ["gitStatus", "cacheBreaker", "scratchpadDirectory"], ["git", "scratchpad", "system-context"]),
  step("prompt-parts", "context", "System prompt parts are assembled", "src/agent/context/prompting.py::build_system_prompt_parts", ["new"], "System prompt parts combine default instructions, tool guidance, agent guidance, runtime guidance, environment info, and optional overrides.", ["PromptContext", "tool registry", "available agents"], ["system_prompt_parts"], ["tool guidance", "runtime guidance", "environment section"], ["prompt-context", "tool-registry", "system-prompt"]),
  step("tool-registry", "context", "Tool registry is selected for the prompt", "src/agent/tools/registry.py::build_effective_tool_registry", ["new", "resume", "tool"], "The built-in registry is expanded with plugin aliases and virtual tools. Large Ollama schemas can be compacted.", ["built-in tools", "plugin aliases", "virtual tools", "model base_url"], ["effective registry", "OpenAI function tool specs"], ["tool names", "tool schemas"], ["tool-registry", "tool-specs"]),
  step("append-user", "prompt", "Effective prompt becomes a user message", "src/agent/models/session.py::AgentSessionState.append_user", ["new", "resume", "tool"], "The effective prompt is appended after local slash handling and hook rewrites.", ["effective_prompt", "AgentSessionState"], ["session user message"], ["user message"], ["session", "messages"]),
  step("run-state", "budget", "PromptRunState initializes counters", "src/agent/runtime/agent.py::_build_prompt_run_state", ["new", "resume", "tool"], "PromptRunState tracks usage, cost, previous tool calls, delegated tasks, file history, model calls, stream events, and turn index.", ["session", "file history", "stored resume budget state"], ["PromptRunState"], ["usage", "cost", "file_history", "stream_events"], ["runtime", "events"]),
  step("pressure-passes", "budget", "Context pressure passes run before each model call", "src/agent/context/pressure.py", ["new", "resume", "tool"], "Snipping and compaction passes reduce older context before prompt-length preflight when thresholds require it.", ["session messages", "runtime thresholds"], ["possibly reduced session", "stream events"], ["snipped_message", "compact_boundary"], ["messages", "budget", "events"]),
  step("prompt-preflight", "budget", "Token budget preflight checks the next request", "src/agent/context/pressure.py::preflight_prompt_length", ["new", "resume", "tool"], "Preflight calculates projected prompt tokens, tries pressure reduction or summary compaction, and blocks calls that still exceed hard limits.", ["session", "model", "BudgetConfig", "output schema"], ["PromptPreflightResult", "budget events"], ["projected_input_tokens", "soft limit", "hard limit"], ["budget", "messages"]),
  step("model-request", "model", "Model request is formed", "src/agent/runtime/model_turn.py::query_model_turn", ["new", "resume", "tool"], "The session messages, tool specs, streaming setting, and optional output schema are sent to the LLM client.", ["OpenAI messages", "tool specs", "output schema"], ["AssistantTurn"], ["messages payload", "tool specs"], ["messages", "tool-specs", "llm"]),
  step("assistant-turn", "model", "Assistant turn is normalized", "src/agent/runtime/agent.py::_process_model_turn", ["new", "resume", "tool"], "The runtime accumulates usage and cost, checks budgets again, and chooses final output, continuation, or tool execution.", ["AssistantTurn", "UsageStats"], ["TurnLoopDirective"], ["assistant content", "tool calls", "finish_reason"], ["llm", "events"]),
  step("tool-execution", "tools", "Tool calls are executed", "src/agent/runtime/tool_calls.py::execute_runtime_tool_call", ["tool", "new", "resume"], "Tool execution emits events, applies plugin and hook-policy checks, streams output, finalizes the result, and records metadata.", ["ToolCall", "tool registry", "ToolExecutionContext", "hooks"], ["ToolExecutionResult", "ToolCallExecutionOutcome"], ["tool result JSON", "tool metadata", "permission denial events"], ["tool-specs", "tool-result", "events"]),
  step("runtime-refresh", "tools", "Runtime views refresh after state-changing tools", "src/agent/runtime/agent.py::_refresh_runtime_views_for_tool_result", ["tool", "new", "resume"], "State-changing tools reload affected runtime views so later prompts see fresh search, remote, account, config, task, plan, team, workflow, or worktree state.", ["tool name", "tool result metadata"], ["updated runtime dependencies", "updated ToolExecutionContext"], ["fresh runtime summaries"], ["tool-result", "workspace-state", "user-context"]),
  step("follow-up-context", "tools", "Tool result becomes follow-up context", "src/agent/models/session.py::AgentSessionState.append_tool", ["tool", "new", "resume"], "The finalized tool result is serialized as a tool message. Plugins or hook policy may add follow-up messages before the next loop.", ["ToolExecutionResult", "plugin messages", "hook policy messages"], ["tool message", "optional runtime user message"], ["tool output", "runtime follow-up message"], ["tool-result", "messages", "events"]),
  step("finish-or-loop", "model", "Run loops or finalizes", "src/agent/runtime/agent.py::_run_prompt", ["new", "resume", "tool"], "If there are tool calls, the loop continues with richer context. Otherwise output is finalized and after-turn events are added.", ["TurnLoopDirective", "PromptRunState"], ["AgentRunResult"], ["final output", "events", "transcript"], ["messages", "events", "persisted"]),
  step("persist-session", "tools", "Session persistence saves the run", "src/agent/runtime/persistence.py::persist_agent_run", ["new", "resume", "tool"], "Persistence writes transcript, prompt parts, context dictionaries, messages, file history, usage, cost, permissions, and plugin state for resume.", ["AgentSessionState", "AgentRunResult"], ["StoredAgentSession"], ["stored context", "stored messages", "stored plugin state"], ["session", "persisted"]),
];

const graphNodes = [
  { id: "runtime", label: "AgentRuntimeConfig", kind: "config", x: 40, y: 40 },
  { id: "model", label: "ModelConfig", kind: "config", x: 40, y: 150 },
  { id: "scratchpad", label: "Scratchpad", kind: "workspace", x: 40, y: 260 },
  { id: "memory", label: "CLAUDE.md + rules", kind: "memory", x: 260, y: 35 },
  { id: "git", label: "Git snapshot", kind: "system", x: 260, y: 125 },
  { id: "plugin", label: "Plugins + hooks", kind: "runtime", x: 260, y: 215 },
  { id: "mcp", label: "MCP + integrations", kind: "runtime", x: 260, y: 305 },
  { id: "workspace-state", label: "Tasks, plan, team, workflow, remote", kind: "state", x: 260, y: 395 },
  { id: "user-context", label: "user_context", kind: "dict", x: 505, y: 90 },
  { id: "system-context", label: "system_context", kind: "dict", x: 505, y: 250 },
  { id: "prompt-context", label: "PromptContext", kind: "dataclass", x: 705, y: 85 },
  { id: "tool-registry", label: "Effective tool registry", kind: "tools", x: 705, y: 205 },
  { id: "system-prompt", label: "system_prompt_parts", kind: "prompt", x: 905, y: 75 },
  { id: "session", label: "AgentSessionState", kind: "session", x: 905, y: 195 },
  { id: "messages", label: "OpenAI messages", kind: "payload", x: 1115, y: 120 },
  { id: "tool-specs", label: "Tool specs", kind: "payload", x: 1115, y: 250 },
  { id: "budget", label: "Token budget", kind: "guard", x: 905, y: 355 },
  { id: "llm", label: "LLM request", kind: "model", x: 1310, y: 165 },
  { id: "tool-result", label: "Tool result", kind: "feedback", x: 1310, y: 315 },
  { id: "events", label: "Stream events", kind: "telemetry", x: 1115, y: 410 },
  { id: "persisted", label: "Stored session", kind: "storage", x: 1310, y: 440 },
];

const graphEdges = [
  ["runtime", "user-context"], ["runtime", "system-context"], ["runtime", "prompt-context"],
  ["model", "prompt-context"], ["scratchpad", "user-context"], ["scratchpad", "system-context"],
  ["memory", "user-context"], ["git", "system-context"], ["plugin", "user-context"],
  ["mcp", "user-context"], ["workspace-state", "user-context"], ["user-context", "prompt-context"],
  ["system-context", "prompt-context"], ["prompt-context", "system-prompt"], ["tool-registry", "system-prompt"],
  ["tool-registry", "tool-specs"], ["system-prompt", "session"], ["prompt-context", "session"],
  ["session", "messages"], ["session", "budget"], ["budget", "messages"], ["messages", "llm"],
  ["tool-specs", "llm"], ["llm", "tool-result"], ["tool-result", "messages"],
  ["tool-result", "workspace-state"], ["tool-result", "events"], ["llm", "events"],
  ["session", "persisted"], ["events", "persisted"],
];

const budgetFlow = [
  {
    title: "Build request",
    detail: "Messages, tool specs, model config, budget config, and optional output schema are ready.",
    event: "session.to_openai_messages()",
    tone: "blue",
  },
  {
    title: "Preflight",
    detail: "Projected input tokens are compared with soft and hard limits before the backend call.",
    event: "prompt_length_check",
    tone: "green",
  },
  {
    title: "Reduce pressure",
    detail: "Older context is snipped or compacted while recent messages and tool-call structure are preserved.",
    event: "snip_boundary / compact_boundary",
    tone: "amber",
  },
  {
    title: "Retry sizing",
    detail: "The reduced session is measured again. Prompt-too-long failures can trigger reactive compaction.",
    event: "prompt_length_recovery",
    tone: "blue",
  },
  {
    title: "Proceed or block",
    detail: "The model call proceeds when under limits. If hard limits remain exceeded, the run stops first.",
    event: "prompt_length_warning",
    tone: "red",
  },
];

const budgetLimits = [
  {
    label: "Token caps",
    value: "max_total_tokens, max_input_tokens, max_output_tokens, max_reasoning_tokens",
    source: "src/agent/context/pressure.py",
  },
  {
    label: "Run caps",
    value: "max_budget_usd, max_tool_calls, max_delegated_tasks, max_model_calls, max_session_turns",
    source: "src/agent/models/types.py",
  },
];

const eventGroups = [
  {
    title: "Prompt and model",
    events: ["content_delta", "message_stop", "continuation_request"],
    source: "src/agent/runtime/model_turn.py",
  },
  {
    title: "Budget and pressure",
    events: ["prompt_length_check", "prompt_length_warning", "snip_boundary", "compact_boundary"],
    source: "src/agent/context/pressure.py",
  },
  {
    title: "Tool execution",
    events: ["tool_start", "tool_delta", "tool_result", "tool_permission_denial"],
    source: "src/agent/runtime/tool_calls.py",
  },
  {
    title: "Plugins, hooks, delegation",
    events: ["plugin_tool_preflight", "hook_policy_tool_block", "delegate_subtask_start"],
    source: "src/agent/runtime/agent.py",
  },
];

const modules = [
  module("src/agent/runtime/", "Core runtime", "LocalCodingAgent, prompt loop, resume, context pressure, tool execution, runtime refresh, and persistence.", "agent"),
  module("src/agent/context/", "Context package", "Context snapshots, prompt assembly, usage accounting, and budget pressure helpers.", "context"),
  module("src/agent/tools/", "Tool package", "Tool schema registry, execution handlers, bash security, and permission-aware contexts.", "tools"),
  module("src/cli/", "CLI package", "Argparse construction, command dispatch, config builders, and live agent rendering.", "cli"),
  module("src/features/integration/", "Integration package", "Plugin, MCP, search, remote, remote trigger, and LSP runtimes.", "integrations"),
  module("src/features/orchestration/", "Orchestration package", "Plan, task, workflow, and worktree runtime state.", "orchestration"),
  module("src/features/system/", "System package", "Account, background, config, doctor, env, hook policy, and tokenizer surfaces.", "system"),
  module("src/features/collaboration/", "Collaboration package", "Ask-user and team state with CLI, slash, and tool access.", "collaboration"),
  module("src/session/", "Session package", "Session store, transcript, compaction, microcompaction, and resume data.", "session"),
  module("src/ui/ and src/textual_ui.py", "UI layer", "Textual state, conversation history, slash command UI, formatting, and app orchestration.", "ui"),
  module("src/llm/", "LLM layer", "LiteLLM client, backend selection, parsers, and transport errors.", "llm"),
  module("tests/ and benchmarks/", "Verification", "Unit coverage and benchmark suites for runtime, tools, CLI, orchestration, and integrations.", "verification"),
];

const permissionModel = [
  {
    id: "read",
    label: "Read/default",
    rule: "Workspace inspection and status-style tools can run in read-oriented sessions.",
    gates: ["no write flag", "no shell flag", "safe for planning"],
    examples: ["read_file", "grep_search", "mcp_read_resource", "lsp-hover"],
    source: "src/agent/tools/registry.py",
  },
  {
    id: "write",
    label: "Write-capable",
    rule: "File, config, task, team, worktree, and account mutations require write permission.",
    gates: ["--no-write blocks", "CLAW_NO_WRITE=1 blocks", "state-changing"],
    examples: ["write_file", "multi_edit", "config_set", "task_update"],
    source: "src/agent/tools/execution.py",
  },
  {
    id: "shell",
    label: "Shell",
    rule: "Shell execution is separately gated and destructive commands require unsafe mode.",
    gates: ["--no-shell blocks", "CLAW_NO_SHELL=1 blocks", "--unsafe for destructive"],
    examples: ["bash", "git status", "python -m unittest"],
    source: "src/agent/tools/bash_security.py",
  },
  {
    id: "agent",
    label: "Runtime action",
    rule: "Delegation, MCP calls, workflow runs, and ask-user flows act through runtime services.",
    gates: ["tool preflight", "hook policy", "delegation budget"],
    examples: ["Agent", "mcp_call_tool", "workflow_run", "ask_user_question"],
    source: "src/agent/runtime/tool_calls.py",
  },
];

const stateFlow = [
  {
    tone: "green",
    title: "Turn state",
    badge: "in memory",
    detail: "The active run tracks messages, usage, cost, file history, stream events, and tool-call loop state.",
    artifacts: ["AgentSessionState", "PromptRunState", "ToolExecutionContext"],
    source: "src/agent/runtime/state.py",
  },
  {
    tone: "blue",
    title: "Session artifact",
    badge: ".port_sessions/agent",
    detail: "Completed runs persist transcript data, prompt parts, context dictionaries, messages, file history, permissions, and plugin state.",
    artifacts: ["transcript", "system_prompt_parts", "user_context", "plugin_state"],
    source: "src/agent/runtime/persistence.py",
  },
  {
    tone: "amber",
    title: "Scratchpad",
    badge: ".port_sessions/scratchpad",
    detail: "Scratchpad storage gives runs a workspace-specific place for notes and guidance without mixing it into source files.",
    artifacts: ["scratchpadDirectory", "workspace notes", "session guidance"],
    source: "src/cli/agent_cli_config.py",
  },
  {
    tone: "violet",
    title: "Resume",
    badge: "resume()",
    detail: "Resume rebuilds the session from stored prompt parts, context dictionaries, messages, file history, and compaction replay.",
    artifacts: ["stored messages", "file history replay", "base session"],
    source: "src/agent/runtime/agent.py::resume",
  },
];

const controlSurfaces = [
  {
    surface: "CLI command",
    entry: "src/cli/parser_build.py",
    route: "src/cli/command_dispatch.py",
    targets: ["agent one-shot", "feature runtimes", "diagnostics"],
  },
  {
    surface: "Slash command",
    entry: "src/agent/commands/slash.py",
    route: "preprocess_slash_command()",
    targets: ["local answer", "rewritten prompt", "runtime status"],
  },
  {
    surface: "Textual UI",
    entry: "src/textual_ui.py",
    route: "src/ui/",
    targets: ["conversation state", "prompt composer", "agent session"],
  },
  {
    surface: "Model tool call",
    entry: "src/agent/tools/registry.py",
    route: "execute_runtime_tool_call()",
    targets: ["workspace tools", "integrations", "orchestration state"],
  },
  {
    surface: "Background run",
    entry: "agent-bg / daemon",
    route: "src/features/system/background_runtime.py",
    targets: ["stored output", "logs", "attach/kill controls"],
  },
];

const state = {
  commandCategory: "All",
  commandQuery: "",
  toolCategory: "All",
  toolQuery: "",
  mode: "new",
  phase: "all",
  stepQuery: "",
  selectedStepId: "prompt-entry",
};

const els = {
  themeToggle: document.querySelector("#theme-toggle"),
  themeLabel: document.querySelector("#theme-label"),
  metrics: document.querySelector("#repo-metrics"),
  architecture: document.querySelector("#architecture-map"),
  capability: document.querySelector("#capability-map"),
  permission: document.querySelector("#permission-matrix"),
  stateMap: document.querySelector("#state-map"),
  controlMap: document.querySelector("#control-map"),
  commandSearch: document.querySelector("#command-search"),
  commandFilters: document.querySelector("#command-filters"),
  commandGrid: document.querySelector("#command-grid"),
  commandCount: document.querySelector("#command-count"),
  toolSearch: document.querySelector("#tool-search"),
  toolFilters: document.querySelector("#tool-filters"),
  toolGrid: document.querySelector("#tool-grid"),
  toolCount: document.querySelector("#tool-count"),
  mode: document.querySelector("#prompt-mode"),
  phaseFilters: document.querySelector("#phase-filters"),
  stepSearch: document.querySelector("#step-search"),
  timeline: document.querySelector("#timeline"),
  detail: document.querySelector("#step-detail"),
  visibleCount: document.querySelector("#visible-count"),
  graph: document.querySelector("#context-graph"),
  resetGraph: document.querySelector("#reset-graph"),
  budget: document.querySelector("#budget-visual"),
  events: document.querySelector("#event-groups"),
  modules: document.querySelector("#module-map"),
};

function init() {
  syncThemeControl();
  renderMetrics();
  renderArchitecture();
  renderCapabilities();
  renderPermissionModel();
  renderStateFlow();
  renderControlMap();
  renderCommandExplorer();
  renderToolExplorer();
  renderPipelineControls();
  renderTimeline();
  renderBudget();
  renderEvents();
  renderModules();
  renderGraph();
  bindEvents();
}

function bindEvents() {
  els.themeToggle.addEventListener("click", toggleTheme);
  els.commandSearch.addEventListener("input", () => {
    state.commandQuery = els.commandSearch.value.trim().toLowerCase();
    renderCommandCards();
  });
  els.toolSearch.addEventListener("input", () => {
    state.toolQuery = els.toolSearch.value.trim().toLowerCase();
    renderToolCards();
  });
  els.mode.addEventListener("change", () => {
    state.mode = els.mode.value;
    const visible = getVisibleSteps();
    state.selectedStepId = visible[0]?.id || null;
    renderTimeline();
    renderGraph();
  });
  els.stepSearch.addEventListener("input", () => {
    state.stepQuery = els.stepSearch.value.trim().toLowerCase();
    renderTimeline();
  });
  els.resetGraph.addEventListener("click", () => {
    state.selectedStepId = getVisibleSteps()[0]?.id || "prompt-entry";
    renderTimeline();
    renderGraph();
  });
}

function toggleTheme() {
  const current = getTheme();
  setTheme(current === "dark" ? "light" : "dark");
}

function getTheme() {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem("claw-docs-theme", theme);
  } catch {
    // Theme persistence is optional; the visible toggle still updates.
  }
  syncThemeControl();
}

function syncThemeControl() {
  const theme = getTheme();
  const next = theme === "dark" ? "light" : "dark";
  els.themeToggle.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
  els.themeToggle.setAttribute("aria-label", `Switch to ${next} theme`);
  els.themeToggle.title = `Switch to ${next} theme`;
  els.themeLabel.textContent = next === "dark" ? "Dark" : "Light";
}

function renderMetrics() {
  els.metrics.replaceChildren();
  for (const metric of repoMetrics) {
    const card = document.createElement("article");
    card.className = "metric-card";
    const value = document.createElement("strong");
    value.textContent = metric.value;
    const label = document.createElement("span");
    label.textContent = metric.label;
    const detail = document.createElement("span");
    detail.textContent = metric.detail;
    card.append(value, label, detail);
    els.metrics.append(card);
  }
}

function renderArchitecture() {
  els.architecture.replaceChildren();
  for (const row of architectureRows) {
    const wrapper = document.createElement("div");
    wrapper.className = "arch-row";
    const label = document.createElement("div");
    label.className = "arch-label";
    label.textContent = row.label;
    const cells = document.createElement("div");
    cells.className = "arch-cells";
    for (const [title, body, tone] of row.cells) {
      const cell = document.createElement("article");
      cell.className = "arch-cell";
      cell.dataset.tone = tone;
      const strong = document.createElement("strong");
      strong.textContent = title;
      const span = document.createElement("span");
      span.textContent = body;
      cell.append(strong, span);
      cells.append(cell);
    }
    wrapper.append(label, cells);
    els.architecture.append(wrapper);
  }
}

function renderCapabilities() {
  els.capability.replaceChildren();
  for (const capability of capabilities) {
    const card = document.createElement("article");
    card.className = "capability-card";
    const title = document.createElement("h3");
    title.textContent = capability.name;
    const summary = document.createElement("p");
    summary.textContent = capability.summary;
    const tags = document.createElement("div");
    tags.className = "capability-tags";
    for (const item of capability.items) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = item;
      tags.append(tag);
    }
    card.append(title, summary, tags);
    els.capability.append(card);
  }
}

function renderPermissionModel() {
  els.permission.replaceChildren();
  for (const item of permissionModel) {
    const lane = document.createElement("article");
    lane.className = "permission-lane";
    lane.dataset.permission = item.id;

    const head = document.createElement("header");
    head.className = "permission-head";
    const title = document.createElement("h3");
    title.textContent = item.label;
    const badge = document.createElement("span");
    badge.className = `permission-badge tag ${item.id}`;
    badge.textContent = permissionLabel(item.id);
    head.append(title, badge);

    const rule = document.createElement("p");
    rule.textContent = item.rule;

    const gates = document.createElement("div");
    gates.className = "permission-rule-list";
    for (const gate of item.gates) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = gate;
      gates.append(tag);
    }

    const examples = document.createElement("div");
    examples.className = "permission-examples";
    for (const example of item.examples) {
      const code = document.createElement("code");
      code.textContent = example;
      examples.append(code);
    }

    const source = document.createElement("p");
    source.className = "detail-meta";
    source.textContent = item.source;
    lane.append(head, rule, gates, examples, source);
    els.permission.append(lane);
  }
}

function renderStateFlow() {
  els.stateMap.replaceChildren();
  stateFlow.forEach((item, index) => {
    const card = document.createElement("article");
    card.className = "state-card";
    card.dataset.tone = item.tone;

    const head = document.createElement("header");
    head.className = "state-head";
    const title = document.createElement("h3");
    title.textContent = item.title;
    const badge = document.createElement("span");
    badge.className = "state-badge";
    badge.textContent = item.badge;
    head.append(title, badge);

    const detail = document.createElement("p");
    detail.textContent = item.detail;

    const artifacts = document.createElement("div");
    artifacts.className = "state-artifacts";
    for (const artifact of item.artifacts) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = artifact;
      artifacts.append(tag);
    }

    const source = document.createElement("code");
    source.textContent = item.source;
    card.append(head, detail, artifacts, source);
    els.stateMap.append(card);

    if (index < stateFlow.length - 1) {
      const arrow = document.createElement("span");
      arrow.className = "state-arrow";
      arrow.textContent = ">";
      els.stateMap.append(arrow);
    }
  });
}

function renderControlMap() {
  els.controlMap.replaceChildren();
  for (const item of controlSurfaces) {
    const row = document.createElement("article");
    row.className = "control-row";

    row.append(
      controlNode("Surface", item.surface, item.entry),
      controlArrow(),
      controlNode("Route", item.route, "shared runtime boundary"),
      controlArrow(),
      controlTargetNode(item.targets),
    );
    els.controlMap.append(row);
  }
}

function controlNode(label, title, detail) {
  const node = document.createElement("div");
  node.className = "control-node";
  const head = document.createElement("header");
  const h3 = document.createElement("h3");
  h3.textContent = title;
  const badge = document.createElement("span");
  badge.className = "control-badge";
  badge.textContent = label;
  head.append(h3, badge);
  const code = document.createElement("code");
  code.textContent = detail;
  node.append(head, code);
  return node;
}

function controlTargetNode(targets) {
  const node = document.createElement("div");
  node.className = "control-node";
  const head = document.createElement("header");
  const title = document.createElement("h3");
  title.textContent = "Runtime target";
  const badge = document.createElement("span");
  badge.className = "control-badge";
  badge.textContent = "Effect";
  head.append(title, badge);
  const list = document.createElement("div");
  list.className = "control-targets";
  for (const target of targets) {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = target;
    list.append(tag);
  }
  node.append(head, list);
  return node;
}

function controlArrow() {
  const arrow = document.createElement("span");
  arrow.className = "control-arrow";
  arrow.textContent = ">";
  return arrow;
}

function renderCommandExplorer() {
  const categories = ["All", ...unique(allCommandItems.map((item) => item.category))];
  renderFilterButtons(els.commandFilters, categories, state.commandCategory, (category) => {
    state.commandCategory = category;
    renderCommandExplorer();
  });
  renderCommandCards();
}

function renderCommandCards() {
  const visible = allCommandItems.filter((item) => {
    if (state.commandCategory !== "All" && item.category !== state.commandCategory) {
      return false;
    }
    if (!state.commandQuery) {
      return true;
    }
    return [item.name, item.category, item.description, item.source, item.aliases, item.kind]
      .join(" ")
      .toLowerCase()
      .includes(state.commandQuery);
  });
  els.commandGrid.replaceChildren();
  els.commandCount.textContent = `${visible.length} commands shown`;
  if (!visible.length) {
    els.commandGrid.append(emptyState("No commands match the current filters."));
    return;
  }
  for (const item of visible) {
    const card = document.createElement("article");
    card.className = "command-card";
    const header = document.createElement("header");
    const name = document.createElement("strong");
    name.innerHTML = `<code>${escapeHtml(item.name)}</code>`;
    const category = document.createElement("span");
    category.className = "tag";
    category.textContent = item.category;
    header.append(name, category);
    const desc = document.createElement("p");
    desc.textContent = item.description;
    const source = document.createElement("p");
    source.className = "detail-meta";
    source.textContent = item.source;
    card.append(header, desc);
    if (item.aliases) {
      const aliases = document.createElement("p");
      aliases.className = "detail-meta";
      aliases.textContent = `Aliases: ${item.aliases}`;
      card.append(aliases);
    }
    if (item.kind) {
      const kind = document.createElement("p");
      kind.className = "detail-meta";
      kind.textContent = item.kind;
      card.append(kind);
    }
    card.append(source);
    els.commandGrid.append(card);
  }
}

function renderToolExplorer() {
  const categories = ["All", ...unique(toolItems.map((item) => item.category))];
  renderFilterButtons(els.toolFilters, categories, state.toolCategory, (category) => {
    state.toolCategory = category;
    renderToolExplorer();
  });
  renderToolCards();
}

function renderToolCards() {
  const visible = toolItems.filter((item) => {
    if (state.toolCategory !== "All" && item.category !== state.toolCategory) {
      return false;
    }
    if (!state.toolQuery) {
      return true;
    }
    return [item.name, item.category, item.description, item.permission]
      .join(" ")
      .toLowerCase()
      .includes(state.toolQuery);
  });
  els.toolGrid.replaceChildren();
  els.toolCount.textContent = `${visible.length} tools shown`;
  if (!visible.length) {
    els.toolGrid.append(emptyState("No tools match the current filters."));
    return;
  }
  for (const item of visible) {
    const card = document.createElement("article");
    card.className = "tool-card";
    const header = document.createElement("header");
    const name = document.createElement("strong");
    name.innerHTML = `<code>${escapeHtml(item.name)}</code>`;
    const category = document.createElement("span");
    category.className = "tag";
    category.textContent = item.category;
    header.append(name, category);
    const desc = document.createElement("p");
    desc.textContent = item.description;
    const meta = document.createElement("div");
    meta.className = "tool-meta";
    const permission = document.createElement("span");
    permission.className = `tag ${item.permission}`;
    permission.textContent = permissionLabel(item.permission);
    const source = document.createElement("span");
    source.className = "tag";
    source.textContent = "src/agent/tools";
    meta.append(permission, source);
    card.append(header, desc, meta);
    els.toolGrid.append(card);
  }
}

function renderPipelineControls() {
  els.mode.replaceChildren();
  for (const mode of promptModes) {
    const option = document.createElement("option");
    option.value = mode.id;
    option.textContent = mode.label;
    option.title = mode.note;
    els.mode.append(option);
  }
  renderFilterButtons(els.phaseFilters, ["all", ...phases], state.phase, (phase) => {
    state.phase = phase;
    renderTimeline();
  }, titleCase);
}

function renderTimeline() {
  const visible = getVisibleSteps();
  if (!visible.some((item) => item.id === state.selectedStepId)) {
    state.selectedStepId = visible[0]?.id || null;
  }
  els.timeline.replaceChildren();
  els.visibleCount.textContent = `${visible.length} steps shown`;
  if (!visible.length) {
    els.timeline.append(emptyState("No pipeline steps match the current filters."));
    renderDetail(null);
    return;
  }
  visible.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "step-card";
    button.dataset.phase = item.phase;
    button.classList.toggle("is-selected", item.id === state.selectedStepId);
    button.addEventListener("click", () => {
      state.selectedStepId = item.id;
      renderTimeline();
      renderGraph();
    });
    const stepIndex = document.createElement("span");
    stepIndex.className = "step-index";
    stepIndex.textContent = String(index + 1);
    const main = document.createElement("span");
    main.className = "step-main";
    const title = document.createElement("span");
    title.className = "step-title";
    title.textContent = item.title;
    const meta = document.createElement("span");
    meta.className = "step-meta";
    meta.textContent = item.source;
    main.append(title, meta);
    const phase = document.createElement("span");
    phase.className = "step-phase";
    phase.textContent = item.phase;
    button.append(stepIndex, main, phase);
    els.timeline.append(button);
  });
  renderDetail(steps.find((item) => item.id === state.selectedStepId));
}

function getVisibleSteps() {
  return steps.filter((item) => {
    if (!item.modes.includes(state.mode)) {
      return false;
    }
    if (state.phase !== "all" && item.phase !== state.phase) {
      return false;
    }
    if (!state.stepQuery) {
      return true;
    }
    return [item.title, item.phase, item.source, item.summary, ...item.inputs, ...item.outputs, ...item.contextKeys]
      .join(" ")
      .toLowerCase()
      .includes(state.stepQuery);
  });
}

function renderDetail(item) {
  els.detail.replaceChildren();
  if (!item) {
    els.detail.append(emptyState("Select a step to inspect its inputs and outputs."));
    return;
  }
  const overview = section("What happens", [item.summary]);
  const io = document.createElement("div");
  io.className = "detail-section";
  io.append(detailHeading("Inputs"), list(item.inputs), detailHeading("Outputs"), list(item.outputs));
  const context = section("Context keys touched", item.contextKeys);
  const source = document.createElement("div");
  source.className = "detail-section";
  source.innerHTML = `<h3>Source</h3><p class="detail-meta"><code>${escapeHtml(item.source)}</code></p>`;
  els.detail.append(overview, io, context, source);
}

function section(title, items) {
  const block = document.createElement("div");
  block.className = "detail-section";
  block.append(detailHeading(title), list(items));
  return block;
}

function detailHeading(text) {
  const heading = document.createElement("h3");
  heading.textContent = text;
  return heading;
}

function list(items) {
  const ul = document.createElement("ul");
  ul.className = "detail-list";
  for (const item of items) {
    const li = document.createElement("li");
    li.textContent = item;
    ul.append(li);
  }
  return ul;
}

function renderGraph() {
  const selected = steps.find((item) => item.id === state.selectedStepId);
  const activeNodes = new Set(selected?.nodes || []);
  const width = 1510;
  const height = 560;
  els.graph.setAttribute("viewBox", `0 0 ${width} ${height}`);
  els.graph.replaceChildren();

  const defs = svg("defs");
  const marker = svg("marker", {
    id: "arrow",
    markerWidth: "10",
    markerHeight: "10",
    refX: "9",
    refY: "3",
    orient: "auto",
    markerUnits: "strokeWidth",
  });
  marker.append(svg("path", { d: "M0,0 L0,6 L9,3 z", fill: "#8ea09a" }));
  defs.append(marker);
  els.graph.append(defs);

  const nodeById = new Map(graphNodes.map((node) => [node.id, node]));
  for (const [from, to] of graphEdges) {
    const a = nodeById.get(from);
    const b = nodeById.get(to);
    if (!a || !b) {
      continue;
    }
    const active = activeNodes.has(from) && activeNodes.has(to);
    els.graph.append(svg("path", {
      class: `graph-edge${active ? " is-active" : ""}`,
      d: edgePath(a, b),
      "marker-end": "url(#arrow)",
    }));
  }

  for (const node of graphNodes) {
    const group = svg("g", {
      class: `graph-node${activeNodes.has(node.id) ? " is-active" : ""}`,
      transform: `translate(${node.x} ${node.y})`,
      tabindex: "0",
      role: "button",
    });
    group.addEventListener("click", () => selectStepByNode(node.id));
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectStepByNode(node.id);
      }
    });
    group.append(svg("rect", { width: "160", height: "70" }));
    addWrappedText(group, node.label, 12, 24, 136, "node-title");
    const kind = svg("text", { x: "12", y: "58", class: "node-kind" });
    kind.textContent = node.kind;
    group.append(kind);
    els.graph.append(group);
  }
}

function selectStepByNode(nodeId) {
  const visible = getVisibleSteps();
  const next = visible.find((item) => item.nodes.includes(nodeId));
  if (next) {
    state.selectedStepId = next.id;
    renderTimeline();
    renderGraph();
  }
}

function edgePath(a, b) {
  const startX = a.x + 160;
  const startY = a.y + 35;
  const endX = b.x;
  const endY = b.y + 35;
  const midX = startX + Math.max(40, (endX - startX) / 2);
  return `M${startX},${startY} C${midX},${startY} ${midX},${endY} ${endX},${endY}`;
}

function addWrappedText(group, text, x, y, maxWidth, className) {
  const words = text.split(/\s+/);
  const lineHeight = 15;
  let line = "";
  let lineIndex = 0;
  const textNode = svg("text", { x: String(x), y: String(y), class: className });
  for (const word of words) {
    const test = line ? `${line} ${word}` : word;
    if (test.length * 6.8 > maxWidth && line) {
      const tspan = svg("tspan", { x: String(x), dy: lineIndex === 0 ? "0" : String(lineHeight) });
      tspan.textContent = line;
      textNode.append(tspan);
      line = word;
      lineIndex += 1;
    } else {
      line = test;
    }
  }
  const tspan = svg("tspan", { x: String(x), dy: lineIndex === 0 ? "0" : String(lineHeight) });
  tspan.textContent = line;
  textNode.append(tspan);
  group.append(textNode);
}

function renderBudget() {
  els.budget.replaceChildren();
  const flow = document.createElement("div");
  flow.className = "budget-flow";
  budgetFlow.forEach((item, index) => {
    const card = document.createElement("article");
    card.className = "flow-node";
    card.dataset.tone = item.tone;
    const number = document.createElement("span");
    number.className = "flow-number";
    number.textContent = String(index + 1);
    const title = document.createElement("h3");
    title.textContent = item.title;
    const detail = document.createElement("p");
    detail.textContent = item.detail;
    const event = document.createElement("code");
    event.textContent = item.event;
    card.append(number, title, detail, event);
    flow.append(card);
    if (index < budgetFlow.length - 1) {
      const arrow = document.createElement("span");
      arrow.className = "flow-arrow";
      arrow.textContent = ">";
      flow.append(arrow);
    }
  });

  const limits = document.createElement("div");
  limits.className = "limit-grid";
  for (const limit of budgetLimits) {
    const item = document.createElement("article");
    item.className = "limit-card";
    const title = document.createElement("h3");
    title.textContent = limit.label;
    const value = document.createElement("p");
    value.textContent = limit.value;
    const source = document.createElement("code");
    source.textContent = limit.source;
    item.append(title, value, source);
    limits.append(item);
  }
  els.budget.append(flow, limits);
}

function renderEvents() {
  els.events.replaceChildren();
  for (const group of eventGroups) {
    const lane = document.createElement("article");
    lane.className = "event-lane";
    const title = document.createElement("h3");
    title.textContent = group.title;
    const sequence = document.createElement("div");
    sequence.className = "event-sequence";
    group.events.forEach((eventName, index) => {
      const chip = document.createElement("span");
      chip.className = "event-chip";
      chip.textContent = eventName;
      sequence.append(chip);
      if (index < group.events.length - 1) {
        const arrow = document.createElement("span");
        arrow.className = "event-arrow";
        arrow.textContent = ">";
        sequence.append(arrow);
      }
    });
    const source = document.createElement("code");
    source.textContent = group.source;
    lane.append(title, sequence, source);
    els.events.append(lane);
  }
}

function renderModules() {
  els.modules.replaceChildren();
  for (const item of modules) {
    const card = document.createElement("article");
    card.className = "module-card";
    const file = document.createElement("strong");
    file.textContent = item.file;
    const meta = document.createElement("code");
    meta.textContent = item.meta;
    const role = document.createElement("span");
    role.textContent = item.role;
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = item.area;
    card.append(file, meta, role, tag);
    els.modules.append(card);
  }
}

function renderFilterButtons(container, labels, active, onSelect, labelFormatter = (value) => value) {
  container.replaceChildren();
  for (const label of labels) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = labelFormatter(label);
    button.classList.toggle("is-active", label === active);
    button.addEventListener("click", () => onSelect(label));
    container.append(button);
  }
}

function emptyState(message) {
  const block = document.createElement("p");
  block.className = "empty-state";
  block.textContent = message;
  return block;
}

function cli(name, category, description) {
  return { name, category, description, source: "src/cli/parser_build.py", kind: "", aliases: "" };
}

function slash(name, aliases, description) {
  return {
    name: `/${name}`,
    aliases: aliases ? aliases.split(",").map((item) => `/${item.trim()}`).join(", ") : "",
    category: "Slash",
    description,
    source: "src/agent/commands/slash.py",
    kind: "local slash command",
  };
}

function tool(name, category, description, permission) {
  return { name, category, description, permission };
}

function step(id, phase, title, source, modes, summary, inputs, outputs, contextKeys, nodes) {
  return { id, phase, title, source, modes, summary, inputs, outputs, contextKeys, nodes };
}

function module(file, meta, role, area) {
  return { file, meta, role, area };
}

function permissionLabel(permission) {
  const labels = {
    read: "read/default",
    write: "write-capable",
    shell: "shell",
    agent: "runtime",
  };
  return labels[permission] || permission;
}

function unique(values) {
  return [...new Set(values)].sort((a, b) => a.localeCompare(b));
}

function svg(name, attrs = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) {
    element.setAttribute(key, value);
  }
  return element;
}

function titleCase(value) {
  return value === "all" ? "All" : value.slice(0, 1).toUpperCase() + value.slice(1);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

init();
