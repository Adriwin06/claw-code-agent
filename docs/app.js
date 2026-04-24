const promptModes = [
  {
    id: "new",
    label: "New prompt",
    note: "Fresh run() path with a new session id and scratchpad.",
  },
  {
    id: "resume",
    label: "Resume prompt",
    note: "resume() path that hydrates persisted messages and plugin state first.",
  },
  {
    id: "slash",
    label: "Slash command",
    note: "A slash command can return before any model call when should_query is false.",
  },
  {
    id: "tool",
    label: "Tool follow-up",
    note: "A model turn requested tools, so tool results become the next context.",
  },
];

const phases = ["prompt", "context", "budget", "model", "tools"];

const steps = [
  {
    id: "prompt-entry",
    phase: "prompt",
    title: "Prompt enters LocalCodingAgent",
    source: "src/agent/runtime/agent.py:421",
    modes: ["new", "resume", "slash", "tool"],
    summary:
      "run() creates a session id and scratchpad. resume() restores a persisted AgentSessionState, file history, compaction replay, and plugin session state before re-entering _run_prompt().",
    inputs: ["user prompt", "runtime_config", "model_config"],
    outputs: ["session_id", "scratchpad_directory", "base_session or None"],
    contextKeys: ["scratchpadDirectory", "session id"],
    nodes: ["runtime", "scratchpad"],
  },
  {
    id: "slash-command",
    phase: "prompt",
    title: "Slash command preprocessing",
    source: "src/agent/runtime/agent.py:489",
    modes: ["new", "resume", "slash"],
    summary:
      "preprocess_slash_command() can handle local commands before the LLM is contacted. If it is handled and should_query is false, the run returns immediately.",
    inputs: ["raw prompt"],
    outputs: ["slash_result.prompt", "slash_result.output", "should_query"],
    contextKeys: ["command transcript when handled locally"],
    nodes: ["runtime"],
  },
  {
    id: "before-hooks",
    phase: "prompt",
    title: "Before-prompt hooks rewrite the prompt",
    source: "src/agent/runtime/agent.py:510",
    modes: ["new", "resume", "tool"],
    summary:
      "Hook policy messages, plugin before-prompt messages, and resume hooks can wrap the user prompt in system-reminder blocks before it is appended.",
    inputs: ["slash_result.prompt or raw prompt", "hook policy runtime", "plugin runtime"],
    outputs: ["effective_prompt"],
    contextKeys: ["plugin before-prompt hooks", "hook policy reminders", "resume hooks"],
    nodes: ["plugin", "user-context"],
  },
  {
    id: "managed-agent",
    phase: "prompt",
    title: "Managed agent run is recorded",
    source: "src/agent/runtime/agent.py:519",
    modes: ["new", "resume", "tool"],
    summary:
      "AgentManager.start_agent() records the prompt, parent/child metadata, group id, label, and resume source so later reports can show the active run.",
    inputs: ["effective_prompt", "parent_agent_id", "managed labels"],
    outputs: ["managed_agent_id"],
    contextKeys: ["agent manager summary"],
    nodes: ["runtime"],
  },
  {
    id: "build-session",
    phase: "context",
    title: "Session and prompt context are built",
    source: "src/agent/runtime/agent.py:296",
    modes: ["new"],
    summary:
      "build_session() calls build_prompt_context(), then build_system_prompt_parts(), then creates AgentSessionState with system prompt parts plus captured user and system context dictionaries.",
    inputs: ["runtime_config", "model_config", "scratchpad_directory"],
    outputs: ["AgentSessionState", "PromptContext", "system_prompt_parts"],
    contextKeys: ["user_context", "system_context", "system prompt parts"],
    nodes: ["runtime", "model", "prompt-context", "system-prompt", "session"],
  },
  {
    id: "resume-session",
    phase: "context",
    title: "Persisted session is hydrated",
    source: "src/agent/runtime/agent.py:445",
    modes: ["resume"],
    summary:
      "resume() reconstructs AgentSessionState from stored system prompt parts, stored context dictionaries, and stored messages, then appends file history and compaction replay when needed.",
    inputs: ["StoredAgentSession", "stored plugin_state", "stored file_history"],
    outputs: ["base_session", "resume_source_session_id"],
    contextKeys: ["stored user_context", "stored system_context", "file history replay"],
    nodes: ["session", "persisted"],
  },
  {
    id: "context-snapshot",
    phase: "context",
    title: "AgentContextSnapshot captures environment",
    source: "src/agent/context/snapshot.py:75",
    modes: ["new"],
    summary:
      "build_context_snapshot() resolves cwd, shell, platform, OS version, current date, git repo/worktree flags, scratchpad directory, additional working directories, user context, and system context.",
    inputs: ["AgentRuntimeConfig", "scratchpad_directory", "AgentRuntimeDependencies"],
    outputs: ["AgentContextSnapshot"],
    contextKeys: ["cwd", "shell", "platform", "currentDate", "git flags"],
    nodes: ["runtime", "scratchpad", "system-context", "user-context"],
  },
  {
    id: "user-context",
    phase: "context",
    title: "User context sources are gathered",
    source: "src/agent/context/snapshot.py:217",
    modes: ["new"],
    summary:
      "_build_user_context() injects current date, scratchpad guidance, CLAUDE.md memory, plugin cache/runtime, hook policy, MCP, remote, triggers, search, account, ask-user, config, LSP, plan, tasks, team, workflow, and worktree summaries when present.",
    inputs: ["cwd", "additional working directories", "runtime dependencies"],
    outputs: ["user_context dictionary"],
    contextKeys: [
      "currentDate",
      "claudeMd",
      "pluginRuntime",
      "hookPolicy",
      "mcpRuntime",
      "searchRuntime",
      "taskRuntime",
    ],
    nodes: ["memory", "plugin", "mcp", "workspace-state", "user-context"],
  },
  {
    id: "system-context",
    phase: "context",
    title: "System context is gathered",
    source: "src/agent/context/snapshot.py:189",
    modes: ["new"],
    summary:
      "_get_system_context_cached() adds git status, optional cache breaker, and scratchpad directory. Git status includes branch, default branch, user, short status, and recent commits.",
    inputs: ["cwd", "scratchpad_directory", "system prompt injection"],
    outputs: ["system_context dictionary"],
    contextKeys: ["gitStatus", "cacheBreaker", "scratchpadDirectory"],
    nodes: ["git", "scratchpad", "system-context"],
  },
  {
    id: "prompt-parts",
    phase: "context",
    title: "System prompt parts are assembled",
    source: "src/agent/context/prompting.py:81",
    modes: ["new"],
    summary:
      "build_system_prompt_parts() combines default instruction sections, tool-aware guidance, agent guidance, runtime-specific guidance, and environment info. override_system_prompt replaces this list, while append_system_prompt adds to it.",
    inputs: ["PromptContext", "tool registry for prompt", "available agents"],
    outputs: ["system_prompt_parts"],
    contextKeys: ["tool guidance", "runtime guidance", "environment section"],
    nodes: ["prompt-context", "tool-registry", "system-prompt"],
  },
  {
    id: "tool-registry",
    phase: "context",
    title: "Tool registry is selected for the prompt",
    source: "src/agent/tools/registry.py:70",
    modes: ["new", "resume", "tool"],
    summary:
      "default_tool_registry() defines built-in tools. build_effective_tool_registry() overlays plugin aliases and virtual tools. For large Ollama schemas, the runtime can compact the visible tool set.",
    inputs: ["built-in tools", "plugin aliases", "virtual tools", "model base_url"],
    outputs: ["effective tool registry", "OpenAI function tool specs"],
    contextKeys: ["tool names", "tool schemas"],
    nodes: ["tool-registry", "tool-specs"],
  },
  {
    id: "append-user",
    phase: "prompt",
    title: "Effective prompt becomes a user message",
    source: "src/agent/runtime/agent.py:532",
    modes: ["new", "resume", "tool"],
    summary:
      "The effective prompt is appended to the session after local slash handling and hook rewrites. This is the first message of the new turn that the LLM sees.",
    inputs: ["effective_prompt", "AgentSessionState"],
    outputs: ["session message with role=user"],
    contextKeys: ["user message"],
    nodes: ["session", "messages"],
  },
  {
    id: "run-state",
    phase: "budget",
    title: "PromptRunState initializes counters",
    source: "src/agent/runtime/agent.py:701",
    modes: ["new", "resume", "tool"],
    summary:
      "_build_prompt_run_state() tracks starting usage, cost, previous tool calls, delegated tasks, file history, model calls, stream events, and turn index.",
    inputs: ["session", "existing_file_history", "stored resume budget state"],
    outputs: ["PromptRunState"],
    contextKeys: ["usage", "cost", "file_history", "stream_events"],
    nodes: ["runtime", "events"],
  },
  {
    id: "pressure-passes",
    phase: "budget",
    title: "Context pressure passes run before each model call",
    source: "src/agent/runtime/agent.py:553",
    modes: ["new", "resume", "tool"],
    summary:
      "Each loop can run microcompact, auto-snip, and auto-compact before the prompt-length preflight. These passes replace old messages with compact reminders when thresholds require it.",
    inputs: ["session messages", "runtime compaction thresholds"],
    outputs: ["possibly reduced session", "stream events"],
    contextKeys: ["microcompact", "snipped_message", "compact_boundary"],
    nodes: ["messages", "budget", "events"],
  },
  {
    id: "prompt-preflight",
    phase: "budget",
    title: "Token budget preflight checks the next request",
    source: "src/agent/context/pressure.py:13",
    modes: ["new", "resume", "tool"],
    summary:
      "preflight_prompt_length() calculates projected prompt tokens. If over limits, it tries heuristic pressure reduction and optional summary compaction before blocking a too-long model call.",
    inputs: ["session", "model", "BudgetConfig", "output schema"],
    outputs: ["PromptPreflightResult", "budget stream events"],
    contextKeys: ["projected_input_tokens", "soft limit", "hard limit"],
    nodes: ["budget", "messages"],
  },
  {
    id: "model-request",
    phase: "model",
    title: "Model request is formed",
    source: "src/agent/runtime/model_turn.py:15",
    modes: ["new", "resume", "tool"],
    summary:
      "query_model_turn() sends session.to_openai_messages(), tool specs, stream setting, and optional output schema to the LLM client. Streaming responses update the assistant message as deltas arrive.",
    inputs: ["OpenAI messages", "tool specs", "output schema"],
    outputs: ["AssistantTurn"],
    contextKeys: ["messages payload", "tool specs"],
    nodes: ["messages", "tool-specs", "llm"],
  },
  {
    id: "assistant-turn",
    phase: "model",
    title: "Assistant turn is normalized",
    source: "src/agent/runtime/agent.py:814",
    modes: ["new", "resume", "tool"],
    summary:
      "_process_model_turn() accumulates usage and cost, checks budgets again, and chooses either final assistant output, continuation, or tool execution.",
    inputs: ["AssistantTurn", "UsageStats"],
    outputs: ["TurnLoopDirective"],
    contextKeys: ["assistant content", "tool calls", "finish_reason"],
    nodes: ["llm", "events"],
  },
  {
    id: "tool-execution",
    phase: "tools",
    title: "Tool calls are executed",
    source: "src/agent/runtime/tool_calls.py:38",
    modes: ["tool", "new", "resume"],
    summary:
      "execute_runtime_tool_call() creates a tool message, emits tool_start, applies plugin and hook-policy preflight or block checks, streams tool output, finalizes the tool result, and records events.",
    inputs: ["ToolCall", "tool registry", "ToolExecutionContext", "hooks"],
    outputs: ["ToolExecutionResult", "ToolCallExecutionOutcome"],
    contextKeys: ["tool result JSON", "tool metadata", "permission denial events"],
    nodes: ["tool-specs", "tool-result", "events"],
  },
  {
    id: "runtime-refresh",
    phase: "tools",
    title: "Runtime views refresh after state-changing tools",
    source: "src/agent/runtime/agent.py:2860",
    modes: ["tool", "new", "resume"],
    summary:
      "_refresh_runtime_views_for_tool_result() reloads search, remote, account, config, task, plan, team, workflow, or worktree runtime state after tools that mutate them.",
    inputs: ["tool name", "tool result metadata"],
    outputs: ["updated runtime dependencies", "updated ToolExecutionContext"],
    contextKeys: ["fresh runtime summaries for later prompts"],
    nodes: ["tool-result", "workspace-state", "user-context"],
  },
  {
    id: "follow-up-context",
    phase: "tools",
    title: "Tool result becomes follow-up context",
    source: "src/agent/runtime/tool_calls.py:214",
    modes: ["tool", "new", "resume"],
    summary:
      "The finalized tool result is serialized into the session as a tool message. Plugin or hook-policy runtime messages can also be appended as user messages before the next loop.",
    inputs: ["ToolExecutionResult", "plugin messages", "hook policy messages"],
    outputs: ["tool message", "optional plugin_tool_runtime user message"],
    contextKeys: ["tool output", "runtime follow-up message"],
    nodes: ["tool-result", "messages", "events"],
  },
  {
    id: "finish-or-loop",
    phase: "model",
    title: "Run loops or finalizes",
    source: "src/agent/runtime/agent.py:683",
    modes: ["new", "resume", "tool"],
    summary:
      "If there are tool calls, the loop continues with richer context. If there are none, assistant output is finalized, after-turn hooks can add events, and the session is persisted.",
    inputs: ["TurnLoopDirective", "PromptRunState"],
    outputs: ["AgentRunResult"],
    contextKeys: ["final output", "events", "transcript"],
    nodes: ["messages", "events", "persisted"],
  },
  {
    id: "persist-session",
    phase: "tools",
    title: "Session persistence saves the run",
    source: "src/agent/runtime/agent.py:2063",
    modes: ["new", "resume", "tool"],
    summary:
      "_persist_session() writes the transcript, system prompt parts, context dictionaries, messages, file history, usage, cost, and plugin state so resume() can rebuild the next prompt path.",
    inputs: ["AgentSessionState", "AgentRunResult"],
    outputs: ["StoredAgentSession"],
    contextKeys: ["stored context", "stored messages", "stored plugin state"],
    nodes: ["session", "persisted"],
  },
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
  ["runtime", "user-context"],
  ["runtime", "system-context"],
  ["runtime", "prompt-context"],
  ["model", "prompt-context"],
  ["scratchpad", "user-context"],
  ["scratchpad", "system-context"],
  ["memory", "user-context"],
  ["git", "system-context"],
  ["plugin", "user-context"],
  ["mcp", "user-context"],
  ["workspace-state", "user-context"],
  ["user-context", "prompt-context"],
  ["system-context", "prompt-context"],
  ["prompt-context", "system-prompt"],
  ["tool-registry", "system-prompt"],
  ["tool-registry", "tool-specs"],
  ["system-prompt", "session"],
  ["prompt-context", "session"],
  ["session", "messages"],
  ["session", "budget"],
  ["budget", "messages"],
  ["messages", "llm"],
  ["tool-specs", "llm"],
  ["llm", "tool-result"],
  ["tool-result", "messages"],
  ["tool-result", "workspace-state"],
  ["tool-result", "events"],
  ["llm", "events"],
  ["session", "persisted"],
  ["events", "persisted"],
];

const budgetRows = [
  {
    label: "Fresh prompt",
    total: 100,
    values: { system: 38, context: 24, history: 18, tools: 20 },
  },
  {
    label: "After several tool loops",
    total: 100,
    values: { system: 23, context: 18, history: 42, tools: 17 },
  },
  {
    label: "Pressure recovery target",
    total: 100,
    values: { system: 28, context: 20, history: 27, tools: 25 },
  },
];

const eventLanes = [
  {
    name: "Prompt setup",
    dots: [
      { at: 8, kind: "prompt", label: "slash" },
      { at: 18, kind: "prompt", label: "hooks" },
      { at: 27, kind: "prompt", label: "append" },
    ],
  },
  {
    name: "Budget",
    dots: [
      { at: 35, kind: "budget", label: "check" },
      { at: 44, kind: "budget", label: "compact" },
    ],
  },
  {
    name: "Model",
    dots: [
      { at: 56, kind: "model", label: "request" },
      { at: 65, kind: "model", label: "stream" },
      { at: 72, kind: "model", label: "usage" },
    ],
  },
  {
    name: "Tools",
    dots: [
      { at: 78, kind: "tools", label: "start" },
      { at: 86, kind: "tools", label: "result" },
      { at: 92, kind: "tools", label: "refresh" },
    ],
  },
];

const modules = [
  {
    file: "src/agent/runtime/agent.py",
    role: "Owns the turn loop, resume path, runtime refresh, budget checks, persistence, and final AgentRunResult assembly.",
  },
  {
    file: "src/agent/context/snapshot.py",
    role: "Builds the source dictionaries that become user_context and system_context for the session.",
  },
  {
    file: "src/agent/context/prompting.py",
    role: "Turns PromptContext and enabled tools into ordered system prompt sections.",
  },
  {
    file: "src/agent/tools/registry.py",
    role: "Defines built-in tool schemas and centralizes plugin-expanded tool registry construction.",
  },
  {
    file: "src/agent/runtime/tool_calls.py",
    role: "Executes model-requested tools and writes tool messages, stream events, and plugin or policy follow-up context.",
  },
  {
    file: "src/agent/context/pressure.py",
    role: "Runs token budget preflight, snipping, compaction, and prompt-too-long recovery.",
  },
  {
    file: "src/agent/runtime/model_turn.py",
    role: "Converts session messages and tool specs into non-streaming or streaming LLM calls.",
  },
  {
    file: "src/agent/models/session.py",
    role: "Defines AgentSessionState and message mutation history used by the turn loop and persistence layer.",
  },
  {
    file: "src/agent/commands/slash.py",
    role: "Parses slash commands and routes local commands before the model path is entered.",
  },
  {
    file: "src/agent/profiles/registry.py",
    role: "Loads built-in and workspace-defined subagent profiles for prompt guidance and delegation.",
  },
  {
    file: "src/session/session_store.py",
    role: "Stores enough session state for resume() to rebuild the next prompt path.",
  },
];

const state = {
  mode: "new",
  phase: "all",
  query: "",
  selectedStepId: "prompt-entry",
};

const els = {
  mode: document.querySelector("#prompt-mode"),
  filters: document.querySelector("#phase-filters"),
  search: document.querySelector("#step-search"),
  timeline: document.querySelector("#timeline"),
  detail: document.querySelector("#step-detail"),
  count: document.querySelector("#visible-count"),
  graph: document.querySelector("#context-graph"),
  resetGraph: document.querySelector("#reset-graph"),
  budget: document.querySelector("#budget-bars"),
  events: document.querySelector("#event-lanes"),
  modules: document.querySelector("#module-map"),
  toolCount: document.querySelector("#tool-count"),
};

function init() {
  renderControls();
  renderTimeline();
  renderBudget();
  renderEvents();
  renderModules();
  renderGraph();

  els.mode.addEventListener("change", () => {
    state.mode = els.mode.value;
    const visible = getVisibleSteps();
    state.selectedStepId = visible[0]?.id || null;
    renderTimeline();
    renderGraph();
  });

  els.search.addEventListener("input", () => {
    state.query = els.search.value.trim().toLowerCase();
    renderTimeline();
  });

  els.resetGraph.addEventListener("click", () => {
    state.selectedStepId = getVisibleSteps()[0]?.id || "prompt-entry";
    renderTimeline();
    renderGraph();
  });
}

function renderControls() {
  for (const mode of promptModes) {
    const option = document.createElement("option");
    option.value = mode.id;
    option.textContent = mode.label;
    els.mode.append(option);
  }

  const allButton = phaseButton("all", "All");
  els.filters.append(allButton);
  for (const phase of phases) {
    els.filters.append(phaseButton(phase, titleCase(phase)));
  }
}

function phaseButton(id, label) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "filter-button";
  button.textContent = label;
  button.dataset.phase = id;
  button.addEventListener("click", () => {
    state.phase = id;
    for (const item of els.filters.querySelectorAll(".filter-button")) {
      item.classList.toggle("is-active", item.dataset.phase === state.phase);
    }
    renderTimeline();
  });
  button.classList.toggle("is-active", id === state.phase);
  return button;
}

function getVisibleSteps() {
  return steps.filter((step) => {
    if (!step.modes.includes(state.mode)) {
      return false;
    }
    if (state.phase !== "all" && step.phase !== state.phase) {
      return false;
    }
    if (!state.query) {
      return true;
    }
    const haystack = [
      step.title,
      step.phase,
      step.source,
      step.summary,
      ...step.inputs,
      ...step.outputs,
      ...step.contextKeys,
    ]
      .join(" ")
      .toLowerCase();
    return haystack.includes(state.query);
  });
}

function renderTimeline() {
  const visible = getVisibleSteps();
  if (!visible.some((step) => step.id === state.selectedStepId)) {
    state.selectedStepId = visible[0]?.id || null;
  }
  els.timeline.replaceChildren();
  els.count.textContent = `${visible.length} steps`;

  if (!visible.length) {
    const empty = document.createElement("p");
    empty.className = "detail-meta";
    empty.textContent = "No steps match the current filters.";
    els.timeline.append(empty);
    renderDetail(null);
    return;
  }

  visible.forEach((step, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "step-card";
    button.dataset.phase = step.phase;
    button.classList.toggle("is-selected", step.id === state.selectedStepId);
    button.addEventListener("click", () => {
      state.selectedStepId = step.id;
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
    title.textContent = step.title;
    const meta = document.createElement("span");
    meta.className = "step-meta";
    meta.textContent = step.source;
    main.append(title, meta);

    const phase = document.createElement("span");
    phase.className = "step-phase";
    phase.textContent = step.phase;

    button.append(stepIndex, main, phase);
    els.timeline.append(button);
  });

  renderDetail(steps.find((step) => step.id === state.selectedStepId));
}

function renderDetail(step) {
  els.detail.replaceChildren();
  if (!step) {
    els.detail.textContent = "Select a step to inspect its inputs and outputs.";
    return;
  }

  const overview = section("What happens", [step.summary]);
  const io = document.createElement("div");
  io.className = "detail-section";
  io.append(
    detailHeading("Inputs"),
    list(step.inputs),
    detailHeading("Outputs"),
    list(step.outputs)
  );

  const context = section("Context keys touched", step.contextKeys);
  const source = document.createElement("div");
  source.className = "detail-section";
  source.innerHTML = `<h3>Source</h3><p class="detail-meta"><code>${escapeHtml(
    step.source
  )}</code></p>`;

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
  const selected = steps.find((step) => step.id === state.selectedStepId);
  const activeNodes = new Set(selected?.nodes || []);
  const width = 1510;
  const height = 540;
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
  marker.append(svg("path", { d: "M0,0 L0,6 L9,3 z", fill: "#8fa19b" }));
  defs.append(marker);
  els.graph.append(defs);

  const nodeById = new Map(graphNodes.map((node) => [node.id, node]));
  for (const [from, to] of graphEdges) {
    const a = nodeById.get(from);
    const b = nodeById.get(to);
    if (!a || !b) {
      continue;
    }
    const active = activeNodes.has(from) || activeNodes.has(to);
    const path = svg("path", {
      class: `graph-edge${active ? " is-active" : ""}`,
      d: edgePath(a, b),
      "marker-end": "url(#arrow)",
    });
    els.graph.append(path);
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
  const next = visible.find((step) => step.nodes.includes(nodeId));
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
      const tspan = svg("tspan", {
        x: String(x),
        dy: lineIndex === 0 ? "0" : String(lineHeight),
      });
      tspan.textContent = line;
      textNode.append(tspan);
      line = word;
      lineIndex += 1;
    } else {
      line = test;
    }
  }
  const tspan = svg("tspan", {
    x: String(x),
    dy: lineIndex === 0 ? "0" : String(lineHeight),
  });
  tspan.textContent = line;
  textNode.append(tspan);
  group.append(textNode);
}

function renderBudget() {
  els.budget.replaceChildren();
  for (const row of budgetRows) {
    const wrapper = document.createElement("div");
    wrapper.className = "budget-row";
    const label = document.createElement("div");
    label.className = "budget-label";
    label.innerHTML = `<span>${escapeHtml(row.label)}</span><span>relative token share</span>`;
    const track = document.createElement("div");
    track.className = "budget-track";
    for (const [name, value] of Object.entries(row.values)) {
      const segment = document.createElement("span");
      segment.className = `budget-segment ${name}`;
      segment.style.width = `${(value / row.total) * 100}%`;
      segment.title = `${titleCase(name)}: ${value}%`;
      track.append(segment);
    }
    wrapper.append(label, track);
    els.budget.append(wrapper);
  }
}

function renderEvents() {
  els.events.replaceChildren();
  for (const lane of eventLanes) {
    const row = document.createElement("div");
    row.className = "lane";
    const name = document.createElement("div");
    name.className = "lane-name";
    name.textContent = lane.name;
    const track = document.createElement("div");
    track.className = "lane-track";
    for (const dot of lane.dots) {
      const item = document.createElement("span");
      item.className = "event-dot";
      item.dataset.kind = dot.kind;
      item.style.left = `${dot.at}%`;
      item.title = dot.label;
      track.append(item);
    }
    row.append(name, track);
    els.events.append(row);
  }
}

function renderModules() {
  els.modules.replaceChildren();
  for (const item of modules) {
    const card = document.createElement("article");
    card.className = "module-card";
    const file = document.createElement("strong");
    file.textContent = item.file;
    const role = document.createElement("span");
    role.textContent = item.role;
    card.append(file, role);
    els.modules.append(card);
  }
}

function svg(name, attrs = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) {
    element.setAttribute(key, value);
  }
  return element;
}

function titleCase(value) {
  return value.slice(0, 1).toUpperCase() + value.slice(1);
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
