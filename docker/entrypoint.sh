#!/usr/bin/env bash
set -euo pipefail

bool_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

AGENT_COMMAND="${AGENT_COMMAND:-agent-chat}"
AGENT_CWD="${AGENT_CWD:-/workspace}"
AGENT_READ_ONLY="${AGENT_READ_ONLY:-false}"

case "$AGENT_COMMAND" in
  agent|agent-bg|agent-chat|agent-tui|agent-prompt|agent-context|agent-context-raw|token-budget|doctor)
    ;;
  *)
    echo "Unsupported AGENT_COMMAND: $AGENT_COMMAND" >&2
    exit 2
    ;;
esac

cmd=(python -m src.main "$AGENT_COMMAND")

if [[ -n "${AGENT_PROMPT:-}" ]]; then
  cmd+=("${AGENT_PROMPT}")
elif [[ "$AGENT_COMMAND" == "agent" || "$AGENT_COMMAND" == "agent-bg" ]]; then
  echo "AGENT_PROMPT must be set when AGENT_COMMAND is '$AGENT_COMMAND'." >&2
  exit 2
fi

cmd+=(--cwd "$AGENT_CWD")

if [[ -n "${OPENAI_MODEL:-}" ]]; then
  cmd+=(--model "$OPENAI_MODEL")
fi

if [[ "$AGENT_COMMAND" == "agent" || "$AGENT_COMMAND" == "agent-bg" || "$AGENT_COMMAND" == "agent-chat" || "$AGENT_COMMAND" == "agent-tui" || "$AGENT_COMMAND" == "doctor" ]]; then
  if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
    cmd+=(--base-url "$OPENAI_BASE_URL")
  fi
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    cmd+=(--api-key "$OPENAI_API_KEY")
  fi
  if [[ "$AGENT_COMMAND" != "doctor" && -n "${AGENT_TEMPERATURE:-}" ]]; then
    cmd+=(--temperature "$AGENT_TEMPERATURE")
  fi
  if [[ -n "${AGENT_TIMEOUT_SECONDS:-}" ]]; then
    cmd+=(--timeout-seconds "$AGENT_TIMEOUT_SECONDS")
  fi
fi

if [[ "$AGENT_COMMAND" == "doctor" ]]; then
  if bool_true "${AGENT_SKIP_BACKEND:-false}"; then
    cmd+=(--skip-backend)
  fi
  if bool_true "${AGENT_SKIP_TUI:-false}"; then
    cmd+=(--skip-tui)
  fi
fi

if [[ -n "${AGENT_MAX_TURNS:-}" ]]; then
  cmd+=(--max-turns "$AGENT_MAX_TURNS")
fi

if bool_true "${AGENT_SHOW_TRANSCRIPT:-false}"; then
  case "$AGENT_COMMAND" in
    agent|agent-bg|agent-chat)
      cmd+=(--show-transcript)
      ;;
  esac
fi

if bool_true "${AGENT_DISABLE_CLAUDE_MD:-false}"; then
  cmd+=(--disable-claude-md)
fi
if ! bool_true "${AGENT_READ_ONLY}"; then
  if bool_true "${AGENT_ALLOW_WRITE:-false}"; then
    cmd+=(--allow-write)
  fi
  if bool_true "${AGENT_ALLOW_SHELL:-false}"; then
    cmd+=(--allow-shell)
  fi
  if bool_true "${AGENT_UNSAFE:-false}"; then
    cmd+=(--unsafe)
  fi
fi
if bool_true "${AGENT_STREAM:-false}"; then
  cmd+=(--stream)
fi

if [[ -n "${AGENT_AUTO_SNIP_THRESHOLD:-}" ]]; then
  cmd+=(--auto-snip-threshold "$AGENT_AUTO_SNIP_THRESHOLD")
fi
if [[ -n "${AGENT_AUTO_COMPACT_THRESHOLD:-}" ]]; then
  cmd+=(--auto-compact-threshold "$AGENT_AUTO_COMPACT_THRESHOLD")
fi
if [[ -n "${AGENT_COMPACT_PRESERVE_MESSAGES:-}" ]]; then
  cmd+=(--compact-preserve-messages "$AGENT_COMPACT_PRESERVE_MESSAGES")
fi
if [[ -n "${AGENT_MAX_TOTAL_TOKENS:-}" ]]; then
  cmd+=(--max-total-tokens "$AGENT_MAX_TOTAL_TOKENS")
fi
if [[ -n "${AGENT_MAX_INPUT_TOKENS:-}" ]]; then
  cmd+=(--max-input-tokens "$AGENT_MAX_INPUT_TOKENS")
fi
if [[ -n "${AGENT_MAX_OUTPUT_TOKENS:-}" ]]; then
  cmd+=(--max-output-tokens "$AGENT_MAX_OUTPUT_TOKENS")
fi
if [[ -n "${AGENT_MAX_REASONING_TOKENS:-}" ]]; then
  cmd+=(--max-reasoning-tokens "$AGENT_MAX_REASONING_TOKENS")
fi
if [[ -n "${AGENT_MAX_BUDGET_USD:-}" ]]; then
  cmd+=(--max-budget-usd "$AGENT_MAX_BUDGET_USD")
fi
if [[ -n "${AGENT_MAX_TOOL_CALLS:-}" ]]; then
  cmd+=(--max-tool-calls "$AGENT_MAX_TOOL_CALLS")
fi
if [[ -n "${AGENT_MAX_DELEGATED_TASKS:-}" ]]; then
  cmd+=(--max-delegated-tasks "$AGENT_MAX_DELEGATED_TASKS")
fi
if [[ -n "${AGENT_MAX_MODEL_CALLS:-}" ]]; then
  cmd+=(--max-model-calls "$AGENT_MAX_MODEL_CALLS")
fi
if [[ -n "${AGENT_MAX_SESSION_TURNS:-}" ]]; then
  cmd+=(--max-session-turns "$AGENT_MAX_SESSION_TURNS")
fi
if [[ -n "${AGENT_RESPONSE_SCHEMA_FILE:-}" ]]; then
  cmd+=(--response-schema-file "$AGENT_RESPONSE_SCHEMA_FILE")
fi
if [[ -n "${AGENT_RESPONSE_SCHEMA_NAME:-}" ]]; then
  cmd+=(--response-schema-name "$AGENT_RESPONSE_SCHEMA_NAME")
fi
if bool_true "${AGENT_RESPONSE_SCHEMA_STRICT:-false}"; then
  cmd+=(--response-schema-strict)
fi
if [[ -n "${AGENT_SCRATCHPAD_ROOT:-}" ]]; then
  cmd+=(--scratchpad-root "$AGENT_SCRATCHPAD_ROOT")
fi
if [[ -n "${AGENT_SYSTEM_PROMPT:-}" ]]; then
  cmd+=(--system-prompt "$AGENT_SYSTEM_PROMPT")
fi
if [[ -n "${AGENT_APPEND_SYSTEM_PROMPT:-}" ]]; then
  cmd+=(--append-system-prompt "$AGENT_APPEND_SYSTEM_PROMPT")
fi
if [[ -n "${AGENT_OVERRIDE_SYSTEM_PROMPT:-}" ]]; then
  cmd+=(--override-system-prompt "$AGENT_OVERRIDE_SYSTEM_PROMPT")
fi

echo "Launching: ${cmd[*]}" >&2
exec "${cmd[@]}"
