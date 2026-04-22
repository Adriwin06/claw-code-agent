#!/usr/bin/env bash
set -euo pipefail

# Launch Claw Code Agent against the current directory.
# If this script is not inside the repository tree, set CLAW_CODE_AGENT_ROOT first.
# Optional environment variables:
#   CLAW_AGENT_COMMAND=agent-tui|agent-chat|doctor|...
#   CLAW_REBUILD=1              rebuild the image before launching
#   CLAW_START_SAGEMATH=1       start the optional SageMath sidecar first

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

is_wsl() {
  grep -qi microsoft /proc/version 2>/dev/null
}

read_env_file_value() {
  local env_file="$1"
  local key="$2"

  if [[ ! -f "$env_file" ]]; then
    return 1
  fi

  awk -F'=' -v target="$key" '
    $1 == target {
      value = substr($0, index($0, "=") + 1)
      gsub(/\r$/, "", value)
      print value
      exit
    }
  ' "$env_file"
}

uses_ollama_backend() {
  local base_url="$1"
  [[ "$base_url" == *":11434"* ]]
}

ollama_ready() {
  curl -fsS --max-time 3 "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1
}

ensure_ollama_running() {
  local waited=0

  if ollama_ready; then
    return 0
  fi
  if ! command -v ollama >/dev/null 2>&1; then
    return 1
  fi

  echo "Starting local Ollama service for WSL launch..."
  nohup ollama serve >/tmp/claw-code-agent-ollama.log 2>&1 &

  while (( waited < 15 )); do
    if ollama_ready; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done

  return 1
}

image_has_runtime_deps() {
  local image_name="$1"

  docker run --rm --entrypoint python "$image_name" -c "import litellm, textual" >/dev/null 2>&1
}

image_matches_workspace_venv() {
  local image_name="$1"
  local workspace_dir="$2"
  local venv_cfg="$workspace_dir/venv-linux/pyvenv.cfg"
  local expected_version
  local image_version

  if [[ ! -f "$venv_cfg" ]]; then
    return 0
  fi

  expected_version="$(awk -F'=' '/^version[[:space:]]*=/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); split($2, parts, "."); print parts[1] "." parts[2]; exit}' "$venv_cfg")"
  if [[ -z "$expected_version" ]]; then
    return 0
  fi

  image_version="$(docker run --rm --entrypoint python "$image_name" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
  [[ "$image_version" == "$expected_version" ]]
}

find_repo_root() {
  local search_dir="$1"
  local parent_dir

  while true; do
    if [[ -f "$search_dir/docker-compose.yml" ]]; then
      printf '%s\n' "$search_dir"
      return 0
    fi

    parent_dir="$(cd "$search_dir/.." && pwd)"
    if [[ "$parent_dir" == "$search_dir" ]]; then
      return 1
    fi
    search_dir="$parent_dir"
  done
}

WORKSPACE_DIR="$(pwd)"
REPO_ROOT="${CLAW_CODE_AGENT_ROOT:-}"
LAUNCH_AGENT_COMMAND="${CLAW_AGENT_COMMAND:-agent-tui}"
DOCKER_IMAGE="claw-code-agent-local"

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  COMPOSE_CMD=()
fi

if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ ! -f "$REPO_ROOT/docker-compose.yml" ]]; then
    REPO_ROOT="$(find_repo_root "$REPO_ROOT" || true)"
  fi
fi

if [[ -n "$REPO_ROOT" && "$REPO_ROOT" != "/" && "${REPO_ROOT: -1}" == "/" ]]; then
  REPO_ROOT="${REPO_ROOT%/}"
fi

if [[ -z "$REPO_ROOT" ]]; then
  echo "Could not locate the claw-code-agent repository."
  echo "Set CLAW_CODE_AGENT_ROOT to the repository root, then run this script again."
  exit 1
fi

if [[ ! -f "$REPO_ROOT/docker-compose.yml" ]]; then
  echo "Docker compose file not found at \"$REPO_ROOT/docker-compose.yml\"."
  exit 1
fi

if [[ "${#COMPOSE_CMD[@]}" -eq 0 ]]; then
  echo "Docker Compose is not available. Install 'docker compose' or 'docker-compose' and try again."
  exit 1
fi

ENV_FILE="$REPO_ROOT/.env"
DOCKER_NETWORK_ARGS=(--add-host "host.docker.internal:host-gateway")
DOCKER_ENV_ARGS=()

LLM_API_BASE_VALUE="${LLM_API_BASE:-}"
LLM_MODEL_VALUE="${LLM_MODEL:-}"
LLM_API_KEY_VALUE="${LLM_API_KEY:-}"
if [[ -f "$ENV_FILE" ]]; then
  LLM_API_BASE_VALUE="$(read_env_file_value "$ENV_FILE" LLM_API_BASE || printf '%s' "$LLM_API_BASE_VALUE")"
  LLM_MODEL_VALUE="$(read_env_file_value "$ENV_FILE" LLM_MODEL || printf '%s' "$LLM_MODEL_VALUE")"
  LLM_API_KEY_VALUE="$(read_env_file_value "$ENV_FILE" LLM_API_KEY || printf '%s' "$LLM_API_KEY_VALUE")"
fi

if is_wsl && uses_ollama_backend "$LLM_API_BASE_VALUE"; then
  if ensure_ollama_running; then
    echo "Detected WSL-hosted Ollama. Using Docker host networking for the backend."
    DOCKER_NETWORK_ARGS=(--network host)
    DOCKER_ENV_ARGS+=(-e "LLM_API_BASE=http://127.0.0.1:11434/v1")
    if [[ -z "$LLM_API_KEY_VALUE" ]]; then
      DOCKER_ENV_ARGS+=(-e "LLM_API_KEY=ollama")
    fi
    if [[ -n "$LLM_MODEL_VALUE" ]]; then
      if [[ "$LLM_MODEL_VALUE" == */* ]]; then
        DOCKER_ENV_ARGS+=(-e "LLM_MODEL=$LLM_MODEL_VALUE")
      else
        echo "Adjusting LLM_MODEL for LiteLLM Ollama routing: openai/$LLM_MODEL_VALUE"
        DOCKER_ENV_ARGS+=(-e "LLM_MODEL=openai/$LLM_MODEL_VALUE")
      fi
    fi
  else
    echo "Ollama backend is configured, but it is not reachable on WSL localhost."
    echo "Start it with 'ollama serve' and make sure a model is available via 'ollama list'."
  fi
fi

echo "Launching Claw Code Agent"
echo "  repo: $REPO_ROOT"
echo "  workspace: $WORKSPACE_DIR"
echo "  command: $LAUNCH_AGENT_COMMAND"
if bool_true "${CLAW_REBUILD:-}"; then
  echo "  image rebuild: enabled"
else
  echo "  image rebuild: skipped"
fi

if bool_true "${CLAW_REBUILD:-}"; then
  echo "Building Docker image $DOCKER_IMAGE..."
  docker build -t "$DOCKER_IMAGE" -f "$REPO_ROOT/docker/Dockerfile" "$REPO_ROOT"
else
  if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    echo "Building Docker image $DOCKER_IMAGE..."
    docker build -t "$DOCKER_IMAGE" -f "$REPO_ROOT/docker/Dockerfile" "$REPO_ROOT"
  elif ! image_matches_workspace_venv "$DOCKER_IMAGE" "$WORKSPACE_DIR"; then
    echo "Cached Docker image Python version does not match workspace venv-linux. Rebuilding $DOCKER_IMAGE..."
    docker build -t "$DOCKER_IMAGE" -f "$REPO_ROOT/docker/Dockerfile" "$REPO_ROOT"
  elif ! image_has_runtime_deps "$DOCKER_IMAGE"; then
    echo "Cached Docker image is missing required Python packages. Rebuilding $DOCKER_IMAGE..."
    docker build -t "$DOCKER_IMAGE" -f "$REPO_ROOT/docker/Dockerfile" "$REPO_ROOT"
  fi
fi

if bool_true "${CLAW_START_SAGEMATH:-}"; then
  echo "Starting optional SageMath sidecar..."
  "${COMPOSE_CMD[@]}" -f "$REPO_ROOT/docker-compose.yml" --project-directory "$REPO_ROOT" up -d sagemath
  if [[ -f "$ENV_FILE" ]]; then
    docker run --rm -it \
      --env-file "$ENV_FILE" \
      "${DOCKER_NETWORK_ARGS[@]}" \
      "${DOCKER_ENV_ARGS[@]}" \
      -e "AGENT_COMMAND=$LAUNCH_AGENT_COMMAND" \
      -e "AGENT_CWD=/workspace" \
      -e "AGENT_READ_ONLY=false" \
      -e "AGENT_ALLOW_WRITE=true" \
      -e "AGENT_ALLOW_SHELL=false" \
      -e "AGENT_UNSAFE=false" \
      -e "SAGEMATH_MCP_URL=http://sagemath:8000/mcp" \
      -v "$WORKSPACE_DIR:/workspace" \
      -w /workspace \
      "$DOCKER_IMAGE"
  else
    docker run --rm -it \
      "${DOCKER_NETWORK_ARGS[@]}" \
      "${DOCKER_ENV_ARGS[@]}" \
      -e "AGENT_COMMAND=$LAUNCH_AGENT_COMMAND" \
      -e "AGENT_CWD=/workspace" \
      -e "AGENT_READ_ONLY=false" \
      -e "AGENT_ALLOW_WRITE=true" \
      -e "AGENT_ALLOW_SHELL=false" \
      -e "AGENT_UNSAFE=false" \
      -e "SAGEMATH_MCP_URL=http://sagemath:8000/mcp" \
      -v "$WORKSPACE_DIR:/workspace" \
      -w /workspace \
      "$DOCKER_IMAGE"
  fi
else
  echo "Skipping SageMath sidecar. Set CLAW_START_SAGEMATH=1 to enable it."
  if [[ -f "$ENV_FILE" ]]; then
    docker run --rm -it \
      --env-file "$ENV_FILE" \
      "${DOCKER_NETWORK_ARGS[@]}" \
      "${DOCKER_ENV_ARGS[@]}" \
      -e "AGENT_COMMAND=$LAUNCH_AGENT_COMMAND" \
      -e "AGENT_CWD=/workspace" \
      -e "AGENT_READ_ONLY=false" \
      -e "AGENT_ALLOW_WRITE=true" \
      -e "AGENT_ALLOW_SHELL=false" \
      -e "AGENT_UNSAFE=false" \
      -v "$WORKSPACE_DIR:/workspace" \
      -w /workspace \
      "$DOCKER_IMAGE"
  else
    docker run --rm -it \
      "${DOCKER_NETWORK_ARGS[@]}" \
      "${DOCKER_ENV_ARGS[@]}" \
      -e "AGENT_COMMAND=$LAUNCH_AGENT_COMMAND" \
      -e "AGENT_CWD=/workspace" \
      -e "AGENT_READ_ONLY=false" \
      -e "AGENT_ALLOW_WRITE=true" \
      -e "AGENT_ALLOW_SHELL=false" \
      -e "AGENT_UNSAFE=false" \
      -v "$WORKSPACE_DIR:/workspace" \
      -w /workspace \
      "$DOCKER_IMAGE"
  fi
fi
