#!/usr/bin/env bash
set -euo pipefail

# Launch Claw Code Agent against the current directory.
# If this script is not inside the repository tree, set CLAW_CODE_AGENT_ROOT first.
# Optional environment variables:
#   CLAW_AGENT_COMMAND=agent-tui|agent-chat|doctor|...
#   CLAW_REBUILD=1              rebuild the image before launching
#   CLAW_START_SAGEMATH=0       disable the default SageMath sidecar
#   CLAW_START_SEARCH=0         disable the default SearXNG sidecar
#   CLAW_HOST_CODE_HOME=...     host directory for persistent TUI history
#   CLAW_HOST_ATTACHMENTS_ROOT=... host directory mounted read-only for pasted file paths
#   SAGEMATH_HOST_PORT=18000    host port published by the SageMath sidecar
#   SEARXNG_HOST_PORT=8080      host port published by the SearXNG sidecar

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

windows_user_profile_from_wsl() {
  local profile

  if ! is_wsl || ! command -v cmd.exe >/dev/null 2>&1; then
    return 1
  fi
  profile="$(cmd.exe /C "echo %USERPROFILE%" 2>/dev/null \
    | tr -d '\r' \
    | awk 'NF { print; exit }')"
  if [[ -z "$profile" ]]; then
    return 1
  fi
  printf '%s\n' "$profile"
}

windows_user_profile_from_wsl_workspace() {
  local workspace_dir="$1"

  if ! is_wsl; then
    return 1
  fi

  awk -F/ '
    BEGIN { IGNORECASE = 1 }
    $2 == "mnt" && tolower($4) == "users" && $5 != "" {
      printf "%s:\\Users\\%s\n", toupper($3), $5
      found = 1
      exit
    }
    END {
      if (!found) {
        exit 1
      }
    }
  ' <<<"$workspace_dir"
}

mount_root_for_host_path() {
  local host_path="$1"

  if is_wsl && [[ "$host_path" =~ ^[A-Za-z]:\\ ]] && command -v wslpath >/dev/null 2>&1; then
    wslpath -u "$host_path"
    return
  fi

  printf '%s\n' "$host_path"
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
      found = 1
      exit
    }
    END {
      if (!found) {
        exit 1
      }
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

wait_http_url() {
  local label="$1"
  local url="$2"
  local timeout_seconds="${3:-60}"
  local waited=0

  while (( waited < timeout_seconds )); do
    if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then
      echo "$label is reachable at $url"
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done

  echo "Warning: $label did not become reachable at $url within ${timeout_seconds}s."
  return 1
}

image_has_runtime_deps() {
  local image_name="$1"

  docker run --rm --entrypoint python "$image_name" -c "import litellm, textual; from PIL import Image" >/dev/null 2>&1
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
CLAW_START_SAGEMATH="${CLAW_START_SAGEMATH:-1}"
CLAW_START_SEARCH="${CLAW_START_SEARCH:-1}"
SAGEMATH_HOST_PORT="${SAGEMATH_HOST_PORT:-18000}"
SEARXNG_HOST_PORT="${SEARXNG_HOST_PORT:-8080}"
DOCKER_IMAGE="claw-code-agent-local"
CONTAINER_AGENT_SOURCE_ROOT="/agent-source"
HOST_CLAW_CODE_HOME="${CLAW_HOST_CODE_HOME:-${CLAW_CODE_HOME:-$HOME/.claw-code}}"
if [[ -n "${CLAW_HOST_ATTACHMENTS_ROOT:-}" ]]; then
  HOST_ATTACHMENTS_ROOT="$CLAW_HOST_ATTACHMENTS_ROOT"
elif WINDOWS_USER_PROFILE="$(windows_user_profile_from_wsl)"; then
  HOST_ATTACHMENTS_ROOT="$WINDOWS_USER_PROFILE"
elif WINDOWS_USER_PROFILE="$(windows_user_profile_from_wsl_workspace "$WORKSPACE_DIR")"; then
  HOST_ATTACHMENTS_ROOT="$WINDOWS_USER_PROFILE"
else
  HOST_ATTACHMENTS_ROOT="$HOME"
fi
HOST_ATTACHMENTS_MOUNT_ROOT="${CLAW_HOST_ATTACHMENTS_MOUNT_ROOT:-$(mount_root_for_host_path "$HOST_ATTACHMENTS_ROOT")}"
if [[ ! -d "$HOST_ATTACHMENTS_MOUNT_ROOT" ]]; then
  echo "Warning: host attachment root is not readable: $HOST_ATTACHMENTS_MOUNT_ROOT"
  echo "Falling back to WSL home for attachment path mounts."
  HOST_ATTACHMENTS_ROOT="$HOME"
  HOST_ATTACHMENTS_MOUNT_ROOT="$HOME"
fi

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
  SAGEMATH_HOST_PORT="$(read_env_file_value "$ENV_FILE" SAGEMATH_HOST_PORT || printf '%s' "$SAGEMATH_HOST_PORT")"
  SEARXNG_HOST_PORT="$(read_env_file_value "$ENV_FILE" SEARXNG_HOST_PORT || printf '%s' "$SEARXNG_HOST_PORT")"
fi
export SAGEMATH_HOST_PORT
export SEARXNG_HOST_PORT

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

AGENT_SIDECAR_HOST="host.docker.internal"
if [[ "${DOCKER_NETWORK_ARGS[*]}" == "--network host" ]]; then
  AGENT_SIDECAR_HOST="127.0.0.1"
fi
AGENT_SAGEMATH_MCP_URL="${CLAW_AGENT_SAGEMATH_MCP_URL:-http://${AGENT_SIDECAR_HOST}:${SAGEMATH_HOST_PORT}/mcp}"
AGENT_SEARXNG_BASE_URL="${CLAW_AGENT_SEARXNG_BASE_URL:-http://${AGENT_SIDECAR_HOST}:${SEARXNG_HOST_PORT}}"

echo "Launching Claw Code Agent"
echo "  repo: $REPO_ROOT"
echo "  workspace: $WORKSPACE_DIR"
echo "  source: $REPO_ROOT -> $CONTAINER_AGENT_SOURCE_ROOT"
echo "  history: $HOST_CLAW_CODE_HOME"
echo "  host attachments: $HOST_ATTACHMENTS_ROOT -> $HOST_ATTACHMENTS_MOUNT_ROOT"
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
    echo "Cached Docker image is missing required Python packages for the TUI image preview. Rebuilding $DOCKER_IMAGE..."
    docker build -t "$DOCKER_IMAGE" -f "$REPO_ROOT/docker/Dockerfile" "$REPO_ROOT"
  fi
fi

SIDECAR_SERVICES=()
if bool_true "${CLAW_START_SAGEMATH:-}"; then
  SIDECAR_SERVICES+=(sagemath)
else
  echo "Skipping SageMath sidecar. Set CLAW_START_SAGEMATH=1 to enable it."
fi
if bool_true "${CLAW_START_SEARCH:-}"; then
  SIDECAR_SERVICES+=(searxng)
else
  echo "Skipping SearXNG sidecar. Set CLAW_START_SEARCH=1 to enable it."
fi

if [[ "${#SIDECAR_SERVICES[@]}" -gt 0 ]]; then
  echo "Starting optional sidecars: ${SIDECAR_SERVICES[*]}..."
  "${COMPOSE_CMD[@]}" -f "$REPO_ROOT/docker-compose.yml" --project-directory "$REPO_ROOT" up -d "${SIDECAR_SERVICES[@]}"
  if bool_true "${CLAW_START_SAGEMATH:-}"; then
    wait_http_url "SageMath MCP" "http://127.0.0.1:${SAGEMATH_HOST_PORT}/health" 90 || true
  fi
  if bool_true "${CLAW_START_SEARCH:-}"; then
    wait_http_url "SearXNG" "http://127.0.0.1:${SEARXNG_HOST_PORT}/" 60 || true
  fi
fi

ENV_FILE_ARGS=()
if [[ -f "$ENV_FILE" ]]; then
  ENV_FILE_ARGS=(--env-file "$ENV_FILE")
fi

mkdir -p "$HOST_CLAW_CODE_HOME"

RUN_ENV_ARGS=(
  -e "AGENT_COMMAND=$LAUNCH_AGENT_COMMAND"
  -e "AGENT_CWD=/workspace"
  -e "AGENT_SOURCE_ROOT=$CONTAINER_AGENT_SOURCE_ROOT"
  -e "CLAW_CODE_HOME=/root/.claw-code"
  -e "CLAW_HOST_WORKSPACE=$WORKSPACE_DIR"
  -e "CLAW_CONTAINER_WORKSPACE=/workspace"
  -e "CLAW_HOST_ATTACHMENTS_ROOT=$HOST_ATTACHMENTS_ROOT"
  -e "CLAW_HOST_ATTACHMENTS_MOUNT_ROOT=$HOST_ATTACHMENTS_MOUNT_ROOT"
  -e "CLAW_CONTAINER_HOST_ATTACHMENTS_ROOT=/host-attachments"
  -e "AGENT_READ_ONLY=false"
  -e "AGENT_ALLOW_WRITE=true"
  -e "AGENT_ALLOW_SHELL=false"
  -e "AGENT_UNSAFE=false"
)
if bool_true "${CLAW_START_SAGEMATH:-}"; then
  RUN_ENV_ARGS+=(-e "SAGEMATH_MCP_URL=$AGENT_SAGEMATH_MCP_URL")
fi
if bool_true "${CLAW_START_SEARCH:-}"; then
  RUN_ENV_ARGS+=(
    -e "SEARXNG_BASE_URL=$AGENT_SEARXNG_BASE_URL"
    -e "WEB_SEARCH_ENABLED=True"
  )
else
  RUN_ENV_ARGS+=(-e "WEB_SEARCH_ENABLED=False")
fi

DOCKER_RUN_ARGS=(--rm -i)
if [[ -t 0 && -t 1 ]]; then
  DOCKER_RUN_ARGS=(--rm -it)
fi

docker run \
  "${DOCKER_RUN_ARGS[@]}" \
  "${ENV_FILE_ARGS[@]}" \
  "${DOCKER_NETWORK_ARGS[@]}" \
  "${DOCKER_ENV_ARGS[@]}" \
  "${RUN_ENV_ARGS[@]}" \
  --entrypoint bash \
  -v "$REPO_ROOT:$CONTAINER_AGENT_SOURCE_ROOT:ro" \
  -v "$HOST_CLAW_CODE_HOME:/root/.claw-code" \
  -v "$HOST_ATTACHMENTS_MOUNT_ROOT:/host-attachments:ro" \
  -v "$WORKSPACE_DIR:/workspace" \
  -w /workspace \
  "$DOCKER_IMAGE" \
  "$CONTAINER_AGENT_SOURCE_ROOT/docker/entrypoint.sh"
