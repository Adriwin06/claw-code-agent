@echo off
setlocal
goto main

:read_env_file_value
set "READ_ENV_FILE=%~1"
set "READ_ENV_KEY=%~2"
set "READ_ENV_TARGET=%~3"
for /f "usebackq tokens=1,* delims==" %%A in ("%READ_ENV_FILE%") do (
  if /i "%%~A"=="%READ_ENV_KEY%" (
    set "%READ_ENV_TARGET%=%%~B"
    exit /b 0
  )
)
exit /b 1

:uses_ollama_backend
echo(%~1 | findstr /I /C:":11434" /C:"ollama" >nul 2>&1
exit /b %errorlevel%

:host_ollama_ready
curl.exe -fsS --max-time 3 "http://127.0.0.1:11434/api/tags" >nul 2>&1
exit /b %errorlevel%

:wait_http_url
set "WAIT_LABEL=%~1"
set "WAIT_URL=%~2"
set "WAIT_SECONDS=%~3"
if not defined WAIT_SECONDS set "WAIT_SECONDS=60"
for /l %%I in (1,1,%WAIT_SECONDS%) do (
  curl.exe -fsS --max-time 3 "%WAIT_URL%" >nul 2>&1
  if not errorlevel 1 (
    echo %WAIT_LABEL% is reachable at %WAIT_URL%
    exit /b 0
  )
  timeout /t 1 /nobreak >nul
)
echo Warning: %WAIT_LABEL% did not become reachable at %WAIT_URL% within %WAIT_SECONDS%s.
exit /b 1

:find_repo_root
set "SEARCH_DIR=%~1"

:find_repo_root_loop
if exist "%SEARCH_DIR%\docker-compose.yml" (
  set "REPO_ROOT=%SEARCH_DIR%"
  exit /b 0
)

for %%I in ("%SEARCH_DIR%\..") do set "PARENT_DIR=%%~fI"
if /i "%PARENT_DIR%"=="%SEARCH_DIR%" exit /b 1
set "SEARCH_DIR=%PARENT_DIR%"
goto find_repo_root_loop

:main

rem Launch Claw Code Agent against the current directory.
rem If this script is not inside the repository tree, set CLAW_CODE_AGENT_ROOT first.
rem Optional environment variables:
rem   CLAW_AGENT_COMMAND=agent-tui|agent-chat|doctor|...
rem   CLAW_REBUILD=1              rebuild the image before launching
rem   CLAW_START_SAGEMATH=0       disable the default SageMath sidecar
rem   CLAW_START_SEARCH=0         disable the default SearXNG sidecar
rem   CLAW_HOST_CODE_HOME=...     host directory for persistent TUI history
rem   CLAW_HOST_ATTACHMENTS_ROOT=... host directory mounted read-only for pasted file paths
rem   SAGEMATH_HOST_PORT=18000    host port published by the SageMath sidecar
rem   SEARXNG_HOST_PORT=8080      host port published by the SearXNG sidecar

set "WORKSPACE_DIR=%CD%"
set "REPO_ROOT=%CLAW_CODE_AGENT_ROOT%"
set "LAUNCH_AGENT_COMMAND=%CLAW_AGENT_COMMAND%"
if not defined LAUNCH_AGENT_COMMAND set "LAUNCH_AGENT_COMMAND=agent-tui"
if not defined CLAW_START_SAGEMATH set "CLAW_START_SAGEMATH=1"
if not defined CLAW_START_SEARCH set "CLAW_START_SEARCH=1"
if not defined SAGEMATH_HOST_PORT set "SAGEMATH_HOST_PORT=18000"
if not defined SEARXNG_HOST_PORT set "SEARXNG_HOST_PORT=8080"
set "DOCKER_IMAGE=claw-code-agent-local"
set "CONTAINER_AGENT_SOURCE_ROOT=/agent-source"
set "HOST_CLAW_CODE_HOME=%CLAW_HOST_CODE_HOME%"
if not defined HOST_CLAW_CODE_HOME (
  if defined USERPROFILE (
    set "HOST_CLAW_CODE_HOME=%USERPROFILE%\.claw-code"
  ) else (
    set "HOST_CLAW_CODE_HOME=%HOMEDRIVE%%HOMEPATH%\.claw-code"
  )
)
set "HOST_ATTACHMENTS_ROOT=%CLAW_HOST_ATTACHMENTS_ROOT%"
if not defined HOST_ATTACHMENTS_ROOT if defined USERPROFILE set "HOST_ATTACHMENTS_ROOT=%USERPROFILE%"
if not defined HOST_ATTACHMENTS_ROOT set "HOST_ATTACHMENTS_ROOT=%WORKSPACE_DIR%"

if not defined REPO_ROOT (
  set "REPO_ROOT=%~dp0"
  if not exist "%REPO_ROOT%\docker-compose.yml" (
    call :find_repo_root "%REPO_ROOT%"
  )
)

if defined REPO_ROOT (
  if "%REPO_ROOT:~-1%"=="\" (
    set "REPO_ROOT=%REPO_ROOT:~0,-1%"
  )
)

if not defined REPO_ROOT (
  echo Could not locate the claw-code-agent repository.
  echo Set CLAW_CODE_AGENT_ROOT to the repository root, then run this script again.
  exit /b 1
)

if not exist "%REPO_ROOT%\docker-compose.yml" (
  echo Docker compose file not found at "%REPO_ROOT%\docker-compose.yml".
  exit /b 1
)

set "ENV_FILE=%REPO_ROOT%\.env"
set "AGENT_SIDECAR_HOST=host.docker.internal"
set "AGENT_SAGEMATH_MCP_URL=%CLAW_AGENT_SAGEMATH_MCP_URL%"
if not defined AGENT_SAGEMATH_MCP_URL set "AGENT_SAGEMATH_MCP_URL=http://%AGENT_SIDECAR_HOST%:%SAGEMATH_HOST_PORT%/mcp"
set "AGENT_SEARXNG_BASE_URL=%CLAW_AGENT_SEARXNG_BASE_URL%"
if not defined AGENT_SEARXNG_BASE_URL set "AGENT_SEARXNG_BASE_URL=http://%AGENT_SIDECAR_HOST%:%SEARXNG_HOST_PORT%"
set "RUN_LLM_API_BASE_ENV="
set "RUN_LLM_API_KEY_ENV="
set "LLM_API_BASE_VALUE=%LLM_API_BASE%"
set "LLM_API_KEY_VALUE=%LLM_API_KEY%"
if exist "%ENV_FILE%" (
  call :read_env_file_value "%ENV_FILE%" "LLM_API_BASE" LLM_API_BASE_VALUE
  call :read_env_file_value "%ENV_FILE%" "LLM_API_KEY" LLM_API_KEY_VALUE
)
call :uses_ollama_backend "%LLM_API_BASE_VALUE%"
if not errorlevel 1 (
  set "RUN_LLM_API_BASE_ENV=-e "LLM_API_BASE=http://host.docker.internal:11434/v1""
  if not defined LLM_API_KEY_VALUE set "RUN_LLM_API_KEY_ENV=-e "LLM_API_KEY=ollama""
  call :host_ollama_ready
  if errorlevel 1 (
    echo Warning: Ollama is configured, but http://127.0.0.1:11434/api/tags is not reachable.
    echo Start Ollama and make sure the configured model is pulled.
  ) else (
    echo Detected host Ollama. Using host.docker.internal for the Docker backend URL.
  )
)

echo Launching Claw Code Agent
echo   repo: %REPO_ROOT%
echo   workspace: %WORKSPACE_DIR%
echo   source: %REPO_ROOT% -^> %CONTAINER_AGENT_SOURCE_ROOT%
echo   history: %HOST_CLAW_CODE_HOME%
if defined HOST_ATTACHMENTS_ROOT echo   host attachments: %HOST_ATTACHMENTS_ROOT%
echo   command: %LAUNCH_AGENT_COMMAND%
if /i "%CLAW_REBUILD%"=="1" (
  echo   image rebuild: enabled
) else (
  echo   image rebuild: skipped
)

if /i "%CLAW_REBUILD%"=="1" goto build_image
docker image inspect "%DOCKER_IMAGE%" >nul 2>&1
if errorlevel 1 goto build_image
docker run --rm --entrypoint python "%DOCKER_IMAGE%" -c "import litellm, textual; from PIL import Image" >nul 2>&1
if errorlevel 1 (
  echo Cached Docker image is missing required Python packages for the TUI image preview. Rebuilding %DOCKER_IMAGE%...
  goto build_image
)
docker run --rm -e "AGENT_COMMAND=doctor" -e "AGENT_SKIP_BACKEND=true" -e "AGENT_SKIP_TUI=true" "%DOCKER_IMAGE%" >nul 2>&1
if errorlevel 1 (
  echo Cached Docker image entrypoint is not runnable. Rebuilding %DOCKER_IMAGE%...
  goto build_image
)
goto image_ready

:build_image
echo Building Docker image %DOCKER_IMAGE%...
docker build -t "%DOCKER_IMAGE%" -f "%REPO_ROOT%\docker\Dockerfile" "%REPO_ROOT%"
if errorlevel 1 exit /b %errorlevel%

:image_ready
set "SIDECAR_SERVICES="
if /i "%CLAW_START_SAGEMATH%"=="1" (
  set "SIDECAR_SERVICES=%SIDECAR_SERVICES% sagemath"
) else (
  echo Skipping SageMath sidecar. Set CLAW_START_SAGEMATH=1 to enable it.
)
if /i "%CLAW_START_SEARCH%"=="1" (
  set "SIDECAR_SERVICES=%SIDECAR_SERVICES% searxng"
) else (
  echo Skipping SearXNG sidecar. Set CLAW_START_SEARCH=1 to enable it.
)

if defined SIDECAR_SERVICES (
  echo Starting optional sidecars:%SIDECAR_SERVICES%...
  docker compose -f "%REPO_ROOT%\docker-compose.yml" --project-directory "%REPO_ROOT%" up -d %SIDECAR_SERVICES%
  if errorlevel 1 exit /b %errorlevel%
  if /i "%CLAW_START_SAGEMATH%"=="1" call :wait_http_url "SageMath MCP" "http://127.0.0.1:%SAGEMATH_HOST_PORT%/health" 90
  if /i "%CLAW_START_SEARCH%"=="1" call :wait_http_url "SearXNG" "http://127.0.0.1:%SEARXNG_HOST_PORT%/" 60
)

if not exist "%HOST_CLAW_CODE_HOME%" mkdir "%HOST_CLAW_CODE_HOME%"

set "ENV_FILE_ARGS="
if exist "%ENV_FILE%" set "ENV_FILE_ARGS=--env-file "%ENV_FILE%""
set "RUN_SAGEMATH_ENV="
if /i "%CLAW_START_SAGEMATH%"=="1" set "RUN_SAGEMATH_ENV=-e "SAGEMATH_MCP_URL=%AGENT_SAGEMATH_MCP_URL%""
set "RUN_SEARCH_ENV=-e "WEB_SEARCH_ENABLED=False""
if /i "%CLAW_START_SEARCH%"=="1" (
  set "RUN_SEARCH_ENV=-e "SEARXNG_BASE_URL=%AGENT_SEARXNG_BASE_URL%" -e "WEB_SEARCH_ENABLED=True""
)

if defined ENV_FILE_ARGS (
  docker run --rm -it ^
    %ENV_FILE_ARGS% ^
    --add-host "host.docker.internal:host-gateway" ^
    --entrypoint bash ^
    -e "AGENT_COMMAND=%LAUNCH_AGENT_COMMAND%" ^
    -e "AGENT_CWD=/workspace" ^
    -e "AGENT_SOURCE_ROOT=%CONTAINER_AGENT_SOURCE_ROOT%" ^
    -e "CLAW_CODE_HOME=/root/.claw-code" ^
    -e "CLAW_HOST_WORKSPACE=%WORKSPACE_DIR%" ^
    -e "CLAW_CONTAINER_WORKSPACE=/workspace" ^
    -e "CLAW_HOST_ATTACHMENTS_ROOT=%HOST_ATTACHMENTS_ROOT%" ^
    -e "CLAW_HOST_ATTACHMENTS_MOUNT_ROOT=%HOST_ATTACHMENTS_ROOT%" ^
    -e "CLAW_CONTAINER_HOST_ATTACHMENTS_ROOT=/host-attachments" ^
    -e "AGENT_READ_ONLY=false" ^
    -e "AGENT_ALLOW_WRITE=true" ^
    -e "AGENT_ALLOW_SHELL=false" ^
    -e "AGENT_UNSAFE=false" ^
    %RUN_SAGEMATH_ENV% ^
    %RUN_SEARCH_ENV% ^
    %RUN_LLM_API_BASE_ENV% ^
    %RUN_LLM_API_KEY_ENV% ^
    -v "%REPO_ROOT%:%CONTAINER_AGENT_SOURCE_ROOT%:ro" ^
    -v "%HOST_CLAW_CODE_HOME%:/root/.claw-code" ^
    -v "%HOST_ATTACHMENTS_ROOT%:/host-attachments:ro" ^
    -v "%WORKSPACE_DIR%:/workspace" ^
    -w /workspace ^
    "%DOCKER_IMAGE%" ^
    "%CONTAINER_AGENT_SOURCE_ROOT%/docker/entrypoint.sh"
) else (
  docker run --rm -it ^
    --add-host "host.docker.internal:host-gateway" ^
    --entrypoint bash ^
    -e "AGENT_COMMAND=%LAUNCH_AGENT_COMMAND%" ^
    -e "AGENT_CWD=/workspace" ^
    -e "AGENT_SOURCE_ROOT=%CONTAINER_AGENT_SOURCE_ROOT%" ^
    -e "CLAW_CODE_HOME=/root/.claw-code" ^
    -e "CLAW_HOST_WORKSPACE=%WORKSPACE_DIR%" ^
    -e "CLAW_CONTAINER_WORKSPACE=/workspace" ^
    -e "CLAW_HOST_ATTACHMENTS_ROOT=%HOST_ATTACHMENTS_ROOT%" ^
    -e "CLAW_HOST_ATTACHMENTS_MOUNT_ROOT=%HOST_ATTACHMENTS_ROOT%" ^
    -e "CLAW_CONTAINER_HOST_ATTACHMENTS_ROOT=/host-attachments" ^
    -e "AGENT_READ_ONLY=false" ^
    -e "AGENT_ALLOW_WRITE=true" ^
    -e "AGENT_ALLOW_SHELL=false" ^
    -e "AGENT_UNSAFE=false" ^
    %RUN_SAGEMATH_ENV% ^
    %RUN_SEARCH_ENV% ^
    %RUN_LLM_API_BASE_ENV% ^
    %RUN_LLM_API_KEY_ENV% ^
    -v "%REPO_ROOT%:%CONTAINER_AGENT_SOURCE_ROOT%:ro" ^
    -v "%HOST_CLAW_CODE_HOME%:/root/.claw-code" ^
    -v "%HOST_ATTACHMENTS_ROOT%:/host-attachments:ro" ^
    -v "%WORKSPACE_DIR%:/workspace" ^
    -w /workspace ^
    "%DOCKER_IMAGE%" ^
    "%CONTAINER_AGENT_SOURCE_ROOT%/docker/entrypoint.sh"
)

exit /b %errorlevel%
