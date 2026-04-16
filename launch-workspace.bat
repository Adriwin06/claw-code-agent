@echo off
setlocal

rem Launch Claw Code Agent against the current directory.
rem If this script is not inside the repository tree, set CLAW_CODE_AGENT_ROOT first.
rem Optional environment variables:
rem   CLAW_AGENT_COMMAND=agent-tui|agent-chat|doctor|...
rem   CLAW_REBUILD=1              rebuild the image before launching
rem   CLAW_START_SAGEMATH=1       start the optional SageMath sidecar first

set "WORKSPACE_DIR=%CD%"
set "REPO_ROOT=%CLAW_CODE_AGENT_ROOT%"
set "LAUNCH_AGENT_COMMAND=%CLAW_AGENT_COMMAND%"
if not defined LAUNCH_AGENT_COMMAND set "LAUNCH_AGENT_COMMAND=agent-tui"
set "DOCKER_IMAGE=claw-code-agent-local"

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

echo Launching Claw Code Agent
echo   repo: %REPO_ROOT%
echo   workspace: %WORKSPACE_DIR%
echo   command: %LAUNCH_AGENT_COMMAND%
if /i "%CLAW_REBUILD%"=="1" (
  echo   image rebuild: enabled
) else (
  echo   image rebuild: skipped
)

if /i "%CLAW_REBUILD%"=="1" goto build_image
docker image inspect "%DOCKER_IMAGE%" >nul 2>&1
if errorlevel 1 goto build_image
goto image_ready

:build_image
echo Building Docker image %DOCKER_IMAGE%...
docker build -t "%DOCKER_IMAGE%" -f "%REPO_ROOT%\docker\Dockerfile" "%REPO_ROOT%"
if errorlevel 1 exit /b %errorlevel%

:image_ready
if /i "%CLAW_START_SAGEMATH%"=="1" (
  echo Starting optional SageMath sidecar...
  docker compose -f "%REPO_ROOT%\docker-compose.yml" --project-directory "%REPO_ROOT%" up -d sagemath
  if errorlevel 1 exit /b %errorlevel%
  if exist "%ENV_FILE%" (
    docker run --rm -it ^
      --env-file "%ENV_FILE%" ^
      --add-host "host.docker.internal:host-gateway" ^
      -e "AGENT_COMMAND=%LAUNCH_AGENT_COMMAND%" ^
      -e "AGENT_CWD=/workspace" ^
      -e "AGENT_READ_ONLY=false" ^
      -e "AGENT_ALLOW_WRITE=true" ^
      -e "AGENT_ALLOW_SHELL=false" ^
      -e "AGENT_UNSAFE=false" ^
      -e "SAGEMATH_MCP_URL=http://sagemath:8000/mcp" ^
      -v "%WORKSPACE_DIR%:/workspace" ^
      -w /workspace ^
      "%DOCKER_IMAGE%"
  ) else (
    docker run --rm -it ^
      --add-host "host.docker.internal:host-gateway" ^
      -e "AGENT_COMMAND=%LAUNCH_AGENT_COMMAND%" ^
      -e "AGENT_CWD=/workspace" ^
      -e "AGENT_READ_ONLY=false" ^
      -e "AGENT_ALLOW_WRITE=true" ^
      -e "AGENT_ALLOW_SHELL=false" ^
      -e "AGENT_UNSAFE=false" ^
      -e "SAGEMATH_MCP_URL=http://sagemath:8000/mcp" ^
      -v "%WORKSPACE_DIR%:/workspace" ^
      -w /workspace ^
      "%DOCKER_IMAGE%"
  )
) else (
  echo Skipping SageMath sidecar. Set CLAW_START_SAGEMATH=1 to enable it.
  if exist "%ENV_FILE%" (
    docker run --rm -it ^
      --env-file "%ENV_FILE%" ^
      --add-host "host.docker.internal:host-gateway" ^
      -e "AGENT_COMMAND=%LAUNCH_AGENT_COMMAND%" ^
      -e "AGENT_CWD=/workspace" ^
      -e "AGENT_READ_ONLY=false" ^
      -e "AGENT_ALLOW_WRITE=true" ^
      -e "AGENT_ALLOW_SHELL=false" ^
      -e "AGENT_UNSAFE=false" ^
      -v "%WORKSPACE_DIR%:/workspace" ^
      -w /workspace ^
      "%DOCKER_IMAGE%"
  ) else (
    docker run --rm -it ^
      --add-host "host.docker.internal:host-gateway" ^
      -e "AGENT_COMMAND=%LAUNCH_AGENT_COMMAND%" ^
      -e "AGENT_CWD=/workspace" ^
      -e "AGENT_READ_ONLY=false" ^
      -e "AGENT_ALLOW_WRITE=true" ^
      -e "AGENT_ALLOW_SHELL=false" ^
      -e "AGENT_UNSAFE=false" ^
      -v "%WORKSPACE_DIR%:/workspace" ^
      -w /workspace ^
      "%DOCKER_IMAGE%"
  )
)

exit /b %errorlevel%

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
