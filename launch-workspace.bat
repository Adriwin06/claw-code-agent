@echo off
setlocal

rem Launch Claw Code Agent against the current directory.
rem If this script is not inside the repository tree, set CLAW_CODE_AGENT_ROOT first.

set "WORKSPACE_DIR=%CD%"
set "REPO_ROOT=%CLAW_CODE_AGENT_ROOT%"

if not defined REPO_ROOT (
  set "REPO_ROOT=%~dp0"
  if not exist "%REPO_ROOT%\docker-compose.yml" (
    call :find_repo_root "%REPO_ROOT%"
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

docker compose -f "%REPO_ROOT%\docker-compose.yml" --project-directory "%REPO_ROOT%" run --build --rm ^
  -e "AGENT_COMMAND=agent-tui" ^
  -e "AGENT_CWD=/workspace" ^
  -e "AGENT_READ_ONLY=false" ^
  -e "AGENT_ALLOW_WRITE=true" ^
  -e "AGENT_ALLOW_SHELL=false" ^
  -e "AGENT_UNSAFE=false" ^
  -e "HOST_WORKSPACE_DIR=%WORKSPACE_DIR%" ^
  claw-agent

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
