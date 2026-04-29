@echo off
setlocal

rem Copy this file into any folder and run it from there.
rem The folder containing this script becomes the agent workspace.
rem If you move the claw-code-agent repository, either update DEFAULT_AGENT_ROOT
rem below or set CLAW_CODE_AGENT_ROOT before running this script.

set "DEFAULT_AGENT_ROOT=C:\Users\adri1\Documents\_Stage\claw-code-agent"
set "WORKSPACE_DIR=%~dp0"

if "%WORKSPACE_DIR:~-1%"=="\" (
  set "WORKSPACE_DIR=%WORKSPACE_DIR:~0,-1%"
)

if not defined CLAW_CODE_AGENT_ROOT (
  set "CLAW_CODE_AGENT_ROOT=%DEFAULT_AGENT_ROOT%"
)

if "%CLAW_CODE_AGENT_ROOT:~-1%"=="\" (
  set "CLAW_CODE_AGENT_ROOT=%CLAW_CODE_AGENT_ROOT:~0,-1%"
)

if not exist "%CLAW_CODE_AGENT_ROOT%\launch-workspace.bat" (
  echo Could not find launch-workspace.bat at:
  echo   %CLAW_CODE_AGENT_ROOT%
  echo.
  echo Update DEFAULT_AGENT_ROOT inside this script or set CLAW_CODE_AGENT_ROOT.
  exit /b 1
)

pushd "%WORKSPACE_DIR%"
call "%CLAW_CODE_AGENT_ROOT%\launch-workspace.bat"
set "EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %EXIT_CODE%
