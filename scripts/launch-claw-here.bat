@echo off
setlocal

rem Copy this file into any folder and run it from there, or call it from a
rem terminal opened in the folder you want to use.
rem The current terminal folder becomes the workspace; Explorer double-clicks
rem fall back to the folder containing this script.
rem If you move the claw-code-agent repository, either update DEFAULT_AGENT_ROOT
rem below or set CLAW_CODE_AGENT_ROOT before running this script.

set "DEFAULT_AGENT_ROOT=C:\Users\adri1\Documents\_Stage\claw-code-agent"
set "SCRIPT_DIR=%~dp0"
set "WORKSPACE_DIR=%CD%"

rem When double-clicked from Explorer, CD is normally the script folder.
rem When invoked from a terminal through a central copy of this script, CD is
rem the folder the user actually wants to open.
if /i "%WORKSPACE_DIR%"=="%SystemRoot%\System32" set "WORKSPACE_DIR=%SCRIPT_DIR%"
if not defined WORKSPACE_DIR set "WORKSPACE_DIR=%SCRIPT_DIR%"

if "%WORKSPACE_DIR:~-1%"=="\" (
  set "WORKSPACE_DIR=%WORKSPACE_DIR:~0,-1%"
)

if not defined CLAW_CODE_AGENT_ROOT (
  set "CLAW_CODE_AGENT_ROOT=%DEFAULT_AGENT_ROOT%"
)

if "%CLAW_CODE_AGENT_ROOT:~-1%"=="\" (
  set "CLAW_CODE_AGENT_ROOT=%CLAW_CODE_AGENT_ROOT:~0,-1%"
)

if not exist "%CLAW_CODE_AGENT_ROOT%\scripts\launch-workspace.bat" (
  echo Could not find scripts\launch-workspace.bat at:
  echo   %CLAW_CODE_AGENT_ROOT%
  echo.
  echo Update DEFAULT_AGENT_ROOT inside this script or set CLAW_CODE_AGENT_ROOT.
  exit /b 1
)

pushd "%WORKSPACE_DIR%"
call "%CLAW_CODE_AGENT_ROOT%\scripts\launch-workspace.bat"
set "EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %EXIT_CODE%
