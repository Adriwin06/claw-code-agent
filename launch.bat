@echo off
setlocal

call "%~dp0scripts\launch-workspace.bat"
exit /b %errorlevel%
