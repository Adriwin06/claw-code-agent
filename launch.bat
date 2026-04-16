@echo off
setlocal

call "%~dp0launch-workspace.bat"
exit /b %errorlevel%
