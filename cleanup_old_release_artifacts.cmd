@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\cleanup_old_release_artifacts.ps1"
exit /b %errorlevel%
