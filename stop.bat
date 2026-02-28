@echo off
REM Double-click this file to stop Video Translator (Windows)
cd /d "%~dp0"
docker compose down
echo.
echo Video Translator stopped
pause
