@echo off
REM Double-click this file to start Video Translator (Windows)
cd /d "%~dp0"
docker compose up -d
timeout /t 2 >nul
start http://localhost:7860
echo.
echo Video Translator running at http://localhost:7860
echo Close this window anytime - it keeps running in background
echo To stop: double-click stop.bat
pause
