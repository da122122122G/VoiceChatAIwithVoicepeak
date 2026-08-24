@echo off
setlocal
chcp 65001 > nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Python virtual environment was not found.
    echo Run setup.ps1 first.
    pause
    exit /b 1
)

if not exist "app_config.json" (
    echo app_config.json was not found.
    echo Run setup.ps1 and configure the VOICEPEAK path first.
    pause
    exit /b 1
)

if "%GEMINI_API_KEY%"=="" (
    echo GEMINI_API_KEY is not set.
    echo Set the API key and reopen the terminal.
    pause
    exit /b 1
)

if /i "%~1"=="--check" (
    echo Startup checks passed.
    exit /b 0
)

".venv\Scripts\python.exe" "voice_chat_gemini.py"
set "VOICE_CHAT_EXIT_CODE=%ERRORLEVEL%"

if not "%VOICE_CHAT_EXIT_CODE%"=="0" pause
exit /b %VOICE_CHAT_EXIT_CODE%
