@echo off
setlocal
chcp 65001 > nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Python仮想環境が見つかりません。
    echo 先にsetup.ps1を実行してください。
    pause
    exit /b 1
)

if not exist "app_config.json" (
    echo app_config.jsonが見つかりません。
    echo 先にsetup.ps1を実行し、VOICEPEAKの場所を設定してください。
    pause
    exit /b 1
)

if "%GEMINI_API_KEY%"=="" (
    echo 環境変数GEMINI_API_KEYが設定されていません。
    echo APIキーを設定してから、新しいターミナルで再実行してください。
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "voice_chat_gemini.py"
set "VOICE_CHAT_EXIT_CODE=%ERRORLEVEL%"

if not "%VOICE_CHAT_EXIT_CODE%"=="0" pause
exit /b %VOICE_CHAT_EXIT_CODE%
