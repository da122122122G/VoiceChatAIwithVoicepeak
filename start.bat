@echo off
setlocal EnableExtensions
chcp 65001 > nul
cd /d "%~dp0"

set "CHECK_ONLY=0"
set "FORCED_BACKEND="

:parse_arguments
if "%~1"=="" goto arguments_done
if /i "%~1"=="--check" set "CHECK_ONLY=1"
if /i "%~1"=="--cpu" set "FORCED_BACKEND=cpu"
if /i "%~1"=="--cuda" set "FORCED_BACKEND=cuda"
shift
goto parse_arguments

:arguments_done
set "WHISPER_BACKEND=cpu"
set "CUDA_ARCH="
set "WHISPER_BIN=whisper.cpp\build-voice-chat\bin\Release"

if /i "%FORCED_BACKEND%"=="cpu" goto backend_ready
if /i "%FORCED_BACKEND%"=="cuda" goto enable_cuda

where.exe nvidia-smi > nul 2>&1
if errorlevel 1 goto backend_ready

where.exe nvcc > nul 2>&1
if errorlevel 1 goto backend_ready

:enable_cuda
set "WHISPER_BACKEND=cuda"
set "WHISPER_BIN=whisper.cpp\build-voice-chat\bin"

for /f "tokens=1 delims=," %%G in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul') do (
    if not defined CUDA_ARCH set "CUDA_ARCH=%%G"
)

set "CUDA_ARCH=%CUDA_ARCH:.=%"
set "CUDA_ARCH=%CUDA_ARCH: =%"

:backend_ready
set "SETUP_REQUIRED=0"

if not exist ".venv\Scripts\python.exe" set "SETUP_REQUIRED=1"
if not exist "%WHISPER_BIN%\whisper-server.exe" set "SETUP_REQUIRED=1"
if not exist "whisper.cpp\models\ggml-small.bin" set "SETUP_REQUIRED=1"
if not exist "external\VoicepeakProxyCore\VoicepeakProxyCore.dll" set "SETUP_REQUIRED=1"
if not exist "voicepeak_proxy_test\bin\Release\net48\VoicepeakProxyTest.exe" set "SETUP_REQUIRED=1"
if not exist "whisper.cpp\build-voice-chat\CMakeCache.txt" set "SETUP_REQUIRED=1"
if not exist "whisper.cpp\build-voice-chat\.voice-chat-backend" set "SETUP_REQUIRED=1"

if /i "%WHISPER_BACKEND%"=="cuda" (
    findstr.exe /x /c:"GGML_CUDA:BOOL=ON" "whisper.cpp\build-voice-chat\CMakeCache.txt" > nul 2>&1
    if errorlevel 1 set "SETUP_REQUIRED=1"
    findstr.exe /x /c:"cuda" "whisper.cpp\build-voice-chat\.voice-chat-backend" > nul 2>&1
    if errorlevel 1 set "SETUP_REQUIRED=1"
    findstr.exe /x /c:"CMAKE_GENERATOR:INTERNAL=Ninja" "whisper.cpp\build-voice-chat\CMakeCache.txt" > nul 2>&1
    if errorlevel 1 set "SETUP_REQUIRED=1"
    if not exist "%WHISPER_BIN%\ggml-cuda.dll" set "SETUP_REQUIRED=1"
) else (
    findstr.exe /x /c:"GGML_CUDA:BOOL=OFF" "whisper.cpp\build-voice-chat\CMakeCache.txt" > nul 2>&1
    if errorlevel 1 set "SETUP_REQUIRED=1"
    findstr.exe /x /c:"cpu" "whisper.cpp\build-voice-chat\.voice-chat-backend" > nul 2>&1
    if errorlevel 1 set "SETUP_REQUIRED=1"
    findstr.exe /x /c:"CMAKE_GENERATOR:INTERNAL=Visual Studio 17 2022" "whisper.cpp\build-voice-chat\CMakeCache.txt" > nul 2>&1
    if errorlevel 1 set "SETUP_REQUIRED=1"
)

if "%CHECK_ONLY%"=="1" (
    if "%SETUP_REQUIRED%"=="1" (
        echo Setup is required for Whisper backend: %WHISPER_BACKEND%
        exit /b 1
    )
)

if "%SETUP_REQUIRED%"=="1" (
    echo Preparing Whisper backend: %WHISPER_BACKEND%
    echo Close VOICEPEAK and VoicepeakProxyTest before setup.

    if /i "%WHISPER_BACKEND%"=="cuda" (
        if defined CUDA_ARCH (
            powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" -WhisperBackend cuda -CudaArchitectures "%CUDA_ARCH%"
        ) else (
            powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" -WhisperBackend cuda
        )
    ) else (
        powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" -WhisperBackend cpu
    )

    if errorlevel 1 (
        echo Automatic setup failed.
        pause
        exit /b 1
    )
)

if not exist "app_config.json" (
    echo app_config.json was not found.
    echo Run setup.ps1 and configure the VOICEPEAK path first.
    pause
    exit /b 1
)

findstr.exe /i /c:"Path\\To\\VOICEPEAK" "app_config.json" > nul 2>&1
if not errorlevel 1 (
    echo Configure voicepeak_exe in app_config.json first.
    pause
    exit /b 1
)

if "%GEMINI_API_KEY%"=="" (
    echo GEMINI_API_KEY is not set.
    echo Set the API key and reopen the terminal.
    pause
    exit /b 1
)

if "%CHECK_ONLY%"=="1" (
    echo Startup checks passed. Whisper backend: %WHISPER_BACKEND%
    exit /b 0
)

".venv\Scripts\python.exe" "voice_chat_gemini.py"
set "VOICE_CHAT_EXIT_CODE=%ERRORLEVEL%"

if not "%VOICE_CHAT_EXIT_CODE%"=="0" pause
exit /b %VOICE_CHAT_EXIT_CODE%
