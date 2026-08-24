[CmdletBinding()]
param(
    [ValidateSet("cpu", "cuda")]
    [string]$WhisperBackend = "cpu",

    [string]$CudaArchitectures = "",

    [ValidateRange(1, 128)]
    [int]$BuildJobs = [Environment]::ProcessorCount,

    [switch]$ForceDownload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepositoryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
$RequirementsFile = Join-Path $RepositoryRoot "requirements.txt"
$AppConfig = Join-Path $RepositoryRoot "app_config.json"
$AppConfigExample = Join-Path $RepositoryRoot "app_config.example.json"
$WhisperDirectory = Join-Path $RepositoryRoot "whisper.cpp"
$WhisperBuildDirectory = Join-Path $WhisperDirectory "build-voice-chat"
$WhisperServer = Join-Path $WhisperBuildDirectory "bin\Release\whisper-server.exe"
$WhisperModel = Join-Path $WhisperDirectory "models\ggml-small.bin"
$VoicepeakCoreDirectory = Join-Path $RepositoryRoot "external\VoicepeakProxyCore"
$VoicepeakCoreDll = Join-Path $VoicepeakCoreDirectory "VoicepeakProxyCore.dll"
$BridgeProject = Join-Path $RepositoryRoot "voicepeak_proxy_test\VoicepeakProxyTest.csproj"
$BridgeOutput = Join-Path $RepositoryRoot "voicepeak_proxy_test\bin\Release\net48"
$BridgeExecutable = Join-Path $BridgeOutput "VoicepeakProxyTest.exe"

$WhisperRepository = "https://github.com/ggml-org/whisper.cpp.git"
$WhisperRevision = "4834a2327d008ace3ec5a9ed00f51454bcabbc1c"
$WhisperModelUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"
$WhisperModelSha1 = "55356645c2b361a969dfd0ef2c5a50d530afd8d5"
$VoicepeakProxyVersion = "1.2.1"
$VoicepeakReleaseApi = "https://api.github.com/repos/rotensyo/VoicepeakProxy/releases/tags/v$VoicepeakProxyVersion"


function Write-Step {
    param([string]$Message)

    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}


function Require-Command {
    param(
        [string]$Name,
        [string]$InstallHint
    )

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name が見つかりません。$InstallHint"
    }
}


function Invoke-Checked {
    param(
        [string]$Command,
        [string[]]$Arguments
    )

    & $Command @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "$Command の実行に失敗しました。終了コード: $LASTEXITCODE"
    }
}


function Get-PythonCommand {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return [pscustomobject]@{
            Executable = "python"
            PrefixArguments = @()
        }
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        return [pscustomobject]@{
            Executable = "py"
            PrefixArguments = @("-3")
        }
    }

    throw "Pythonが見つかりません。Python 3.12以降をインストールしてください。"
}


Write-Host "================================"
Write-Host " Voice Chat AI setup"
Write-Host "================================"
Write-Host "Repository: $RepositoryRoot"
Write-Host "Whisper backend: $WhisperBackend"

Write-Step "前提ツールを確認"
Require-Command "git" "Git for Windowsをインストールしてください。"
Require-Command "cmake" "CMakeをインストールしてください。"
Require-Command "dotnet" ".NET SDKと.NET Framework 4.8 Developer Packをインストールしてください。"
$PythonCommand = Get-PythonCommand
$VersionArguments = @($PythonCommand.PrefixArguments) + @(
    "-c",
    "import sys; print('.'.join(map(str, sys.version_info[:3])))"
)
$PythonVersionText = & $PythonCommand.Executable @VersionArguments

if ($LASTEXITCODE -ne 0) {
    throw "Pythonのバージョンを確認できませんでした。"
}

$PythonVersion = [version]$PythonVersionText.Trim()

if ($PythonVersion -lt [version]"3.12") {
    throw "Python 3.12以降が必要です。現在: $PythonVersion"
}

Write-Host "Python: $PythonVersion"

$ProcessesToClose = @(
    Get-Process -Name "voicepeak", "VoicepeakProxyTest" -ErrorAction SilentlyContinue
)

if ($ProcessesToClose.Count -gt 0) {
    $ProcessNames = (
        $ProcessesToClose |
        ForEach-Object { "$($_.ProcessName) (PID=$($_.Id))" }
    ) -join ", "

    throw (
        "セットアップ中はVOICEPEAKとVoicepeakProxyTestを終了してください。" +
        " 実行中: $ProcessNames"
    )
}

Write-Step "app_config.jsonを準備"
if (-not (Test-Path -LiteralPath $AppConfig)) {
    Copy-Item -LiteralPath $AppConfigExample -Destination $AppConfig
    Write-Warning "app_config.jsonを作成しました。setup完了後、voicepeak_exeを実際の場所へ変更してください。"
}
else {
    Write-Host "既存のapp_config.jsonを使用します。"
}

Write-Step "Python仮想環境と依存パッケージを準備"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    $PythonExecutable = $PythonCommand.Executable
    $PythonArguments = @($PythonCommand.PrefixArguments)
    $PythonArguments += @("-m", "venv", (Join-Path $RepositoryRoot ".venv"))
    Invoke-Checked $PythonExecutable $PythonArguments
}

Invoke-Checked $VenvPython @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $VenvPython @("-m", "pip", "install", "-r", $RequirementsFile)

Write-Step "whisper.cppを準備"
if (-not (Test-Path -LiteralPath $WhisperDirectory)) {
    Invoke-Checked "git" @("clone", "--filter=blob:none", "--no-checkout", $WhisperRepository, $WhisperDirectory)
    Invoke-Checked "git" @("-C", $WhisperDirectory, "checkout", "--detach", $WhisperRevision)
}
elseif (-not (Test-Path -LiteralPath (Join-Path $WhisperDirectory ".git"))) {
    throw "whisper.cppフォルダーはありますがGitリポジトリではありません: $WhisperDirectory"
}
else {
    Write-Host "既存のwhisper.cppを使用します。"
}

$CMakeArguments = @(
    "-S", $WhisperDirectory,
    "-B", $WhisperBuildDirectory,
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DWHISPER_BUILD_EXAMPLES=ON",
    "-DWHISPER_BUILD_SERVER=ON"
)

if ($WhisperBackend -eq "cuda") {
    $CMakeArguments += "-DGGML_CUDA=ON"

    if ($CudaArchitectures) {
        $CMakeArguments += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
    }
}
else {
    $CMakeArguments += "-DGGML_CUDA=OFF"
}

Invoke-Checked "cmake" $CMakeArguments
Invoke-Checked "cmake" @(
    "--build", $WhisperBuildDirectory,
    "--config", "Release",
    "--target", "whisper-server",
    "-j", $BuildJobs.ToString()
)

if (-not (Test-Path -LiteralPath $WhisperServer)) {
    throw "whisper-server.exeが生成されませんでした: $WhisperServer"
}

Write-Step "Whisper smallモデルを準備"
$DownloadModel = $ForceDownload -or -not (Test-Path -LiteralPath $WhisperModel)

if (-not $DownloadModel) {
    $CurrentHash = (Get-FileHash -Algorithm SHA1 -LiteralPath $WhisperModel).Hash.ToLowerInvariant()

    if ($CurrentHash -ne $WhisperModelSha1) {
        throw "既存モデルのSHA-1が一致しません。再取得する場合は -ForceDownload を指定してください: $WhisperModel"
    }

    Write-Host "既存モデルのSHA-1を確認しました。"
}
else {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $WhisperModel) | Out-Null
    $TemporaryModel = "$WhisperModel.download"

    try {
        Invoke-WebRequest -Uri $WhisperModelUrl -OutFile $TemporaryModel
        $DownloadedHash = (Get-FileHash -Algorithm SHA1 -LiteralPath $TemporaryModel).Hash.ToLowerInvariant()

        if ($DownloadedHash -ne $WhisperModelSha1) {
            throw "ダウンロードしたWhisperモデルのSHA-1が一致しません。"
        }

        Move-Item -LiteralPath $TemporaryModel -Destination $WhisperModel -Force
    }
    finally {
        if (Test-Path -LiteralPath $TemporaryModel) {
            Remove-Item -LiteralPath $TemporaryModel -Force
        }
    }
}

Write-Step "VoicepeakProxyCoreを準備"
if ($ForceDownload -or -not (Test-Path -LiteralPath $VoicepeakCoreDll)) {
    $Headers = @{ "User-Agent" = "VoiceChatAIwithVoicepeak-setup" }
    $Release = Invoke-RestMethod -Uri $VoicepeakReleaseApi -Headers $Headers
    $Asset = @($Release.assets) |
        Where-Object { $_.name -eq "VoicepeakProxyCore-$VoicepeakProxyVersion.zip" } |
        Select-Object -First 1

    if ($null -eq $Asset) {
        throw "最新ReleaseにVoicepeakProxyCoreのZIPが見つかりません。"
    }

    $TemporaryZip = Join-Path (
        [System.IO.Path]::GetTempPath()
    ) ("VoicepeakProxyCore-" + [guid]::NewGuid().ToString("N") + ".zip")
    $TemporaryExtract = Join-Path (
        [System.IO.Path]::GetTempPath()
    ) ("VoicepeakProxyCore-" + [guid]::NewGuid().ToString("N"))

    try {
        Invoke-WebRequest -Uri $Asset.browser_download_url -OutFile $TemporaryZip -Headers $Headers
        New-Item -ItemType Directory -Path $TemporaryExtract | Out-Null
        Expand-Archive -LiteralPath $TemporaryZip -DestinationPath $TemporaryExtract -Force

        $DownloadedDll = Get-ChildItem -LiteralPath $TemporaryExtract -Recurse -Filter "VoicepeakProxyCore.dll" |
            Select-Object -First 1

        if ($null -eq $DownloadedDll) {
            throw "VoicepeakProxyCore.dllがZIP内に見つかりません。"
        }

        New-Item -ItemType Directory -Force -Path $VoicepeakCoreDirectory | Out-Null
        Copy-Item -Path (Join-Path $DownloadedDll.Directory.FullName "*") -Destination $VoicepeakCoreDirectory -Recurse -Force
    }
    finally {
        if (Test-Path -LiteralPath $TemporaryZip) {
            Remove-Item -LiteralPath $TemporaryZip -Force
        }

        if (Test-Path -LiteralPath $TemporaryExtract) {
            $TempRoot = [System.IO.Path]::GetFullPath(
                [System.IO.Path]::GetTempPath()
            )
            $ResolvedExtract = [System.IO.Path]::GetFullPath(
                $TemporaryExtract
            )

            if (-not $ResolvedExtract.StartsWith(
                $TempRoot,
                [System.StringComparison]::OrdinalIgnoreCase
            )) {
                throw "一時展開先が想定範囲外です: $ResolvedExtract"
            }

            Remove-Item -LiteralPath $ResolvedExtract -Recurse -Force
        }
    }
}
else {
    Write-Host "既存のVoicepeakProxyCoreを使用します。"
}

Write-Step "VOICEPEAK Bridgeをビルド"
Invoke-Checked "dotnet" @("build", $BridgeProject, "-c", "Release")
New-Item -ItemType Directory -Force -Path $BridgeOutput | Out-Null
Copy-Item -Path (Join-Path $VoicepeakCoreDirectory "*") -Destination $BridgeOutput -Recurse -Force
Get-ChildItem -LiteralPath $BridgeOutput -Recurse -File | Unblock-File

if (-not (Test-Path -LiteralPath $BridgeExecutable)) {
    throw "VOICEPEAK Bridgeが生成されませんでした: $BridgeExecutable"
}

Write-Step "セットアップ完了"
Write-Host ""
Write-Host "次の作業:"
Write-Host "1. app_config.jsonのvoicepeak_exeを確認"
Write-Host "2. GEMINI_API_KEYをWindowsの環境変数へ設定"
Write-Host "3. start.batを実行"
