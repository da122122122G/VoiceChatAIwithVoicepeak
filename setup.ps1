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
$RequirementsSnapshot = Join-Path $RepositoryRoot ".venv\.requirements.txt"
$RequirementsFile = Join-Path $RepositoryRoot "requirements.txt"
$AppConfig = Join-Path $RepositoryRoot "app_config.json"
$AppConfigExample = Join-Path $RepositoryRoot "app_config.example.json"
$WhisperDirectory = Join-Path $RepositoryRoot "whisper.cpp"
$WhisperBuildDirectory = Join-Path $WhisperDirectory "build-voice-chat"
$WhisperBackendMarker = Join-Path $WhisperBuildDirectory ".voice-chat-backend"
$WhisperModel = Join-Path $WhisperDirectory "models\ggml-small.bin"
$VoicepeakCoreDirectory = Join-Path $RepositoryRoot "external\VoicepeakProxyCore"
$VoicepeakCoreDll = Join-Path $VoicepeakCoreDirectory "VoicepeakProxyCore.dll"
$BridgeProject = Join-Path $RepositoryRoot "voicepeak_proxy_test\VoicepeakProxyTest.csproj"
$BridgeOutput = Join-Path $RepositoryRoot "voicepeak_proxy_test\bin\Release\net48"
$BridgeExecutable = Join-Path $BridgeOutput "VoicepeakProxyTest.exe"
$BridgeCoreDll = Join-Path $BridgeOutput "VoicepeakProxyCore.dll"
$BridgeProgram = Join-Path (Split-Path -Parent $BridgeProject) "Program.cs"
$BridgeProgramSnapshot = Join-Path $BridgeOutput ".Program.cs.snapshot"
$BridgeProjectSnapshot = Join-Path $BridgeOutput ".VoicepeakProxyTest.csproj.snapshot"

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


function Test-RequiredCommand {
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


function Initialize-VisualStudioBuildEnvironment {
    $VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"

    if (-not (Test-Path -LiteralPath $VsWhere)) {
        throw "vswhere.exeが見つかりません。Visual Studio 2022 Build ToolsのC++ビルドツールをインストールしてください。"
    }

    $VisualStudioPath = (& $VsWhere `
        -latest `
        -products "*" `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath).Trim()

    if (-not $VisualStudioPath) {
        throw "Visual Studio 2022のC++ビルドツールが見つかりません。"
    }

    $VsDevCmd = Join-Path $VisualStudioPath "Common7\Tools\VsDevCmd.bat"

    if (-not (Test-Path -LiteralPath $VsDevCmd)) {
        throw "VsDevCmd.batが見つかりません: $VsDevCmd"
    }

    $CommandLine = "`"$VsDevCmd`" -no_logo -arch=x64 -host_arch=x64 >nul && set"
    $EnvironmentLines = & $env:ComSpec /d /s /c $CommandLine

    if ($LASTEXITCODE -ne 0) {
        throw "Visual Studioのビルド環境を初期化できませんでした。"
    }

    foreach ($Line in $EnvironmentLines) {
        $Separator = $Line.IndexOf("=")

        if ($Separator -le 0) {
            continue
        }

        $Name = $Line.Substring(0, $Separator)
        $Value = $Line.Substring($Separator + 1)
        Set-Item -LiteralPath "Env:$Name" -Value $Value
    }

    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "Visual Studio環境の初期化後もcl.exeが見つかりません。"
    }

    Write-Host "Visual Studio C++ビルド環境を初期化しました。"
}


function Reset-WhisperBuildDirectory {
    if (-not (Test-Path -LiteralPath $WhisperBuildDirectory)) {
        return
    }

    $WhisperRoot = [System.IO.Path]::GetFullPath($WhisperDirectory).TrimEnd("\") + "\"
    $BuildRoot = [System.IO.Path]::GetFullPath($WhisperBuildDirectory)

    if (
        -not $BuildRoot.StartsWith(
            $WhisperRoot,
            [System.StringComparison]::OrdinalIgnoreCase
        ) -or
        [System.IO.Path]::GetFileName($BuildRoot) -ne "build-voice-chat"
    ) {
        throw "Whisperビルド先が想定範囲外です: $BuildRoot"
    }

    Write-Host "ビルド方式を切り替えるため、既存の生成物を再作成します。"
    Remove-Item -LiteralPath $BuildRoot -Recurse -Force
}


Write-Host "================================"
Write-Host " Voice Chat AI setup"
Write-Host "================================"
Write-Host "Repository: $RepositoryRoot"
Write-Host "Whisper backend: $WhisperBackend"

Write-Step "前提ツールを確認"
Test-RequiredCommand "git" "Git for Windowsをインストールしてください。"
Test-RequiredCommand "cmake" "CMakeをインストールしてください。"
Test-RequiredCommand "dotnet" ".NET SDKと.NET Framework 4.8 Developer Packをインストールしてください。"
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

$DependenciesReady = $false

if (Test-Path -LiteralPath $RequirementsSnapshot) {
    $RequirementsHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $RequirementsFile
    ).Hash
    $StoredRequirementsHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $RequirementsSnapshot
    ).Hash

    if ($StoredRequirementsHash -eq $RequirementsHash) {
        & $VenvPython -c "import google.genai, numpy, requests, sounddevice" *> $null

        if ($LASTEXITCODE -eq 0) {
            & $VenvPython -m pip check *> $null
            $DependenciesReady = $LASTEXITCODE -eq 0
        }
    }
}

if ($DependenciesReady) {
    Write-Host "Python依存パッケージは更新不要です。"
}
else {
    Invoke-Checked $VenvPython @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-Checked $VenvPython @("-m", "pip", "install", "-r", $RequirementsFile)
    Copy-Item -LiteralPath $RequirementsFile -Destination $RequirementsSnapshot -Force
}

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

if ($WhisperBackend -eq "cuda") {
    Test-RequiredCommand "nvcc" "CUDA Toolkitをインストールしてください。"
    Test-RequiredCommand "ninja" "Ninja Buildをインストールしてください。"
    Initialize-VisualStudioBuildEnvironment
}

$WhisperGenerator = if ($WhisperBackend -eq "cuda") {
    "Ninja"
}
else {
    "Visual Studio 17 2022"
}

$ExistingGenerator = ""
$CMakeCache = Join-Path $WhisperBuildDirectory "CMakeCache.txt"

if (Test-Path -LiteralPath $CMakeCache) {
    $GeneratorLine = Get-Content -LiteralPath $CMakeCache |
        Where-Object { $_ -like "CMAKE_GENERATOR:INTERNAL=*" } |
        Select-Object -First 1

    if ($GeneratorLine) {
        $ExistingGenerator = $GeneratorLine.Substring(
            "CMAKE_GENERATOR:INTERNAL=".Length
        )
    }
}

if ($ExistingGenerator -and $ExistingGenerator -ne $WhisperGenerator) {
    Reset-WhisperBuildDirectory
}

if (Test-Path -LiteralPath $WhisperBackendMarker) {
    Remove-Item -LiteralPath $WhisperBackendMarker -Force
}

$CMakeArguments = @(
    "-S", $WhisperDirectory,
    "-B", $WhisperBuildDirectory,
    "-G", $WhisperGenerator,
    "-DWHISPER_BUILD_EXAMPLES=ON",
    "-DWHISPER_BUILD_SERVER=ON"
)

if ($WhisperBackend -eq "cuda") {
    $CMakeArguments += "-DCMAKE_BUILD_TYPE=Release"
    $CMakeArguments += "-DGGML_CUDA=ON"

    if ($CudaArchitectures) {
        $CMakeArguments += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
    }
}
else {
    $CMakeArguments += @("-A", "x64")
    $CMakeArguments += "-DGGML_CUDA=OFF"
}

Invoke-Checked "cmake" $CMakeArguments
Invoke-Checked "cmake" @(
    "--build", $WhisperBuildDirectory,
    "--config", "Release",
    "--target", "whisper-server",
    "-j", $BuildJobs.ToString()
)

$WhisperBinDirectory = if ($WhisperBackend -eq "cuda") {
    Join-Path $WhisperBuildDirectory "bin"
}
else {
    Join-Path $WhisperBuildDirectory "bin\Release"
}
$WhisperServer = Join-Path $WhisperBinDirectory "whisper-server.exe"
$WhisperCudaDll = Join-Path $WhisperBinDirectory "ggml-cuda.dll"

if (-not (Test-Path -LiteralPath $WhisperServer)) {
    throw "whisper-server.exeが生成されませんでした: $WhisperServer"
}

if ($WhisperBackend -eq "cuda" -and -not (Test-Path -LiteralPath $WhisperCudaDll)) {
    throw "CUDA版のggml-cuda.dllが生成されませんでした: $WhisperCudaDll"
}

Set-Content -LiteralPath $WhisperBackendMarker -Value $WhisperBackend -Encoding ASCII

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

Write-Step "VOICEPEAK Bridgeを確認"
$BridgeBuildRequired = (
    -not (Test-Path -LiteralPath $BridgeExecutable) -or
    -not (Test-Path -LiteralPath $BridgeCoreDll) -or
    -not (Test-Path -LiteralPath $BridgeProgramSnapshot) -or
    -not (Test-Path -LiteralPath $BridgeProjectSnapshot)
)

if (-not $BridgeBuildRequired) {
    $SourceCoreHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $VoicepeakCoreDll
    ).Hash
    $OutputCoreHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $BridgeCoreDll
    ).Hash
    $BridgeBuildRequired = $SourceCoreHash -ne $OutputCoreHash
}

if (-not $BridgeBuildRequired) {
    $BridgeTimestamp = (Get-Item -LiteralPath $BridgeExecutable).LastWriteTimeUtc
    $BridgeInputs = @(
        Get-Item -LiteralPath $BridgeProject
        Get-Item -LiteralPath $BridgeProgram
        Get-ChildItem -LiteralPath $VoicepeakCoreDirectory -Recurse -File
    )
    $BridgeBuildRequired = @(
        $BridgeInputs |
        Where-Object { $_.LastWriteTimeUtc -gt $BridgeTimestamp }
    ).Count -gt 0
}

if ($BridgeBuildRequired) {
    $ProcessesToClose = @(
        Get-Process -Name "voicepeak", "VoicepeakProxyTest" -ErrorAction SilentlyContinue
    )

    if ($ProcessesToClose.Count -gt 0) {
        $ProcessNames = (
            $ProcessesToClose |
            ForEach-Object { "$($_.ProcessName) (PID=$($_.Id))" }
        ) -join ", "

        throw (
            "Bridgeの更新時はVOICEPEAKとVoicepeakProxyTestを終了してください。" +
            " 実行中: $ProcessNames"
        )
    }

    Invoke-Checked "dotnet" @("build", $BridgeProject, "-c", "Release")
    New-Item -ItemType Directory -Force -Path $BridgeOutput | Out-Null
    Copy-Item -Path (Join-Path $VoicepeakCoreDirectory "*") -Destination $BridgeOutput -Recurse -Force
    Get-ChildItem -LiteralPath $BridgeOutput -Recurse -File | Unblock-File
    Copy-Item -LiteralPath $BridgeProgram -Destination $BridgeProgramSnapshot -Force
    Copy-Item -LiteralPath $BridgeProject -Destination $BridgeProjectSnapshot -Force
}
else {
    Write-Host "VOICEPEAK Bridgeは更新不要です。"
}

if (-not (Test-Path -LiteralPath $BridgeExecutable)) {
    throw "VOICEPEAK Bridgeが生成されませんでした: $BridgeExecutable"
}

Write-Step "セットアップ完了"
Write-Host ""
Write-Host "次の作業:"
Write-Host "1. app_config.jsonのvoicepeak_exeを確認"
Write-Host "2. GEMINI_API_KEYをWindowsの環境変数へ設定"
Write-Host "3. start.batを実行"
