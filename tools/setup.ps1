<#
.SYNOPSIS
    Reproducible bootstrap for AI Clip Factory M0.

.DESCRIPTION
    Creates the Python 3.12 virtualenv, installs pinned/binary-only dependencies,
    freezes the verified set into requirements.lock.txt, downloads the pinned
    FFmpeg build into tools\ffmpeg (git-ignored) and runs the environment check.

    Safe to re-run: existing venv and FFmpeg binaries are reused.

.PARAMETER SkipFfmpeg
    Do not download FFmpeg (useful when ACF_FFMPEG already points at a binary).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\setup.ps1
#>
[CmdletBinding()]
param(
    [switch]$SkipFfmpeg
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$ffmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-11-13-12-43/ffmpeg-n8.1.3-35-gf1acb0a266-win64-gpl-8.1.zip"
$ffmpegSha256 = "096d0596ba87ee534c6357797dc68882e714b53aac76859fca4be9c012a6a91c"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

# --- 1. Python 3.12 ---------------------------------------------------------
Write-Step "Checking Python 3.12"
python -c "print(1)" 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "The 'python' alias is the Microsoft Store stub; using 'py -3.12' instead." -ForegroundColor Yellow
}
$pythonVersion = & py -3.12 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0 -or $pythonVersion.Trim() -ne "3.12") {
    throw "Python 3.12 not found via 'py -3.12'. Install it from python.org and retry."
}
Write-Host "Python $pythonVersion found."

# --- 2. Virtualenv ----------------------------------------------------------
Write-Step "Preparing .venv"
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed." }
}
& $venvPython -m pip install -U "pip>=25" --quiet
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

# --- 3. Dependencies --------------------------------------------------------
Write-Step "Installing dependencies (binary wheels only)"
& $venvPython -m pip install --only-binary=:all: -r requirements-dev.in
if ($LASTEXITCODE -ne 0) { throw "dependency installation failed." }

Write-Step "Freezing verified versions"
& $venvPython -m pip freeze | Set-Content -Path requirements.lock.txt -Encoding utf8
Write-Host "requirements.lock.txt written."

# --- 4. FFmpeg --------------------------------------------------------------
Write-Step "Preparing FFmpeg"
$ffmpegExe = Join-Path $root "tools\ffmpeg\bin\ffmpeg.exe"
if ($SkipFfmpeg) {
    Write-Host "Skipped (-SkipFfmpeg)."
} elseif (Test-Path $ffmpegExe) {
    Write-Host "FFmpeg already present at $ffmpegExe"
} else {
    $downloads = Join-Path $root "tools\downloads"
    New-Item -ItemType Directory -Force -Path $downloads | Out-Null
    $zipPath = Join-Path $downloads "ffmpeg-n8.1.3-win64-gpl.zip"

    if (-not (Test-Path $zipPath)) {
        Write-Host "Downloading $ffmpegUrl"
        & curl.exe -L --fail --retry 3 -o $zipPath $ffmpegUrl
        if ($LASTEXITCODE -ne 0) { throw "FFmpeg download failed." }
    }

    $actualHash = (Get-FileHash -Path $zipPath -Algorithm SHA256).Hash.ToLower()
    if ($actualHash -ne $ffmpegSha256) {
        throw "FFmpeg archive checksum mismatch. Expected $ffmpegSha256, got $actualHash."
    }
    Write-Host "Checksum verified."

    $extractDir = Join-Path $root "tools\_extract"
    if (Test-Path $extractDir) { Remove-Item -Recurse -Force $extractDir }
    New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
    & tar.exe -xf $zipPath -C $extractDir
    if ($LASTEXITCODE -ne 0) { throw "FFmpeg extraction failed." }

    $binDir = Get-ChildItem -Path $extractDir -Directory |
        ForEach-Object { Join-Path $_.FullName "bin" } |
        Where-Object { Test-Path (Join-Path $_ "ffmpeg.exe") } |
        Select-Object -First 1
    if (-not $binDir) { throw "Could not locate ffmpeg.exe inside the archive." }

    $targetDir = Join-Path $root "tools\ffmpeg"
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    Copy-Item -Path $binDir -Destination $targetDir -Recurse -Force
    Remove-Item -Recurse -Force $extractDir
    Write-Host "FFmpeg installed into tools\ffmpeg\bin"
}

# --- 5. Verify --------------------------------------------------------------
Write-Step "Verifying environment"
& $venvPython tools\verify_env.py
$verifyExit = $LASTEXITCODE

Write-Host ""
if ($verifyExit -eq 0) {
    Write-Host "Bootstrap complete. Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
} else {
    Write-Host "Bootstrap finished with environment problems (exit $verifyExit). See output above." -ForegroundColor Yellow
}
exit $verifyExit
