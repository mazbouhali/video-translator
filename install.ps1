# Video Translator - One-line installer for Windows
# Usage: irm https://raw.githubusercontent.com/mazbouhali/video-translator/main/install.ps1 | iex

$ErrorActionPreference = "Stop"

Write-Host "🎬 Video Translator Installer" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan

# Check Docker
try {
    $null = docker --version
} catch {
    Write-Host "❌ Docker not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Docker Desktop first:"
    Write-Host "  https://docs.docker.com/desktop/install/windows-install/"
    Write-Host ""
    exit 1
}

# Check Docker running
try {
    $null = docker info 2>$null
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop and try again." -ForegroundColor Red
    exit 1
}

Write-Host "✓ Docker found" -ForegroundColor Green

# Clone or update
$InstallDir = "$env:USERPROFILE\video-translator"

if (Test-Path $InstallDir) {
    Write-Host "→ Updating existing installation..."
    Set-Location $InstallDir
    git pull
} else {
    Write-Host "→ Downloading..."
    git clone https://github.com/mazbouhali/video-translator.git $InstallDir
    Set-Location $InstallDir
}

Write-Host "→ Building (this may take a few minutes on first run)..."
docker compose build

Write-Host ""
Write-Host "✅ Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host "📁 Installed to: $InstallDir"
Write-Host ""
Write-Host "To start:"
Write-Host "  • Double-click start.bat"
Write-Host "  • Or: cd $InstallDir; docker compose up"
Write-Host ""
Write-Host "Opens at: http://localhost:7860"
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
