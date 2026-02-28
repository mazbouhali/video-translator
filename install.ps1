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
Write-Host "  cd $InstallDir"
Write-Host "  docker compose up"
Write-Host ""
Write-Host "Then open: http://localhost:7860"
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host ""

$response = Read-Host "Start now? [Y/n]"
if ($response -ne "n" -and $response -ne "N") {
    Write-Host "→ Starting web UI..."
    docker compose up -d
    Start-Sleep -Seconds 3
    
    # Open browser
    Start-Process "http://localhost:7860"
    
    Write-Host ""
    Write-Host "🎬 Video Translator is running!" -ForegroundColor Green
    Write-Host "   Open: http://localhost:7860"
    Write-Host "   Stop: cd $InstallDir; docker compose down"
}
