#!/bin/bash
# Video Translator - One-line installer for Linux/Mac
# Usage: curl -fsSL https://raw.githubusercontent.com/mazbouhali/video-translator/main/install.sh | bash

set -e

echo "🎬 Video Translator Installer"
echo "=============================="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found!"
    echo ""
    echo "Please install Docker first:"
    echo "  Mac:   https://docs.docker.com/desktop/install/mac-install/"
    echo "  Linux: https://docs.docker.com/engine/install/"
    echo ""
    exit 1
fi

# Check Docker running, start if needed
if ! docker info &> /dev/null; then
    echo "→ Docker not running, starting it..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open -a Docker
        echo "  Waiting for Docker Desktop to start..."
        while ! docker info &> /dev/null; do
            sleep 2
        done
    else
        sudo systemctl start docker
        if ! docker info &> /dev/null; then
            echo "❌ Failed to start Docker. Try: sudo systemctl start docker"
            exit 1
        fi
    fi
    echo "✓ Docker started"
fi

echo "✓ Docker found"

# Clone or update
INSTALL_DIR="$HOME/video-translator"

if [ -d "$INSTALL_DIR" ]; then
    echo "→ Updating existing installation..."
    cd "$INSTALL_DIR"
    git pull
else
    echo "→ Downloading..."
    git clone https://github.com/mazbouhali/video-translator.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Clean up files for other operating systems
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "→ Cleaning up Windows/Linux files..."
    rm -f start.bat stop.bat install.ps1 start.sh stop.sh start-amd.sh VideoTranslator.desktop 2>/dev/null
else
    echo "→ Cleaning up Windows/Mac files..."
    rm -f start.bat stop.bat install.ps1 start.command stop.command 2>/dev/null
fi

# Detect GPU type and build appropriate image
if lspci 2>/dev/null | grep -i "vga\|3d" | grep -iq "amd\|radeon"; then
    echo "→ AMD GPU detected — building ROCm version..."
    docker compose -f docker-compose.yml -f docker-compose.amd.yml build
    # Set AMD as default for start.sh
    echo '#!/bin/bash
cd "$(dirname "$0")"
docker compose -f docker-compose.yml -f docker-compose.amd.yml up -d
sleep 2
xdg-open http://localhost:7860 2>/dev/null || echo "Open http://localhost:7860"
echo "✓ Video Translator running (AMD GPU) at http://localhost:7860"' > start.sh
    chmod +x start.sh
elif lspci 2>/dev/null | grep -i "vga\|3d" | grep -iq "nvidia"; then
    echo "→ NVIDIA GPU detected — building CUDA version..."
    docker compose build
else
    echo "→ No dedicated GPU detected — building CPU version..."
    docker compose build
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 Installed to: $INSTALL_DIR"
echo ""
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "To start: Double-click start.command"
else
    echo "To start:"
    echo "  cd $INSTALL_DIR"
    echo "  ./start.sh"
fi
echo ""
echo "Opens at: http://localhost:7860"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
