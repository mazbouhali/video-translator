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

# Check Docker running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker Desktop and try again."
    exit 1
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

echo "→ Building (this may take a few minutes on first run)..."
docker compose build

echo ""
echo "✅ Installation complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 Installed to: $INSTALL_DIR"
echo ""
echo "To start:"
echo "  • Mac: Double-click start.command"
echo "  • Linux: ./start.command"
echo "  • Or: cd $INSTALL_DIR && docker compose up"
echo ""
echo "Opens at: http://localhost:7860"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
