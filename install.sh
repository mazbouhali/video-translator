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
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  https://docs.docker.com/desktop/install/mac-install/"
    else
        echo "  https://docs.docker.com/engine/install/"
    fi
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
        sleep 2
        if ! docker info &> /dev/null; then
            echo "❌ Failed to start Docker."
            exit 1
        fi
    fi
    echo "✓ Docker started"
fi

echo "✓ Docker running"

# Clone or update
INSTALL_DIR="$HOME/video-translator"

if [ -d "$INSTALL_DIR" ]; then
    echo "→ Updating existing installation..."
    cd "$INSTALL_DIR"
    git pull --quiet
else
    echo "→ Downloading..."
    git clone --quiet https://github.com/mazbouhali/video-translator.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Detect GPU
GPU_TYPE="cpu"
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac - check for Apple Silicon
    if sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -q "Apple"; then
        GPU_TYPE="apple"
    fi
else
    # Linux - check lspci
    if command -v lspci &> /dev/null; then
        if lspci | grep -iE "vga|3d|display" | grep -iq "amd\|radeon"; then
            GPU_TYPE="amd"
        elif lspci | grep -iE "vga|3d|display" | grep -iq "nvidia"; then
            GPU_TYPE="nvidia"
        fi
    fi
fi

echo "→ Detected: $GPU_TYPE"

# Build appropriate image
case $GPU_TYPE in
    amd)
        echo "→ Building for AMD GPU (ROCm)..."
        docker compose -f docker-compose.yml -f docker-compose.amd.yml build
        COMPOSE_CMD="docker compose -f docker-compose.yml -f docker-compose.amd.yml"
        ;;
    nvidia)
        echo "→ Building for NVIDIA GPU (CUDA)..."
        docker compose -f docker-compose.yml -f docker-compose.nvidia.yml build
        COMPOSE_CMD="docker compose -f docker-compose.yml -f docker-compose.nvidia.yml"
        ;;
    apple)
        echo "→ Building for Apple Silicon (MPS)..."
        docker compose build
        COMPOSE_CMD="docker compose"
        ;;
    *)
        echo "→ Building for CPU..."
        docker compose build
        COMPOSE_CMD="docker compose"
        ;;
esac

# Create start script for this OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac - keep start.command, remove others
    rm -f start.sh stop.sh start-amd.sh start.bat stop.bat install.ps1 VideoTranslator.desktop 2>/dev/null
else
    # Linux - create proper start.sh
    rm -f start.command stop.command start.bat stop.bat install.ps1 2>/dev/null
    
    cat > start.sh << EOF
#!/bin/bash
cd "\$(dirname "\$0")"
$COMPOSE_CMD up -d
sleep 3
echo "✓ Video Translator running at http://localhost:7860"
xdg-open http://localhost:7860 2>/dev/null || echo "Open http://localhost:7860 in your browser"
EOF
    chmod +x start.sh
    
    cat > stop.sh << EOF
#!/bin/bash
cd "\$(dirname "\$0")"
$COMPOSE_CMD down
echo "✓ Stopped"
EOF
    chmod +x stop.sh

    # Create .desktop file for app menu
    cat > "$HOME/.local/share/applications/video-translator.desktop" << EOF
[Desktop Entry]
Name=Video Translator
Comment=Translate Arabic videos to English
Exec=bash -c 'cd $INSTALL_DIR && ./start.sh'
Icon=video-x-generic
Terminal=true
Type=Application
Categories=AudioVideo;Video;
EOF
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 Installed to: $INSTALL_DIR"
echo ""
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "To start: Double-click start.command in Finder"
else
    echo "To start:"
    echo "  cd $INSTALL_DIR && ./start.sh"
    echo ""
    echo "Or find 'Video Translator' in your app menu"
fi
echo ""
echo "Opens at: http://localhost:7860"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
