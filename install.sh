#!/bin/bash
# Video Translator - One-line installer for Linux/Mac
# Usage: curl -fsSL https://raw.githubusercontent.com/mazbouhali/video-translator/main/install.sh | bash
#
# Works out of the box for:
# - NVIDIA GPUs (CUDA)
# - AMD GPUs (ROCm) - auto-installs required packages
# - Apple Silicon (MPS)
# - CPU fallback

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()    { echo -e "${GREEN}✓${NC} $1"; }
warn()   { echo -e "${YELLOW}→${NC} $1"; }
error()  { echo -e "${RED}✗${NC} $1"; }
info()   { echo -e "${BLUE}ℹ${NC} $1"; }

echo ""
echo -e "${BLUE}🎬 Video Translator Installer${NC}"
echo "=============================="
echo ""

INSTALL_DIR="${VIDEO_TRANSLATOR_DIR:-$HOME/video-translator}"

# =============================================================================
# Docker Check
# =============================================================================
install_docker() {
    warn "Docker not found. Installing..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        error "Please install Docker Desktop manually:"
        echo "  https://docs.docker.com/desktop/install/mac-install/"
        exit 1
    fi
    
    # Linux - detect distro and install Docker
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        case "$ID" in
            ubuntu|debian|pop|linuxmint|elementary)
                sudo apt-get update
                sudo apt-get install -y docker.io docker-compose-plugin
                sudo systemctl enable docker
                sudo systemctl start docker
                sudo usermod -aG docker "$USER"
                ;;
            arch|manjaro|endeavouros)
                sudo pacman -S --noconfirm docker docker-compose
                sudo systemctl enable docker
                sudo systemctl start docker
                sudo usermod -aG docker "$USER"
                ;;
            fedora)
                sudo dnf install -y docker docker-compose
                sudo systemctl enable docker
                sudo systemctl start docker
                sudo usermod -aG docker "$USER"
                ;;
            *)
                error "Unsupported distro: $ID"
                echo "Please install Docker manually: https://docs.docker.com/engine/install/"
                exit 1
                ;;
        esac
        
        log "Docker installed"
        warn "You may need to log out and back in for group changes to take effect."
        warn "Then run this installer again."
        
        # Try newgrp to avoid logout
        if groups | grep -q docker; then
            :
        else
            exec sg docker "$0"
        fi
    else
        error "Cannot detect Linux distribution"
        exit 1
    fi
}

if ! command -v docker &> /dev/null; then
    install_docker
fi

# Check Docker daemon running
if ! docker info &> /dev/null; then
    warn "Docker not running, starting it..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open -a Docker
        echo "  Waiting for Docker Desktop to start..."
        for i in {1..30}; do
            sleep 2
            if docker info &> /dev/null; then break; fi
        done
    else
        sudo systemctl start docker
        sleep 2
    fi
    
    if ! docker info &> /dev/null; then
        error "Failed to start Docker"
        exit 1
    fi
fi

log "Docker running"

# =============================================================================
# GPU Detection
# =============================================================================
detect_gpu() {
    GPU_TYPE="cpu"
    GPU_NAME=""
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -q "Apple"; then
            GPU_TYPE="apple"
            GPU_NAME="Apple Silicon"
        fi
    else
        # Linux GPU detection
        if command -v lspci &> /dev/null; then
            GPU_LINE=$(lspci | grep -iE "vga|3d|display" | head -1)
            
            if echo "$GPU_LINE" | grep -iq "nvidia"; then
                GPU_TYPE="nvidia"
                GPU_NAME=$(echo "$GPU_LINE" | sed 's/.*: //')
            elif echo "$GPU_LINE" | grep -iq "amd\|radeon"; then
                GPU_TYPE="amd"
                GPU_NAME=$(echo "$GPU_LINE" | sed 's/.*: //')
                
                # Detect RDNA generation for gfx version
                if echo "$GPU_NAME" | grep -iqE "9070|9080|9090"; then
                    GPU_GEN="rdna4"
                    GFX_VERSION="12.0.0"
                elif echo "$GPU_NAME" | grep -iqE "7900|7800|7700|7600"; then
                    GPU_GEN="rdna3"
                    GFX_VERSION="11.0.0"
                elif echo "$GPU_NAME" | grep -iqE "6900|6800|6700|6600|6500"; then
                    GPU_GEN="rdna2"
                    GFX_VERSION="10.3.0"
                else
                    GPU_GEN="unknown"
                    GFX_VERSION=""
                fi
            fi
        fi
    fi
    
    echo "GPU_TYPE='$GPU_TYPE'"
    echo "GPU_NAME='$GPU_NAME'"
    echo "GPU_GEN='${GPU_GEN:-}'"
    echo "GFX_VERSION='${GFX_VERSION:-}'"
}

eval "$(detect_gpu)"

if [ -n "$GPU_NAME" ]; then
    log "Detected GPU: $GPU_NAME"
else
    warn "No dedicated GPU detected, using CPU"
fi

# =============================================================================
# AMD ROCm Setup (Linux only)
# =============================================================================
setup_amd_rocm() {
    info "Setting up AMD ROCm support..."
    
    # Check kernel version for RDNA4
    KERNEL_VERSION=$(uname -r | cut -d. -f1-2)
    KERNEL_MAJOR=$(echo "$KERNEL_VERSION" | cut -d. -f1)
    KERNEL_MINOR=$(echo "$KERNEL_VERSION" | cut -d. -f2)
    
    if [ "$GPU_GEN" = "rdna4" ]; then
        # RDNA4 needs kernel 6.12+ for proper support
        if [ "$KERNEL_MAJOR" -lt 6 ] || ([ "$KERNEL_MAJOR" -eq 6 ] && [ "$KERNEL_MINOR" -lt 8 ]); then
            warn "RDNA4 GPUs work best with kernel 6.8+"
            warn "Your kernel: $KERNEL_VERSION"
            warn "Continuing anyway (may need HSA_OVERRIDE_GFX_VERSION)"
        fi
    fi
    
    # Check if /dev/kfd exists (ROCm kernel support)
    if [ ! -e /dev/kfd ]; then
        warn "/dev/kfd not found - loading amdgpu kernel module..."
        sudo modprobe amdgpu
        sleep 1
        
        if [ ! -e /dev/kfd ]; then
            error "/dev/kfd still not available"
            error "Your kernel may not have ROCm support compiled in."
            echo ""
            echo "Options:"
            echo "  1. Install a newer kernel (6.8+ recommended)"
            echo "  2. Install ROCm DKMS: https://rocm.docs.amd.com/projects/install-on-linux/"
            echo "  3. Continue with CPU mode"
            echo ""
            read -p "Continue with CPU mode? [Y/n] " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Nn]$ ]]; then
                exit 1
            fi
            GPU_TYPE="cpu"
            return
        fi
    fi
    
    log "/dev/kfd available"
    
    # Check and fix group membership
    NEEDS_GROUP=false
    
    if [ -e /dev/kfd ]; then
        KFD_GROUP=$(stat -c '%G' /dev/kfd 2>/dev/null || stat -f '%Sg' /dev/kfd 2>/dev/null)
        if ! groups | grep -qw "$KFD_GROUP"; then
            warn "Adding user to '$KFD_GROUP' group for GPU access..."
            sudo usermod -aG "$KFD_GROUP" "$USER"
            NEEDS_GROUP=true
        fi
    fi
    
    # Also add to video and render groups if they exist
    for group in video render; do
        if getent group "$group" &>/dev/null; then
            if ! groups | grep -qw "$group"; then
                sudo usermod -aG "$group" "$USER"
                NEEDS_GROUP=true
            fi
        fi
    done
    
    if [ "$NEEDS_GROUP" = true ]; then
        log "Added user to required groups"
        warn "Group changes require logout/login to take effect."
        warn "If GPU doesn't work, log out and back in, then run ./start.sh"
    fi
    
    # Test ROCm access
    if docker run --rm --device=/dev/kfd --device=/dev/dri rocm/pytorch:latest rocm-smi &>/dev/null; then
        log "ROCm Docker test passed"
    else
        warn "ROCm Docker test failed - may need group changes to take effect"
    fi
}

if [ "$GPU_TYPE" = "amd" ]; then
    setup_amd_rocm
fi

# =============================================================================
# NVIDIA Container Toolkit Setup (Linux only)
# =============================================================================
setup_nvidia() {
    info "Setting up NVIDIA GPU support..."
    
    # Check for nvidia-smi
    if ! command -v nvidia-smi &>/dev/null; then
        warn "NVIDIA driver not found"
        echo "Please install NVIDIA drivers first, then run this installer again."
        GPU_TYPE="cpu"
        return
    fi
    
    # Check for nvidia-container-toolkit
    if ! docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi &>/dev/null; then
        warn "Installing NVIDIA Container Toolkit..."
        
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            case "$ID" in
                ubuntu|debian|pop|linuxmint)
                    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
                    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
                        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
                    sudo apt-get update
                    sudo apt-get install -y nvidia-container-toolkit
                    sudo nvidia-ctk runtime configure --runtime=docker
                    sudo systemctl restart docker
                    ;;
                arch|manjaro|endeavouros)
                    sudo pacman -S --noconfirm nvidia-container-toolkit
                    sudo nvidia-ctk runtime configure --runtime=docker
                    sudo systemctl restart docker
                    ;;
                fedora)
                    curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
                        sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
                    sudo dnf install -y nvidia-container-toolkit
                    sudo nvidia-ctk runtime configure --runtime=docker
                    sudo systemctl restart docker
                    ;;
                *)
                    error "Please install nvidia-container-toolkit manually"
                    echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
                    GPU_TYPE="cpu"
                    return
                    ;;
            esac
            
            log "NVIDIA Container Toolkit installed"
        fi
    else
        log "NVIDIA Container Toolkit working"
    fi
}

if [ "$GPU_TYPE" = "nvidia" ]; then
    setup_nvidia
fi

# =============================================================================
# Clone/Update Repository
# =============================================================================
if [ -d "$INSTALL_DIR" ]; then
    warn "Updating existing installation..."
    cd "$INSTALL_DIR"
    git pull --quiet
else
    warn "Downloading..."
    git clone --quiet https://github.com/mazbouhali/video-translator.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

log "Repository ready"

# =============================================================================
# Build Docker Image
# =============================================================================
case $GPU_TYPE in
    amd)
        info "Building for AMD GPU (ROCm)..."
        
        # Update docker-compose.amd.yml with correct gfx version
        if [ -n "$GFX_VERSION" ]; then
            sed -i "s/HSA_OVERRIDE_GFX_VERSION=.*/HSA_OVERRIDE_GFX_VERSION=$GFX_VERSION/" docker-compose.amd.yml 2>/dev/null || true
        fi
        
        docker compose -f docker-compose.yml -f docker-compose.amd.yml build
        COMPOSE_CMD="docker compose -f docker-compose.yml -f docker-compose.amd.yml"
        ;;
    nvidia)
        info "Building for NVIDIA GPU (CUDA)..."
        docker compose -f docker-compose.yml -f docker-compose.nvidia.yml build
        COMPOSE_CMD="docker compose -f docker-compose.yml -f docker-compose.nvidia.yml"
        ;;
    apple)
        info "Building for Apple Silicon (MPS)..."
        docker compose build
        COMPOSE_CMD="docker compose"
        ;;
    *)
        info "Building for CPU..."
        docker compose -f docker-compose.yml -f docker-compose.cpu.yml build
        COMPOSE_CMD="docker compose -f docker-compose.yml -f docker-compose.cpu.yml"
        ;;
esac

log "Docker image built"

# =============================================================================
# Create Platform-Specific Launchers
# =============================================================================
mkdir -p input output

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - create .command files
    cat > start.command << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
docker compose up -d
sleep 3
open http://localhost:7860
echo "✓ Video Translator running at http://localhost:7860"
echo "Press Enter to close this window..."
read
EOF
    chmod +x start.command
    
    cat > stop.command << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
docker compose down
echo "✓ Stopped"
EOF
    chmod +x stop.command
    
    # Remove Linux/Windows files
    rm -f start.sh stop.sh start-amd.sh start.bat stop.bat install.ps1 2>/dev/null
    
else
    # Linux - create shell scripts
    cat > start.sh << EOF
#!/bin/bash
cd "\$(dirname "\$0")"
$COMPOSE_CMD up -d
sleep 3
echo "✓ Video Translator running at http://localhost:7860"
xdg-open http://localhost:7860 2>/dev/null || echo "Open http://localhost:7860 in your browser"
EOF
    chmod +x start.sh
    
    cat > stop.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
docker compose down 2>/dev/null
docker compose -f docker-compose.yml -f docker-compose.amd.yml down 2>/dev/null
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml down 2>/dev/null
echo "✓ Stopped"
EOF
    chmod +x stop.sh
    
    # Create .desktop file for app menu
    mkdir -p "$HOME/.local/share/applications"
    cat > "$HOME/.local/share/applications/video-translator.desktop" << EOF
[Desktop Entry]
Name=Video Translator
Comment=Translate foreign videos to English subtitles
Exec=bash -c 'cd $INSTALL_DIR && ./start.sh'
Icon=video-x-generic
Terminal=true
Type=Application
Categories=AudioVideo;Video;
EOF
    
    # Remove Mac/Windows files
    rm -f start.command stop.command start.bat stop.bat install.ps1 2>/dev/null
fi

# Save GPU config for reference
cat > .gpu-config << EOF
GPU_TYPE=$GPU_TYPE
GPU_NAME=$GPU_NAME
GPU_GEN=${GPU_GEN:-}
GFX_VERSION=${GFX_VERSION:-}
COMPOSE_CMD=$COMPOSE_CMD
INSTALLED=$(date -Iseconds)
EOF

log "Installation complete!"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "📁 Installed to: ${BLUE}$INSTALL_DIR${NC}"
echo ""
echo "🖥️  GPU: $GPU_TYPE"
[ -n "$GPU_NAME" ] && echo "   ($GPU_NAME)"
echo ""
echo "▶️  To start:"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "   Double-click start.command in Finder"
    echo "   Or: cd $INSTALL_DIR && ./start.command"
else
    echo "   cd $INSTALL_DIR && ./start.sh"
    echo "   Or find 'Video Translator' in your app menu"
fi
echo ""
echo "🌐 Opens at: http://localhost:7860"
echo ""
echo "📥 Drop videos in: $INSTALL_DIR/input/"
echo "📤 Find results in: $INSTALL_DIR/output/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# First run hint
if [ ! -f .first-run-done ]; then
    echo ""
    info "First run will download AI models (~5GB) - this takes a few minutes"
    touch .first-run-done
fi
