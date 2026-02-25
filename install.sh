#!/bin/bash
#
# Voyager Evolved - Linux Installation Script
# =====================================================
# 
# This project is Linux-only for optimal performance.
# Supports: Ubuntu/Debian, Fedora/RHEL, Arch Linux
#

set -e  # Exit on error

echo "==========================================="
echo "  Voyager Evolved Installation Script"
echo "           (Linux Only)"
echo "==========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions for colored output
print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_info() { echo -e "${BLUE}[i]${NC} $1"; }

# Detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        DISTRO_VERSION=$VERSION_ID
    elif [ -f /etc/debian_version ]; then
        DISTRO="debian"
    elif [ -f /etc/fedora-release ]; then
        DISTRO="fedora"
    elif [ -f /etc/arch-release ]; then
        DISTRO="arch"
    else
        DISTRO="unknown"
    fi
    echo $DISTRO
}

# Install system dependencies based on distro
install_system_deps() {
    local distro=$(detect_distro)
    print_info "Detected distribution: $distro"
    
    case $distro in
        ubuntu|debian|linuxmint|pop)
            print_info "Installing dependencies via apt..."
            sudo apt-get update
            sudo apt-get install -y \
                python3 python3-pip python3-venv \
                nodejs npm \
                git curl wget \
                build-essential \
                libffi-dev libssl-dev \
                xvfb libxcomposite1 libxdamage1 libxrandr2 \
                libasound2 libatk1.0-0 libcups2 libpango-1.0-0
            ;;
        fedora|rhel|centos|rocky|almalinux)
            print_info "Installing dependencies via dnf..."
            sudo dnf install -y \
                python3 python3-pip python3-virtualenv \
                nodejs npm \
                git curl wget \
                gcc gcc-c++ make \
                libffi-devel openssl-devel \
                xorg-x11-server-Xvfb
            ;;
        arch|manjaro|endeavouros)
            print_info "Installing dependencies via pacman..."
            sudo pacman -Syu --noconfirm \
                python python-pip python-virtualenv \
                nodejs npm \
                git curl wget \
                base-devel \
                xorg-server-xvfb
            ;;
        *)
            print_warning "Unknown distribution. Please install manually:"
            echo "  - Python 3.9+"
            echo "  - Node.js 18+"
            echo "  - npm"
            echo "  - git, curl, wget"
            echo "  - Build tools (gcc, make)"
            ;;
    esac
}

# Check Python version
check_python() {
    echo ""
    print_info "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 9 ]; then
            print_status "Python $PYTHON_VERSION found"
            return 0
        else
            print_error "Python 3.9+ required (found $PYTHON_VERSION)"
            return 1
        fi
    else
        print_error "Python 3 not found"
        return 1
    fi
}

# Check Node.js version
check_nodejs() {
    print_info "Checking Node.js version..."
    
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node -v | cut -d'v' -f2)
        NODE_MAJOR=$(echo $NODE_VERSION | cut -d. -f1)
        
        if [ "$NODE_MAJOR" -ge 16 ]; then
            print_status "Node.js $NODE_VERSION found"
            return 0
        else
            print_warning "Node.js 16+ recommended (found $NODE_VERSION)"
        fi
    else
        print_warning "Node.js not found"
        return 1
    fi
}

# Install Node.js via nvm if needed
install_nodejs() {
    print_info "Installing Node.js via nvm..."
    
    # Install nvm
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
    
    # Load nvm
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    
    # Install Node.js LTS
    nvm install --lts
    nvm use --lts
    
    print_status "Node.js installed via nvm"
}

# Setup virtual environment
setup_venv() {
    echo ""
    read -p "Create a Python virtual environment? (recommended) [Y/n]: " CREATE_VENV
    CREATE_VENV=${CREATE_VENV:-Y}
    
    if [[ $CREATE_VENV =~ ^[Yy]$ ]]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        print_status "Virtual environment created and activated"
        echo ""
        print_warning "Remember to activate with: source venv/bin/activate"
    fi
}

# Install Python packages
install_python_packages() {
    echo ""
    print_info "Upgrading pip..."
    python3 -m pip install --upgrade pip wheel setuptools
    print_status "pip upgraded"
    
    print_info "Installing Voyager Evolved..."
    pip install -e .
    print_status "Voyager Evolved installed"
    
    # Install optional performance packages
    print_info "Installing optional performance packages..."
    pip install psutil numpy scikit-learn || print_warning "Some optional packages failed to install"
}

# Install Mineflayer
install_mineflayer() {
    echo ""
    print_info "Installing Mineflayer (Minecraft bot framework)..."
    
    if [ -d "voyager/env/mineflayer" ]; then
        cd voyager/env/mineflayer
        npm install
        cd ../../..
        print_status "Mineflayer installed"
    else
        print_warning "Mineflayer directory not found, skipping npm install"
    fi
}

# Setup Ollama
setup_ollama() {
    echo ""
    print_info "Checking Ollama installation..."
    
    if command -v ollama &> /dev/null; then
        print_status "Ollama is already installed"
    else
        read -p "Install Ollama? (required for LLM) [Y/n]: " INSTALL_OLLAMA
        INSTALL_OLLAMA=${INSTALL_OLLAMA:-Y}
        
        if [[ $INSTALL_OLLAMA =~ ^[Yy]$ ]]; then
            print_info "Installing Ollama..."
            curl -fsSL https://ollama.ai/install.sh | sh
            print_status "Ollama installed"
        fi
    fi
    
    # Pull recommended models
    if command -v ollama &> /dev/null; then
        read -p "Pull recommended Ollama models? (llama2, nomic-embed-text) [Y/n]: " PULL_MODELS
        PULL_MODELS=${PULL_MODELS:-Y}
        
        if [[ $PULL_MODELS =~ ^[Yy]$ ]]; then
            print_info "Pulling llama2 model..."
            ollama pull llama2 || print_warning "Failed to pull llama2"
            
            print_info "Pulling nomic-embed-text model..."
            ollama pull nomic-embed-text || print_warning "Failed to pull nomic-embed-text"
        fi
    fi
}

# Create config
setup_config() {
    echo ""
    mkdir -p configs
    
    if [ ! -f "configs/config.yaml" ]; then
        print_info "Creating default configuration..."
        cp configs/config.example.yaml configs/config.yaml 2>/dev/null || true
        print_status "Config file created at configs/config.yaml"
    else
        print_info "Config file already exists"
    fi
}

# Apply Linux optimizations
apply_linux_optimizations() {
    echo ""
    print_info "Applying Linux optimizations..."
    
    # Set environment variables
    export MALLOC_ARENA_MAX=2
    
    # Add to bashrc for persistence
    if ! grep -q "MALLOC_ARENA_MAX" ~/.bashrc 2>/dev/null; then
        echo "export MALLOC_ARENA_MAX=2" >> ~/.bashrc
    fi
    
    print_status "Linux optimizations applied"
}

# Main installation
main() {
    # Check if running on Linux
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        print_error "This project is Linux-only!"
        print_info "For optimal performance, please run on a Linux system."
        exit 1
    fi
    
    # Ask about system dependencies
    read -p "Install system dependencies? (requires sudo) [y/N]: " INSTALL_DEPS
    if [[ $INSTALL_DEPS =~ ^[Yy]$ ]]; then
        install_system_deps
    fi
    
    # Check Python
    if ! check_python; then
        print_error "Please install Python 3.9+ and try again"
        exit 1
    fi
    
    # Check Node.js
    if ! check_nodejs; then
        read -p "Install Node.js via nvm? [Y/n]: " INSTALL_NODE
        INSTALL_NODE=${INSTALL_NODE:-Y}
        if [[ $INSTALL_NODE =~ ^[Yy]$ ]]; then
            install_nodejs
        fi
    fi
    
    # Setup virtual environment
    setup_venv
    
    # Install Python packages
    install_python_packages
    
    # Install Mineflayer
    install_mineflayer
    
    # Setup Ollama
    setup_ollama
    
    # Setup config
    setup_config
    
    # Apply optimizations
    apply_linux_optimizations
    
    # Done!
    echo ""
    echo "==========================================="
    print_status "Installation complete!"
    echo "==========================================="
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Start Ollama server:"
    echo "     ollama serve"
    echo ""
    echo "  2. Edit configuration:"
    echo "     nano configs/config.yaml"
    echo ""
    echo "  3. Run Voyager Evolved:"
    echo "     python run_voyager.py"
    echo ""
    echo "  4. For performance tuning, see:"
    echo "     docs/PERFORMANCE_TUNING.md"
    echo ""
    echo "For more info, see README.md"
}

# Run main
main "$@"
