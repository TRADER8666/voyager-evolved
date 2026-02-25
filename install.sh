#!/bin/bash
#
# Voyager Evolved - Installation Script for Linux/macOS
# =====================================================
#
# This script will set up Voyager Evolved and all its dependencies.
#

set -e  # Exit on error

echo "==========================================="
echo "  Voyager Evolved Installation Script"
echo "==========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 9 ]; then
        print_status "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.9 or higher is required (found $PYTHON_VERSION)"
        exit 1
    fi
else
    print_error "Python 3 is not installed"
    exit 1
fi

# Check Node.js version
echo "Checking Node.js version..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node -v | cut -d'v' -f2)
    print_status "Node.js $NODE_VERSION found"
else
    print_warning "Node.js is not installed. Installing..."
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
        sudo apt-get install -y nodejs
    elif command -v brew &> /dev/null; then
        # macOS with Homebrew
        brew install node
    else
        print_error "Please install Node.js manually from https://nodejs.org/"
        exit 1
    fi
fi

# Create virtual environment (optional)
echo ""
read -p "Create a virtual environment? (recommended) [Y/n]: " CREATE_VENV
CREATE_VENV=${CREATE_VENV:-Y}

if [[ $CREATE_VENV =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    print_status "Virtual environment created and activated"
    echo ""
    print_warning "Remember to activate the venv with: source venv/bin/activate"
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
python3 -m pip install --upgrade pip
print_status "pip upgraded"

# Install Python package
echo ""
echo "Installing Voyager Evolved..."
pip install -e .
print_status "Voyager Evolved installed"

# Install Mineflayer dependencies
echo ""
echo "Installing Mineflayer (Minecraft bot framework)..."
if [ -d "voyager/env/mineflayer" ]; then
    cd voyager/env/mineflayer
    npm install
    cd ../../..
    print_status "Mineflayer installed"
else
    print_warning "Mineflayer directory not found, skipping npm install"
fi

# Create config directory
mkdir -p configs

# Check for config
if [ ! -f "configs/config.yaml" ]; then
    print_warning "No config file found. Creating example config..."
    cp configs/config.example.yaml configs/config.yaml 2>/dev/null || true
fi

echo ""
echo "==========================================="
print_status "Installation complete!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "  1. Set your OpenAI API key:"
echo "     export OPENAI_API_KEY='your-api-key-here'"
echo ""
echo "  2. Edit the config file:"
echo "     nano configs/config.yaml"
echo ""
echo "  3. Run Voyager Evolved:"
echo "     python run_voyager.py"
echo ""
echo "For more info, see README.md"
