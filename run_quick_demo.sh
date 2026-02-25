#!/bin/bash
#
# Quick Demo Script for Voyager Evolved
# Run this after installation to test the setup
#

echo "==========================================="
echo "  Voyager Evolved - Quick Demo"
echo "==========================================="
echo ""

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set"
    echo ""
    echo "Please set your API key first:"
    echo "  export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "[✓] Virtual environment activated"
fi

echo ""
echo "Running Voyager Evolved demo..."
echo "This will run 5 learning iterations."
echo ""
echo "Press Ctrl+C to stop at any time."
echo ""

python run_voyager.py --evolved --iterations 5
