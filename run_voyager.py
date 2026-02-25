#!/usr/bin/env python3
"""
Voyager Evolved - Quick Start Runner
====================================

This script provides a simple way to start the Voyager Evolved agent.
It handles configuration loading and provides sensible defaults.

Uses Ollama (local LLM) - no API key required!

Usage:
    python run_voyager.py [--evolved] [--model MODEL]
"""

import argparse
import os
import sys
from pathlib import Path


def check_ollama_available():
    """Check if Ollama server is running."""
    import urllib.request
    import urllib.error
    
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run Voyager Evolved - An enhanced Minecraft AI agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_voyager.py --evolved              # Run with default model (llama2)
  python run_voyager.py --evolved --model mistral  # Run with mistral model
  python run_voyager.py --port 55555           # Specify Minecraft port

Ollama Setup:
  1. Install: https://ollama.ai
  2. Pull model: ollama pull llama2
  3. Start server: ollama serve

Environment Variables:
  MC_PORT           Minecraft server port (default: 25565)
  OLLAMA_BASE_URL   Ollama server URL (default: http://localhost:11434)
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--evolved", "-e",
        action="store_true",
        help="Use Voyager Evolved with enhanced features"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="llama2",
        help="Ollama model to use (default: llama2)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Minecraft server port"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=10,
        help="Number of iterations to run (default: 10)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    
    args = parser.parse_args()
    
    # Check if Ollama server is running
    if not check_ollama_available():
        print("ERROR: Ollama server is not running or not accessible.")
        print("")
        print("Please start Ollama:")
        print("  1. Install Ollama: https://ollama.ai")
        print("  2. Pull a model: ollama pull llama2")
        print("  3. Start server: ollama serve")
        sys.exit(1)
    
    mc_port = args.port or int(os.environ.get("MC_PORT", 25565))
    model = args.model
    
    # Determine which Voyager to use
    if args.evolved:
        print("=" * 50)
        print("🚀 Starting Voyager Evolved (Enhanced Mode)")
        print("=" * 50)
        print(f"LLM: Ollama")
        print(f"Model: {model}")
        print(f"Minecraft Port: {mc_port}")
        print(f"Iterations: {args.iterations}")
        print("")
        print("Features: Player observation, evolutionary goals, human-like behavior")
        print("-" * 50)
        
        from voyager.evolved import VoyagerEvolved
        
        voyager = VoyagerEvolved(
            action_agent_model_name=model,
            curriculum_agent_model_name=model,
            critic_agent_model_name=model,
            mc_port=mc_port,
            resume=args.resume,
        )
    else:
        print("=" * 50)
        print("🎮 Starting Voyager (Standard Mode)")
        print("=" * 50)
        print(f"LLM: Ollama")
        print(f"Model: {model}")
        print(f"Minecraft Port: {mc_port}")
        print("")
        print("Tip: Use --evolved flag for enhanced features!")
        print("-" * 50)
        
        from voyager import Voyager
        
        voyager = Voyager(
            action_agent_model_name=model,
            curriculum_agent_model_name=model,
            critic_agent_model_name=model,
            mc_port=mc_port,
        )
    
    try:
        # Run the learning loop
        voyager.learn(max_iterations=args.iterations)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving progress...")
    finally:
        print("\nVoyager session ended.")


if __name__ == "__main__":
    main()
