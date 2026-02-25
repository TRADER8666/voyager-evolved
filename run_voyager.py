#!/usr/bin/env python3
"""
Voyager Evolved - Quick Start Runner
====================================

This script provides a simple way to start the Voyager Evolved agent.
It handles configuration loading and provides sensible defaults.

Now supports Ollama (default) - no API key required for local LLMs!

Usage:
    python run_voyager.py [--evolved] [--provider ollama|openai]
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
  python run_voyager.py --evolved              # Run with Ollama (default, no API key!)
  python run_voyager.py --evolved --provider openai  # Run with OpenAI
  python run_voyager.py --port 55555           # Specify Minecraft port

LLM Providers:
  ollama (default)  - Free, local LLM. No API key required!
                      Install: https://ollama.ai
                      Pull model: ollama pull llama2
                      Start server: ollama serve
  
  openai            - Cloud LLM. Requires OPENAI_API_KEY environment variable.

Environment Variables:
  OPENAI_API_KEY    Your OpenAI API key (only if using --provider openai)
  MC_PORT           Minecraft server port (default: 25565)
  LLM_PROVIDER      Default LLM provider (ollama or openai)
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
        "--provider",
        type=str,
        choices=["ollama", "openai"],
        default=os.environ.get("LLM_PROVIDER", "ollama"),
        help="LLM provider: 'ollama' (default, free, local) or 'openai' (cloud)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model to use (default: llama2 for Ollama, gpt-4 for OpenAI)"
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
    
    # Validate provider-specific requirements
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if args.provider == "openai":
        if not api_key:
            print("ERROR: OPENAI_API_KEY environment variable is not set.")
            print("")
            print("For OpenAI, please set it with:")
            print("  export OPENAI_API_KEY='your-api-key-here'  # Linux/Mac")
            print("  set OPENAI_API_KEY=your-api-key-here       # Windows")
            print("")
            print("Or use Ollama (free, local) instead:")
            print("  python run_voyager.py --evolved --provider ollama")
            sys.exit(1)
        model = args.model or "gpt-4"
    else:
        # Ollama - check if server is running
        if not check_ollama_available():
            print("WARNING: Ollama server is not running or not accessible.")
            print("")
            print("Please start Ollama:")
            print("  1. Install Ollama: https://ollama.ai")
            print("  2. Pull a model: ollama pull llama2")
            print("  3. Start server: ollama serve")
            print("")
            print("Or use OpenAI instead:")
            print("  python run_voyager.py --evolved --provider openai")
            sys.exit(1)
        model = args.model or "llama2"
    
    mc_port = args.port or int(os.environ.get("MC_PORT", 25565))
    
    # Determine which Voyager to use
    if args.evolved:
        print("=" * 50)
        print("🚀 Starting Voyager Evolved (Enhanced Mode)")
        print("=" * 50)
        print(f"LLM Provider: {args.provider.upper()}")
        print(f"Model: {model}")
        print(f"Minecraft Port: {mc_port}")
        print(f"Iterations: {args.iterations}")
        print("")
        print("Features: Player observation, evolutionary goals, human-like behavior")
        print("-" * 50)
        
        from voyager.evolved import VoyagerEvolved
        
        voyager = VoyagerEvolved(
            llm_provider=args.provider,
            openai_api_key=api_key,
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
        print(f"LLM Provider: {args.provider.upper()}")
        print(f"Model: {model}")
        print(f"Minecraft Port: {mc_port}")
        print("")
        print("Tip: Use --evolved flag for enhanced features!")
        print("-" * 50)
        
        from voyager import Voyager
        
        voyager = Voyager(
            llm_provider=args.provider,
            openai_api_key=api_key,
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
