#!/usr/bin/env python3
"""
Voyager Evolved - Quick Start Runner
====================================

This script provides a simple way to start the Voyager Evolved agent.
It handles configuration loading and provides sensible defaults.

Usage:
    python run_voyager.py [--config CONFIG_FILE] [--evolved]
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run Voyager Evolved - An enhanced Minecraft AI agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_voyager.py                    # Run with default settings
  python run_voyager.py --evolved          # Run enhanced Voyager Evolved
  python run_voyager.py --config my.yaml   # Run with custom config

Environment Variables:
  OPENAI_API_KEY    Your OpenAI API key (required)
  MC_PORT           Minecraft server port (default: 25565)
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
        default="gpt-4",
        help="OpenAI model to use (default: gpt-4)"
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
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("")
        print("Please set it with:")
        print("  export OPENAI_API_KEY='your-api-key-here'  # Linux/Mac")
        print("  set OPENAI_API_KEY=your-api-key-here       # Windows")
        sys.exit(1)
    
    # Determine which Voyager to use
    if args.evolved:
        print("Starting Voyager Evolved (Enhanced Mode)...")
        print("Features: Player observation, evolutionary goals, human-like behavior")
        print("")
        
        from voyager.evolved import VoyagerEvolved, EvolvedConfig
        
        config = EvolvedConfig(
            openai_api_key=api_key,
            model_name=args.model,
            mc_port=args.port or int(os.environ.get("MC_PORT", 25565)),
            resume=args.resume,
        )
        
        voyager = VoyagerEvolved(config)
    else:
        print("Starting Voyager (Standard Mode)...")
        print("Use --evolved flag for enhanced features.")
        print("")
        
        from voyager import Voyager
        
        voyager = Voyager(
            openai_api_key=api_key,
            mc_port=args.port or int(os.environ.get("MC_PORT", 25565)),
        )
    
    print(f"Model: {args.model}")
    print(f"Iterations: {args.iterations}")
    print("-" * 40)
    
    try:
        # Run the learning loop
        voyager.learn(max_iterations=args.iterations)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving progress...")
    finally:
        print("\nVoyager session ended.")


if __name__ == "__main__":
    main()
