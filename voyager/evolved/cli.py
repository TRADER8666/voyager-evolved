#!/usr/bin/env python3
"""
Voyager Evolved CLI - Command Line Interface
============================================

This module provides the command-line interface for Voyager Evolved.
"""

import argparse
import os
import sys


def main():
    """Main entry point for the Voyager Evolved CLI."""
    parser = argparse.ArgumentParser(
        prog="voyager-evolved",
        description="Voyager Evolved - An enhanced Minecraft AI agent with human-like behaviors",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the Voyager agent")
    run_parser.add_argument(
        "--model", "-m",
        default="gpt-4",
        help="OpenAI model to use"
    )
    run_parser.add_argument(
        "--port", "-p",
        type=int,
        default=25565,
        help="Minecraft server port"
    )
    run_parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=10,
        help="Number of learning iterations"
    )
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration utilities")
    config_parser.add_argument(
        "--show", action="store_true",
        help="Show current configuration"
    )
    config_parser.add_argument(
        "--generate", action="store_true",
        help="Generate example configuration"
    )
    
    # Version command
    subparsers.add_parser("version", help="Show version information")
    
    args = parser.parse_args()
    
    if args.command == "run":
        _run_voyager(args)
    elif args.command == "config":
        _handle_config(args)
    elif args.command == "version":
        print("Voyager Evolved v1.0.0")
    else:
        parser.print_help()


def _run_voyager(args):
    """Run the Voyager agent."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    from .voyager_evolved import VoyagerEvolved
    from .config import EvolvedConfig
    
    config = EvolvedConfig(
        openai_api_key=api_key,
        model_name=args.model,
        mc_port=args.port,
    )
    
    voyager = VoyagerEvolved(config)
    voyager.learn(max_iterations=args.iterations)


def _handle_config(args):
    """Handle configuration commands."""
    if args.generate:
        print("""# Voyager Evolved Configuration
# ==============================

openai:
  api_key: "${OPENAI_API_KEY}"  # Set via environment variable
  model: "gpt-4"

minecraft:
  port: 25565
  host: "localhost"

evolved:
  enable_player_observation: true
  enable_evolutionary_goals: true
  enable_human_behavior: true
  personality_traits:
    curiosity: 0.8
    caution: 0.5
    social: 0.6
""")
    elif args.show:
        print("Current configuration: (from environment)")
        print(f"  OPENAI_API_KEY: {'set' if os.environ.get('OPENAI_API_KEY') else 'not set'}")
        print(f"  MC_PORT: {os.environ.get('MC_PORT', '25565 (default)')}")


if __name__ == "__main__":
    main()
