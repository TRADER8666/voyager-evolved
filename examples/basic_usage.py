#!/usr/bin/env python3
"""
Basic Usage Example for Voyager Evolved
=======================================

This example shows the simplest way to run Voyager.
"""

import os
from voyager import Voyager

def main():
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Create Voyager instance with minimal configuration
    voyager = Voyager(
        openai_api_key=api_key,
        mc_port=25565,  # Default Minecraft port
    )
    
    # Run 5 learning iterations
    voyager.learn(max_iterations=5)


if __name__ == "__main__":
    main()
