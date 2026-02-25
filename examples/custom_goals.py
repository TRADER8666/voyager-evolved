#!/usr/bin/env python3
"""
Custom Goals Example
====================

This example shows how to create custom goals for Voyager Evolved.
"""

import os
from voyager.evolved import VoyagerEvolved, EvolvedConfig


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    config = EvolvedConfig(
        openai_api_key=api_key,
        mc_port=25565,
    )
    
    voyager = VoyagerEvolved(config)
    
    # Define custom goals
    custom_goals = [
        {
            "name": "Build a Shelter",
            "description": "Construct a basic shelter with walls, roof, door, and a bed",
            "priority": 1,
            "requirements": ["wood", "crafting_table"],
        },
        {
            "name": "Start a Farm",
            "description": "Create a small farm with wheat, carrots, or potatoes",
            "priority": 2,
            "requirements": ["hoe", "water_nearby"],
        },
        {
            "name": "Explore a Cave",
            "description": "Find and explore a cave system, collect ores",
            "priority": 3,
            "requirements": ["torch", "pickaxe"],
        },
    ]
    
    print("Adding custom goals...")
    for goal in custom_goals:
        voyager.add_goal(goal)
        print(f"  Added: {goal['name']}")
    
    print("\nStarting Voyager with custom goals...")
    voyager.learn(max_iterations=30)


if __name__ == "__main__":
    main()
