#!/usr/bin/env python3
"""
Voyager Evolved Features Example
================================

This example demonstrates all the enhanced features of Voyager Evolved,
including player observation, evolutionary goals, and human-like behaviors.
"""

import os
from voyager.evolved import (
    VoyagerEvolved,
    EvolvedConfig,
    PersonalityEngine,
    PlayerObserver,
    EvolutionaryGoalSystem,
    HumanBehaviorSystem,
)


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Create a custom configuration
    config = EvolvedConfig(
        openai_api_key=api_key,
        model_name="gpt-4",
        mc_port=25565,
        
        # Evolved features
        enable_player_observation=True,
        enable_evolutionary_goals=True,
        enable_human_behavior=True,
        
        # Personality customization
        personality_traits={
            "curiosity": 0.9,      # Very curious - loves exploring
            "caution": 0.3,        # Brave - takes risks
            "social": 0.8,         # Friendly - interacts with players
            "creativity": 0.85,    # Creative problem solver
            "persistence": 0.7,    # Determined but flexible
        }
    )
    
    print("Creating Voyager Evolved instance...")
    print(f"Personality: {config.personality_traits}")
    
    # Create and run Voyager Evolved
    voyager = VoyagerEvolved(config)
    
    print("\nStarting learning session...")
    print("The agent will:")
    print("  - Observe and learn from nearby players")
    print("  - Evolve its goals based on experience")
    print("  - Exhibit human-like behaviors")
    print()
    
    try:
        voyager.learn(max_iterations=20)
    except KeyboardInterrupt:
        print("\nSession interrupted by user")
    
    print("\nSession complete!")


def demonstrate_components():
    """
    Demonstrate individual components of Voyager Evolved.
    """
    print("\n=== Demonstrating Individual Components ===")
    
    # 1. Personality Engine
    print("\n--- Personality Engine ---")
    personality = PersonalityEngine(
        curiosity=0.8,
        caution=0.4,
        social=0.7
    )
    print(f"Personality created: {personality}")
    
    # 2. Player Observer
    print("\n--- Player Observer ---")
    observer = PlayerObserver(observation_radius=50)
    print(f"Observer ready with radius: {observer.observation_radius}")
    
    # 3. Evolutionary Goals
    print("\n--- Evolutionary Goal System ---")
    goals = EvolutionaryGoalSystem(mutation_rate=0.1)
    print(f"Goal system initialized with mutation rate: {goals.mutation_rate}")
    
    # 4. Human Behavior System
    print("\n--- Human Behavior System ---")
    behavior = HumanBehaviorSystem(enable_pauses=True)
    print(f"Behavior system ready: {behavior}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--components":
        demonstrate_components()
    else:
        main()
