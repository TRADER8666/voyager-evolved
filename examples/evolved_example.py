"""Example usage of Voyager Evolved.

This script demonstrates how to use the enhanced Voyager agent
with emergent behaviors and observational learning.
"""

import os
from voyager.evolved import VoyagerEvolved, EvolvedConfig


def main():
    """Run Voyager Evolved with custom configuration."""
    
    # Option 1: Use default configuration
    # agent = VoyagerEvolved(
    #     mc_port=25565,
    #     openai_api_key=os.environ.get("OPENAI_API_KEY"),
    # )
    
    # Option 2: Use a preset configuration
    # config = EvolvedConfig.create_curious_explorer()
    # config = EvolvedConfig.create_social_learner()
    # config = EvolvedConfig.create_survivor()
    # config = EvolvedConfig.create_natural_player()
    
    # Option 3: Create custom configuration
    config = EvolvedConfig()
    
    # Customize personality
    config.personality.curiosity = 0.8          # More curious
    config.personality.caution = 0.4            # Less cautious
    config.personality.sociability = 0.7        # Interested in other players
    config.personality.persistence = 0.8        # Determined
    config.personality.creativity = 0.6         # Somewhat creative
    config.personality.imitation_tendency = 0.7  # Will learn from others
    
    # Customize observation settings
    config.observation.observation_radius = 40   # Wider observation range
    config.observation.update_frequency = 0.5    # More frequent updates
    config.observation.min_observation_confidence = 0.65
    
    # Customize goal generation
    config.evolutionary_goals.survival_weight = 0.25
    config.evolutionary_goals.exploration_weight = 0.3
    config.evolutionary_goals.social_learning_weight = 0.25
    config.evolutionary_goals.skill_development_weight = 0.15
    config.evolutionary_goals.creativity_weight = 0.05
    
    # Customize human-like behavior
    config.human_behavior.path_deviation_chance = 0.2
    config.human_behavior.thinking_pause_chance = 0.12
    config.human_behavior.mistake_chance = 0.03
    config.human_behavior.look_around_frequency = 0.25
    
    # Logging and debugging
    config.log_observations = True
    config.log_goal_evolution = True
    config.verbose_mode = True
    
    # Save this configuration for later use
    config.save("my_evolved_config.json")
    
    # Create the evolved agent
    agent = VoyagerEvolved(
        mc_port=25565,                    # Minecraft server port
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        ckpt_dir="ckpt_evolved",          # Checkpoint directory
        resume=False,                      # Start fresh or resume
        evolved_config=config,             # Our custom config
    )
    
    print("\n" + "="*60)
    print("Voyager Evolved Configuration:")
    print("="*60)
    print(f"Personality: {config.personality}")
    print(f"Observation radius: {config.observation.observation_radius}")
    print(f"Goal weights: survival={config.evolutionary_goals.survival_weight}, "
          f"exploration={config.evolutionary_goals.exploration_weight}, "
          f"social={config.evolutionary_goals.social_learning_weight}")
    print("="*60 + "\n")
    
    try:
        # Start the learning loop
        result = agent.learn()
        
        # Print final results
        print("\n" + "="*60)
        print("Learning Complete!")
        print("="*60)
        print(f"Completed tasks: {len(result['completed_tasks'])}")
        print(f"Failed tasks: {len(result['failed_tasks'])}")
        print(f"Skills learned: {len(result['skills'])}")
        
        # Print evolved statistics
        evolved_stats = result.get('evolved_stats', {})
        if 'personality' in evolved_stats:
            print(f"\nFinal mood: {evolved_stats['personality']['overall_mood']:.2f}")
        if 'goals' in evolved_stats:
            print(f"Total goals generated: {evolved_stats['goals']['total_goals_generated']}")
        if 'learning' in evolved_stats:
            print(f"Strategies learned: {evolved_stats['learning']['total_strategies_learned']}")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving state...")
    finally:
        agent.close()
        print("Agent closed.")


def example_with_specific_task():
    """Example: Run agent on a specific task."""
    config = EvolvedConfig.create_natural_player()
    
    agent = VoyagerEvolved(
        mc_port=25565,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        evolved_config=config,
    )
    
    try:
        # Run inference on a specific task
        agent.inference(task="Build a small house with a door")
    finally:
        agent.close()


def example_observing_multiplayer():
    """Example: Focus on observing and learning from other players."""
    config = EvolvedConfig()
    
    # Maximize social learning
    config.personality.sociability = 0.95
    config.personality.imitation_tendency = 0.9
    config.personality.curiosity = 0.4  # Less self-driven exploration
    
    # Enhanced observation
    config.observation.observation_radius = 64
    config.observation.update_frequency = 0.5
    config.observation.max_tracked_players = 20
    
    # Heavy weight on social learning goals
    config.evolutionary_goals.social_learning_weight = 0.5
    config.evolutionary_goals.exploration_weight = 0.15
    
    agent = VoyagerEvolved(
        mc_port=25565,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        evolved_config=config,
    )
    
    try:
        agent.learn()
    finally:
        agent.close()


if __name__ == "__main__":
    main()
