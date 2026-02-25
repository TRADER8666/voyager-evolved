# Voyager Evolved 🚀

**An Enhanced Minecraft AI Agent with Emergent Behaviors and Observational Learning**

Voyager Evolved builds upon the original [Voyager](https://github.com/MineDojo/Voyager) project, adding sophisticated systems for emergent goal generation, player observation, human-like behavior, and social learning. The agent develops its own personality through experience and learns by watching other players on the server.

## 🌟 Key Features

### 1. Player Observation System
The agent can detect, track, and learn from other players on the Minecraft server:
- **Player Detection**: Identifies players within configurable radius
- **Activity Classification**: Recognizes activities (mining, building, farming, fighting, etc.)
- **Behavior Recording**: Stores observed strategies with confidence scores
- **Player Profiling**: Builds profiles of observed players over time

### 2. Evolutionary Goal System
Replace or enhance automatic curriculum with emergent goal generation:
- **Survival Instincts**: Prioritizes food, health, and shelter when needed
- **Curiosity-Driven Exploration**: Explores based on novelty and interest
- **Social Learning Goals**: Generates goals to mimic successful player behaviors
- **Evolutionary Fitness**: Adapts goal preferences based on success/failure history
- **Personality Influence**: Goals weighted by personality traits

### 3. Human-like Behavior Patterns
Makes the agent's actions more natural and believable:
- **Path Deviation**: Slight randomness in movement (not perfectly optimal)
- **Thinking Pauses**: Occasional pauses before actions
- **Minor Mistakes**: Rare, recoverable errors for authenticity
- **Natural Camera Movement**: Head turns and looking at interesting things
- **Action Pacing**: Variable timing based on mood and energy

### 4. Personality System
Dynamic personality that evolves through experience:
- **Core Traits**: Curiosity, caution, sociability, persistence, creativity, impulsivity
- **Mood System**: Happiness, frustration, excitement, confidence, boredom
- **Learning Traits**: Imitation tendency, self-reliance
- **Experience-Based Evolution**: Traits subtly change based on successes/failures

### 5. Observational Learning Integration
Connects observation to skill acquisition:
- **Strategy Extraction**: Learns patterns from multiple observations
- **Contextual Adaptation**: Adapts learned strategies to current situation
- **Skill Library Integration**: Adds successful observed strategies to skills
- **Prioritization**: Ranks strategies by frequency and success rate

## 📁 Project Structure

```
voyager_evolved/
├── voyager/
│   ├── evolved/                    # New evolved features
│   │   ├── __init__.py
│   │   ├── config.py               # Configuration system
│   │   ├── player_observer.py      # Player observation
│   │   ├── evolutionary_goals.py   # Goal generation
│   │   ├── human_behavior.py       # Human-like behaviors
│   │   ├── personality.py          # Personality engine
│   │   ├── observational_learning.py # Learning integration
│   │   └── voyager_evolved.py      # Main evolved agent
│   ├── agents/                     # Original Voyager agents
│   ├── env/                        # Environment interface
│   ├── prompts/                    # LLM prompts
│   └── utils/                      # Utilities
├── skill_library/                  # Pre-built skills
├── README.md                       # Original README
└── README_EVOLVED.md               # This file
```

## 🚀 Quick Start

### Prerequisites

1. Follow the original Voyager installation instructions in `README.md`
2. Ensure you have:
   - Python 3.9+
   - Node.js 16+
   - Minecraft Java Edition
   - OpenAI API key

### Basic Usage

```python
from voyager.evolved import VoyagerEvolved, EvolvedConfig

# Create with default evolved features
agent = VoyagerEvolved(
    mc_port=25565,
    openai_api_key="your-api-key",
)

# Start learning with evolved behaviors
agent.learn()
```

### With Custom Configuration

```python
from voyager.evolved import VoyagerEvolved, EvolvedConfig

# Create custom configuration
config = EvolvedConfig()

# Adjust personality
config.personality.curiosity = 0.9  # Very curious
config.personality.caution = 0.3   # Risk-taking
config.personality.sociability = 0.8  # Interested in other players

# Adjust observation
config.observation.observation_radius = 48
config.observation.min_observation_confidence = 0.7

# Adjust goals
config.evolutionary_goals.survival_weight = 0.2
config.evolutionary_goals.exploration_weight = 0.4
config.evolutionary_goals.social_learning_weight = 0.3

# Adjust human-like behavior
config.human_behavior.path_deviation_chance = 0.2
config.human_behavior.thinking_pause_chance = 0.15

# Create agent
agent = VoyagerEvolved(
    mc_port=25565,
    openai_api_key="your-api-key",
    evolved_config=config,
)

agent.learn()
```

### Using Preset Configurations

```python
from voyager.evolved import VoyagerEvolved, EvolvedConfig

# Create a curious explorer
config = EvolvedConfig.create_curious_explorer()

# Or a social learner
config = EvolvedConfig.create_social_learner()

# Or a survival-focused agent
config = EvolvedConfig.create_survivor()

# Or maximize human-like behavior
config = EvolvedConfig.create_natural_player()

agent = VoyagerEvolved(
    mc_port=25565,
    openai_api_key="your-api-key",
    evolved_config=config,
)
```

### Loading/Saving Configuration

```python
# Save configuration
config.save("my_config.json")

# Load from file
agent = VoyagerEvolved(
    mc_port=25565,
    openai_api_key="your-api-key",
    evolved_config_path="my_config.json",
)
```

## ⚙️ Configuration Reference

### PersonalityConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `curiosity` | float | 0.7 | Drive to explore and try new things (0-1) |
| `caution` | float | 0.5 | Tendency to avoid risks (0-1) |
| `sociability` | float | 0.6 | Interest in observing/interacting with players (0-1) |
| `persistence` | float | 0.7 | Determination to complete tasks (0-1) |
| `creativity` | float | 0.5 | Willingness to try unconventional approaches (0-1) |
| `impulsivity` | float | 0.3 | Tendency to act without full planning (0-1) |
| `imitation_tendency` | float | 0.6 | How much to copy other players (0-1) |
| `self_reliance` | float | 0.5 | Trust in own abilities vs learned behaviors (0-1) |
| `current_mood` | float | 0.5 | Initial mood (-1 to 1) |
| `energy_level` | float | 0.8 | Initial energy level (0-1) |

### ObservationConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `observation_radius` | int | 32 | Blocks radius to detect players |
| `update_frequency` | float | 1.0 | Seconds between observation updates |
| `max_tracked_players` | int | 10 | Maximum players to track simultaneously |
| `activity_detection_window` | float | 5.0 | Seconds to analyze for activity type |
| `min_observation_confidence` | float | 0.6 | Minimum confidence to record behavior |
| `max_observation_memory` | int | 1000 | Maximum stored observations |
| `observation_decay_rate` | float | 0.1 | How fast old observations lose weight |
| `build_player_profiles` | bool | True | Create detailed player profiles |
| `track_player_skills` | bool | True | Track what skills players demonstrate |

### EvolutionaryGoalConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `survival_weight` | float | 0.3 | Weight for survival goals |
| `exploration_weight` | float | 0.25 | Weight for exploration goals |
| `social_learning_weight` | float | 0.2 | Weight for social learning goals |
| `skill_development_weight` | float | 0.15 | Weight for skill improvement goals |
| `creativity_weight` | float | 0.1 | Weight for creative/novel goals |
| `hunger_threshold` | float | 8.0 | Hunger level that triggers food-seeking |
| `health_threshold` | float | 10.0 | Health level that triggers safety-seeking |
| `shelter_priority_at_night` | bool | True | Prioritize shelter at night |
| `novelty_seeking_radius` | int | 100 | Blocks to consider for novelty |
| `goal_mutation_rate` | float | 0.1 | Rate of random goal modifications |
| `fitness_memory_size` | int | 100 | Number of past goals to track fitness |
| `adaptation_rate` | float | 0.05 | How fast to adapt goal preferences |
| `observe_before_imitate` | int | 3 | Times to observe before copying |
| `success_imitation_boost` | float | 0.3 | Boost for observed successful actions |

### HumanBehaviorConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path_deviation_chance` | float | 0.15 | Chance to deviate from optimal path |
| `path_deviation_amount` | float | 2.0 | Max blocks to deviate |
| `look_around_frequency` | float | 0.2 | Chance to look around per action |
| `thinking_pause_chance` | float | 0.1 | Chance to pause before actions |
| `thinking_pause_duration` | tuple | (0.5, 2.0) | Min/max pause duration in seconds |
| `decision_hesitation` | float | 0.05 | Chance to reconsider decisions |
| `mistake_chance` | float | 0.05 | Chance of making minor mistakes |
| `suboptimal_choice_chance` | float | 0.1 | Chance of non-optimal but valid choice |
| `recovery_from_mistake` | float | 0.8 | Chance to correct mistakes |
| `natural_head_movement` | bool | True | Enable natural looking behavior |
| `look_at_interesting_things` | bool | True | Look at mobs, players, etc. |
| `head_turn_smoothness` | float | 0.3 | Lower = smoother head turns |
| `action_speed_variation` | float | 0.2 | Variation in action timing |
| `break_between_tasks` | bool | True | Take breaks between major tasks |
| `break_duration` | tuple | (1.0, 5.0) | Min/max break duration in seconds |

### EvolvedConfig (Main)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_player_observation` | bool | True | Enable player observation system |
| `enable_evolutionary_goals` | bool | True | Enable evolutionary goal system |
| `enable_human_behavior` | bool | True | Enable human-like behaviors |
| `enable_observational_learning` | bool | True | Enable learning from observations |
| `log_observations` | bool | True | Log observation events |
| `log_goal_evolution` | bool | True | Log goal generation events |
| `log_behavior_decisions` | bool | False | Log behavior decisions (verbose) |
| `verbose_mode` | bool | False | Enable verbose statistics output |
| `save_personality_state` | bool | True | Persist personality state |
| `save_learned_behaviors` | bool | True | Persist learned strategies |
| `checkpoint_interval` | int | 50 | Save state every N iterations |

## 🔄 How It Works

### Goal Generation Flow

```
1. Check Survival State
   ├── Low health → Generate heal/food goal
   ├── Low hunger → Generate food goal
   ├── Night time → Generate shelter goal
   └── Threats nearby → Generate safety goal

2. Generate Candidate Goals
   ├── Survival goals (if needed)
   ├── Exploration goals (based on curiosity)
   ├── Social learning goals (from observations)
   ├── Skill development goals
   ├── Tool progression goals
   └── Creative mutations of past successes

3. Score and Select
   ├── Apply category weights
   ├── Apply fitness (historical success)
   ├── Apply personality modifiers
   ├── Apply freshness bonus
   └── Softmax selection with impulsivity

4. Execute and Learn
   ├── Execute selected goal
   ├── Record success/failure
   ├── Update fitness scores
   └── Adapt category weights
```

### Observation and Learning Flow

```
1. Detect Players
   └── Scan for players within observation radius

2. Track Movement
   └── Record position, velocity, yaw, pitch, held item

3. Classify Activity
   └── Analyze patterns to classify (mining, building, etc.)

4. Extract Strategies
   └── Group observations by activity
   └── Find common patterns (tools, blocks, duration)

5. Add to Learning
   └── Create LearnedStrategy objects
   └── Prioritize by frequency and success

6. Apply to Decisions
   └── Retrieve relevant strategies for tasks
   └── Adapt to current context
   └── Integrate with skill library
```

## 📊 Statistics and Monitoring

The agent tracks various statistics accessible via:

```python
# Get evolved statistics
stats = agent._get_evolved_stats()

# Personality summary
print(stats['personality'])
# - traits
# - mood
# - energy
# - consecutive_successes/failures
# - top_preferences
# - avoided_situations

# Observation summary
print(stats['observation'])
# - total_players_observed
# - total_behaviors_recorded
# - activity_distribution
# - player_profiles

# Goal statistics
print(stats['goals'])
# - total_goals_generated
# - completed/failed counts
# - category_weights
# - category_fitness
# - exploration_stats

# Learning summary
print(stats['learning'])
# - total_strategies_learned
# - total_strategy_uses
# - strategies_by_activity
# - top_strategies
```

## 🧪 Example Scenarios

### Creating a Cautious Gatherer

```python
config = EvolvedConfig()
config.personality.caution = 0.9
config.personality.curiosity = 0.3
config.evolutionary_goals.survival_weight = 0.5
config.evolutionary_goals.exploration_weight = 0.1
config.human_behavior.mistake_chance = 0.02  # Very few mistakes
```

### Creating a Social Mimic

```python
config = EvolvedConfig()
config.personality.sociability = 0.95
config.personality.imitation_tendency = 0.9
config.personality.self_reliance = 0.2
config.observation.observation_radius = 64
config.evolutionary_goals.social_learning_weight = 0.5
```

### Creating a Creative Builder

```python
config = EvolvedConfig()
config.personality.creativity = 0.9
config.personality.impulsivity = 0.6
config.evolutionary_goals.creativity_weight = 0.4
config.evolutionary_goals.skill_development_weight = 0.3
```

## 🔧 Troubleshooting

### Agent Not Observing Players
- Ensure `enable_player_observation` is True
- Increase `observation_radius` 
- Check that players are within range
- Verify the server allows entity data

### Goals Not Evolving
- Ensure `enable_evolutionary_goals` is True
- Check `fitness_memory_size` is adequate
- Increase `adaptation_rate` for faster evolution
- Verify goals are completing (success/failure recording)

### Behavior Too Mechanical
- Increase `path_deviation_chance`
- Increase `thinking_pause_chance`
- Enable `natural_head_movement`
- Increase `action_speed_variation`

### Not Learning from Observations
- Ensure `enable_observational_learning` is True
- Check `observe_before_imitate` threshold
- Increase `min_observation_confidence` for quality over quantity
- Verify observations are being recorded

## 📝 License

This project extends the original Voyager which is licensed under MIT. The evolved additions are also MIT licensed.

## 🙏 Acknowledgments

- Original Voyager team at MineDojo
- OpenAI for GPT models
- Mineflayer community

## 📧 Contact

For issues related to evolved features, please open an issue with the `[EVOLVED]` tag.
