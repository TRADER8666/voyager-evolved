# 🚀 Voyager Evolved

[![Linux Only](https://img.shields.io/badge/platform-Linux-blue.svg)](https://www.linux.org/)
[![Ollama Powered](https://img.shields.io/badge/LLM-Ollama-green.svg)](https://ollama.ai/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An enhanced, human-like Minecraft AI agent powered by local LLMs via Ollama.**

> ⚠️ **Linux Only**: This project is optimized exclusively for Linux systems for maximum performance.

## 🌟 Overview

Voyager Evolved is a significant enhancement of the original Voyager project, featuring:

- **🧠 Ollama-Powered**: 100% local LLM inference with no API keys needed
- **🐧 Linux-Optimized**: Performance tuned specifically for Linux
- **👀 Player Observation**: Watch and learn from other players on the server
- **🎯 Evolutionary Goals**: Emergent goal generation with priorities and chaining
- **🎭 Human-like Behavior**: Fatigue, attention span, and emotional responses
- **📈 Skill System**: Difficulty tracking, prerequisites, and versioning
- **⚡ Performance**: LLM caching, batch processing, async operations

## ✨ Key Features

### Player Observation System
- Track nearby players and classify their activities
- Pattern recognition using DBSCAN clustering
- Memory decay for relevance-weighted observations
- Build player profiles over time

### Evolutionary Goal System
- **Priority System**: Survival > Safety > Tools > Resources > Skills > Exploration
- **Goal Chaining**: Complete prerequisites to unlock advanced goals
- **Goal Mutations**: Create variations of successful strategies
- **Fitness Tracking**: Adapt based on success/failure history

### Human-like Behavior
- **Fatigue System**: Actions slow down with prolonged activity
- **Attention Span**: Natural focus shifts and distractions
- **Emotional Responses**: React to events (excitement, caution, frustration)
- **Learning Curve**: Improve at tasks over time
- **Idle Behaviors**: Natural looking around and small movements

### Performance Optimizations
- LRU cache for LLM responses (50MB default)
- Batch processing for observations
- Async task manager with thread pool
- Linux-specific memory management
- Configurable performance profiles (fast/balanced/quality)

## 📋 Requirements

- **OS**: Linux (Ubuntu 20.04+, Debian 11+, Fedora 35+, Arch)
- **Python**: 3.9+
- **Node.js**: 16+
- **RAM**: 8GB minimum, 16GB recommended
- **Ollama**: Latest version

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/voyager_evolved.git
cd voyager_evolved
```

### 2. Run the Installer

```bash
chmod +x install.sh
./install.sh
```

### 3. Setup Ollama

```bash
# Install Ollama (if not installed by script)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# Pull recommended models (in another terminal)
ollama pull llama2
ollama pull nomic-embed-text
```

### 4. Configure

```bash
# Copy example config
cp configs/config.example.yaml configs/config.yaml

# Edit as needed
nano configs/config.yaml
```

### 5. Run

```bash
# Activate virtual environment
source venv/bin/activate

# Run Voyager Evolved
python run_voyager.py
```

## ⚙️ Configuration

### Performance Profiles

```python
from voyager.evolved.config import EvolvedConfig, PerformanceProfile

# Fast mode (prioritize speed)
config = EvolvedConfig.create_fast_profile()

# Balanced mode (default)
config = EvolvedConfig()

# Quality mode (better decisions)
config = EvolvedConfig.create_quality_profile()

# Debug mode (detailed logging)
config = EvolvedConfig.create_debug_profile()

# Linux-optimized
config = EvolvedConfig.create_linux_optimized()
```

### Personality Presets

```python
# Curious explorer
config = EvolvedConfig.create_curious_explorer()

# Social learner
config = EvolvedConfig.create_social_learner()

# Survival focused
config = EvolvedConfig.create_survivor()

# Maximum human-likeness
config = EvolvedConfig.create_natural_player()
```

### Key Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `observation.update_frequency` | 1.0s | How often to scan for players |
| `observation.max_tracked_players` | 10 | Maximum players to track |
| `evolutionary_goals.survival_weight` | 0.3 | Weight for survival goals |
| `human_behavior.enable_fatigue` | true | Enable fatigue system |
| `human_behavior.enable_emotions` | true | Enable emotional responses |
| `performance.enable_llm_cache` | true | Cache LLM responses |
| `performance.max_async_workers` | 4 | Thread pool size |

## 📁 Project Structure

```
voyager_evolved/
├── voyager/
│   ├── agents/           # Action, Critic, Curriculum agents
│   │   └── skill.py      # Enhanced skill manager
│   ├── evolved/          # Evolved features
│   │   ├── config.py     # Configuration system
│   │   ├── evolutionary_goals.py  # Goal system
│   │   ├── human_behavior.py      # Human-like behavior
│   │   ├── performance.py         # Performance optimizations
│   │   ├── personality.py         # Personality engine
│   │   └── player_observer.py     # Player observation
│   ├── llm/             # Ollama LLM integration
│   └── env/             # Minecraft environment
├── configs/             # Configuration files
├── docs/               # Documentation
├── tests/              # Test suite
├── install.sh          # Linux installer
└── run_voyager.py      # Main entry point
```

## 🔧 Performance Tuning

### Memory Usage

```yaml
performance:
  memory_limit_percent: 75.0
  cache_max_size_mb: 50.0
  cache_max_entries: 500
```

### LLM Response Time

```yaml
ollama:
  model: "llama2"  # Use smaller models for speed
  # model: "mistral"  # Alternative fast model
  request_timeout: 120
```

### Batch Processing

```yaml
performance:
  batch_processing: true
  batch_size: 10
  batch_interval: 1.0
```

### Linux Optimizations

The installer automatically applies these optimizations:

```bash
export MALLOC_ARENA_MAX=2  # Reduce memory fragmentation
```

For more detailed performance tuning, see [docs/PERFORMANCE_TUNING.md](docs/PERFORMANCE_TUNING.md).

## 🐛 Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
systemctl restart ollama
# or
ollama serve
```

### High Memory Usage

```bash
# Monitor memory
htop

# Reduce cache size in config
performance:
  cache_max_size_mb: 25.0
  memory_limit_percent: 70.0
```

### Slow Performance

1. Use the `fast` performance profile
2. Reduce `observation.max_tracked_players`
3. Disable pattern recognition: `observation.enable_pattern_recognition: false`
4. Use a smaller model: `ollama.model: "mistral"`

### Minecraft Connection

```bash
# Ensure Minecraft server is running
# Check server logs for mineflayer connection

# Verify Node.js modules
cd voyager/env/mineflayer
npm install
```

## 📊 Monitoring

### Performance Report

```python
from voyager.evolved.performance import get_performance_report

report = get_performance_report()
print(report)
```

### Goal Statistics

```python
stats = voyager.goal_system.get_goal_statistics()
print(stats)
```

### Skill Library Stats

```python
stats = voyager.skill_manager.get_skill_stats()
print(stats)
```

## 🤝 Contributing

Contributions are welcome! Please ensure:

1. Code works on Linux
2. Tests pass: `pytest tests/`
3. Code is formatted: `black .`
4. Add documentation for new features

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Original Voyager](https://github.com/MineDojo/Voyager) by MineDojo
- [Ollama](https://ollama.ai/) for local LLM inference
- [Mineflayer](https://github.com/PrismarineJS/mineflayer) for Minecraft automation
- [LangChain](https://langchain.com/) for LLM abstractions

---

**Note**: This is a research project. Use responsibly and respect Minecraft server rules.
