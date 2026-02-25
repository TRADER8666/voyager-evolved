# 🚀 Voyager Evolved

<div align="center">

**An Enhanced Minecraft AI Agent with Evolutionary Goals and Human-like Behaviors**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Node.js 16+](https://img.shields.io/badge/Node.js-16+-green.svg)](https://nodejs.org/)

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 📖 Overview

**Voyager Evolved** is an enhanced version of the original [Voyager](https://github.com/MineDojo/Voyager) project - the first LLM-powered embodied lifelong learning agent in Minecraft. This evolution adds sophisticated human-like behaviors, player observation capabilities, and evolutionary goal systems.

### What's New in Voyager Evolved?

- 🔭 **Player Observation System** - Learn from watching other players
- 🧬 **Evolutionary Goals** - Goals that adapt and evolve based on experience
- 🎭 **Human-like Behaviors** - Natural pauses, emotional responses, realistic patterns
- 🧠 **Personality Engine** - Customizable personality traits that affect decision-making
- 📊 **Observational Learning** - Extract and apply knowledge from observed behaviors

---

## ✨ Features

### Core Features (from Original Voyager)
- **Automatic Curriculum** - Maximizes exploration through intelligent task generation
- **Skill Library** - Ever-growing collection of executable code for complex behaviors
- **Iterative Prompting** - Self-verification with environment feedback

### Evolved Features (New!)

| Feature | Description |
|---------|-------------|
| **Player Observer** | Monitors and learns from nearby player actions |
| **Evolutionary Goals** | Goals mutate and evolve based on success/failure |
| **Human Behavior** | Realistic delays, curiosity-driven exploration |
| **Personality Traits** | Configurable curiosity, caution, social traits |
| **Observational Learning** | Converts observations into actionable skills |

---

## 🛠 Installation

### Prerequisites

- **Python** 3.9 or higher
- **Node.js** 16.13.0 or higher
- **Minecraft Java Edition** (1.19.x recommended)
- **OpenAI API Key** with GPT-4 access

### Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/voyager-evolved.git
cd voyager-evolved

# Run the installer
chmod +x install.sh
./install.sh
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/voyager-evolved.git
cd voyager-evolved

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install Python dependencies
pip install -e .

# Install Node.js dependencies (if mineflayer directory exists)
cd voyager/env/mineflayer
npm install
cd ../../..
```

### Windows Installation

```batch
git clone https://github.com/YOUR_USERNAME/voyager-evolved.git
cd voyager-evolved
install.bat
```

---

## 🚀 Quick Start

### 1. Set Up Your API Key

```bash
# Linux/Mac
export OPENAI_API_KEY='your-openai-api-key-here'

# Windows
set OPENAI_API_KEY=your-openai-api-key-here
```

### 2. Start Minecraft

1. Launch Minecraft Java Edition
2. Create a new world in Creative or Survival mode
3. Open to LAN (Esc → Open to LAN → Start LAN World)
4. Note the port number displayed

### 3. Run Voyager Evolved

```bash
# Basic run
python run_voyager.py --evolved

# With custom settings
python run_voyager.py --evolved --port 55555 --iterations 20

# Quick demo (5 iterations)
./run_quick_demo.sh
```

### 4. Python API Usage

```python
import os
from voyager.evolved import VoyagerEvolved, EvolvedConfig

config = EvolvedConfig(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    mc_port=55555,  # Your LAN port
    enable_player_observation=True,
    enable_evolutionary_goals=True,
    personality_traits={
        "curiosity": 0.8,
        "caution": 0.5,
        "social": 0.7,
    }
)

voyager = VoyagerEvolved(config)
voyager.learn(max_iterations=10)
```

---

## 📚 Documentation

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `openai_api_key` | str | - | Your OpenAI API key (required) |
| `mc_port` | int | 25565 | Minecraft server port |
| `model_name` | str | "gpt-4" | OpenAI model to use |
| `enable_player_observation` | bool | True | Enable player watching |
| `enable_evolutionary_goals` | bool | True | Enable goal evolution |
| `enable_human_behavior` | bool | True | Enable human-like actions |
| `personality_traits` | dict | {} | Custom personality values |

### Personality Traits

| Trait | Range | Description |
|-------|-------|-------------|
| `curiosity` | 0.0-1.0 | Exploration vs. task focus |
| `caution` | 0.0-1.0 | Risk aversion level |
| `social` | 0.0-1.0 | Interaction tendency |
| `creativity` | 0.0-1.0 | Novel solution preference |
| `persistence` | 0.0-1.0 | Task completion drive |

### Example Configurations

See the [`configs/`](configs/) directory for example configuration files:

- `config.example.yaml` - Full configuration with all options
- Copy to `config.yaml` and customize for your setup

---

## 🎮 Usage Examples

### Basic Usage
```python
from voyager import Voyager

voyager = Voyager(
    openai_api_key="your-key",
    mc_port=25565,
)
voyager.learn()
```

### Evolved Usage with Custom Personality
```python
from voyager.evolved import VoyagerEvolved, EvolvedConfig

config = EvolvedConfig(
    openai_api_key="your-key",
    mc_port=25565,
    personality_traits={
        "curiosity": 0.9,   # Very exploratory
        "caution": 0.2,     # Risk-taker
        "social": 0.8,      # Friendly
    }
)

voyager = VoyagerEvolved(config)
voyager.learn(max_iterations=50)
```

### Adding Custom Goals
```python
voyager.add_goal({
    "name": "Build a Castle",
    "description": "Construct a castle with towers and walls",
    "priority": 1,
})
```

---

## 🔧 Troubleshooting

### Common Issues

#### "Connection refused" error
- Ensure Minecraft is running and open to LAN
- Verify the port number matches your configuration
- Check if firewall is blocking the connection

#### "OPENAI_API_KEY not set"
```bash
export OPENAI_API_KEY='sk-your-actual-key-here'
```

#### Bot not moving / stuck
- Ensure you have GPT-4 access (not just GPT-3.5)
- Check the Minecraft world is not paused
- Verify Node.js dependencies are installed

#### Import errors
```bash
pip install -e .  # Reinstall the package
```

### Debug Mode

Enable verbose logging:
```python
config = EvolvedConfig(
    ...,
    debug=True,
)
```

Or set environment variable:
```bash
export VOYAGER_DEBUG=1
```

---

## 📁 Project Structure

```
voyager-evolved/
├── voyager/                 # Main package
│   ├── agents/             # AI agent components
│   ├── env/                # Minecraft environment
│   ├── evolved/            # ⭐ Evolved features
│   │   ├── player_observer.py
│   │   ├── evolutionary_goals.py
│   │   ├── human_behavior.py
│   │   ├── personality.py
│   │   └── voyager_evolved.py
│   ├── prompts/            # LLM prompts
│   └── utils/              # Utilities
├── examples/               # Usage examples
├── configs/                # Configuration files
├── skill_library/          # Learned skills storage
├── install.sh             # Linux/Mac installer
├── install.bat            # Windows installer
├── run_voyager.py         # Quick start script
└── README.md              # This file
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Original [Voyager](https://github.com/MineDojo/Voyager) project by MineDojo Team
- [MineDojo](https://minedojo.org/) for Minecraft ML research
- [Mineflayer](https://github.com/PrismarineJS/mineflayer) for Minecraft bot framework

---

## 📬 Support

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/voyager-evolved/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/voyager-evolved/discussions)

---

<div align="center">

**[⬆ Back to Top](#-voyager-evolved)**

Made with ❤️ for the Minecraft AI community

</div>
