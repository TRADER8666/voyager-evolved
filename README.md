# 🚀 Voyager Evolved

<div align="center">

**An Enhanced Minecraft AI Agent with Evolutionary Goals and Human-like Behaviors**

**Powered by Ollama - Run Locally Without API Keys! 🎉**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Node.js 16+](https://img.shields.io/badge/Node.js-16+-green.svg)](https://nodejs.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange.svg)](https://ollama.ai/)

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Ollama Setup](#-ollama-setup) • [Documentation](#-documentation)

</div>

---

## 📖 Overview

**Voyager Evolved** is an enhanced version of the original [Voyager](https://github.com/MineDojo/Voyager) project - the first LLM-powered embodied lifelong learning agent in Minecraft. This evolution adds sophisticated human-like behaviors, player observation capabilities, and evolutionary goal systems.

### 🦙 Powered by Local LLM

**No API keys required!** Voyager Evolved uses [Ollama](https://ollama.ai/) for local LLM inference, allowing you to run the agent completely locally with free, open-source models like LLaMA 2, Mistral, and CodeLlama.

### What's New in Voyager Evolved?

- 🦙 **Ollama-Powered** - Run locally with free, open-source LLMs
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
- **Ollama** (free, local LLM)

### Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/TRADER8666/voyager-evolved.git
cd voyager-evolved

# Run the installer
chmod +x install.sh
./install.sh
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/TRADER8666/voyager-evolved.git
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
git clone https://github.com/TRADER8666/voyager-evolved.git
cd voyager-evolved
install.bat
```

---

## 🦙 Ollama Setup

Ollama allows you to run Voyager Evolved **completely locally** with no API keys or cloud services!

### Step 1: Install Ollama

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**macOS:**
```bash
brew install ollama
# Or download from https://ollama.ai/download
```

**Windows:**
Download from [https://ollama.ai/download](https://ollama.ai/download)

### Step 2: Pull Required Models

```bash
# Pull the main LLM model (choose one)
ollama pull llama2        # Good general-purpose (7B, ~4GB)
ollama pull mistral       # Fast and capable (7B, ~4GB)
ollama pull codellama     # Great for code generation (7B, ~4GB)
ollama pull llama2:13b    # Better reasoning (13B, ~8GB RAM needed)

# Pull the embedding model (required for skill library)
ollama pull nomic-embed-text
```

### Step 3: Start Ollama

```bash
# Start the Ollama server (runs on http://localhost:11434)
ollama serve
```

**Note:** Keep this running in a separate terminal while using Voyager Evolved.

### Recommended Models for Minecraft Agent

| Model | Size | RAM Needed | Best For |
|-------|------|------------|----------|
| `llama2` | 7B | ~4GB | General tasks (default) |
| `mistral` | 7B | ~4GB | Fast responses |
| `codellama` | 7B | ~4GB | Code generation |
| `llama2:13b` | 13B | ~8GB | Complex reasoning |
| `mixtral` | 47B | ~26GB | Best quality (if you have RAM) |

### Troubleshooting Ollama

**Connection Refused:**
```bash
# Make sure Ollama is running
ollama serve

# Check if it's responding
curl http://localhost:11434/api/tags
```

**Model Not Found:**
```bash
# List installed models
ollama list

# Pull the missing model
ollama pull llama2
```

**Slow Performance:**
- Use a smaller model (llama2 vs llama2:13b)
- Close other applications
- Consider GPU acceleration (Ollama auto-detects NVIDIA GPUs)

---

## 🚀 Quick Start

### 1. Start Ollama

```bash
# In a separate terminal
ollama serve
```

### 2. Start Minecraft

1. Launch Minecraft Java Edition
2. Create a new world in Creative or Survival mode
3. Open to LAN (Esc → Open to LAN → Start LAN World)
4. Note the port number displayed

### 3. Run Voyager Evolved

```bash
python run_voyager.py --evolved --port 55555
```

### Python API Usage

```python
from voyager.evolved import VoyagerEvolved

# Create agent (uses Ollama by default - no API key needed!)
voyager = VoyagerEvolved(
    mc_port=55555,  # Your LAN port
    # action_agent_model_name="llama2",  # Optional: specify model
)

# Start learning
voyager.learn(max_iterations=10)
```

### Using a Different Model

```python
from voyager.evolved import VoyagerEvolved

voyager = VoyagerEvolved(
    mc_port=55555,
    action_agent_model_name="mistral",
    curriculum_agent_model_name="mistral",
    critic_agent_model_name="mistral",
)
voyager.learn(max_iterations=10)
```

---

## 📚 Documentation

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mc_port` | int | 25565 | Minecraft server port |
| `action_agent_model_name` | str | "llama2" | Model for action generation |
| `curriculum_agent_model_name` | str | "llama2" | Model for curriculum |
| `critic_agent_model_name` | str | "llama2" | Model for task evaluation |
| `enable_player_observation` | bool | True | Enable player watching |
| `enable_evolutionary_goals` | bool | True | Enable goal evolution |
| `enable_human_behavior` | bool | True | Enable human-like actions |

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

voyager = Voyager(mc_port=25565)
voyager.learn()
```

### Evolved Usage with Custom Personality
```python
from voyager.evolved import VoyagerEvolved, EvolvedConfig

config = EvolvedConfig(
    personality_traits={
        "curiosity": 0.9,   # Very exploratory
        "caution": 0.2,     # Risk-taker
        "social": 0.8,      # Friendly
    }
)

voyager = VoyagerEvolved(
    mc_port=25565,
    evolved_config=config,
)
voyager.learn(max_iterations=50)
```

---

## 🔧 Troubleshooting

### Common Issues

#### "Connection refused" error
- Ensure Minecraft is running and open to LAN
- Verify the port number matches your configuration
- Check if firewall is blocking the connection

#### "Ollama not running"
```bash
# Start Ollama server
ollama serve
```

#### Bot not moving / stuck
- Check if the Ollama model is responding
- Try a smaller/faster model like mistral
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
│   ├── llm/                # Ollama LLM integration
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
- [Ollama](https://ollama.ai/) for local LLM inference
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
