# GitHub Repository Setup Guide

This guide explains how to upload Voyager Evolved to GitHub and set up your repository for distribution.

---

## Table of Contents

1. [Create a New GitHub Repository](#1-create-a-new-github-repository)
2. [Push Your Code](#2-push-your-code)
3. [Configure Repository Settings](#3-configure-repository-settings)
4. [Set Up Releases](#4-set-up-releases)
5. [Enable GitHub Actions](#5-enable-github-actions-optional)
6. [Additional Setup](#6-additional-setup)

---

## 1. Create a New GitHub Repository

### Option A: Using GitHub Web Interface

1. Go to [github.com](https://github.com) and sign in
2. Click the **+** icon in the top right → **New repository**
3. Fill in the details:
   - **Repository name**: `voyager-evolved`
   - **Description**: "An enhanced Minecraft AI agent with evolutionary goals and human-like behaviors"
   - **Visibility**: Public (for open source) or Private
   - ❌ Do NOT initialize with README, .gitignore, or license (we have these)
4. Click **Create repository**

### Option B: Using GitHub CLI

```bash
# Install GitHub CLI if not already installed
# See: https://cli.github.com/

# Create repository
gh repo create voyager-evolved --public --description "Enhanced Minecraft AI agent"
```

---

## 2. Push Your Code

### If Starting Fresh

```bash
# Navigate to your project directory
cd /path/to/voyager_evolved

# Initialize git (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Voyager Evolved v1.0.0"

# Add GitHub as remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/voyager-evolved.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### If You Already Have a Git Repository

```bash
# Add GitHub as remote
git remote add origin https://github.com/YOUR_USERNAME/voyager-evolved.git

# Or update existing remote
git remote set-url origin https://github.com/YOUR_USERNAME/voyager-evolved.git

# Push all branches
git push -u origin main
```

### Pushing with Authentication

If prompted for credentials:

#### Option 1: Personal Access Token (Recommended)

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` scope
3. Use token as password when pushing

#### Option 2: SSH Key

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your-email@example.com"

# Add to SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings → SSH and GPG keys → New SSH key

# Change remote to SSH
git remote set-url origin git@github.com:YOUR_USERNAME/voyager-evolved.git
```

---

## 3. Configure Repository Settings

### Set Up Repository Topics

1. Go to your repository on GitHub
2. Click the ⚙️ gear icon next to "About"
3. Add topics: `minecraft`, `ai-agent`, `llm`, `gpt-4`, `openai`, `python`, `machine-learning`

### Configure Branch Protection (Recommended)

1. Go to Settings → Branches
2. Add rule for `main` branch:
   - ✅ Require pull request reviews
   - ✅ Require status checks to pass
   - ✅ Require branches to be up to date

### Enable Issues and Discussions

1. Go to Settings → General
2. Under Features:
   - ✅ Issues
   - ✅ Discussions
   - ✅ Projects

---

## 4. Set Up Releases

### Create Your First Release

1. Go to your repository → **Releases** → **Create a new release**

2. Fill in the details:
   - **Tag**: `v1.0.0` (click "Create new tag")
   - **Release title**: `Voyager Evolved v1.0.0`
   - **Description**: (use template below)

3. Click **Publish release**

### Release Notes Template

```markdown
## Voyager Evolved v1.0.0

🚀 **Initial Release**

### Features

- 🔭 Player Observation System - Learn from watching other players
- 🧬 Evolutionary Goals - Goals that adapt based on experience  
- 🎭 Human-like Behaviors - Natural pauses, emotional responses
- 🧠 Personality Engine - Customizable personality traits
- 📊 Observational Learning - Convert observations to skills

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/voyager-evolved.git
cd voyager-evolved
pip install -e .
```

### Requirements

- Python 3.9+
- Node.js 16+
- OpenAI API Key with GPT-4 access

### Documentation

See [README.md](README.md) for full documentation.
```

### Semantic Versioning

Follow [Semantic Versioning](https://semver.org/):

- `MAJOR.MINOR.PATCH`
- `1.0.0` → `1.0.1` (bug fixes)
- `1.0.0` → `1.1.0` (new features, backward compatible)
- `1.0.0` → `2.0.0` (breaking changes)

---

## 5. Enable GitHub Actions (Optional)

The repository includes GitHub Actions workflows in `.github/workflows/`. To enable:

1. Go to Settings → Actions → General
2. Select "Allow all actions and reusable workflows"
3. Save

### Available Workflows

- **CI** (`ci.yml`): Runs tests on push/PR
- **Release** (`release.yml`): Auto-creates releases on tag

### Creating a CI Workflow

If not present, create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run tests
      run: pytest
    
    - name: Check code style
      run: |
        black --check .
        flake8 .
```

---

## 6. Additional Setup

### Add Badges to README

Update your README with dynamic badges:

```markdown
[![CI](https://docs.github.com/assets/cb-40551/images/help/actions/superlinter-workflow-sidebar.png)
[![Release](https://img.shields.io/github/v/release/YOUR_USERNAME/voyager-evolved)](https://github.com/YOUR_USERNAME/voyager-evolved/releases)
[![License](https://img.shields.io/github/license/YOUR_USERNAME/voyager-evolved)](LICENSE)
```

### Set Up Issue Templates

Create `.github/ISSUE_TEMPLATE/bug_report.md`:

```markdown
---
name: Bug Report
about: Report a bug
title: '[BUG] '
labels: bug
---

## Description
A clear description of the bug.

## Steps to Reproduce
1. ...
2. ...

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Environment
- OS: 
- Python version: 
- Node.js version: 
```

### Set Up Pull Request Template

Create `.github/PULL_REQUEST_TEMPLATE.md`:

```markdown
## Description
Describe your changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update

## Checklist
- [ ] Tests pass
- [ ] Code follows style guidelines
- [ ] Documentation updated
```

---

## Quick Reference Commands

```bash
# Check remote
git remote -v

# Push tags
git tag v1.0.0
git push origin v1.0.0

# Push all tags
git push --tags

# Create and push new branch
git checkout -b feature/new-feature
git push -u origin feature/new-feature

# Sync with upstream (if forked)
git fetch upstream
git merge upstream/main
```

---

## Checklist

- [ ] Repository created on GitHub
- [ ] Code pushed successfully
- [ ] Topics/tags configured
- [ ] First release created
- [ ] README displays correctly
- [ ] License is visible
- [ ] Issues enabled
- [ ] (Optional) GitHub Actions enabled
- [ ] (Optional) Branch protection configured

---

**Your repository is now ready for collaboration!** 🎉

Share your repository URL:
```
https://github.com/YOUR_USERNAME/voyager-evolved
```

Users can now install with:
```bash
git clone https://github.com/YOUR_USERNAME/voyager-evolved.git
cd voyager-evolved
pip install -e .
```
