# Contributing to Voyager Evolved

Thank you for your interest in contributing to Voyager Evolved! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please:

- Be respectful of differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 16+
- Git
- A GitHub account

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/voyager-evolved.git
cd voyager-evolved
```

3. Add the upstream remote:

```bash
git remote add upstream https://github.com/ORIGINAL_OWNER/voyager-evolved.git
```

---

## Development Setup

### Create a Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### Verify Setup

```bash
# Run tests
pytest

# Check code style
black --check .
flake8 .
```

---

## Making Changes

### Create a Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
```

### Branch Naming Conventions

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring
- `test/description` - Test additions/changes

### Commit Messages

Write clear, concise commit messages:

```
type: short description

Longer description if needed. Explain what and why,
not how (the code shows how).

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

---

## Submitting Changes

### Before Submitting

1. **Update your branch** with the latest upstream changes:

```bash
git fetch upstream
git rebase upstream/main
```

2. **Run tests** and ensure they pass:

```bash
pytest
```

3. **Check code style**:

```bash
black .
flake8 .
```

4. **Update documentation** if needed

### Create a Pull Request

1. Push your branch:

```bash
git push origin feature/your-feature-name
```

2. Go to GitHub and create a Pull Request

3. Fill out the PR template:
   - Clear description of changes
   - Link to related issues
   - Screenshots if applicable

### PR Checklist

- [ ] Code follows the project style guidelines
- [ ] Tests added for new functionality
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

---

## Code Style

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with these additions:

- **Line length**: 100 characters max
- **Formatter**: Use `black` for formatting
- **Import order**: Use `isort`

```python
# Good
def calculate_distance(point_a: tuple, point_b: tuple) -> float:
    """Calculate Euclidean distance between two points.
    
    Args:
        point_a: First point (x, y, z)
        point_b: Second point (x, y, z)
        
    Returns:
        Distance between the points
    """
    ...
```

### Type Hints

Use type hints for function arguments and return values:

```python
from typing import Optional, List, Dict

def process_data(items: List[str], config: Optional[Dict] = None) -> bool:
    ...
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=voyager

# Run specific test file
pytest tests/test_evolved.py

# Run specific test
pytest tests/test_evolved.py::test_player_observer
```

### Writing Tests

Place tests in the `tests/` directory:

```python
# tests/test_example.py
import pytest
from voyager.evolved import PlayerObserver


class TestPlayerObserver:
    def test_initialization(self):
        observer = PlayerObserver(radius=50)
        assert observer.observation_radius == 50
    
    def test_invalid_radius(self):
        with pytest.raises(ValueError):
            PlayerObserver(radius=-1)
```

---

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of function.
    
    Longer description if needed. Explain what the function
    does and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something is wrong
        
    Example:
        >>> example_function("test", 42)
        True
    """
    ...
```

### README Updates

When adding features, update the README.md:

- Add to feature list if significant
- Update configuration options
- Add usage examples

---

## Project Areas

### Areas for Contribution

1. **Core Features**
   - Improve agent decision making
   - Add new skill primitives
   - Enhance error handling

2. **Evolved Features**
   - Player observation improvements
   - New personality traits
   - Better goal evolution algorithms

3. **Documentation**
   - Improve tutorials
   - Add examples
   - Fix typos

4. **Testing**
   - Increase test coverage
   - Add integration tests

5. **Performance**
   - Optimize API calls
   - Reduce memory usage

---

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/YOUR_USERNAME/voyager-evolved/discussions)
- **Bugs**: Open an [Issue](https://github.com/YOUR_USERNAME/voyager-evolved/issues)
- **Security**: Email security concerns privately

---

## Recognition

Contributors will be recognized in:

- [CONTRIBUTORS.md](CONTRIBUTORS.md) file
- Release notes
- Project documentation

Thank you for contributing to Voyager Evolved! 🚀
