#!/usr/bin/env python3
"""Voyager Evolved - Enhanced Minecraft AI Agent with Human-like Behaviors.

This package provides an advanced AI agent for Minecraft that combines
LLM-powered decision making with evolutionary goals, player observation,
and emergent personality traits.
"""

import os
import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8")

def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements_file = HERE / "requirements.txt"
    if requirements_file.exists():
        with requirements_file.open(encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="voyager-evolved",
    version="1.0.0",
    author="Voyager Evolved Contributors",
    author_email="voyager@example.com",
    description="An enhanced Minecraft AI agent with evolutionary goals and human-like behaviors",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/voyager-evolved",
    project_urls={
        "Bug Tracker": "https://github.com/YOUR_USERNAME/voyager-evolved/issues",
        "Documentation": "https://github.com/YOUR_USERNAME/voyager-evolved#readme",
        "Source Code": "https://github.com/YOUR_USERNAME/voyager-evolved",
    },
    keywords=[
        "minecraft",
        "ai-agent",
        "llm",
        "openai",
        "gpt-4",
        "reinforcement-learning",
        "embodied-agents",
        "lifelong-learning",
        "open-ended-learning",
    ],
    license="MIT",
    packages=find_packages(include=["voyager", "voyager.*"]),
    include_package_data=True,
    package_data={
        "voyager": [
            "control_primitives/*.js",
            "control_primitives_context/*.js",
            "prompts/*.txt",
        ],
    },
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "voyager-evolved=voyager.evolved.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment",
    ],
    zip_safe=False,
)
