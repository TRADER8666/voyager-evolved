#!/usr/bin/env python3
"""Basic tests for Voyager Evolved."""

import pytest


def test_import_voyager():
    """Test that voyager module can be imported."""
    from voyager import Voyager
    assert Voyager is not None


def test_import_evolved():
    """Test that evolved module can be imported."""
    from voyager.evolved import (
        VoyagerEvolved,
        EvolvedConfig,
        PlayerObserver,
        EvolutionaryGoalSystem,
        HumanBehaviorSystem,
        PersonalityEngine,
    )
    assert VoyagerEvolved is not None
    assert EvolvedConfig is not None


def test_config_creation():
    """Test EvolvedConfig creation."""
    from voyager.evolved import EvolvedConfig
    
    config = EvolvedConfig(
        openai_api_key="test-key",
        mc_port=25565,
    )
    assert config.openai_api_key == "test-key"
    assert config.mc_port == 25565


class TestPersonalityEngine:
    """Tests for PersonalityEngine."""
    
    def test_creation(self):
        """Test PersonalityEngine can be created."""
        from voyager.evolved import PersonalityEngine
        
        personality = PersonalityEngine(
            curiosity=0.8,
            caution=0.5,
        )
        assert personality is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
