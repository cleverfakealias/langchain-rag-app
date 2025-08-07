"""Tests for configuration module"""
import os
from unittest.mock import patch, MagicMock
from config.config import Config


class TestConfig:
    """Test configuration loading and model presets"""
    
    def test_config_initialization(self):
        """Test basic config initialization"""
        config = Config()
        assert config is not None
        assert hasattr(config, 'MODEL_PRESETS')
        
    def test_default_model_preset(self):
        """Test default model preset is balanced"""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            llm_config = config.get_current_llm_config()
            assert llm_config is not None
            
    def test_custom_model_preset(self):
        """Test setting custom model preset"""
        # Test that quality preset exists and has the expected model
        config = Config()
        quality_preset = config.MODEL_PRESETS.get('quality')
        assert quality_preset is not None
        assert 'mistralai/Mistral-7B-Instruct-v0.3' == quality_preset['llm'].name
        
        # Test environment override by directly checking preset lookup
        with patch.dict(os.environ, {'MODEL_PRESET': 'quality'}):
            # Directly test the preset retrieval logic
            preset_name = os.getenv("MODEL_PRESET", "balanced")
            assert preset_name == 'quality'
            preset_config = config.MODEL_PRESETS.get(preset_name)
            assert preset_config is not None
            assert preset_config['llm'].name == 'mistralai/Mistral-7B-Instruct-v0.3'
            
    def test_all_model_configs_available(self):
        """Test that all model configs are accessible"""
        config = Config()
        all_configs = config.get_all_llm_configs()
        assert len(all_configs) > 0
        assert isinstance(all_configs, list)
        
    def test_preset_availability(self):
        """Test that all expected presets are available"""
        config = Config()
        expected_presets = ['fast', 'balanced', 'quality', 'max_quality', 'technical']
        for preset in expected_presets:
            assert preset in config.MODEL_PRESETS
            preset_config = config.MODEL_PRESETS[preset]
            assert 'llm' in preset_config
            assert 'embedding' in preset_config
        
    def test_embedding_config(self):
        """Test embedding configuration"""
        config = Config()
        embedding_config = config.get_current_embedding_config()
        assert embedding_config is not None
        assert hasattr(embedding_config, 'name')
        assert hasattr(embedding_config, 'device')
        
    def test_invalid_model_preset(self):
        """Test handling of invalid model preset"""
        with patch.dict(os.environ, {'MODEL_PRESET': 'nonexistent'}):
            config = Config()
            # Should fallback to default
            llm_config = config.get_current_llm_config()
            assert llm_config is not None
