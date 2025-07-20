#!/usr/bin/env python3
"""
Configuration utility for LangChain RAG Application
Shows current configuration and available options
"""

import os
import sys
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.config import Config

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_section(title: str):
    """Print a formatted section"""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")

def show_current_config():
    """Show current configuration"""
    print_header("CURRENT CONFIGURATION")
    
    try:
        config = Config()
        config.print_current_config()
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False
    
    return True

def show_available_presets():
    """Show all available model presets"""
    print_header("AVAILABLE MODEL PRESETS")
    
    presets = Config.get_available_presets()
    
    for preset_name, preset_info in presets.items():
        print(f"\n🎯 **{preset_name.upper()}**")
        print(f"   Description: {preset_info['description']}")
        print(f"   LLM Model: {preset_info['llm_model']}")
        print(f"   Embedding Model: {preset_info['embedding_model']}")
        print(f"   Memory Usage: {preset_info['memory_usage']}")
        print(f"   Speed: {preset_info['speed']}")
        
        # Show if this is the current preset
        if preset_name == Config.CURRENT_PRESET:
            print(f"   ✅ CURRENT SELECTION")

def show_usage_examples():
    """Show usage examples"""
    print_header("USAGE EXAMPLES")
    
    print("\n🔧 **Change Model Preset**")
    print("   Set in .env file:")
    print("   MODEL_PRESET=fast          # Microsoft Phi-2 (fast, 3GB RAM)")
    print("   MODEL_PRESET=balanced      # Microsoft Phi-2 (default, 3GB RAM)")
    print("   MODEL_PRESET=quality       # Mistral 7B (excellent, 8GB RAM)")
    print("   MODEL_PRESET=max_quality   # Mixtral 8x7B (best, 16GB RAM)")
    print("   MODEL_PRESET=technical     # Code Llama 7B (coding/docs, 8GB RAM)")
    print("   MODEL_PRESET=mistral       # Mistral 7B (alternative)")
    print("   MODEL_PRESET=mixtral       # Mixtral 8x7B (alternative)")
    print("   MODEL_PRESET=phi           # Microsoft Phi-2 (alternative)")
    print("   MODEL_PRESET=gemma         # Google Gemma 7B (8GB RAM)")
    
    print("\n🔧 **Use Custom Models**")
    print("   Set in .env file:")
    print("   CUSTOM_LLM_MODEL=your-custom-model")
    print("   CUSTOM_EMBEDDING_MODEL=your-custom-embedding")
    
    print("\n🔧 **Override Advanced Settings**")
    print("   Set in .env file:")
    print("   TEMPERATURE=0.5")
    print("   MAX_NEW_TOKENS=256")
    print("   LOAD_IN_8BIT=false")
    print("   DEVICE_MAP=cpu")
    
    print("\n🔧 **Document Processing Settings**")
    print("   Set in .env file:")
    print("   CHUNK_SIZE=500")
    print("   CHUNK_OVERLAP=100")
    print("   NUM_RETRIEVED_DOCS=6")

def show_environment_variables():
    """Show all available environment variables"""
    print_header("ENVIRONMENT VARIABLES")
    
    env_vars = {
        "MODEL_PRESET": "Choose model preset (fast, balanced, quality, max_quality, technical, mistral, mixtral, phi, gemma, gpt2, llama2)",
        "CUSTOM_LLM_MODEL": "Override with custom LLM model name",
        "CUSTOM_EMBEDDING_MODEL": "Override with custom embedding model name",
        "TEMPERATURE": "Control response randomness (0.0-1.0)",
        "MAX_NEW_TOKENS": "Maximum tokens in response",
        "LOAD_IN_8BIT": "Use 8-bit quantization (true/false)",
        "DEVICE_MAP": "Device mapping (auto, cpu, cuda)",
        "TORCH_DTYPE": "Torch data type (auto, float16, float32)",
        "CHUNK_SIZE": "Document chunk size for processing",
        "CHUNK_OVERLAP": "Overlap between chunks",
        "NUM_RETRIEVED_DOCS": "Number of documents to retrieve",
        "VECTOR_DB_PATH": "Path to vector database",
        "DOCUMENTS_PATH": "Path to documents directory",
        "LOG_LEVEL": "Logging level (DEBUG, INFO, WARNING, ERROR)",
        "HF_HOME": "HuggingFace cache directory"
    }
    
    for var, description in env_vars.items():
        current_value = os.getenv(var, "Not set")
        print(f"\n📝 {var}")
        print(f"   Description: {description}")
        print(f"   Current Value: {current_value}")

def main():
    """Main function"""
    print("🤖 LangChain RAG Application - Configuration Utility")
    
    # Show current configuration
    if not show_current_config():
        return
    
    # Show available presets
    show_available_presets()
    
    # Show usage examples
    show_usage_examples()
    
    # Show environment variables
    show_environment_variables()
    
    print_header("QUICK START")
    print("\n🚀 To change your configuration:")
    print("1. Edit the .env file in your project root")
    print("2. Set MODEL_PRESET=fast for quick testing")
    print("3. Set MODEL_PRESET=quality for better responses")
    print("4. Restart the application")
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main() 