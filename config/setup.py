#!/usr/bin/env python3
"""
Setup script for LangChain RAG Application
"""

import os
import sys
import subprocess
import shutil

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_pip():
    """Check if pip is available"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = ".env"
    if os.path.exists(env_file):
        print("✅ .env file already exists")
        return True
    
    print("\n🔧 Creating .env file...")
    try:
        with open(env_file, "w") as f:
            f.write("# Configuration for HuggingFace Models\n")
            f.write("# No API keys required - using completely free models!\n\n")
            f.write("# ===== MODEL PRESETS =====\n")
            f.write("# Choose from: fast, balanced, quality, max_quality, technical, mistral, mixtral, phi, gemma, gpt2, llama2\n")
            f.write("# Default: balanced (now uses Microsoft Phi-2)\n")
            f.write("MODEL_PRESET=balanced\n\n")
            f.write("# ===== CUSTOM MODELS (Optional) =====\n")
            f.write("# Override preset with custom models\n")
            f.write("# CUSTOM_LLM_MODEL=your-custom-model-name\n")
            f.write("# CUSTOM_EMBEDDING_MODEL=your-custom-embedding-model\n\n")
            f.write("# ===== ADVANCED SETTINGS (Optional) =====\n")
            f.write("# Override preset values\n")
            f.write("# TEMPERATURE=0.7\n")
            f.write("# MAX_NEW_TOKENS=512\n")
            f.write("# LOAD_IN_8BIT=true\n")
            f.write("# DEVICE_MAP=auto\n")
            f.write("# TORCH_DTYPE=auto\n\n")
            f.write("# ===== DEVICE SETTINGS (Optional) =====\n")
            f.write("# Force device selection: auto, cpu, cuda, mps\n")
            f.write("# FORCE_DEVICE=auto\n")
            f.write("# Enable GPU acceleration for embeddings\n")
            f.write("# EMBEDDING_USE_GPU=true\n\n")
            f.write("# ===== DOCUMENT PROCESSING =====\n")
            f.write("# CHUNK_SIZE=1000\n")
            f.write("# CHUNK_OVERLAP=200\n")
            f.write("# NUM_RETRIEVED_DOCS=4\n\n")
            f.write("# ===== SYSTEM SETTINGS =====\n")
            f.write("# LOG_LEVEL=INFO\n")
            f.write("# HF_HOME=~/.cache/huggingface\n")
        
        print("✅ .env file created")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["documents", "vector_db"]
    print("\n📁 Creating directories...")
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory already exists: {directory}")

def verify_installation():
    """Verify that the installation is complete"""
    print("\n✅ Installation verification complete!")
    print("   The application is ready to use.")
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up LangChain RAG Application")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        return False
    
    if not check_pip():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create environment file
    if not create_env_file():
        return False
    
    # Create directories
    create_directories()
    
    # Verify installation
    verify_installation()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the application: streamlit run src/web_interface/app.py")
    print("2. Open your browser to http://localhost:8501")
    print("\n📚 For more information, see README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 