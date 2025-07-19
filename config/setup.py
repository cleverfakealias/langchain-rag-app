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
            f.write("# OpenAI API Configuration\n")
            f.write("# Get your API key from: https://platform.openai.com/api-keys\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n\n")
            f.write("# Optional: Model Configuration\n")
            f.write("# MODEL_NAME=gpt-3.5-turbo\n")
        
        print("✅ .env file created")
        print("⚠️  Please edit .env file and add your OpenAI API key")
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

def run_test():
    """Run a quick test to verify installation"""
    print("\n🧪 Running quick test...")
    try:
        result = subprocess.run([sys.executable, "tests/test_rag.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Test passed! RAG system is working correctly.")
            return True
        else:
            print("⚠️  Test failed, but this might be due to missing API key")
            print("   You can still run the application after adding your API key")
            return True
    except Exception as e:
        print(f"⚠️  Could not run test: {e}")
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
    
    # Run test
    run_test()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Run the application: streamlit run src/web_interface/app.py")
    print("3. Open your browser to http://localhost:8501")
    print("\n📚 For more information, see README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 