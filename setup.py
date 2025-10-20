#!/usr/bin/env python3
"""
Sammy AI - Setup Script
One-command setup for the AI-powered Hebrew summarizer
"""

import subprocess
import sys
import platform
from pathlib import Path

def get_python_command():
    """Get the correct Python command for this platform."""
    commands = ['python', 'python3', 'py']
    
    for cmd in commands:
        try:
            result = subprocess.run([cmd, '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0 and 'Python 3.' in result.stdout:
                version = result.stdout.strip().split()[1]
                major, minor = version.split('.')[:2]
                if int(major) >= 3 and int(minor) >= 8:
                    return cmd
        except FileNotFoundError:
            continue
    
    return None

def check_system():
    """Check system compatibility."""
    print("🔍 Checking system compatibility...")
    
    # Check OS
    os_name = platform.system()
    print(f"   OS: {os_name} {platform.release()}")
    
    # Check Python
    python_cmd = get_python_command()
    if not python_cmd:
        print("❌ Python 3.8+ not found!")
        print("   Please install Python from python.org")
        return False, None
    
    # Get Python version
    result = subprocess.run([python_cmd, '--version'], 
                          capture_output=True, text=True)
    version = result.stdout.strip()
    print(f"   Python: {version} ✅")
    
    return True, python_cmd

def install_dependencies(python_cmd):
    """Install Python dependencies."""
    print("\n📦 Installing AI dependencies...")
    
    project_root = Path(__file__).parent
    requirements_file = project_root / "backend" / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    try:
        # Install requirements
        pip_cmd = f'{python_cmd} -m pip'
        cmd = pip_cmd.split() + ['install', '-r', str(requirements_file)]
        result = subprocess.run(cmd, check=True)
        
        # Install additional AI packages
        additional = ['protobuf', 'sentencepiece']
        for package in additional:
            cmd = pip_cmd.split() + ['install', package]
            subprocess.run(cmd, check=True)
        
        print("✅ All AI dependencies installed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("\n💡 Try manual installation:")
        print(f"   {python_cmd} -m pip install -r backend/requirements.txt")
        return False

def download_ai_models(python_cmd):
    """Download AI models using model manager."""
    print("\n🤖 Downloading AI models...")
    
    project_root = Path(__file__).parent
    model_manager = project_root / "backend" / "model_manager.py"
    
    if not model_manager.exists():
        print("❌ Model manager not found!")
        return False
    
    try:
        # Download models using model manager
        result = subprocess.run([python_cmd, str(model_manager), 'download'], 
                              cwd=project_root, check=True)
        print("✅ AI models downloaded successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download models: {e}")
        print("\n💡 Try manual download:")
        print(f"   {python_cmd} backend/model_manager.py download")
        return False

def test_ai_installation(python_cmd):
    """Test the AI installation."""
    print("\n🧪 Testing AI installation...")
    
    # Test AI imports
    test_script = '''
import torch
import transformers
from transformers import AutoTokenizer, AutoModel
import flask
import flask_cors
import numpy
import sklearn
import networkx
print("✅ All AI packages imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
'''
    
    try:
        result = subprocess.run([python_cmd, '-c', test_script], 
                              check=True, capture_output=True, text=True)
        print(result.stdout.strip())
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ AI import test failed: {e}")
        return False

def show_completion_message(python_cmd):
    """Show completion message with next steps."""
    print("\n" + "="*60)
    print("🎉 SAMMY AI SETUP COMPLETE!")
    print("="*60)
    
    print(f"\n🚀 Start the AI system:")
    print(f"   {python_cmd} run.py")
    
    print(f"\n🌐 Install Chrome Extension:")
    print(f"   1. Open Chrome → chrome://extensions/")
    print(f"   2. Enable 'Developer mode'")
    print(f"   3. Click 'Load unpacked'")
    print(f"   4. Select the 'frontend' folder")
    
    print(f"\n🧪 Test on Hebrew websites:")
    print(f"   • Visit Ynet, Haaretz, or any Hebrew site")
    print(f"   • Click Sammy extension icon")
    print(f"   • Choose summarization method:")
    print(f"     - חילוץ חכם (Extractive) - Fast & reliable")
    print(f"     - יצירה חדשה (Abstractive) - Creative & slower")
    print(f"   • Click 'סכם דף' and enjoy AI-powered summaries!")
    
    print(f"\n📊 System Info:")
    print(f"   • Python: {python_cmd}")
    print(f"   • AI Models: AlephBERT Hebrew NLP (~1.6GB)")
    print(f"   • Server: http://localhost:5002")
    print(f"   • Methods: Extractive & Abstractive")
    
    print(f"\n💡 Need help? Check README.md or config/README.md")

def main():
    """Main setup function."""
    print("🤖 SAMMY AI - AI-Powered Hebrew Summarizer Setup")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.machine()}")
    print()
    
    # Check system
    compatible, python_cmd = check_system()
    if not compatible:
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies(python_cmd):
        print("\n⚠️  Continuing with model download...")
        print("   You may need to install dependencies manually")
    
    # Download AI models
    if not download_ai_models(python_cmd):
        print("\n⚠️  You can download models later with:")
        print(f"   {python_cmd} backend/model_manager.py download")
    
    # Test installation
    if test_ai_installation(python_cmd):
        show_completion_message(python_cmd)
    else:
        print("\n⚠️  Installation may have issues")
        print("   Check dependencies and try again")

if __name__ == "__main__":
    main()