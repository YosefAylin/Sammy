#!/usr/bin/env python3
"""
Universal Setup Script for Sammy AI
Handles cross-platform installation and setup
"""

import sys
import os
import platform
import subprocess
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

def get_pip_command(python_cmd):
    """Get the correct pip command for this platform."""
    pip_commands = [
        f'{python_cmd} -m pip',
        'pip',
        'pip3'
    ]
    
    for cmd in pip_commands:
        try:
            result = subprocess.run(cmd.split() + ['--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
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
        return False, None, None
    
    # Get Python version
    result = subprocess.run([python_cmd, '--version'], 
                          capture_output=True, text=True)
    version = result.stdout.strip()
    print(f"   Python: {version} ✅")
    
    # Check pip
    pip_cmd = get_pip_command(python_cmd)
    if not pip_cmd:
        print("❌ pip not found!")
        print(f"   Try: {python_cmd} -m ensurepip --upgrade")
        return False, None, None
    
    print(f"   pip: Available ✅")
    
    return True, python_cmd, pip_cmd

def install_dependencies(pip_cmd):
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")
    
    requirements_file = Path("backend/requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    try:
        # Install requirements
        cmd = pip_cmd.split() + ['install', '-r', str(requirements_file)]
        result = subprocess.run(cmd, check=True)
        
        # Install additional packages
        additional = ['protobuf', 'sentencepiece']
        for package in additional:
            cmd = pip_cmd.split() + ['install', package]
            subprocess.run(cmd, check=True)
        
        print("✅ All dependencies installed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("\n💡 Try manual installation:")
        print(f"   {pip_cmd} install -r backend/requirements.txt")
        print(f"   {pip_cmd} install protobuf sentencepiece")
        return False

def download_models(python_cmd):
    """Download AI models."""
    print("\n🤖 Downloading AI models...")
    
    try:
        result = subprocess.run([python_cmd, 'preload_models.py'], 
                              check=True)
        print("✅ Models downloaded successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download models: {e}")
        print("\n💡 Try manual download:")
        print(f"   {python_cmd} preload_models.py")
        return False

def test_installation(python_cmd):
    """Test the installation."""
    print("\n🧪 Testing installation...")
    
    # Test imports
    test_script = '''
import torch
import transformers
import flask
import numpy
import sklearn
print("✅ All packages imported successfully!")
'''
    
    try:
        result = subprocess.run([python_cmd, '-c', test_script], 
                              check=True, capture_output=True, text=True)
        print(result.stdout.strip())
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Import test failed: {e}")
        return False

def show_next_steps(python_cmd):
    """Show next steps to user."""
    print("\n" + "="*60)
    print("🎉 SAMMY AI SETUP COMPLETE!")
    print("="*60)
    
    print(f"\n🚀 Start the system:")
    print(f"   {python_cmd} start_sammy.py")
    
    print(f"\n🌐 Install Chrome Extension:")
    print(f"   1. Open Chrome → chrome://extensions/")
    print(f"   2. Enable 'Developer mode'")
    print(f"   3. Click 'Load unpacked'")
    print(f"   4. Select the 'frontend' folder")
    
    print(f"\n🧪 Test on Hebrew websites:")
    print(f"   • Open any Hebrew website")
    print(f"   • Click Sammy extension icon")
    print(f"   • Choose summarization method")
    print(f"   • Click 'סכם דף' and enjoy!")
    
    print(f"\n📊 System Info:")
    print(f"   • Python: {python_cmd}")
    print(f"   • Models: ./models/ (~1.6GB)")
    print(f"   • Server: http://localhost:5002")
    
    print(f"\n💡 Need help? Check CROSS_PLATFORM_GUIDE.md")

def main():
    """Main setup function."""
    print("🤖 SAMMY AI - Universal Setup")
    print("=" * 50)
    print(f"Platform: {platform.system()} {platform.machine()}")
    print()
    
    # Check system
    compatible, python_cmd, pip_cmd = check_system()
    if not compatible:
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies(pip_cmd):
        print("\n⚠️  Continuing with model download...")
        print("   You may need to install dependencies manually")
    
    # Download models
    if not download_models(python_cmd):
        print("\n⚠️  You can download models later with:")
        print(f"   {python_cmd} preload_models.py")
    
    # Test installation
    if test_installation(python_cmd):
        show_next_steps(python_cmd)
    else:
        print("\n⚠️  Installation may have issues")
        print("   Check CROSS_PLATFORM_GUIDE.md for troubleshooting")

if __name__ == "__main__":
    main()