#!/usr/bin/env python3
"""
Sammy AI - Run Script
One-command startup for the AI-powered Hebrew summarizer
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def check_models():
    """Check if AI models are available."""
    print("🤖 Checking AI models...")
    
    try:
        # Check if model_manager exists and models are available
        project_root = Path(__file__).parent
        model_manager = project_root / "backend" / "model_manager.py"
        
        if not model_manager.exists():
            print("❌ Model manager not found!")
            return False
        
        # Check model status
        result = subprocess.run([sys.executable, str(model_manager), 'list'], 
                              capture_output=True, text=True, cwd=project_root)
        
        if "✅ alephbert" in result.stdout:
            print("✅ AlephBERT model available")
            return True
        else:
            print("⚠️  AlephBERT model not found")
            print("💡 Run setup first: python setup.py")
            return False
            
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False

def start_ai_server():
    """Start the AI server directly."""
    print("🚀 Starting AI-powered Sammy server...")
    
    project_root = Path(__file__).parent
    server_script = project_root / "backend" / "ai_server.py"
    
    if not server_script.exists():
        print("❌ AI server script not found!")
        print(f"Expected: {server_script}")
        return False
    
    try:
        # Start server
        print("📍 Server starting on: http://localhost:5002")
        print("🧠 Models: AlephBERT Hebrew NLP")
        print("⚡ Methods: Extractive & Abstractive")
        print("⏹️  Press Ctrl+C to stop")
        print("-" * 50)
        
        result = subprocess.run([sys.executable, str(server_script)], 
                              cwd=project_root)
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print("\n🛑 Sammy AI stopped by user")
        return True
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        print("\n💡 Try running setup first:")
        print("   python setup.py")
        return False

def show_usage_instructions():
    """Show usage instructions."""
    print("\n" + "="*60)
    print("🎉 SAMMY AI SERVER READY!")
    print("="*60)
    print("\n📱 Install Chrome Extension:")
    print("1. Open Chrome → chrome://extensions/")
    print("2. Enable 'Developer mode'")
    print("3. Click 'Load unpacked'")
    print("4. Select the 'frontend' folder")
    
    print("\n🧪 Test the Extension:")
    print("1. Visit any Hebrew website")
    print("2. Click Sammy extension icon")
    print("3. Choose method:")
    print("   • חילוץ חכם (Extractive) - Fast, reliable")
    print("   • יצירה חדשה (Abstractive) - Creative, slower")
    print("4. Click 'סכם דף' and enjoy!")
    
    print("\n🔧 API Endpoints:")
    print("• Health: GET http://localhost:5002/health")
    print("• Summarize: POST http://localhost:5002/summarize")
    print("• Models: GET http://localhost:5002/models/status")

def main():
    """Run the Sammy AI application."""
    print("🤖 SAMMY AI - Hebrew Summarizer")
    print("=" * 50)
    
    # Check if models are available
    if not check_models():
        print("\n💡 Run setup first to download models:")
        print("   python setup.py")
        return False
    
    # Show usage instructions
    show_usage_instructions()
    
    # Start the AI server
    return start_ai_server()

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)