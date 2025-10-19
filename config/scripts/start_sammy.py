#!/usr/bin/env python3
"""
Sammy AI Hebrew Summarizer - Startup Script
One-command startup for the complete AI system
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'torch', 'transformers', 'flask', 'flask_cors', 
        'numpy', 'sklearn', 'networkx'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r backend/requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True

def preload_models():
    """Preload AI models if not already cached."""
    print("🤖 Checking AI models...")
    
    try:
        # Quick test to see if models are cached
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('onlplab/alephbert-base')
        print("✅ AlephBERT model cached")
        
        tokenizer = AutoTokenizer.from_pretrained('Dicta-IL/dictalm2.0')
        print("✅ Dicta Hebrew model cached")
        
        return True
        
    except Exception as e:
        print("📥 Models not cached, downloading...")
        result = subprocess.run([sys.executable, 'preload_models.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Models downloaded successfully")
            return True
        else:
            print(f"❌ Model download failed: {result.stderr}")
            return False

def start_ai_server():
    """Start the AI server."""
    print("🚀 Starting AI server...")
    
    # Start server in background
    process = subprocess.Popen([
        sys.executable, 'backend/ai_server.py'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get('http://localhost:5002/health', timeout=1)
            if response.status_code == 200:
                print("✅ AI server started successfully")
                print("📍 Server running on: http://localhost:5002")
                return process
        except:
            pass
        
        time.sleep(1)
        print(f"⏳ Waiting for server... ({i+1}/30)")
    
    print("❌ Server failed to start")
    process.terminate()
    return None

def load_models_in_server():
    """Preload models in the running server."""
    print("🧠 Loading AI models in server...")
    
    try:
        response = requests.post('http://localhost:5002/load_models', timeout=60)
        if response.status_code == 200:
            print("✅ AI models loaded in server")
            return True
        else:
            print(f"❌ Failed to load models: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def test_summarization():
    """Test both summarization methods."""
    print("🧪 Testing summarization...")
    
    test_text = """
    בינה מלאכותית היא תחום מדעי המתמחה בפיתוח מערכות מחשב המסוגלות לבצע משימות הדורשות בדרך כלל אינטליגנציה אנושית. 
    התחום כולל למידת מכונה, עיבוד שפה טבעית, ראייה ממוחשבת ועוד. בשנים האחרונות חלה התפתחות מהירה בתחום, 
    במיוחד בזכות רשתות נוירונים עמוקות. טכנולוגיות אלו משנות את פני התעשייה והחברה.
    """
    
    # Test extractive
    try:
        response = requests.post('http://localhost:5002/summarize', 
                               json={'text': test_text, 'method': 'extractive'},
                               timeout=10)
        if response.status_code == 200:
            print("✅ Extractive summarization working")
        else:
            print("❌ Extractive summarization failed")
    except Exception as e:
        print(f"❌ Extractive test failed: {e}")
    
    # Test abstractive
    try:
        response = requests.post('http://localhost:5002/summarize', 
                               json={'text': test_text, 'method': 'abstractive'},
                               timeout=30)
        if response.status_code == 200:
            print("✅ Abstractive summarization working")
        else:
            print("❌ Abstractive summarization failed")
    except Exception as e:
        print(f"❌ Abstractive test failed: {e}")

def show_extension_instructions():
    """Show Chrome extension installation instructions."""
    print("\n" + "="*60)
    print("🎉 SAMMY AI IS READY!")
    print("="*60)
    print("\n📱 Install Chrome Extension:")
    print("1. Open Chrome and go to: chrome://extensions/")
    print("2. Enable 'Developer mode' (toggle in top right)")
    print("3. Click 'Load unpacked'")
    print("4. Select the 'frontend' folder from this project")
    print("5. Extension installed! Look for Sammy icon in toolbar")
    
    print("\n🧪 Test the Extension:")
    print("1. Open test_extension.html in Chrome")
    print("2. Click the Sammy extension icon")
    print("3. Choose summarization method:")
    print("   • חילוץ חכם (Extractive) - Fast, reliable")
    print("   • יצירה חדשה (Abstractive) - Creative, slower")
    print("4. Click 'סכם דף' and see the magic!")
    
    print("\n🌐 Use on Real Websites:")
    print("Visit any Hebrew website (Ynet, Haaretz, etc.) and summarize!")
    
    print(f"\n🔧 Server Status:")
    print(f"• AI Server: http://localhost:5002")
    print(f"• Models: AlephBERT + mT5 loaded")
    print(f"• Methods: Extractive & Abstractive ready")
    
    print(f"\n⏹️  To stop: Press Ctrl+C")

def main():
    """Main startup function."""
    print("🤖 SAMMY AI HEBREW SUMMARIZER")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Preload models
    if not preload_models():
        return
    
    # Start server
    server_process = start_ai_server()
    if not server_process:
        return
    
    # Load models in server
    if not load_models_in_server():
        server_process.terminate()
        return
    
    # Test functionality
    test_summarization()
    
    # Show instructions
    show_extension_instructions()
    
    # Keep server running
    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Sammy AI...")
        server_process.terminate()
        print("✅ Server stopped")

if __name__ == "__main__":
    main()