#!/usr/bin/env python3
"""
Preload AI Models for Sammy
Downloads models to project directory for Git-friendly distribution
"""

import sys
import os
from pathlib import Path

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent / 'backend'))

from model_manager import ModelManager
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main preloading function."""
    print("🤖 Preloading AI Models for Sammy")
    print("=" * 50)
    print("📁 Models will be downloaded to: ./models/")
    print("🎯 Git-friendly: Each user downloads individually")
    print()
    
    # Initialize model manager
    manager = ModelManager()
    
    # Show current status
    print("📊 Current Model Status:")
    print("-" * 30)
    for model_key, status in manager.list_models().items():
        downloaded = "✅" if status['downloaded'] else "❌"
        print(f"{downloaded} {model_key}: {status['size']}")
    print()
    
    # Download all models
    print("📥 Starting model downloads...")
    start_time = time.time()
    
    try:
        success = manager.download_all_models()
        
        if success:
            total_time = time.time() - start_time
            print("\n" + "=" * 50)
            print("🎉 All models downloaded successfully!")
            print(f"⏱️  Total time: {total_time:.1f} seconds")
            
            # Show final status
            print("\n📊 Final Model Status:")
            print("-" * 30)
            for model_key, status in manager.list_models().items():
                print(f"✅ {model_key}: {status['size']} -> {status['path']}")
            
            print("\n💡 Next steps:")
            print("1. Start AI server: python3 backend/ai_server.py")
            print("2. Install Chrome extension from 'frontend' folder")
            print("3. Test on Hebrew websites!")
            
            print("\n📁 Project structure:")
            print("├── models/")
            print("│   ├── alephbert-base/  (Hebrew understanding)")
            print("│   └── mt5-small/       (Text generation)")
            print("└── backend/ai_server.py (AI engine)")
            
        else:
            print("\n❌ Some models failed to download")
            print("🔧 Check your internet connection and try again")
            
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("🔧 Check your internet connection and try again")

if __name__ == "__main__":
    main()