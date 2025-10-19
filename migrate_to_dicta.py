#!/usr/bin/env python3
"""
Migration Script: mT5 → Dicta Hebrew Model
Removes old mT5 model and downloads new Dicta Hebrew model
"""

import shutil
import sys
from pathlib import Path

def main():
    print("🔄 Migrating from mT5 to Dicta Hebrew Model")
    print("=" * 50)
    
    # Remove old mT5 model
    mt5_path = Path("models/mt5-small")
    if mt5_path.exists():
        print("🗑️  Removing old mT5 model...")
        shutil.rmtree(mt5_path)
        print("✅ Old mT5 model removed")
    else:
        print("✅ No old mT5 model found")
    
    # Download new Dicta model
    print("\n📥 Downloading Dicta Hebrew model...")
    print("⏳ This may take 5-10 minutes (~2GB download)")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, 'config/scripts/preload_models.py'], 
                              check=True)
        print("✅ Dicta Hebrew model downloaded successfully!")
        
    except subprocess.CalledProcessError:
        print("❌ Failed to download Dicta model")
        print("💡 Try manually: python config/scripts/preload_models.py")
        return False
    
    print("\n🎉 Migration Complete!")
    print("🚀 Start with: python run.py")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)