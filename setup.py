#!/usr/bin/env python3
"""
Sammy AI - Setup Script
One-command setup for the entire project
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the complete setup process."""
    print("🤖 Sammy AI - Setup Starting...")
    print("=" * 40)
    
    # Get the project root directory
    project_root = Path(__file__).parent
    setup_script = project_root / "config" / "scripts" / "setup.py"
    
    if not setup_script.exists():
        print("❌ Setup script not found!")
        print(f"Expected: {setup_script}")
        return False
    
    try:
        # Run the actual setup script
        result = subprocess.run([sys.executable, str(setup_script)], 
                              cwd=project_root)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)