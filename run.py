#!/usr/bin/env python3
"""
Sammy AI - Run Script
One-command startup for the application
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Sammy AI application."""
    print("🚀 Starting Sammy AI...")
    
    # Get the project root directory
    project_root = Path(__file__).parent
    run_script = project_root / "config" / "scripts" / "run_sammy.py"
    
    if not run_script.exists():
        print("❌ Run script not found!")
        print(f"Expected: {run_script}")
        print("\n💡 Try running setup first:")
        print("   python setup.py")
        return False
    
    try:
        # Run the actual application
        result = subprocess.run([sys.executable, str(run_script)], 
                              cwd=project_root)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n🛑 Sammy AI stopped by user")
        return True
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)