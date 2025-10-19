#!/usr/bin/env python3
"""
Virtual Environment Wrapper for Sammy AI
Ensures virtual environment is activated before running
"""

import os
import sys
import subprocess
from pathlib import Path

def check_virtual_env():
    """Check if we're in a virtual environment."""
    return hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

def activate_and_run():
    """Activate virtual environment and run start_sammy.py."""
    project_root = Path(__file__).parent
    venv_path = project_root / "venv"
    
    if not venv_path.exists():
        print("❌ Virtual environment not found!")
        print("Run setup first:")
        print("  python3 -m venv venv")
        print("  source venv/bin/activate")
        print("  pip install -r backend/requirements.txt")
        return False
    
    # Determine activation script path
    if os.name == 'nt':  # Windows
        activate_script = venv_path / "Scripts" / "activate"
        python_exe = venv_path / "Scripts" / "python.exe"
    else:  # Unix/Linux/macOS
        activate_script = venv_path / "bin" / "activate"
        python_exe = venv_path / "bin" / "python"
    
    if not python_exe.exists():
        print(f"❌ Python executable not found: {python_exe}")
        return False
    
    print("🔄 Using virtual environment...")
    print(f"📍 Python: {python_exe}")
    
    # Run start_sammy.py with virtual environment Python
    try:
        result = subprocess.run([str(python_exe), "start_sammy.py"], 
                              cwd=project_root)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return True
    except Exception as e:
        print(f"❌ Error running Sammy AI: {e}")
        return False

def main():
    """Main function."""
    if check_virtual_env():
        print("✅ Virtual environment already active")
        # Import and run the original start_sammy
        try:
            import start_sammy
            start_sammy.main()
        except ImportError:
            print("❌ Could not import start_sammy module")
            return False
    else:
        print("🔄 Activating virtual environment...")
        return activate_and_run()

if __name__ == "__main__":
    main()