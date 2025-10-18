# Virtual Environment Setup Guide

This project now uses Python virtual environments for better dependency management and cross-platform compatibility.

## Quick Start

### 1. Activate Virtual Environment
```bash
# Option A: Use the activation script
./activate_env.sh

# Option B: Manual activation
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

### 2. Run Sammy AI
```bash
# Option A: Use the wrapper (automatically handles venv)
python run_sammy.py

# Option B: Manual (after activating venv)
python start_sammy.py
```

## What's Changed

### ✅ Benefits of Virtual Environment
- **Isolated dependencies** - No conflicts between projects
- **Reproducible builds** - Same environment everywhere
- **Easy cleanup** - Delete `venv/` folder to remove everything
- **Cross-platform** - Works identically on Windows, macOS, Linux
- **Version control friendly** - Only track requirements, not packages

### 📁 New Files
- `venv/` - Virtual environment (excluded from git)
- `activate_env.sh` - Easy activation script
- `run_sammy.py` - Wrapper that auto-activates venv
- `cleanup_global_packages.py` - Remove global packages safely

## Commands Reference

### Virtual Environment Management
```bash
# Create virtual environment (already done)
python3 -m venv venv

# Activate
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate          # Windows

# Deactivate
deactivate

# Check if active
which python                   # Should show venv/bin/python
```

### Package Management
```bash
# Install project dependencies (in venv)
pip install -r backend/requirements.txt

# Add new package
pip install package_name
pip freeze > backend/requirements.txt  # Update requirements

# List installed packages
pip list
```

### Running the Project
```bash
# Method 1: Auto-activation wrapper
python run_sammy.py

# Method 2: Manual activation
source venv/bin/activate
python start_sammy.py

# Method 3: Direct execution
./activate_env.sh
python start_sammy.py
```

## Cleaning Up Global Packages

If you want to remove globally installed packages (recommended):

```bash
# Run the cleanup script
python cleanup_global_packages.py

# Or manually remove specific packages
python3 -m pip uninstall torch transformers flask numpy scikit-learn
```

**⚠️ Important:** Only run cleanup AFTER confirming your virtual environment works!

## Troubleshooting

### Virtual Environment Not Found
```bash
# Recreate virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

### Package Import Errors
```bash
# Make sure you're in the virtual environment
which python  # Should show venv/bin/python

# Reinstall requirements
pip install -r backend/requirements.txt
```

### Permission Errors
```bash
# Make scripts executable
chmod +x activate_env.sh
chmod +x run_sammy.py
```

### SSL Warnings
The urllib3 SSL warning is harmless and doesn't affect functionality. It's due to macOS using LibreSSL instead of OpenSSL.

## Development Workflow

### Daily Usage
1. `./activate_env.sh` or `source venv/bin/activate`
2. `python start_sammy.py`
3. `deactivate` when done

### Adding Dependencies
1. Activate virtual environment
2. `pip install new_package`
3. `pip freeze > backend/requirements.txt`
4. Commit the updated requirements.txt

### Sharing with Others
1. Share the project (venv/ is excluded from git)
2. Others run: `python3 -m venv venv`
3. Others run: `pip install -r backend/requirements.txt`
4. Ready to go!

## Cross-Platform Notes

### Windows Users
- Use `venv\Scripts\activate` instead of `source venv/bin/activate`
- Use `python` instead of `python3`
- The `run_sammy.py` wrapper handles these differences automatically

### macOS/Linux Users
- Use `source venv/bin/activate`
- Use `python3` for initial setup
- The activation scripts work out of the box

## Why Virtual Environments?

### Before (Global Installation)
```
System Python
├── torch==2.8.0
├── flask==3.1.2
├── numpy==2.0.2
└── ... (100+ packages)
```
**Problems:** Conflicts, hard to clean, platform differences

### After (Virtual Environment)
```
System Python (clean)
├── pip
├── setuptools
└── wheel

Project venv/
├── torch==2.8.0
├── flask==3.1.2
├── numpy==2.0.2
└── ... (only what you need)
```
**Benefits:** Isolated, reproducible, clean

## Next Steps

1. ✅ Virtual environment created and working
2. ✅ Dependencies installed in isolation
3. 🔄 **Optional:** Clean up global packages with `python cleanup_global_packages.py`
4. 🚀 **Ready:** Run `python run_sammy.py` to start Sammy AI!

Your project is now properly isolated and ready for cross-platform development! 🎉