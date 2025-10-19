# Configuration Files

This directory contains all configuration files for the Sammy AI project.

## Directory Structure

```
config/
├── README.md              # This file
├── .env.example           # Environment variables template
├── vscode/                # VS Code IDE configuration
│   ├── extensions.json    # Recommended extensions
│   ├── launch.json        # Debug configurations
│   ├── settings.json      # IDE settings
│   └── tasks.json         # Build tasks
└── scripts/               # Setup and utility scripts
    ├── activate_env.sh    # Virtual environment activation
    ├── deploy.py          # Deployment script
    ├── run_sammy.py       # Application launcher
    └── setup.py           # Project setup script
```

## Usage

### Environment Setup
```bash
# Copy environment template
cp config/.env.example .env

# Activate virtual environment
./activate_env.sh
# or directly:
./config/scripts/activate_env.sh
```

### VS Code Configuration
The `.vscode` symlink in the root directory points to `config/vscode/`, so VS Code automatically picks up all configurations.

### Scripts
All utility scripts are in `config/scripts/`:
- `run_sammy.py` - Main application launcher with auto-venv activation
- `setup.py` - Complete project setup and installation
- `deploy.py` - Deployment utilities
- `activate_env.sh` - Virtual environment activation helper

## Files Explanation

### .env.example
Template for environment variables. Copy to `.env` and modify as needed:
- Flask configuration (host, port, debug mode)
- AI model settings (device, cache directory)
- Logging configuration
- Performance tuning parameters

### VS Code Configuration
- **extensions.json** - Recommended extensions for Python, AI development
- **launch.json** - Debug configurations for different components
- **settings.json** - Python interpreter, linting, formatting settings
- **tasks.json** - Build tasks for common operations

### Scripts
- **activate_env.sh** - Smart virtual environment activation with status
- **run_sammy.py** - Wrapper that ensures venv is active before running
- **setup.py** - Cross-platform setup with dependency installation
- **deploy.py** - Deployment and distribution utilities

## Maintenance

When adding new configuration files:
1. Place them in the appropriate subdirectory
2. Update this README
3. Create symlinks in root if needed for tool compatibility
4. Update .gitignore if necessary