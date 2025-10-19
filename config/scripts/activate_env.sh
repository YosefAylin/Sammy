#!/bin/bash
# Activation script for Sammy AI virtual environment

echo "🤖 Activating Sammy AI virtual environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run: python3 -m venv venv"
    echo "Then: pip install -r backend/requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo "📦 Python: $(which python)"
echo "📦 Pip: $(which pip)"

# Show installed packages
echo ""
echo "🔍 Key packages installed:"
python -c "
import sys
packages = ['torch', 'transformers', 'flask', 'numpy', 'scikit-learn']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'  ✅ {pkg}')
    except ImportError:
        print(f'  ❌ {pkg} - missing')
"

echo ""
echo "🚀 Ready to run: python start_sammy.py"
echo "🛑 To deactivate: deactivate"