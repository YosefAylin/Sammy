#!/usr/bin/env python3
"""
Production Deployment Script for Sammy AI
Optimizes and packages the application for production use
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
import json

def create_production_build():
    """Create optimized production build."""
    print("🚀 Creating production build...")
    
    # Create build directory
    build_dir = Path("build")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()
    
    # Copy essential files
    essential_files = [
        "backend/ai_server.py",
        "backend/model_manager.py", 
        "backend/config.py",
        "backend/utils.py",
        "backend/requirements.txt",
        "preload_models.py",
        "start_sammy.py",
        "README.md",
        ".gitignore"
    ]
    
    for file_path in essential_files:
        src = Path(file_path)
        if src.exists():
            dst = build_dir / file_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    
    # Copy frontend
    shutil.copytree("frontend", build_dir / "frontend")
    
    print("✅ Production build created in ./build/")
    return build_dir

def optimize_requirements():
    """Create minimal requirements.txt for production."""
    minimal_requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0", 
        "tokenizers>=0.13.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "networkx>=3.0",
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "requests>=2.31.0"
    ]
    
    build_dir = Path("build")
    req_file = build_dir / "backend" / "requirements.txt"
    
    with open(req_file, 'w') as f:
        f.write("# Production requirements for Sammy AI\n")
        f.write("# Minimal dependencies for core functionality\n\n")
        for req in minimal_requirements:
            f.write(f"{req}\n")
    
    print("✅ Optimized requirements.txt for production")

def create_docker_files():
    """Create Docker configuration for containerized deployment."""
    
    dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 5002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:5002/health || exit 1

# Start application
CMD ["python", "backend/ai_server.py"]
"""

    docker_compose_content = """version: '3.8'

services:
  sammy-ai:
    build: .
    ports:
      - "5002:5002"
    volumes:
      - ./models:/app/models
    environment:
      - SAMMY_ENV=production
      - SAMMY_HOST=0.0.0.0
      - SAMMY_PORT=5002
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5002/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
"""

    build_dir = Path("build")
    
    with open(build_dir / "Dockerfile", 'w') as f:
        f.write(dockerfile_content)
    
    with open(build_dir / "docker-compose.yml", 'w') as f:
        f.write(docker_compose_content)
    
    print("✅ Created Docker configuration files")

def create_deployment_guide():
    """Create deployment documentation."""
    
    guide_content = """# Sammy AI - Production Deployment Guide

## Quick Deploy

### Option 1: Direct Python
```bash
# Install dependencies
pip install -r backend/requirements.txt

# Download models
python preload_models.py

# Start production server
SAMMY_ENV=production python backend/ai_server.py
```

### Option 2: Docker
```bash
# Build and run
docker-compose up -d

# Download models (one-time)
docker-compose exec sammy-ai python preload_models.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SAMMY_ENV` | development | Environment (production/development) |
| `SAMMY_HOST` | 0.0.0.0 | Server host |
| `SAMMY_PORT` | 5002 | Server port |
| `SAMMY_DEBUG` | False | Debug mode |

## Production Checklist

- [ ] Models downloaded (~1.6GB)
- [ ] Server accessible on port 5002
- [ ] Chrome extension installed
- [ ] Health check returns 200 OK
- [ ] CORS configured for your domain

## Monitoring

- Health endpoint: `GET /health`
- Model status: `GET /models/status`
- Logs: Check application logs for errors

## Security Notes

- Server binds to localhost by default in production
- No authentication required (local use only)
- Models stored locally (no external API calls)
"""

    build_dir = Path("build")
    with open(build_dir / "DEPLOYMENT.md", 'w') as f:
        f.write(guide_content)
    
    print("✅ Created deployment guide")

def main():
    """Main deployment function."""
    print("📦 SAMMY AI - Production Deployment")
    print("=" * 50)
    
    # Create build
    build_dir = create_production_build()
    
    # Optimize for production
    optimize_requirements()
    
    # Create Docker files
    create_docker_files()
    
    # Create deployment guide
    create_deployment_guide()
    
    # Show summary
    print("\n" + "=" * 50)
    print("🎉 Production build complete!")
    print(f"📁 Build location: {build_dir.absolute()}")
    print("\n📋 Next steps:")
    print("1. cd build/")
    print("2. Follow DEPLOYMENT.md instructions")
    print("3. Test with: python backend/ai_server.py")
    
    # Calculate size
    total_size = sum(f.stat().st_size for f in build_dir.rglob('*') if f.is_file())
    print(f"\n📊 Build size: {total_size / 1024 / 1024:.1f} MB (without models)")

if __name__ == "__main__":
    main()