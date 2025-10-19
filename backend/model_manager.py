#!/usr/bin/env python3
"""
Model Manager for Sammy AI
Downloads and manages AI models within the project directory
"""

import os
import shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages AI model downloads and caching within project directory."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            'alephbert': {
                'name': 'onlplab/alephbert-base',
                'local_path': self.models_dir / 'alephbert-base',
                'type': 'encoder'
            },
            'dicta': {
                'name': 'Dicta-IL/dictalm2.0',
                'local_path': self.models_dir / 'dicta-hebrew',
                'type': 'causal'
            }
        }
    
    def download_model(self, model_key: str, force_download: bool = False):
        """Download a model to the project directory."""
        if model_key not in self.model_configs:
            raise ValueError(f"Unknown model: {model_key}")
        
        config = self.model_configs[model_key]
        local_path = config['local_path']
        
        # Check if already downloaded
        if local_path.exists() and not force_download:
            logger.info(f"✅ {model_key} already downloaded to {local_path}")
            return str(local_path)
        
        logger.info(f"📥 Downloading {model_key} ({config['name']}) to {local_path}")
        
        try:
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config['name'])
            tokenizer.save_pretrained(local_path)
            
            # Download model based on type
            if config['type'] == 'encoder':
                model = AutoModel.from_pretrained(config['name'])
            elif config['type'] == 'seq2seq':
                model = AutoModelForSeq2SeqLM.from_pretrained(config['name'])
            elif config['type'] == 'causal':
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(config['name'])
            else:
                raise ValueError(f"Unknown model type: {config['type']}")
            
            model.save_pretrained(local_path)
            
            logger.info(f"✅ {model_key} downloaded successfully to {local_path}")
            return str(local_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to download {model_key}: {e}")
            # Clean up partial download
            if local_path.exists():
                shutil.rmtree(local_path)
            raise
    
    def get_model_path(self, model_key: str) -> str:
        """Get the local path for a model, download if not exists."""
        if model_key not in self.model_configs:
            raise ValueError(f"Unknown model: {model_key}")
        
        config = self.model_configs[model_key]
        local_path = config['local_path']
        
        if not local_path.exists():
            logger.info(f"Model {model_key} not found locally, downloading...")
            return self.download_model(model_key)
        
        return str(local_path)
    
    def is_model_downloaded(self, model_key: str) -> bool:
        """Check if a model is already downloaded."""
        if model_key not in self.model_configs:
            return False
        
        local_path = self.model_configs[model_key]['local_path']
        return local_path.exists() and any(local_path.iterdir())
    
    def get_model_size(self, model_key: str) -> str:
        """Get the size of a downloaded model."""
        if not self.is_model_downloaded(model_key):
            return "Not downloaded"
        
        local_path = self.model_configs[model_key]['local_path']
        
        # Calculate directory size
        total_size = sum(f.stat().st_size for f in local_path.rglob('*') if f.is_file())
        
        # Convert to human readable
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_size < 1024.0:
                return f"{total_size:.1f} {unit}"
            total_size /= 1024.0
        return f"{total_size:.1f} TB"
    
    def list_models(self) -> dict:
        """List all models and their status."""
        status = {}
        for model_key, config in self.model_configs.items():
            status[model_key] = {
                'name': config['name'],
                'downloaded': self.is_model_downloaded(model_key),
                'size': self.get_model_size(model_key),
                'path': str(config['local_path'])
            }
        return status
    
    def download_all_models(self):
        """Download all required models."""
        logger.info("📥 Downloading all AI models...")
        
        for model_key in self.model_configs.keys():
            try:
                self.download_model(model_key)
            except Exception as e:
                logger.error(f"Failed to download {model_key}: {e}")
                return False
        
        logger.info("✅ All models downloaded successfully!")
        return True
    
    def clean_models(self):
        """Remove all downloaded models."""
        logger.info("🧹 Cleaning all downloaded models...")
        
        if self.models_dir.exists():
            shutil.rmtree(self.models_dir)
            self.models_dir.mkdir(exist_ok=True)
        
        logger.info("✅ All models cleaned")

def main():
    """CLI interface for model management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage Sammy AI models")
    parser.add_argument('action', choices=['download', 'list', 'clean'], 
                       help='Action to perform')
    parser.add_argument('--model', choices=['alephbert', 'mt5', 'all'], 
                       default='all', help='Model to download')
    parser.add_argument('--force', action='store_true', 
                       help='Force re-download even if exists')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(levelname)s: %(message)s')
    
    manager = ModelManager()
    
    if args.action == 'list':
        print("📊 Model Status:")
        print("-" * 50)
        for model_key, status in manager.list_models().items():
            downloaded = "✅" if status['downloaded'] else "❌"
            print(f"{downloaded} {model_key}: {status['name']}")
            print(f"   Size: {status['size']}")
            print(f"   Path: {status['path']}")
            print()
    
    elif args.action == 'download':
        if args.model == 'all':
            manager.download_all_models()
        else:
            manager.download_model(args.model, args.force)
    
    elif args.action == 'clean':
        manager.clean_models()

if __name__ == "__main__":
    main()