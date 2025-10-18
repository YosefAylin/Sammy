#!/usr/bin/env python3
"""
Configuration Management for Sammy AI
Centralized settings and environment variables
"""

import os
from pathlib import Path

class Config:
    """Main configuration class."""
    
    # Server settings
    HOST = os.getenv('SAMMY_HOST', '0.0.0.0')
    PORT = int(os.getenv('SAMMY_PORT', '5002'))
    DEBUG = os.getenv('SAMMY_DEBUG', 'False').lower() == 'true'
    
    # Model settings
    MODELS_DIR = Path(os.getenv('SAMMY_MODELS_DIR', './models'))
    CACHE_SIZE = int(os.getenv('SAMMY_CACHE_SIZE', '128'))
    
    # AI settings
    DEFAULT_TOP_N = int(os.getenv('SAMMY_DEFAULT_TOP_N', '6'))
    MAX_TOP_N = int(os.getenv('SAMMY_MAX_TOP_N', '10'))
    MIN_SENTENCES = int(os.getenv('SAMMY_MIN_SENTENCES', '3'))
    
    # Performance settings
    BATCH_SIZE = int(os.getenv('SAMMY_BATCH_SIZE', '8'))
    MAX_LENGTH = int(os.getenv('SAMMY_MAX_LENGTH', '512'))
    
    # CORS settings
    ALLOWED_ORIGINS = os.getenv('SAMMY_CORS_ORIGINS', 'chrome-extension://*,moz-extension://*').split(',')
    
    @classmethod
    def get_device(cls):
        """Get the best available device."""
        import torch
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')  # Apple Silicon
        else:
            return torch.device('cpu')

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    PORT = 5002

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    HOST = '127.0.0.1'  # More secure for production

# Configuration factory
def get_config():
    """Get configuration based on environment."""
    env = os.getenv('SAMMY_ENV', 'development')
    
    if env == 'production':
        return ProductionConfig()
    else:
        return DevelopmentConfig()