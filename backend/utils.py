#!/usr/bin/env python3
"""
Utility functions for Sammy AI
Common helpers and error handling
"""

import logging
import functools
import time
from typing import Any, Callable

def setup_logging(level=logging.INFO):
    """Setup consistent logging across the application."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Suppress noisy libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

def timer(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger(func.__module__)
        logger.debug(f"{func.__name__} took {end_time - start_time:.3f}s")
        
        return result
    return wrapper

def safe_execute(func: Callable, fallback: Any = None, log_errors: bool = True) -> Any:
    """Safely execute a function with error handling."""
    try:
        return func()
    except Exception as e:
        if log_errors:
            logger = logging.getLogger(__name__)
            logger.error(f"Error in {func.__name__ if hasattr(func, '__name__') else 'function'}: {e}")
        return fallback

def validate_hebrew_text(text: str, min_hebrew_ratio: float = 0.3) -> bool:
    """Validate if text contains sufficient Hebrew content."""
    if not text or len(text.strip()) < 10:
        return False
    
    import re
    hebrew_chars = len(re.findall(r'[\u0590-\u05FF]', text))
    total_chars = len(re.findall(r'\w', text))
    
    if total_chars == 0:
        return False
    
    hebrew_ratio = hebrew_chars / total_chars
    return hebrew_ratio >= min_hebrew_ratio

def clean_text(text: str) -> str:
    """Clean and normalize text for processing."""
    if not text:
        return ""
    
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common web artifacts
    noise_patterns = [
        r'קרא עוד.*?(?=\.|$)',
        r'לחץ כאן.*?(?=\.|$)',
        r'פרסומת.*?(?=\.|$)',
        r'ממומן.*?(?=\.|$)',
        r'מידע נוסף.*?(?=\.|$)',
        r'תגובות.*?(?=\.|$)'
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()

class PerformanceMonitor:
    """Simple performance monitoring."""
    
    def __init__(self):
        self.metrics = {}
    
    def record(self, operation: str, duration: float):
        """Record operation duration."""
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
    
    def get_stats(self, operation: str) -> dict:
        """Get statistics for an operation."""
        if operation not in self.metrics:
            return {}
        
        durations = self.metrics[operation]
        return {
            'count': len(durations),
            'avg': sum(durations) / len(durations),
            'min': min(durations),
            'max': max(durations),
            'total': sum(durations)
        }
    
    def get_all_stats(self) -> dict:
        """Get all performance statistics."""
        return {op: self.get_stats(op) for op in self.metrics.keys()}