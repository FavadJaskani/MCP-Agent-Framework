import logging
import sys
from typing import Optional

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Create a logger with the specified name and level."""
    logger = logging.getLogger(name)
    
    if level is not None:
        logger.setLevel(level)
    elif not logger.hasHandlers():
        # Default configuration for new loggers
        logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
    
    return logger