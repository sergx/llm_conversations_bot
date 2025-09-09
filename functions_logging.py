import logging
import sys
from logging.handlers import RotatingFileHandler

def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions by logging them"""
    if issubclass(exc_type, KeyboardInterrupt):
        # Allow KeyboardInterrupt to behave normally
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    # Get the root logger (or use a specific one)
    logger = logging.getLogger()
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

def setup_logger(name=__name__):
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times in case setup_logger is called more than once
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            'logs.log',
            maxBytes=1024*1024*2,  # 2 MB
            backupCount=10,
            encoding='utf-8',
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set formatter for both handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add both handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
        # Set exception hook only once for the entire application
        if not hasattr(setup_logger, 'exception_hook_set'):
            sys.excepthook = handle_uncaught_exception
            setup_logger.exception_hook_set = True
    
    return logger