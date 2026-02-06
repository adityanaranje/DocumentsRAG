"""
Centralized structured logging with rotation and request ID tracking.
"""
import os
import json
import logging
import threading
from logging.handlers import RotatingFileHandler
from typing import Optional
from datetime import datetime
from config import config

# Thread-local storage for request context
_request_context = threading.local()


class RequestContextFilter(logging.Filter):
    """Add request ID to log records."""
    
    def filter(self, record):
        record.request_id = getattr(_request_context, 'request_id', 'N/A')
        record.user_ip = getattr(_request_context, 'user_ip', 'N/A')
        return True


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'request_id': getattr(record, 'request_id', 'N/A'),
            'user_ip': getattr(record, 'user_ip', 'N/A'),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data
        
        return json.dumps(log_data)


def setup_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """
    Create a logger with both file and console handlers.
    
    Args:
        name: Logger name (typically __name__)
        log_level: Optional override for log level
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Set level
    level = log_level or config.LOG_LEVEL
    logger.setLevel(getattr(logging, level.upper()))
    
    # Add request context filter
    logger.addFilter(RequestContextFilter())
    
    # Console handler (human-readable for development)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if config.DEBUG else logging.INFO)
    
    if config.ENVIRONMENT.value == "production":
        # JSON format for production
        console_handler.setFormatter(JSONFormatter())
    else:
        # Human-readable format for development
        console_format = logging.Formatter(
            '%(asctime)s - [%(request_id)s] - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
    
    logger.addHandler(console_handler)
    
    # File handler with rotation
    try:
        log_dir = os.path.dirname(config.LOG_FILE_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            config.LOG_FILE_PATH,
            maxBytes=config.LOG_MAX_BYTES,
            backupCount=config.LOG_BACKUP_COUNT
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Always use JSON format for file logs
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Failed to setup file logging: {e}")
    
    return logger


def set_request_context(request_id: str, user_ip: Optional[str] = None):
    """Set request context for the current thread."""
    _request_context.request_id = request_id
    _request_context.user_ip = user_ip or 'unknown'


def clear_request_context():
    """Clear request context for the current thread."""
    if hasattr(_request_context, 'request_id'):
        delattr(_request_context, 'request_id')
    if hasattr(_request_context, 'user_ip'):
        delattr(_request_context, 'user_ip')


def log_with_extra(logger: logging.Logger, level: str, message: str, **extra_data):
    """Log with extra structured data."""
    log_method = getattr(logger, level.lower())
    
    # Create a custom log record with extra data
    if extra_data:
        extra_record = {'extra_data': extra_data}
        log_method(message, extra=extra_record)
    else:
        log_method(message)


# Create module-level loggers for common components
app_logger = setup_logger('app')
agent_logger = setup_logger('agents')
retrieval_logger = setup_logger('retrieval')
ingestion_logger = setup_logger('ingestion')
llm_logger = setup_logger('llm')
api_logger = setup_logger('api')
