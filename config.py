"""
Production-grade configuration management with environment-based settings.
"""
import os
from typing import Optional
from dotenv import load_dotenv
from enum import Enum

load_dotenv()


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Config:
    """Base configuration class with defaults."""
    
    # Environment
    ENVIRONMENT: Environment = Environment(os.getenv("ENVIRONMENT", "development"))
    DEBUG: bool = ENVIRONMENT == Environment.DEVELOPMENT
    
    # Application
    APP_NAME: str = "Insurance RAG System"
    VERSION: str = "1.0.0"
    PORT: int = int(os.getenv("PORT", 7860))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    
    # LLM Configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL_SMALL: str = os.getenv("GROQ_MODEL_SMALL", "llama-3.1-8b-instant")
    GROQ_MODEL_MEDIUM: str = os.getenv("GROQ_MODEL_MEDIUM", "llama-3.1-8b-instant")
    GROQ_MODEL_LARGE: str = os.getenv("GROQ_MODEL_LARGE", "llama-3.1-8b-instant")
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", 30))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", 3))
    
    # LangChain / LangSmith Tracing
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_API_KEY: Optional[str] = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "Insurance-RAG")
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "logs/app.log")
    LOG_MAX_BYTES: int = int(os.getenv("LOG_MAX_BYTES", 10485760))  # 10MB
    LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", 5))
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance Configuration
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", 4))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", 30))
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", 300))  # 5 minutes
    CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", 1000))
    
    # Vector Store Configuration
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "rag/faiss_index")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_CACHE_PATH: str = os.getenv("EMBEDDING_CACHE_PATH", "rag/embeddings_cache.json")
    LLM_CACHE_DB_PATH: str = os.getenv("LLM_CACHE_DB_PATH", "rag/llm_cache.db")
    
    # Retrieval Configuration
    DEFAULT_RETRIEVAL_K: int = int(os.getenv("DEFAULT_RETRIEVAL_K", 5))
    MAX_RETRIEVAL_K: int = int(os.getenv("MAX_RETRIEVAL_K", 20))
    RETRIEVAL_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", 0.5))
    
    # Security Configuration
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", 50))
    ALLOWED_FILE_TYPES: list = os.getenv("ALLOWED_FILE_TYPES", "pdf,docx").split(",")
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", 60))
    ENABLE_API_KEY_AUTH: bool = os.getenv("ENABLE_API_KEY_AUTH", "false").lower() == "true"
    API_KEY: Optional[str] = os.getenv("API_KEY", None)
    ENABLE_CORS: bool = os.getenv("ENABLE_CORS", "true").lower() == "true"
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Monitoring Configuration
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", 9090))
    REQUEST_LOG_DB_PATH: str = os.getenv("REQUEST_LOG_DB_PATH", "utils/request_logs.db")
    
    # Circuit Breaker Configuration
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", 5))
    CIRCUIT_BREAKER_TIMEOUT: int = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", 60))
    CIRCUIT_BREAKER_EXPECTED_EXCEPTION: type = Exception
    
    # Document Processing
    DOCS_DIR: str = os.getenv("DOCS_DIR", "docs")
    BROCHURE_CHUNK_SIZE: int = int(os.getenv("BROCHURE_CHUNK_SIZE", 2600))
    BROCHURE_CHUNK_OVERLAP: int = int(os.getenv("BROCHURE_CHUNK_OVERLAP", 400))
    CIS_CHUNK_SIZE: int = int(os.getenv("CIS_CHUNK_SIZE", 1300))
    CIS_CHUNK_OVERLAP: int = int(os.getenv("CIS_CHUNK_OVERLAP", 160))
    TABLE_CHUNK_SIZE: int = int(os.getenv("TABLE_CHUNK_SIZE", 800))
    TABLE_CHUNK_OVERLAP: int = int(os.getenv("TABLE_CHUNK_OVERLAP", 100))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate critical configuration values."""
        errors = []
        
        # Check required API keys
        if not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is not set")
        
        # Validate file size limits
        if cls.MAX_FILE_SIZE_MB <= 0 or cls.MAX_FILE_SIZE_MB > 500:
            errors.append(f"MAX_FILE_SIZE_MB must be between 1 and 500, got {cls.MAX_FILE_SIZE_MB}")
        
        # Validate rate limits
        if cls.RATE_LIMIT_PER_MINUTE <= 0:
            errors.append(f"RATE_LIMIT_PER_MINUTE must be positive, got {cls.RATE_LIMIT_PER_MINUTE}")
        
        # Validate timeouts
        if cls.REQUEST_TIMEOUT <= 0:
            errors.append(f"REQUEST_TIMEOUT must be positive, got {cls.REQUEST_TIMEOUT}")
        
        # Validate cache settings
        if cls.CACHE_TTL < 0:
            errors.append(f"CACHE_TTL cannot be negative, got {cls.CACHE_TTL}")
        
        # Validate API key auth
        if cls.ENABLE_API_KEY_AUTH and not cls.API_KEY:
            errors.append("ENABLE_API_KEY_AUTH is true but API_KEY is not set")
        
        if errors:
            error_msg = "\n".join(f"  - {err}" for err in errors)
            raise ValueError(f"Configuration validation failed:\n{error_msg}")
        
        return True
    
    @classmethod
    def get_summary(cls) -> dict:
        """Get configuration summary for logging."""
        return {
            "environment": cls.ENVIRONMENT.value,
            "debug": cls.DEBUG,
            "app_name": cls.APP_NAME,
            "version": cls.VERSION,
            "port": cls.PORT,
            "log_level": cls.LOG_LEVEL,
            "max_file_size_mb": cls.MAX_FILE_SIZE_MB,
            "rate_limit_per_minute": cls.RATE_LIMIT_PER_MINUTE,
            "cache_ttl": cls.CACHE_TTL,
            "enable_metrics": cls.ENABLE_METRICS,
            "enable_api_key_auth": cls.ENABLE_API_KEY_AUTH,
        }


class DevelopmentConfig(Config):
    """Development-specific configuration."""
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production-specific configuration."""
    DEBUG = False
    LOG_LEVEL = "WARNING"


# Configuration factory
def get_config() -> Config:
    """Get configuration based on environment."""
    env = os.getenv("ENVIRONMENT", "development")
    
    config_map = {
        Environment.DEVELOPMENT: DevelopmentConfig,
        Environment.STAGING: Config,
        Environment.PRODUCTION: ProductionConfig,
    }
    
    return config_map.get(Environment(env), Config)


# Global config instance
config = get_config()

# Validate on import
try:
    config.validate()
except ValueError as e:
    print(f"[CONFIG ERROR] {e}")
    print("[CONFIG] Continuing with invalid configuration - some features may not work correctly")
