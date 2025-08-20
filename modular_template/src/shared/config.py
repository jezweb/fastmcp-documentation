"""
Configuration Module
====================
Central configuration for the MCP server.
"""

import os
from typing import Optional


class Config:
    """Server configuration class."""
    
    # Server settings
    SERVER_NAME: str = os.getenv("MCP_SERVER_NAME", "fastmcp-modular")
    SERVER_VERSION: str = os.getenv("MCP_SERVER_VERSION", "1.0.0")
    ENVIRONMENT: str = os.getenv("MCP_ENVIRONMENT", "development")
    
    # Transport settings
    TRANSPORT: str = os.getenv("MCP_TRANSPORT", "stdio")  # stdio or http
    PORT: int = int(os.getenv("MCP_PORT", "8000"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("MCP_LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("MCP_LOG_FILE")
    
    # API settings
    API_BASE_URL: Optional[str] = os.getenv("MCP_API_BASE_URL")
    API_KEY: Optional[str] = os.getenv("MCP_API_KEY")
    API_SECRET: Optional[str] = os.getenv("MCP_API_SECRET")
    REQUEST_TIMEOUT: int = int(os.getenv("MCP_REQUEST_TIMEOUT", "30"))
    MAX_RETRIES: int = int(os.getenv("MCP_MAX_RETRIES", "3"))
    VERIFY_SSL: bool = os.getenv("MCP_VERIFY_SSL", "true").lower() == "true"
    
    # Cache settings
    ENABLE_CACHE: bool = os.getenv("MCP_CACHE_ENABLED", "true").lower() == "true"
    CACHE_TTL: int = int(os.getenv("MCP_CACHE_TTL", "300"))  # 5 minutes
    CACHE_MAX_SIZE: int = int(os.getenv("MCP_CACHE_MAX_SIZE", "1000"))
    CACHE_EVICTION_POLICY: str = os.getenv("MCP_CACHE_EVICTION", "LRU")
    
    # Feature flags
    ENABLE_ADVANCED_FEATURES: bool = os.getenv("MCP_ADVANCED_FEATURES", "true").lower() == "true"
    ENABLE_RATE_LIMITING: bool = os.getenv("MCP_RATE_LIMITING", "false").lower() == "true"
    ENABLE_MONITORING: bool = os.getenv("MCP_MONITORING", "false").lower() == "true"
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = int(os.getenv("MCP_RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("MCP_RATE_LIMIT_WINDOW", "60"))  # seconds
    
    # File operations
    MAX_FILE_SIZE: int = int(os.getenv("MCP_MAX_FILE_SIZE", "10485760"))  # 10MB
    ALLOWED_FILE_EXTENSIONS: list = os.getenv(
        "MCP_ALLOWED_EXTENSIONS", 
        ".txt,.json,.csv,.xml,.log,.md"
    ).split(",")
    
    # Security
    ENABLE_AUTH: bool = os.getenv("MCP_ENABLE_AUTH", "false").lower() == "true"
    AUTH_TOKEN: Optional[str] = os.getenv("MCP_AUTH_TOKEN")
    ALLOWED_ORIGINS: list = os.getenv("MCP_ALLOWED_ORIGINS", "*").split(",")
    
    @classmethod
    def load_from_file(cls, config_file: str = "config.json"):
        """
        Load configuration from a JSON file.
        
        Args:
            config_file: Path to configuration file
        """
        import json
        from pathlib import Path
        
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Update class attributes
            for section, values in config_data.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        attr_name = f"{section.upper()}_{key.upper()}"
                        if hasattr(cls, attr_name):
                            setattr(cls, attr_name, value)
                else:
                    attr_name = section.upper()
                    if hasattr(cls, attr_name):
                        setattr(cls, attr_name, values)
    
    @classmethod
    def validate(cls):
        """
        Validate configuration settings.
        
        Raises:
            ValueError: If configuration is invalid
        """
        errors = []
        
        # Validate transport
        if cls.TRANSPORT not in ["stdio", "http"]:
            errors.append(f"Invalid transport: {cls.TRANSPORT}")
        
        # Validate port for HTTP transport
        if cls.TRANSPORT == "http" and not (1 <= cls.PORT <= 65535):
            errors.append(f"Invalid port: {cls.PORT}")
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if cls.LOG_LEVEL not in valid_log_levels:
            errors.append(f"Invalid log level: {cls.LOG_LEVEL}")
        
        # Validate cache settings
        if cls.ENABLE_CACHE:
            if cls.CACHE_TTL <= 0:
                errors.append(f"Invalid cache TTL: {cls.CACHE_TTL}")
            if cls.CACHE_MAX_SIZE <= 0:
                errors.append(f"Invalid cache max size: {cls.CACHE_MAX_SIZE}")
        
        # Validate API settings
        if cls.API_BASE_URL and not cls.API_BASE_URL.startswith(("http://", "https://")):
            errors.append(f"Invalid API base URL: {cls.API_BASE_URL}")
        
        # Validate rate limiting
        if cls.ENABLE_RATE_LIMITING:
            if cls.RATE_LIMIT_REQUESTS <= 0:
                errors.append(f"Invalid rate limit requests: {cls.RATE_LIMIT_REQUESTS}")
            if cls.RATE_LIMIT_WINDOW <= 0:
                errors.append(f"Invalid rate limit window: {cls.RATE_LIMIT_WINDOW}")
        
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")
    
    @classmethod
    def to_dict(cls) -> dict:
        """
        Export configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        config = {}
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith('_'):
                value = getattr(cls, attr)
                if not callable(value):
                    config[attr] = value
        return config