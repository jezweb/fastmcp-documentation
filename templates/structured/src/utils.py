"""
Utility Functions
================
All utility functions needed by the server tools and resources.
This file contains everything that would have been in separate shared modules.
"""

import logging
import os
import json
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

class Config:
    """Application configuration from environment variables."""
    
    # Server
    SERVER_NAME = os.getenv("SERVER_NAME", "FastMCP Structured Server")
    SERVER_VERSION = "1.0.0"
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    # Transport
    TRANSPORT = os.getenv("TRANSPORT", "stdio")
    
    # API Configuration
    API_BASE_URL = os.getenv("API_BASE_URL", "")
    API_KEY = os.getenv("API_KEY", "")
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
    
    # Cache Configuration
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes
    
    # Features
    ENABLE_ADVANCED_FEATURES = os.getenv("ENABLE_ADVANCED_FEATURES", "false").lower() == "true"
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# ============================================================
# Response Formatting
# ============================================================

def format_success(data: Any, message: str = "Success") -> Dict[str, Any]:
    """Format a successful response."""
    return {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }


def format_error(error: Any, error_code: str = "GENERAL_ERROR") -> Dict[str, Any]:
    """Format an error response."""
    return {
        "success": False,
        "error": {
            "code": error_code,
            "message": str(error)
        },
        "timestamp": datetime.now().isoformat()
    }


# ============================================================
# Input Validation
# ============================================================

def validate_input(data: Any, required_fields: List[str]) -> Optional[str]:
    """Validate input data has required fields."""
    if not isinstance(data, dict):
        return "Input must be a dictionary"
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return f"Missing required fields: {', '.join(missing_fields)}"
    
    return None


def validate_path(path: str) -> bool:
    """Validate file path is safe."""
    # Basic path validation - extend as needed
    if ".." in path or path.startswith("/"):
        return False
    return True


# ============================================================
# Simple Cache Implementation
# ============================================================

class SimpleCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key not in self._cache:
            return None
        
        # Check if expired
        if time.time() - self._timestamps[key] > Config.CACHE_TTL:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
            return None
        
        return self._cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._timestamps.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        valid_entries = sum(
            1 for ts in self._timestamps.values() 
            if now - ts <= Config.CACHE_TTL
        )
        
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self._cache) - valid_entries,
            "ttl_seconds": Config.CACHE_TTL
        }


# Global cache instance
_cache = SimpleCache() if Config.ENABLE_CACHE else None


async def get_cache_value(key: str) -> Optional[Any]:
    """Get value from cache."""
    if not _cache:
        return None
    return _cache.get(key)


async def set_cache_value(key: str, value: Any) -> None:
    """Set value in cache."""
    if _cache:
        _cache.set(key, value)


async def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    if not _cache:
        return {"enabled": False}
    
    stats = _cache.stats()
    stats["enabled"] = True
    return stats


async def cleanup_cache() -> None:
    """Cleanup cache on shutdown."""
    if _cache:
        _cache.clear()


# ============================================================
# HTTP Client
# ============================================================

class APIClient:
    """HTTP client with connection pooling."""
    
    def __init__(self):
        self._session = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if not self._session or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=Config.API_TIMEOUT)
            headers = {}
            
            if Config.API_KEY:
                headers["Authorization"] = f"Bearer {Config.API_KEY}"
            
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
        
        return self._session
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# Global API client instance
_api_client = APIClient()


async def get_api_client() -> aiohttp.ClientSession:
    """Get HTTP client session."""
    return await _api_client.get_session()


async def close_api_client() -> None:
    """Close API client."""
    await _api_client.close()


async def verify_api_connection() -> bool:
    """Verify API connection is working."""
    if not Config.API_BASE_URL:
        return False
    
    try:
        client = await get_api_client()
        async with client.get(f"{Config.API_BASE_URL}/health") as response:
            return response.status < 400
    except Exception as e:
        logger.warning(f"API connection check failed: {e}")
        return False


# ============================================================
# Initialization Functions
# ============================================================

async def initialize_cache() -> None:
    """Initialize cache system."""
    if Config.ENABLE_CACHE:
        logger.info("Cache system initialized")
    else:
        logger.info("Cache system disabled")