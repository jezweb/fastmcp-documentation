"""
Cache Module
============
Simple in-memory cache implementation.
"""

import logging
import asyncio
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
from collections import OrderedDict

from .config import Config

logger = logging.getLogger(__name__)


class Cache:
    """Simple LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of items in cache
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, datetime] = {}
        self.ttls: Dict[str, int] = {}
        self.hits = 0
        self.misses = 0
        self.lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None
        """
        async with self.lock:
            if key in self.cache:
                # Check if expired
                if self._is_expired(key):
                    del self.cache[key]
                    del self.timestamps[key]
                    del self.ttls[key]
                    self.misses += 1
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            
            self.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if not specified)
        """
        async with self.lock:
            # Remove oldest items if at capacity
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
                del self.ttls[oldest_key]
            
            # Add new item
            self.cache[key] = value
            self.timestamps[key] = datetime.now()
            self.ttls[key] = ttl or self.default_ttl
    
    async def delete(self, key: str) -> bool:
        """
        Delete item from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if item was deleted
        """
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
                del self.ttls[key]
                return True
            return False
    
    async def clear(self):
        """Clear all items from cache."""
        async with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.ttls.clear()
            self.hits = 0
            self.misses = 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        async with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(hit_rate, 2),
                "total_requests": total_requests
            }
    
    def _is_expired(self, key: str) -> bool:
        """
        Check if cache entry is expired.
        
        Args:
            key: Cache key
        
        Returns:
            True if expired
        """
        if key not in self.timestamps:
            return True
        
        age = (datetime.now() - self.timestamps[key]).total_seconds()
        return age > self.ttls[key]


# Global cache instance
_cache: Optional[Cache] = None


async def initialize_cache():
    """Initialize the global cache."""
    global _cache
    if Config.ENABLE_CACHE:
        _cache = Cache(
            max_size=Config.CACHE_MAX_SIZE,
            default_ttl=Config.CACHE_TTL
        )
        logger.info(f"Cache initialized (max_size={Config.CACHE_MAX_SIZE}, ttl={Config.CACHE_TTL})")
    else:
        logger.info("Cache disabled")


async def cleanup_cache():
    """Cleanup the global cache."""
    global _cache
    if _cache:
        await _cache.clear()
        logger.info("Cache cleared")


async def cache_get(key: str) -> Optional[Any]:
    """
    Get value from global cache.
    
    Args:
        key: Cache key
    
    Returns:
        Cached value or None
    """
    if _cache:
        return await _cache.get(key)
    return None


async def cache_set(key: str, value: Any, ttl: Optional[int] = None):
    """
    Set value in global cache.
    
    Args:
        key: Cache key
        value: Value to cache
        ttl: TTL in seconds
    """
    if _cache:
        await _cache.set(key, value, ttl)


async def get_cache_value(key: str) -> Optional[Any]:
    """
    Get value from cache (alias for cache_get).
    
    Args:
        key: Cache key
    
    Returns:
        Cached value or None
    """
    return await cache_get(key)


async def set_cache_value(key: str, value: Any, ttl: Optional[int] = None):
    """
    Set value in cache (alias for cache_set).
    
    Args:
        key: Cache key
        value: Value to cache
        ttl: TTL in seconds
    """
    await cache_set(key, value, ttl)


async def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns:
        Cache statistics or empty dict if cache disabled
    """
    if _cache:
        return await _cache.get_stats()
    return {
        "enabled": False,
        "size": 0,
        "hits": 0,
        "misses": 0
    }