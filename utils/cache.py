"""
LRU cache with TTL support for application data.
"""
import time
import threading
from typing import Any, Optional, Dict
from collections import OrderedDict
from config import config


class LRUCacheWithTTL:
    """Thread-safe LRU cache with TTL (Time To Live) support."""
    
    def __init__(self, max_size: int = None, ttl_seconds: int = None):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of items (default from config)
            ttl_seconds: Time to live in seconds (default from config)
        """
        self.max_size = max_size or config.CACHE_MAX_SIZE
        self.ttl = ttl_seconds or config.CACHE_TTL
        self._cache = OrderedDict()
        self._timestamps = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            # Check if expired
            if time.time() - self._timestamps[key] > self.ttl:
                self._evict(key)
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        with self._lock:
            # Update if exists
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
                self._timestamps[key] = time.time()
                return
            
            # Add new item
            self._cache[key] = value
            self._timestamps[key] = time.time()
            
            # Evict oldest if necessary
            if len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                self._evict(oldest_key)
    
    def _evict(self, key: str):
        """Evict item from cache."""
        if key in self._cache:
            del self._cache[key]
            del self._timestamps[key]
    
    def clear(self):
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def invalidate(self, key: str):
        """Invalidate a specific cache entry."""
        with self._lock:
            self._evict(key)
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / max(1, total_requests)
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "ttl_seconds": self.ttl
            }


class CacheManager:
    """Manages multiple caches for different data types."""
    
    def __init__(self):
        # Cache for plan listings
        self.plan_cache = LRUCacheWithTTL(max_size=100, ttl_seconds=300)  # 5 minutes
        
        # Cache for plan metadata
        self.metadata_cache = LRUCacheWithTTL(max_size=500, ttl_seconds=600)  # 10 minutes
        
        # Cache for query rewrites
        self.query_cache = LRUCacheWithTTL(max_size=1000, ttl_seconds=300)  # 5 minutes
    
    def invalidate_all(self):
        """Invalidate all caches (e.g., after ingestion)."""
        self.plan_cache.clear()
        self.metadata_cache.clear()
        # Don't clear query cache as it's query-dependent
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all caches."""
        return {
            "plan_cache": self.plan_cache.get_stats(),
            "metadata_cache": self.metadata_cache.get_stats(),
            "query_cache": self.query_cache.get_stats(),
        }


# Global cache manager
cache_manager = CacheManager()
