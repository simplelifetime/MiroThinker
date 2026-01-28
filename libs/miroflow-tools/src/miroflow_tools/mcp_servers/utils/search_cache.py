# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Search cache management for serper-based search tools.
Uses SQLite database for thread-safe concurrent access.
"""

import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Optional


class SearchCache:
    """
    Cache manager for search tool results.

    Stores search results in a SQLite database with keys based on tool name and parameters.
    Thread-safe for concurrent access from multiple threads.
    """

    def __init__(self, cache_path: Optional[str] = None, enabled: Optional[bool] = None):
        """
        Initialize the search cache.

        Args:
            cache_path: Path to the cache database file. If None, uses default path.
            enabled: Whether caching is enabled. If None, checks MIROFLOW_SEARCH_CACHE_ENABLED env var.
        """
        # Check if caching is disabled via environment variable
        if enabled is None:
            env_enabled = os.getenv("MIROFLOW_SEARCH_CACHE_ENABLED", "true").lower()
            self.enabled = env_enabled not in ("false", "0", "no", "off")
        else:
            self.enabled = enabled

        # Thread lock for SQLite operations (Python sqlite3 requires locks for multi-threaded access)
        self._lock = threading.Lock()

        # Performance statistics
        self._hit_count = 0
        self._miss_count = 0

        if not self.enabled:
            self.db_path = None
            self._conn = None
            return

        if cache_path is None:
            # Default cache path in user's home directory
            cache_dir = Path.home() / "MiroThinker" / ".miroflow_tools" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / "search_cache.db"

        self.db_path = Path(cache_path)
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite database
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database and create tables if they don't exist."""
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,  # Allow sharing connection across threads with manual locking
            timeout=30.0  # Wait up to 30 seconds for lock
        )

        # Enable WAL mode for better concurrent access
        with self._lock:
            self._conn.execute('PRAGMA journal_mode=WAL')
            self._conn.execute('PRAGMA synchronous=NORMAL')

            # Create cache table
            self._conn.execute('''
                CREATE TABLE IF NOT EXISTS search_cache (
                    cache_key TEXT PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    query TEXT NOT NULL,
                    params TEXT,
                    result TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create statistics table
            self._conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_statistics (
                    id INTEGER PRIMARY KEY,
                    hit_count INTEGER DEFAULT 0,
                    miss_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Initialize statistics if not exists
            cursor = self._conn.execute('SELECT COUNT(*) FROM cache_statistics')
            if cursor.fetchone()[0] == 0:
                self._conn.execute('INSERT INTO cache_statistics (id, hit_count, miss_count) VALUES (1, 0, 0)')
                self._conn.commit()

            # Load existing statistics
            cursor = self._conn.execute('SELECT hit_count, miss_count FROM cache_statistics WHERE id = 1')
            row = cursor.fetchone()
            if row:
                self._hit_count = row[0] or 0
                self._miss_count = row[1] or 0

            # Create indexes for faster queries
            self._conn.execute('CREATE INDEX IF NOT EXISTS idx_tool_name ON search_cache(tool_name)')
            self._conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON search_cache(created_at)')

            self._conn.commit()

    def _update_statistics(self):
        """Update statistics in database."""
        if not self.enabled:
            return
        try:
            with self._lock:
                self._conn.execute('''
                    UPDATE cache_statistics
                    SET hit_count = ?, miss_count = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE id = 1
                ''', (self._hit_count, self._miss_count))
                self._conn.commit()
        except Exception:
            pass  # Ignore errors updating statistics

    def _update_access_time(self, cache_key: str):
        """Update the last accessed time for a cache entry."""
        if not self.enabled:
            return
        try:
            with self._lock:
                self._conn.execute(
                    'UPDATE search_cache SET accessed_at = CURRENT_TIMESTAMP WHERE cache_key = ?',
                    (cache_key,)
                )
                self._conn.commit()
        except Exception:
            pass  # Ignore errors updating access time

    def _generate_cache_key(
        self, tool_name: str, query: str, **kwargs
    ) -> str:
        """
        Generate a unique cache key based on tool name, query, and parameters.

        Args:
            tool_name: Name of the search tool (e.g., 'google_search', 'image_search')
            query: Search query string
            **kwargs: Additional search parameters

        Returns:
            A unique cache key string
        """
        # Normalize parameters
        params = {"query": query.strip()}
        params.update(kwargs)

        # Sort parameters for consistent hashing
        params_str = json.dumps(params, sort_keys=True)

        # Create hash of parameters
        params_hash = hashlib.md5(params_str.encode("utf-8")).hexdigest()[:16]

        # Combine tool name with hash
        cache_key = f"{tool_name}:{params_hash}"
        return cache_key

    def get(self, tool_name: str, query: str, **kwargs) -> Optional[str]:
        """
        Get cached search result if available.

        Args:
            tool_name: Name of the search tool (e.g., 'google_search', 'image_search')
            query: Search query string
            **kwargs: Additional search parameters

        Returns:
            Cached JSON string result, or None if not found or cache is disabled
        """
        if not self.enabled:
            return None

        cache_key = self._generate_cache_key(tool_name, query, **kwargs)

        try:
            with self._lock:
                cursor = self._conn.execute(
                    'SELECT result FROM search_cache WHERE cache_key = ?',
                    (cache_key,)
                )
                row = cursor.fetchone()

                if row:
                    # Cache hit!
                    result = row[0]
                    self._hit_count += 1
                    # Update access time in the same transaction
                    self._conn.execute(
                        'UPDATE search_cache SET accessed_at = CURRENT_TIMESTAMP WHERE cache_key = ?',
                        (cache_key,)
                    )
                    self._update_statistics()
                    self._conn.commit()
                    return result
                else:
                    # Cache miss
                    self._miss_count += 1
                    self._update_statistics()
                    self._conn.commit()
                    return None
        except Exception:
            # On error, count as miss
            self._miss_count += 1
            return None

    def set(self, tool_name: str, query: str, result: str, **kwargs):
        """
        Store search result in cache.

        Args:
            tool_name: Name of the search tool (e.g., 'google_search', 'image_search')
            query: Search query string
            result: JSON string result to cache
            **kwargs: Additional search parameters
        """
        if not self.enabled:
            return

        cache_key = self._generate_cache_key(tool_name, query, **kwargs)

        # Serialize kwargs to JSON for storage
        params_str = json.dumps(kwargs, sort_keys=True) if kwargs else None

        try:
            with self._lock:
                # Use INSERT OR REPLACE to handle both new entries and updates
                self._conn.execute('''
                    INSERT OR REPLACE INTO search_cache (cache_key, tool_name, query, params, result)
                    VALUES (?, ?, ?, ?, ?)
                ''', (cache_key, tool_name, query, params_str, result))
                self._conn.commit()
        except Exception as e:
            print(f"Warning: Failed to save cache entry: {e}")

    def clear(self):
        """Clear all cached results."""
        if not self.enabled:
            return
        try:
            with self._lock:
                self._conn.execute('DELETE FROM search_cache')
                self._conn.commit()
        except Exception as e:
            print(f"Warning: Failed to clear cache: {e}")

    def remove(self, tool_name: str, query: str, **kwargs):
        """
        Remove specific entry from cache.

        Args:
            tool_name: Name of the search tool
            query: Search query string
            **kwargs: Additional search parameters
        """
        if not self.enabled:
            return
        cache_key = self._generate_cache_key(tool_name, query, **kwargs)
        try:
            with self._lock:
                self._conn.execute('DELETE FROM search_cache WHERE cache_key = ?', (cache_key,))
                self._conn.commit()
        except Exception as e:
            print(f"Warning: Failed to remove cache entry: {e}")

    def get_statistics(self) -> Dict[str, int]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary containing:
                - hit_count: Number of cache hits
                - miss_count: Number of cache misses
                - total_queries: Total number of queries (hits + misses)
                - hit_rate: Cache hit rate as percentage (0-100)
                - total_entries: Total number of cached entries
        """
        if not self.enabled:
            return {
                "hit_count": 0,
                "miss_count": 0,
                "total_queries": 0,
                "hit_rate": 0.0,
                "total_entries": 0
            }

        total_queries = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / total_queries * 100) if total_queries > 0 else 0.0

        # Get total entries from cache
        cursor = self._conn.execute('SELECT COUNT(*) FROM search_cache')
        total_entries = cursor.fetchone()[0]

        return {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "total_queries": total_queries,
            "hit_rate": hit_rate,
            "total_entries": total_entries
        }

    def reset_statistics(self):
        """Reset hit/miss counters to zero."""
        if not self.enabled:
            return
        with self._lock:
            self._hit_count = 0
            self._miss_count = 0
            self._update_statistics()
            self._conn.commit()


# Global cache instance (can be configured via environment variable)
_global_cache: Optional[SearchCache] = None


def get_search_cache() -> SearchCache:
    """
    Get the global search cache instance.

    Returns:
        SearchCache instance
    """
    global _global_cache
    if _global_cache is None:
        cache_path = os.getenv("MIROFLOW_SEARCH_CACHE_PATH")
        _global_cache = SearchCache(cache_path=cache_path)
    return _global_cache


def reset_search_cache():
    """Reset the global search cache instance."""
    global _global_cache
    _global_cache = None
