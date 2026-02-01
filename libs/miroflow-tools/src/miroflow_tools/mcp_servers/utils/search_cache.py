# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Search cache management for serper-based search tools.
"""

import hashlib
import json
import os
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional


class SearchCache:
    """
    Cache manager for search tool results.

    Stores search results in a JSON file with keys based on tool name and parameters.
    """

    def __init__(self, cache_path: Optional[str] = None, enabled: Optional[bool] = None):
        """
        Initialize the search cache.

        Args:
            cache_path: Path to the cache file. If None, uses default path.
            enabled: Whether caching is enabled. If None, checks MIROFLOW_SEARCH_CACHE_ENABLED env var.
        """
        env_debug = os.getenv("MIROFLOW_SEARCH_CACHE_DEBUG", "false").lower()
        self._debug = env_debug in ("true", "1", "yes", "on")

        if enabled is None:
            env_enabled = os.getenv("MIROFLOW_SEARCH_CACHE_ENABLED", "true").lower()
            self.enabled = env_enabled not in ("false", "0", "no", "off")
        else:
            self.enabled = enabled

        self._lock = threading.RLock()
        self._cache: Dict[str, Any] = {}
        self.cache_path = None
        self.db_path = None
        self._conn = None

        if not self.enabled:
            self._log("disabled (MIROFLOW_SEARCH_CACHE_ENABLED=false)")
            return

        if cache_path is None:
            cache_dir = Path.home() / "MiroThinker" / ".miroflow_tools" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / "search_cache.json"

        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._load_cache()

    def _load_cache(self):
        """Load cache from file."""
        if self.cache_path is None:
            self._cache = {}
            return

        with self._lock:
            if self.cache_path.exists():
                try:
                    with open(self.cache_path, "r", encoding="utf-8") as f:
                        self._cache = json.load(f)
                except (json.JSONDecodeError, IOError):
                    self._cache = {}
            else:
                self._cache = {}

    def _save_cache(self):
        """Save cache to file."""
        if not self.enabled or self.cache_path is None:
            return

        with self._lock:
            try:
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(self._cache, f, ensure_ascii=False, indent=2)
            except IOError as e:
                self._log(f"save failed path={self.cache_path} err={e}")

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
            self._log(f"lookup skipped (disabled) tool={tool_name}")
            return None

        cache_key = self._generate_cache_key(tool_name, query, **kwargs)
        q_preview = (query or "").strip().replace("\n", " ")[:120]
        self._log(f"lookup tool={tool_name} key={cache_key} q={q_preview!r}")

        with self._lock:
            return self._cache.get(cache_key)

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
            self._log(f"store skipped (disabled) tool={tool_name}")
            return

        cache_key = self._generate_cache_key(tool_name, query, **kwargs)
        with self._lock:
            self._cache[cache_key] = result
        self._save_cache()

        self._log(
            f"store tool={tool_name} key={cache_key} bytes={len(result) if result is not None else 0}"
        )

    def clear(self):
        """Clear all cached results."""
        if not self.enabled:
            return
        with self._lock:
            self._cache = {}
        self._save_cache()

    def remove(self, tool_name: str, query: str, **kwargs):
        """Remove specific entry from cache."""
        if not self.enabled:
            return
        cache_key = self._generate_cache_key(tool_name, query, **kwargs)
        with self._lock:
            self._cache.pop(cache_key, None)
        self._save_cache()

    def _log(self, message: str):
        if not getattr(self, "_debug", False):
            return
        print(f"[SEARCH_CACHE] {message}", file=sys.stderr)

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
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Warning: Failed to save cache to {self.cache_path}: {e}")

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
