# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Search cache management for serper-based search tools.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class SearchCache:
    """
    Cache manager for search tool results.

    Stores search results in a JSON file with keys based on tool name and parameters.
    """

    def __init__(self, cache_path: Optional[str] = None):
        """
        Initialize the search cache.

        Args:
            cache_path: Path to the cache file. If None, uses default path.
        """
        if cache_path is None:
            # Default cache path in user's home directory
            cache_dir = Path.home() / "MiroThinker" / ".miroflow_tools" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / "search_cache.json"

        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Any] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cache from file."""
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

    def get(self, tool_name: str, query: str, **kwargs) -> Optional[str]:
        """
        Get cached search result if available.

        Args:
            tool_name: Name of the search tool (e.g., 'google_search', 'image_search')
            query: Search query string
            **kwargs: Additional search parameters

        Returns:
            Cached JSON string result, or None if not found
        """
        cache_key = self._generate_cache_key(tool_name, query, **kwargs)

        if cache_key in self._cache:
            return self._cache[cache_key]

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
        cache_key = self._generate_cache_key(tool_name, query, **kwargs)
        self._cache[cache_key] = result
        self._save_cache()

    def clear(self):
        """Clear all cached results."""
        self._cache = {}
        self._save_cache()

    def remove(self, tool_name: str, query: str, **kwargs):
        """
        Remove specific entry from cache.

        Args:
            tool_name: Name of the search tool
            query: Search query string
            **kwargs: Additional search parameters
        """
        cache_key = self._generate_cache_key(tool_name, query, **kwargs)
        if cache_key in self._cache:
            del self._cache[cache_key]
            self._save_cache()


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
