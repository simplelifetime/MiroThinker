# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Search cache management for serper-based search tools.
Uses JSON file-based cache with support for multi-task scenarios.

Design:
- Single task: Loads global cache, operates in memory, saves back on task completion
- Multi-task: Each task loads a copy of global cache, operates independently,
  saves to task-specific file (search_cache_task{task_id}.json), then merges all at the end
"""

import hashlib
import json
import os
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Optional

# Context variable to store the current task_id
# This allows task-specific cache to be used throughout the call stack
_current_task_id: ContextVar[str] = ContextVar('_current_task_id', default=None)


class SearchCache:
    """
    JSON file-based cache manager for search tool results.

    Thread-safe for concurrent access from multiple threads within the same process.
    Multi-process safe through per-process cache files with merge on exit.
    """

    def __init__(self, cache_path: Optional[str] = None, enabled: Optional[bool] = None,
                 task_id: Optional[str] = None):
        """
        Initialize the search cache.

        Args:
            cache_path: Path to the global cache JSON file. If None, uses default path.
            enabled: Whether caching is enabled. If None, checks MIROFLOW_SEARCH_CACHE_ENABLED env var.
            task_id: Optional task identifier for task-specific caching. If provided, cache will be
                    saved to search_cache_task{task_id}.json instead of a process-specific file.
        """
        # Check if caching is disabled via environment variable
        if enabled is None:
            env_enabled = os.getenv("MIROFLOW_SEARCH_CACHE_ENABLED", "true").lower()
            self.enabled = env_enabled not in ("false", "0", "no", "off")
        else:
            self.enabled = enabled

        # Thread lock for thread-safe operations
        self._lock = threading.Lock()

        # Performance statistics
        self._hit_count = 0
        self._miss_count = 0

        # Flag to avoid duplicate saves
        self._saved = False

        # Task ID for task-specific caching
        self.task_id = task_id

        if not self.enabled:
            self._memory_cache = None
            self.global_cache_file = None
            self.task_cache_file = None
            return

        # Determine cache directory
        if cache_path is None:
            cache_dir = Path.home() / "MiroThinker" / ".miroflow_tools" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / "search_cache.json"

        self.global_cache_file = Path(cache_path)

        # Determine task-specific cache file if task_id is provided
        cache_dir = self.global_cache_file.parent
        if task_id:
            # Sanitize task_id to make it safe for filename
            safe_task_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(task_id))
            self.task_cache_file = cache_dir / f"search_cache_task{safe_task_id}.json"
        else:
            # Fallback to process-specific file for backward compatibility
            pid = os.getpid()
            self.task_cache_file = cache_dir / f"search_cache_pid{pid}.json"

        # Load global cache into memory (if exists), otherwise start with empty dict
        self._memory_cache = {}
        self._load_cache()

    def _load_cache(self):
        """Load global cache file into memory."""
        if not self.enabled:
            return

        if self.global_cache_file.exists():
            try:
                with open(self.global_cache_file, 'r', encoding='utf-8') as f:
                    self._memory_cache = json.load(f)
                    print(f"[SEARCH_CACHE] Loaded {len(self._memory_cache)} entries from global cache")
            except Exception as e:
                print(f"[SEARCH_CACHE] Warning: Failed to load global cache: {e}")
                self._memory_cache = {}
        else:
            self._memory_cache = {}
            print("[SEARCH_CACHE] No existing global cache found, starting with empty cache")

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
                if cache_key in self._memory_cache:
                    # Cache hit!
                    entry = self._memory_cache[cache_key]
                    result = entry["result"]
                    self._hit_count += 1
                    # Update access time
                    entry["accessed_at"] = self._get_current_timestamp()
                    return result
                else:
                    # Cache miss
                    self._miss_count += 1
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
        current_time = self._get_current_timestamp()

        # Serialize kwargs to JSON for storage
        params_str = json.dumps(kwargs, sort_keys=True) if kwargs else None

        try:
            with self._lock:
                self._memory_cache[cache_key] = {
                    "tool_name": tool_name,
                    "query": query,
                    "params": params_str,
                    "result": result,
                    "created_at": current_time,
                    "accessed_at": current_time
                }
        except Exception as e:
            print(f"[SEARCH_CACHE] Warning: Failed to cache entry: {e}")

    def save_to_file(self):
        """
        Save current memory cache to task-specific JSON file.
        This should be called when the task completes.
        """
        if not self.enabled or not self._memory_cache:
            return

        # Avoid duplicate saves
        if self._saved:
            return

        try:
            # Create parent directory if it doesn't exist
            self.task_cache_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.task_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._memory_cache, f, ensure_ascii=False, indent=2)

            self._saved = True
            print(f"[SEARCH_CACHE] Saved {len(self._memory_cache)} entries to {self.task_cache_file.name}")
        except Exception as e:
            print(f"[SEARCH_CACHE] Warning: Failed to save cache to {self.task_cache_file}: {e}")

    def clear(self):
        """Clear all cached results from memory."""
        if not self.enabled:
            return
        with self._lock:
            self._memory_cache.clear()

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
                if cache_key in self._memory_cache:
                    del self._memory_cache[cache_key]
        except Exception as e:
            print(f"[SEARCH_CACHE] Warning: Failed to remove cache entry: {e}")

    def get_statistics(self) -> Dict[str, Any]:
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

        return {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "total_queries": total_queries,
            "hit_rate": hit_rate,
            "total_entries": len(self._memory_cache)
        }

    def reset_statistics(self):
        """Reset hit/miss counters to zero."""
        if not self.enabled:
            return
        with self._lock:
            self._hit_count = 0
            self._miss_count = 0

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def merge_caches(cache_dir: Optional[Path] = None) -> int:
        """
        Merge all task-specific (and legacy process-specific) cache files into the global cache file.

        This should be called after all tasks have completed.

        Args:
            cache_dir: Directory containing cache files. If None, uses default directory.

        Returns:
            Number of entries in the merged global cache

        Example:
            SearchCache.merge_caches()  # After all tasks complete
        """
        # Determine cache directory
        if cache_dir is None:
            cache_dir = Path.home() / "MiroThinker" / ".miroflow_tools" / "cache"

        cache_dir = Path(cache_dir)
        if not cache_dir.exists():
            print("[SEARCH_CACHE] Cache directory not found")
            return 0

        # Find all task-specific cache files
        task_cache_files = list(cache_dir.glob("search_cache_task*.json"))

        # Also find any legacy process-specific cache files for backward compatibility
        process_cache_files = list(cache_dir.glob("search_cache_pid*.json"))

        # Combine both lists
        all_cache_files = task_cache_files + process_cache_files

        if not all_cache_files:
            print("[SEARCH_CACHE] No task cache files found to merge")
            # Check if global cache exists, return its count
            global_cache = cache_dir / "search_cache.json"
            if global_cache.exists():
                try:
                    with open(global_cache, 'r', encoding='utf-8') as f:
                        merged_cache = json.load(f)
                        return len(merged_cache)
                except Exception:
                    pass
            return 0

        print(f"[SEARCH_CACHE] Found {len(all_cache_files)} cache files to merge "
              f"({len(task_cache_files)} task files, {len(process_cache_files)} legacy process files)")

        # Load global cache if it exists
        global_cache_file = cache_dir / "search_cache.json"
        merged_cache: Dict[str, Any] = {}

        if global_cache_file.exists():
            try:
                with open(global_cache_file, 'r', encoding='utf-8') as f:
                    merged_cache = json.load(f)
                    print(f"[SEARCH_CACHE] Loaded {len(merged_cache)} entries from existing global cache")
            except Exception as e:
                print(f"[SEARCH_CACHE] Warning: Failed to load global cache: {e}")
                merged_cache = {}

        # Merge each cache file
        total_new_entries = 0
        total_updated_entries = 0

        for cache_file in all_cache_files:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    file_cache = json.load(f)

                print(f"[SEARCH_CACHE] Merging {len(file_cache)} entries from {cache_file.name}")

                for cache_key, entry in file_cache.items():
                    if cache_key not in merged_cache:
                        # New entry
                        merged_cache[cache_key] = entry
                        total_new_entries += 1
                    else:
                        # Entry exists, keep the one with more recent access time
                        existing_time = merged_cache[cache_key].get("accessed_at", "")
                        new_time = entry.get("accessed_at", "")
                        if new_time > existing_time:
                            merged_cache[cache_key] = entry
                            total_updated_entries += 1

            except Exception as e:
                print(f"[SEARCH_CACHE] Warning: Failed to process {cache_file.name}: {e}")
                continue

        # Save merged cache to global cache file
        try:
            # Backup existing global cache
            if global_cache_file.exists():
                backup_file = cache_dir / "search_cache.json.backup"
                import shutil
                shutil.copy2(global_cache_file, backup_file)
                print(f"[SEARCH_CACHE] Backed up existing global cache to {backup_file.name}")

            # Write merged cache
            with open(global_cache_file, 'w', encoding='utf-8') as f:
                json.dump(merged_cache, f, ensure_ascii=False, indent=2)

            print(f"[SEARCH_CACHE] Merged cache saved: {len(merged_cache)} total entries "
                  f"({total_new_entries} new, {total_updated_entries} updated)")

        except Exception as e:
            print(f"[SEARCH_CACHE] Error: Failed to save merged cache: {e}")
            # Continue to cleanup anyway

        # Clean up all cache files (even if merge failed)
        try:
            for cache_file in all_cache_files:
                try:
                    cache_file.unlink()
                    print(f"[SEARCH_CACHE] Cleaned up {cache_file.name}")
                except Exception as e:
                    print(f"[SEARCH_CACHE] Warning: Failed to delete {cache_file.name}: {e}")
        except Exception as e:
            print(f"[SEARCH_CACHE] Warning: Error during cleanup: {e}")

        return len(merged_cache)


# Global cache instances
_global_cache: Optional[SearchCache] = None
_task_caches: Dict[str, SearchCache] = {}  # Task-specific cache instances


def set_current_task_id(task_id: str):
    """
    Set the current task_id in the context.

    This should be called at the beginning of a task to ensure all search operations
    use the task-specific cache.

    Args:
        task_id: The task identifier
    """
    _current_task_id.set(task_id)


def get_current_task_id() -> Optional[str]:
    """
    Get the current task_id from the context.

    Returns:
        The current task_id, or None if not set
    """
    return _current_task_id.get(None)


def get_search_cache(task_id: Optional[str] = None) -> SearchCache:
    """
    Get the search cache instance.

    Args:
        task_id: Optional task identifier. If provided, returns (or creates) a task-specific cache instance.
                If None, checks the current context for a task_id. If no task_id in context either,
                returns the global shared cache instance.

    Returns:
        SearchCache instance (cached per task_id for efficiency)
    """
    global _global_cache, _task_caches

    # Determine which task_id to use
    effective_task_id = task_id

    if effective_task_id is None:
        # Check if there's a task_id in the current context
        effective_task_id = get_current_task_id()

    # If task_id is provided (either explicitly or from context), use task-specific cache
    if effective_task_id is not None:
        # Check if we already created a cache instance for this task
        if effective_task_id not in _task_caches:
            cache_path = os.getenv("MIROFLOW_SEARCH_CACHE_PATH")
            _task_caches[effective_task_id] = SearchCache(cache_path=cache_path, task_id=effective_task_id)
            print(f"[SEARCH_CACHE] Created new cache instance for task: {effective_task_id}")
        return _task_caches[effective_task_id]

    # Otherwise, use the global shared cache (for backward compatibility)
    if _global_cache is None:
        cache_path = os.getenv("MIROFLOW_SEARCH_CACHE_PATH")
        _global_cache = SearchCache(cache_path=cache_path)
    return _global_cache


def reset_search_cache():
    """Reset the global search cache instance."""
    global _global_cache, _task_caches
    _global_cache = None
    _task_caches.clear()


def cleanup_task_cache(task_id: str):
    """
    Clean up a task-specific cache instance after the task completes.

    This should be called after saving the task cache to free up memory.

    Args:
        task_id: The task identifier to clean up
    """
    global _task_caches
    if task_id in _task_caches:
        del _task_caches[task_id]
        print(f"[SEARCH_CACHE] Cleaned up cache instance for task: {task_id}")
