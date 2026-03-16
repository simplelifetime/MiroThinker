# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
adapted from
https://github.com/MiroMindAI/MiroRL/blob/5073693549ffe05a157a1886e87650ef3be6606e/mirorl/tools/serper_search.py#L1
"""

import base64
import json
import logging
import os
import re
import time
from typing import Any, Dict, List

import requests
from mcp.server.fastmcp import FastMCP
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .utils import decode_http_urls_in_dict
from .utils.search_cache import get_search_cache, get_current_task_id, set_current_task_id

logger = logging.getLogger(__name__)

# Initialize task-specific cache context if MIROFLOW_TASK_ID is set
# This must be done at module load time to ensure all cache operations use the correct task_id
_task_id = os.getenv("MIROFLOW_TASK_ID")
_cache_enabled_env = os.getenv("MIROFLOW_SEARCH_CACHE_ENABLED", "true")
# logger.info(f"[SEARCH_CACHE] serper_mcp_server module loading, MIROFLOW_TASK_ID env var = {_task_id}")
# logger.info(f"[SEARCH_CACHE] serper_mcp_server module loading, MIROFLOW_SEARCH_CACHE_ENABLED env var = {_cache_enabled_env}")
if _task_id:
    set_current_task_id(_task_id)
    # logger.info(f"[SEARCH_CACHE] MCP server initialized with task_id: {_task_id}")
# else:
#     logger.info(f"[SEARCH_CACHE] MCP server initialized WITHOUT task_id (will use pid-based cache)")
#     # If no task_id is provided, log a warning but still create a cache
    # logger.warning("[SEARCH_CACHE] No MIROFLOW_TASK_ID found, search results will NOT be cached to task-specific file!")

# Create a single cache instance for this MCP server process
# This instance will be reused for all tool calls in this process
# IMPORTANT: If task_id is None, this will create a pid-based cache which won't be merged!
_mcp_server_cache = get_search_cache(task_id=_task_id)
# logger.info(f"[SEARCH_CACHE] Created MCP server cache instance: task_id={_mcp_server_cache.task_id}, file={_mcp_server_cache.task_cache_file}, enabled={_mcp_server_cache.enabled}")


def download_and_encode_images(
    image_results: List[Dict[str, Any]], max_images: int = 5, limit_results: bool = True
) -> List[Dict[str, Any]]:
    """
    Download and encode images to base64 format.

    Args:
        image_results: List of image search results with 'imageUrl' field
        max_images: Maximum number of images to process (default: 5)
        limit_results: If True, only return max_images results; if False, return all results but only encode first max_images (default: True)

    Returns:
        List of image results with added base64 data
    """
    processed_results = []

    for idx, result in enumerate(image_results[:max_images]):
        image_url = result.get("imageUrl") or result.get("link", "")
        if not image_url:
            continue

        try:
            # Download image
            response = requests.get(image_url, timeout=10, stream=True)
            response.raise_for_status()

            # Encode to base64
            image_base64 = base64.b64encode(response.content).decode("utf-8")
            image_base64_with_mime = f"data:image/jpeg;base64,{image_base64}"

            # Add base64 data to result
            result_copy = result.copy()
            result_copy["base64_data"] = image_base64_with_mime

            processed_results.append(result_copy)

        except Exception as e:
            print(f"Warning: Failed to download/encode image {idx + 1}: {str(e)}")
            # Keep the result without base64 data
            processed_results.append(result)

    # Include remaining results without processing (only if limit_results is False)
    if not limit_results and len(image_results) > max_images:
        processed_results.extend(image_results[max_images:])

    return processed_results

SERPER_BASE_URL = os.getenv("SERPER_BASE_URL", "https://google.serper.dev")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

# "serper" (default) or "api_hub"
GOOGLE_SEARCH_PROXY = os.getenv("GOOGLE_SEARCH_PROXY", "serper")

# Separate proxy for image_search / visual_search; defaults to GOOGLE_SEARCH_PROXY
# so existing behaviour is preserved unless explicitly overridden.
IMAGE_SEARCH_PROXY = os.getenv("IMAGE_SEARCH_PROXY", GOOGLE_SEARCH_PROXY)

# APIHub configuration (only used when GOOGLE_SEARCH_PROXY=api_hub)
APIHUB_URL = "https://gpt.bytedance.net/gpt/tool_hub/online/mcp_server/proxy/apihub_google_search/mcp"
APIHUB_API_KEY = os.getenv("APIHUB_API_KEY", "")
APIHUB_USER_EMAIL = os.getenv("APIHUB_USER_EMAIL", "")

# global_search_v2 for ImageSearch/VisualSearch (different MCP server, different api_key)
GLOBAL_SEARCH_URL = "https://gpt.bytedance.net/gpt/tool_hub/online/mcp_server/proxy/global_search_v2/mcp"
GLOBAL_SEARCH_API_KEY = os.getenv("GLOBAL_SEARCH_API_KEY", "c165b7dc-5a6d-4f37-b37d-87fced6ece7a")

_apihub_auth_proxy = None

def _get_apihub_auth_proxy():
    """Lazy-load apihub_auth_proxy to avoid import errors when not using api_hub."""
    global _apihub_auth_proxy
    if _apihub_auth_proxy is None:
        os.environ.setdefault("SEC_TOKEN_PATH", "/etc/tce_dynamic/identity.token")
        os.environ.setdefault("BYTE_REGION", "CN")
        from seed.auth import apihub_auth_proxy
        _apihub_auth_proxy = apihub_auth_proxy
    return _apihub_auth_proxy


def _make_apihub_request(search_request: Dict[str, Any], max_attempts: int = 3) -> Dict[str, Any]:
    """
    Make a synchronous request to APIHub apihub_google_search_bayou.
    Uses the sync apihub_auth_proxy.post() method.

    Returns the parsed inner JSON (search_response_list etc.) or raises an exception.
    """
    auth_proxy = _get_apihub_auth_proxy()
    headers = {
        "api-key": APIHUB_API_KEY,
        "Content-Type": "application/json",
        "project-id": os.getenv("MERLIN_JOB_ID", "0"),
        "user": APIHUB_USER_EMAIL,
    }
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "apihub_google_search_bayou",
            "arguments": {
                "search_request_list": [search_request]
            },
        },
    }

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = auth_proxy.post(url=APIHUB_URL, headers=headers, json=payload, timeout=30)
            resp_json = resp.json()

            if "error" in resp_json and resp_json["error"]:
                raise RuntimeError(f"APIHub error: {resp_json['error']}")

            result = resp_json.get("result", {})
            if result.get("isError"):
                content = result.get("content", [])
                text = next((c["text"] for c in content if c.get("type") == "text"), "unknown error")
                raise RuntimeError(f"APIHub backend error: {text[:300]}")

            mcp_content = result["content"]
            text_item = next(item for item in mcp_content if item.get("type") == "text")
            return json.loads(text_item["text"])

        except Exception as e:
            last_error = e
            logger.warning(f"APIHub request failed (attempt {attempt}/{max_attempts}): {e}")
            if attempt < max_attempts:
                time.sleep(2 * attempt)

    raise last_error


def _apihub_google_search(q: str, gl: str, hl: str, num: int) -> dict:
    """
    Execute Google search via APIHub proxy.
    Converts the APIHub response format to match Serper's format for compatibility.

    Serper organic item fields: title, link, snippet, position, date, sitelinks, ...
    APIHub provides: title, url, snippet. We map url→link and add position.
    Fields not available from APIHub (date, sitelinks) will be absent.
    """
    search_request = {"query": q.strip(), "gl": gl, "hl": hl, "num": num}
    inner = _make_apihub_request(search_request)

    organic = []
    for sr in inner.get("search_response_list", []):
        docs = sorted(sr.get("documents", []), key=lambda d: d.get("rank", 1e9))
        for idx, doc in enumerate(docs):
            di = doc.get("doc_info", doc)
            snippet = di.get("snippet", [{"text": ""}])
            if isinstance(snippet, list) and snippet:
                snippet = snippet[0].get("text", "")
            organic.append({
                "title": di.get("title", ""),
                "link": di.get("url", ""),
                "snippet": snippet if isinstance(snippet, str) else "",
                "position": idx + 1,
            })

    return {
        "searchParameters": {"q": q, "gl": gl, "hl": hl, "num": num, "type": "search"},
        "organic": organic[:num],
    }


def _apihub_scholar_search(q: str, gl: str, hl: str, num: int) -> dict:
    """
    Execute Google Scholar search via APIHub proxy (search_type=scholar).
    Converts the APIHub response format to match Serper's format for compatibility.

    Serper scholar organic item fields: title, link, snippet, position, year,
        publicationInfo, citedBy, ...
    APIHub provides: title, url, snippet, host_info.hostname. We map accordingly.
    Fields not available from APIHub (year, citedBy) will be absent.
    """
    search_request = {"query": q.strip(), "search_type": "scholar", "gl": gl, "hl": hl, "num": num}
    inner = _make_apihub_request(search_request)

    organic = []
    for sr in inner.get("search_response_list", []):
        docs = sorted(sr.get("documents", []), key=lambda d: d.get("rank", 1e9))
        for idx, doc in enumerate(docs):
            di = doc.get("doc_info", doc)
            snippet = di.get("snippet", [{"text": ""}])
            if isinstance(snippet, list) and snippet:
                snippet = snippet[0].get("text", "")
            url = di.get("htmlUrl") or di.get("pdfUrl") or di.get("url", "")
            entry = {
                "title": di.get("title", ""),
                "link": url,
                "snippet": snippet if isinstance(snippet, str) else "",
                "position": idx + 1,
            }
            publication = (
                di.get("publicationInfo")
                or di.get("publication")
                or doc.get("host_info", {}).get("hostname", "")
            )
            if publication:
                entry["publicationInfo"] = publication
            year = di.get("year", di.get("publish_time", ""))
            if year:
                entry["year"] = year
            cited_by = di.get("citedBy", "")
            if cited_by:
                entry["citedBy"] = cited_by
            organic.append(entry)

    return {
        "searchParameters": {"q": q, "gl": gl, "hl": hl, "num": num, "type": "scholar"},
        "organic": organic[:num],
    }


def _make_global_search_request(tool_name: str, arguments: Dict[str, Any], max_attempts: int = 3) -> Dict[str, Any]:
    """
    Make a synchronous request to global_search_v2 MCP server.
    Used for ImageSearch and VisualSearch.
    """
    auth_proxy = _get_apihub_auth_proxy()
    headers = {
        "api-key": GLOBAL_SEARCH_API_KEY,
        "Content-Type": "application/json",
        "call_email": APIHUB_USER_EMAIL,
    }
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = auth_proxy.post(url=GLOBAL_SEARCH_URL, headers=headers, json=payload, timeout=60)
            resp_json = resp.json()

            result = resp_json.get("result", {})
            if result.get("isError"):
                content = result.get("content", [])
                text = next((c["text"] for c in content if c.get("type") == "text"), "unknown error")
                raise RuntimeError(f"global_search_v2 error: {text[:300]}")

            mcp_content = result["content"]
            text_item = next(item for item in mcp_content if item.get("type") == "text")
            inner = json.loads(text_item["text"])
            sr_data = inner.get("result", inner)
            return sr_data

        except Exception as e:
            last_error = e
            logger.warning(f"global_search_v2 {tool_name} failed (attempt {attempt}/{max_attempts}): {e}")
            if attempt < max_attempts:
                time.sleep(2 * attempt)

    raise last_error


def _apihub_image_search(q: str, gl: str, hl: str, num: int) -> dict:
    """
    Execute image search via global_search_v2 ImageSearch.
    Converts response to match Serper /images format.

    Serper images item: title, imageUrl, imageWidth, imageHeight, thumbnailUrl,
                        source, domain, link, position
    """
    sr_data = _make_global_search_request("ImageSearch", {
        "search_request_list": [{"query": q.strip(), "thumbnail_size": "small"}]
    })

    images = []
    for sr in sr_data.get("search_response_list", []):
        for idx, doc in enumerate(sr.get("documents", [])):
            di = doc.get("doc_info", doc)
            title = di.get("title", "")
            link = di.get("url", "")

            image_url = ""
            thumbnail_url = ""
            width = 0
            height = 0
            text_snippet = ""
            for s in di.get("snippet", []):
                if s.get("type") == "image":
                    img = s.get("image", {})
                    image_url = img.get("image_url") or img.get("display_url") or img.get("internal_url", "")
                    thumbnail_url = img.get("thumbnail_internal_url") or img.get("thumbnail_display_url", "")
                    width = img.get("width", 0)
                    height = img.get("height", 0)
                elif s.get("type") == "text":
                    text_snippet = s.get("text", "")

            images.append({
                "title": title,
                "imageUrl": image_url or thumbnail_url,
                "imageWidth": width,
                "imageHeight": height,
                "thumbnailUrl": thumbnail_url,
                "link": link,
                "source": title,
                "position": idx + 1,
            })

    return {
        "searchParameters": {"q": q, "gl": gl, "hl": hl, "num": num, "type": "images"},
        "images": images[:num],
    }


_OSS_PUBLIC_TO_INTERNAL_RE = re.compile(
    r"(\.oss-cn-[a-z]+)(\.)(?!internal)(aliyuncs\.com)"
)


def _to_oss_internal_url(url: str) -> str:
    """Convert Alibaba Cloud OSS public endpoint to VPC-internal endpoint."""
    return _OSS_PUBLIC_TO_INTERNAL_RE.sub(r"\1-internal\2\3", url)


def _apihub_visual_search(image_url: str, gl: str, hl: str, num: int) -> dict:
    """
    Execute visual search via global_search_v2 VisualSearch.
    Converts response to match Serper /lens format.

    Serper lens organic item: title, link, snippet, imageUrl, position
    """
    image_base64 = None
    url = image_url.strip()
    for download_url in (url, _to_oss_internal_url(url)):
        try:
            resp = requests.get(download_url, timeout=(5, 15))
            resp.raise_for_status()
            image_base64 = base64.b64encode(resp.content).decode("utf-8")
            break
        except Exception as e:
            logger.warning(f"Failed to download image from {download_url}: {e}")

    search_request: Dict[str, Any] = {
        "query": "",
        "image_query": {
            "url": image_url.strip(),
            "region_of_interest": {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1},
        },
        "thumbnail_size": "small",
    }
    if image_base64:
        search_request["image_query"]["image_base64"] = image_base64

    sr_data = _make_global_search_request("VisualSearch", {
        "search_request_list": [search_request]
    })

    organic = []
    all_images = []
    for sr in sr_data.get("search_response_list", []):
        for idx, doc in enumerate(sr.get("documents", [])):
            di = doc.get("doc_info", doc)
            title = di.get("title", "")
            link = di.get("url", "")

            image_url_val = ""
            thumbnail_url = ""
            text_snippet = ""
            for s in di.get("snippet", []):
                if s.get("type") == "image":
                    img = s.get("image", {})
                    image_url_val = img.get("image_url") or img.get("display_url") or img.get("internal_url", "")
                    thumbnail_url = img.get("thumbnail_internal_url") or img.get("thumbnail_display_url", "")
                elif s.get("type") == "text":
                    text_snippet = s.get("text", "")

            organic.append({
                "title": title,
                "link": link,
                "snippet": text_snippet,
                "imageUrl": image_url_val or thumbnail_url,
                "position": idx + 1,
            })
            if image_url_val or thumbnail_url:
                all_images.append({
                    "title": title,
                    "link": link,
                    "imageUrl": image_url_val or thumbnail_url,
                    "thumbnailUrl": thumbnail_url,
                    "position": idx + 1,
                })

    return {
        "searchParameters": {"q": image_url, "gl": gl, "hl": hl, "num": num, "type": "lens"},
        "organic": organic[:num],
        "images": all_images[:num],
    }


# Initialize FastMCP server
mcp = FastMCP("serper-mcp-server")


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type(
        (requests.ConnectionError, requests.Timeout, requests.HTTPError)
    ),
    before_sleep=lambda retry_state: logger.warning(
        f"Serper API request failed (attempt {retry_state.attempt_number}), "
        f"retrying in {retry_state.next_action.sleep:.1f}s: {retry_state.outcome.exception()}"
    ),
)
def make_serper_request(
    endpoint: str, payload: Dict[str, Any], headers: Dict[str, str]
) -> requests.Response:
    """Make HTTP request to Serper API with retry logic."""
    response = requests.post(
        f"{SERPER_BASE_URL}/{endpoint}", json=payload, headers=headers, timeout=30
    )
    response.raise_for_status()
    return response


def _is_huggingface_dataset_or_space_url(url):
    """
    Check if the URL is a HuggingFace dataset or space URL.
    :param url: The URL to check
    :return: True if it's a HuggingFace dataset or space URL, False otherwise
    """
    if not url:
        return False
    return "huggingface.co/datasets" in url or "huggingface.co/spaces" in url


@mcp.tool()
def google_search(
    q: str,
    gl: str = "us",
    hl: str = "en",
    location: str | None = None,
    num: int | None = None,
    tbs: str | None = None,
    page: int | None = None,
    autocorrect: bool | None = None,
):
    """
    Tool to perform web searches via Serper API and retrieve rich results.

    It is able to retrieve organic search results, people also ask,
    related searches, and knowledge graph.

    Args:
        q: Search query string
        gl: Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., 'us')
        hl: Optional language code for search results in ISO 639-1 format (e.g., 'en')
        location: Optional location for search results (e.g., 'SoHo, New York, United States', 'California, United States')
        num: Number of results to return (default: 10)
        tbs: Time-based search filter ('qdr:h' for past hour, 'qdr:d' for past day, 'qdr:w' for past week,
            'qdr:m' for past month, 'qdr:y' for past year)
        page: Page number of results to return (default: 1)
        autocorrect: Whether to autocorrect spelling in query

    Returns:
        Dictionary containing search results and metadata.
    """
    # Validate required parameter
    if not q or not q.strip():
        return json.dumps(
            {
                "success": False,
                "error": "Search query 'q' is required and cannot be empty",
                "results": [],
            },
            ensure_ascii=False,
        )

    # Check cache first
    # Use the module-level cache instance instead of creating a new one
    cache = _mcp_server_cache
    # if cache.enabled:
    #     logger.info(f"[SEARCH_CACHE] google_search: using cache task_id={cache.task_id}, cache_file={cache.task_cache_file}")
    # else:
    #     logger.info(f"[SEARCH_CACHE] google_search: cache is disabled, bypassing cache")
    # Normalize parameters to match actual API request
    normalized_num = num if num is not None else 10
    normalized_page = page if page is not None else 1

    cache_params = {
        "gl": gl,
        "hl": hl,
        "num": normalized_num,
        "page": normalized_page,
    }
    # Only include autocorrect if it's explicitly set
    if autocorrect is not None:
        cache_params["autocorrect"] = autocorrect
    if location:
        cache_params["location"] = location
    if tbs:
        cache_params["tbs"] = tbs

    cached_result = cache.get("google_search", q, **cache_params)
    if cached_result is not None:
        # logger.info(f"[SEARCH_CACHE] Cache HIT for google_search: '{q}'")
        return cached_result

    try:
        if GOOGLE_SEARCH_PROXY == "api_hub":
            data = _apihub_google_search(q, gl, hl, normalized_num)
        else:
            if not SERPER_API_KEY:
                return json.dumps(
                    {"success": False, "error": "SERPER_API_KEY environment variable not set", "results": []},
                    ensure_ascii=False,
                )

            payload: dict[str, Any] = {"q": q.strip(), "gl": gl, "hl": hl}
            if location:
                payload["location"] = location
            payload["num"] = num if num is not None else 10
            if tbs:
                payload["tbs"] = tbs
            if page is not None:
                payload["page"] = page
            if autocorrect is not None:
                payload["autocorrect"] = autocorrect

            headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
            response = make_serper_request("search", payload, headers)
            data = response.json()

        # filter out HuggingFace dataset or space urls
        organic_results = []
        if "organic" in data:
            for item in data["organic"]:
                if _is_huggingface_dataset_or_space_url(item.get("link", "")):
                    continue
                organic_results.append(item)

        # Limit organic results to the requested number
        requested_num = num if num is not None else 10
        organic_results = organic_results[:requested_num]

        # Keep all original fields, but overwrite "organic"
        response_data = dict(data)
        response_data["organic"] = organic_results
        response_data = decode_http_urls_in_dict(response_data)

        result_json = json.dumps(response_data, ensure_ascii=False)

        # Cache the result with the same normalized parameters
        # logger.info(f"[SEARCH_CACHE] google_search: calling cache.set for query='{q[:50]}...'")
        cache.set("google_search", q, result_json, **cache_params)
        # if cache.enabled:
        #     logger.info(f"[SEARCH_CACHE] google_search: cache.set completed, cache now has {len(cache._memory_cache)} entries")
        # else:
        #     logger.info(f"[SEARCH_CACHE] google_search: cache.set skipped (cache disabled)")

        # Immediately save to file after caching
        # This is necessary because each tool call runs in a separate process
        cache.save_to_file(force=True)
        # if cache.enabled:
        #     logger.info(f"[SEARCH_CACHE] google_search: saved cache to file")

        return result_json
    except RetryError as e:
        last_exception = e.last_attempt.exception()
        status_code = None
        if isinstance(last_exception, requests.HTTPError) and last_exception.response is not None:
            status_code = last_exception.response.status_code
        error_msg = (
            f"Serper API request failed after retries: "
            f"status_code={status_code}, last_error={str(last_exception)}"
        )
        logger.error(error_msg)
        return json.dumps(
            {"success": False, "retryable": True, "error": error_msg, "results": []},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"Unexpected error in google_search: {str(e)}")
        return json.dumps(
            {"success": False, "error": f"Unexpected error: {str(e)}", "results": []},
            ensure_ascii=False,
        )


@mcp.tool()
def scholar_search(
    q: str,
    gl: str = "us",
    hl: str = "en",
    num: int | None = None,
    page: int | None = None,
):
    """
    Tool to perform academic searches via Google Scholar through Serper API.

    Retrieve scholarly literature including articles, theses, books,
    abstracts, and court opinions from academic publishers, professional
    societies, online repositories, and universities.

    Args:
        q: Search query string for academic literature
        gl: Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., 'us')
        hl: Optional language code for search results in ISO 639-1 format (e.g., 'en')
        num: Number of results to return (default: 10)
        page: Page number of results to return (default: 1)

    Returns:
        Dictionary containing scholarly search results and metadata.
    """
    # Validate required parameter
    if not q or not q.strip():
        return json.dumps(
            {
                "success": False,
                "error": "Search query 'q' is required and cannot be empty",
                "results": [],
            },
            ensure_ascii=False,
        )

    try:
        requested_num = num if num is not None else 10

        if GOOGLE_SEARCH_PROXY == "api_hub":
            data = _apihub_scholar_search(q, gl, hl, requested_num)
        else:
            if not SERPER_API_KEY:
                return json.dumps(
                    {"success": False, "error": "SERPER_API_KEY environment variable not set", "results": []},
                    ensure_ascii=False,
                )

            payload: dict[str, Any] = {"q": q.strip(), "gl": gl, "hl": hl}
            payload["num"] = requested_num
            if page is not None:
                payload["page"] = page

            headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
            response = make_serper_request("scholar", payload, headers)
            data = response.json()

        data = decode_http_urls_in_dict(data)

        if "organic" in data and isinstance(data["organic"], list):
            data["organic"] = data["organic"][:requested_num]

        return json.dumps(data, ensure_ascii=False)

    except Exception as e:
        return json.dumps(
            {"success": False, "error": f"Unexpected error: {str(e)}", "results": []},
            ensure_ascii=False,
        )


@mcp.tool()
def image_search(
    q: str,
    gl: str = "us",
    hl: str = "en",
    location: str | None = None,
    num: int | None = None,
    page: int | None = None,
):
    """
    Tool to perform image searches via Serper API and retrieve visual results.

    Retrieve image search results including thumbnails, titles, and source URLs.
    Returns image metadata with URLs for reference, without downloading images.

    Args:
        q: Search query string for images
        gl: Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., 'us')
        hl: Optional language code for search results in ISO 639-1 format (e.g., 'en')
        location: Optional location for search results (e.g., 'SoHo, New York, United States', 'California, United States')
        num: Number of results to return (default: 5)
        page: Page number of results to return (default: 1)

    Returns:
        Dictionary containing image search results and metadata.
        Images are returned with URLs and metadata, without base64 encoding.
    """
    # Validate required parameter
    if not q or not q.strip():
        return json.dumps(
            {"success": False, "error": "Search query 'q' is required and cannot be empty", "results": []},
            ensure_ascii=False,
        )

    # Check cache first
    # Use the module-level cache instance instead of creating a new one
    cache = _mcp_server_cache
    # if cache.enabled:
    #     logger.info(f"[SEARCH_CACHE] image_search: using cache task_id={cache.task_id}, cache_file={cache.task_cache_file}")
    # else:
    #     logger.info(f"[SEARCH_CACHE] image_search: cache is disabled, bypassing cache")
    # Normalize parameters to match actual API request
    normalized_num = num if num is not None else 5
    normalized_page = page if page is not None else 1

    cache_params = {"gl": gl, "hl": hl, "num": normalized_num, "page": normalized_page}
    if location:
        cache_params["location"] = location

    cached_result = cache.get("image_search", q, **cache_params)
    if cached_result is not None:
        # logger.info(f"[SEARCH_CACHE] Cache HIT for image_search: '{q}'")
        return cached_result

    try:
        if IMAGE_SEARCH_PROXY == "api_hub":
            data = _apihub_image_search(q, gl, hl, normalized_num)
        else:
            if not SERPER_API_KEY:
                return json.dumps(
                    {"success": False, "error": "SERPER_API_KEY environment variable not set", "results": []},
                    ensure_ascii=False,
                )

            payload: dict[str, Any] = {"q": q.strip(), "gl": gl, "hl": hl}
            if location:
                payload["location"] = location
            payload["num"] = num if num is not None else 5
            if page is not None:
                payload["page"] = page

            headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
            response = make_serper_request("images", payload, headers)
            data = response.json()

        data = decode_http_urls_in_dict(data)

        requested_num = num if num is not None else 5
        if "images" in data and isinstance(data["images"], list):
            data["images"] = data["images"][:requested_num]

        result_json = json.dumps(data, ensure_ascii=False)

        # Cache the result with the same normalized parameters
        cache.set("image_search", q, result_json, **cache_params)

        # Immediately save to file after caching
        # This is necessary because each tool call runs in a separate process
        cache.save_to_file(force=True)

        return result_json

    except Exception as e:
        return json.dumps(
            {"success": False, "error": f"Unexpected error: {str(e)}", "results": []},
            ensure_ascii=False,
        )


@mcp.tool()
def visual_search(
    image_url: str,
    gl: str = "us",
    hl: str = "en",
    location: str | None = None,
    num: int | None = None,
    page: int | None = None,
):
    """
    Tool to perform visual searches via Serper Lens API to find similar images.

    Given an image URL, retrieve visually similar images from across the web.
    Returns image metadata with URLs for reference, without downloading images.

    Args:
        image_url: URL of the image to search with
        gl: Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., 'us')
        hl: Optional language code for search results in ISO 639-1 format (e.g., 'en')
        location: Optional location for search results (e.g., 'SoHo, New York, United States', 'California, United States')
        num: Number of results to return (default: 5)
        page: Page number of results to return (default: 1)

    Returns:
        Dictionary containing visually similar image search results and metadata.
        Images are returned with URLs and metadata, without base64 encoding.
    """
    # Validate required parameter
    if not image_url or not image_url.strip():
        return json.dumps(
            {"success": False, "error": "Image URL 'image_url' is required and cannot be empty", "results": []},
            ensure_ascii=False,
        )

    # Basic URL validation
    if not image_url.startswith(("http://", "https://")):
        return json.dumps(
            {"success": False, "error": "Invalid image URL format. URLs must start with http:// or https://", "results": []},
            ensure_ascii=False,
        )

    try:
        requested_num = num if num is not None else 5

        if IMAGE_SEARCH_PROXY == "api_hub":
            data = _apihub_visual_search(image_url, gl, hl, requested_num)
        else:
            if not SERPER_API_KEY:
                return json.dumps(
                    {"success": False, "error": "SERPER_API_KEY environment variable not set", "results": []},
                    ensure_ascii=False,
                )

            payload: dict[str, Any] = {"url": image_url.strip(), "gl": gl, "hl": hl}
            if location:
                payload["location"] = location
            payload["num"] = requested_num
            if page is not None:
                payload["page"] = page

            headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
            response = make_serper_request("lens", payload, headers)
            data = response.json()

        data = decode_http_urls_in_dict(data)

        # Limit organic results to the requested number
        requested_num = num if num is not None else 5
        if "organic" in data and isinstance(data["organic"], list):
            data["organic"] = data["organic"][:requested_num]

        # Limit images to requested number (return metadata only, no download/encoding)
        if "images" in data and isinstance(data["images"], list):
            data["images"] = data["images"][:requested_num]

        return json.dumps(data, ensure_ascii=False)

    except Exception as e:
        return json.dumps(
            {"success": False, "error": f"Unexpected error: {str(e)}", "results": []},
            ensure_ascii=False,
        )


if __name__ == "__main__":
    mcp.run()
