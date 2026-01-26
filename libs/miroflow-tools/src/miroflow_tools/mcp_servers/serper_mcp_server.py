# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
adapted from
https://github.com/MiroMindAI/MiroRL/blob/5073693549ffe05a157a1886e87650ef3be6606e/mirorl/tools/serper_search.py#L1
"""

import base64
import json
import os
from typing import Any, Dict, List

import requests
from mcp.server.fastmcp import FastMCP
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .utils import decode_http_urls_in_dict
from .utils.search_cache import get_search_cache


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

# Initialize FastMCP server
mcp = FastMCP("serper-mcp-server")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (requests.ConnectionError, requests.Timeout, requests.HTTPError)
    ),
)
def make_serper_request(
    endpoint: str, payload: Dict[str, Any], headers: Dict[str, str]
) -> requests.Response:
    """Make HTTP request to Serper API with retry logic."""
    response = requests.post(f"{SERPER_BASE_URL}/{endpoint}", json=payload, headers=headers)
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
    # Check for API key
    if not SERPER_API_KEY:
        return json.dumps(
            {
                "success": False,
                "error": "SERPER_API_KEY environment variable not set",
                "results": [],
            },
            ensure_ascii=False,
        )

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
    cache = get_search_cache()
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
        print(f"[SEARCH_CACHE] Cache HIT for google_search: '{q}'")
        return cached_result

    try:
        # Build payload with all supported parameters
        payload: dict[str, Any] = {
            "q": q.strip(),
            "gl": gl,
            "hl": hl,
        }

        # Add optional parameters if provided
        if location:
            payload["location"] = location
        if num is not None:
            payload["num"] = num
        else:
            payload["num"] = 10  # Default
        if tbs:
            payload["tbs"] = tbs
        if page is not None:
            payload["page"] = page
        if autocorrect is not None:
            payload["autocorrect"] = autocorrect

        # Set up headers
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

        # Make the API request
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
        cache.set("google_search", q, result_json, **cache_params)

        return result_json

    except Exception as e:
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
    # Check for API key
    if not SERPER_API_KEY:
        return json.dumps(
            {
                "success": False,
                "error": "SERPER_API_KEY environment variable not set",
                "results": [],
            },
            ensure_ascii=False,
        )

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
        # Build payload with all supported parameters
        payload: dict[str, Any] = {
            "q": q.strip(),
            "gl": gl,
            "hl": hl,
        }

        # Add optional parameters if provided
        if num is not None:
            payload["num"] = num
        else:
            payload["num"] = 10  # Default
        if page is not None:
            payload["page"] = page

        # Set up headers
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

        # Make the API request to scholar endpoint
        response = make_serper_request("scholar", payload, headers)
        data = response.json()
        data = decode_http_urls_in_dict(data)

        # Limit organic results to the requested number
        requested_num = num if num is not None else 10
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
    # Check for API key
    if not SERPER_API_KEY:
        return json.dumps(
            {
                "success": False,
                "error": "SERPER_API_KEY environment variable not set",
                "results": [],
            },
            ensure_ascii=False,
        )

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
    cache = get_search_cache()
    # Normalize parameters to match actual API request
    normalized_num = num if num is not None else 5
    normalized_page = page if page is not None else 1
    
    cache_params = {
        "gl": gl,
        "hl": hl,
        "num": normalized_num,
        "page": normalized_page,
    }
    if location:
        cache_params["location"] = location

    cached_result = cache.get("image_search", q, **cache_params)
    if cached_result is not None:
        print(f"[SEARCH_CACHE] Cache HIT for image_search: '{q}'")
        return cached_result

    try:
        # Build payload with all supported parameters
        payload: dict[str, Any] = {
            "q": q.strip(),
            "gl": gl,
            "hl": hl,
        }

        # Add optional parameters if provided
        if location:
            payload["location"] = location
        if num is not None:
            payload["num"] = num
        else:
            payload["num"] = 5  # Default
        if page is not None:
            payload["page"] = page

        # Set up headers
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

        # Make the API request to images endpoint
        response = make_serper_request("images", payload, headers)
        data = response.json()
        data = decode_http_urls_in_dict(data)

        # Limit images to requested number (return metadata only, no download/encoding)
        requested_num = num if num is not None else 5
        if "images" in data and isinstance(data["images"], list):
            data["images"] = data["images"][:requested_num]

        result_json = json.dumps(data, ensure_ascii=False)

        # Cache the result with the same normalized parameters
        cache.set("image_search", q, result_json, **cache_params)

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
    # Check for API key
    if not SERPER_API_KEY:
        return json.dumps(
            {
                "success": False,
                "error": "SERPER_API_KEY environment variable not set",
                "results": [],
            },
            ensure_ascii=False,
        )

    # Validate required parameter
    if not image_url or not image_url.strip():
        return json.dumps(
            {
                "success": False,
                "error": "Image URL 'image_url' is required and cannot be empty",
                "results": [],
            },
            ensure_ascii=False,
        )

    # Basic URL validation
    if not image_url.startswith(("http://", "https://")):
        return json.dumps(
            {
                "success": False,
                "error": "Invalid image URL format. URLs must start with http:// or https://",
                "results": [],
            },
            ensure_ascii=False,
        )

    try:
        # Build payload with all supported parameters
        payload: dict[str, Any] = {
            "url": image_url.strip(),
            "gl": gl,
            "hl": hl,
        }

        # Add optional parameters if provided
        if location:
            payload["location"] = location
        if num is not None:
            payload["num"] = num
        else:
            payload["num"] = 5  # Default
        if page is not None:
            payload["page"] = page

        # Set up headers
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

        # Make the API request to lens endpoint
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
