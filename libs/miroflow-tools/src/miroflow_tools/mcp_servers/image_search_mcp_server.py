# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Image search MCP server for multimodal search functionality.

This module provides tools for image search including:
- ImageSearch: Text-based image search using Serper API
- visual_search: Visual search (image-to-image) using SerpAPI Google Lens
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

import requests
from fastmcp import FastMCP
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Initialize FastMCP server
mcp = FastMCP("image-search-mcp-server")

# API Keys
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
SERPER_BASE_URL = os.environ.get("SERPER_BASE_URL", "https://google.serper.dev")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY", "")
SERPAPI_BASE_URL = os.environ.get("SERPAPI_BASE_URL", "https://serpapi.com")


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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (requests.ConnectionError, requests.Timeout, requests.HTTPError)
    ),
)
def make_serpapi_request(params: Dict[str, Any]) -> requests.Response:
    """Make HTTP request to SerpAPI with retry logic."""
    response = requests.get(SERPAPI_BASE_URL + "/search", params=params)
    response.raise_for_status()
    return response


def parse_image_reference(reference_id: str) -> Optional[str]:
    """
    Parse image reference ID from format like '<image: 0>' to extract the image index.

    Args:
        reference_id: Image reference string

    Returns:
        Image index as string, or None if parsing fails
    """
    match = re.match(r'<image:\s*(\d+)>', reference_id)
    if match:
        return match.group(1)
    return None


def parse_region_of_interest(roi_str: str) -> Optional[List[float]]:
    """
    Parse region of interest string to coordinates.

    Args:
        roi_str: ROI string in format '[min_x, min_y, max_x, max_y]'

    Returns:
        List of 4 float coordinates, or None if parsing fails
    """
    try:
        # Remove brackets and split by comma
        coords_str = roi_str.strip('[]')
        coords = [float(x.strip()) for x in coords_str.split(',')]

        if len(coords) == 4 and all(0 <= c <= 1 for c in coords):
            return coords
        return None
    except (ValueError, AttributeError):
        return None


@mcp.tool()
async def image_search(
    query: str,
    gl: str = "us",
    hl: str = "en",
    num: int = 10,
) -> str:
    """
    Perform image search using text query via Serper API.

    This tool searches for images based on text descriptions and returns
    image URLs along with their source webpage information.

    Args:
        query: Search query string for finding images
        gl: Country context for search (e.g., 'us' for United States, 'cn' for China)
        hl: Google interface language (e.g., 'en' for English, 'zh' for Chinese)
        num: Number of image results to return (default: 10, max: 100)

    Returns:
        JSON string containing search results with image URLs and metadata
    """
    if not SERPER_API_KEY:
        return json.dumps({
            "success": False,
            "error": "SERPER_API_KEY environment variable not set",
            "images": []
        }, ensure_ascii=False)

    if not query or not query.strip():
        return json.dumps({
            "success": False,
            "error": "Search query is required and cannot be empty",
            "images": []
        }, ensure_ascii=False)

    try:
        # Build payload for image search
        payload = {
            "q": query.strip(),
            "gl": gl,
            "hl": hl,
            "num": min(num, 100),  # Serper limits to 100 images
            "tbm": "isch"  # Image search
        }

        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

        # Make the API request
        response = make_serper_request("images", payload, headers)
        data = response.json()

        # Process image results
        images = []
        if "images" in data:
            for idx, img_data in enumerate(data["images"][:num]):
                image_info = {
                    "index": idx + 1,
                    "title": img_data.get("title", ""),
                    "imageUrl": img_data.get("imageUrl", ""),
                    "imageWidth": img_data.get("imageWidth", 0),
                    "imageHeight": img_data.get("imageHeight", 0),
                    "thumbnailUrl": img_data.get("thumbnailUrl", ""),
                    "thumbnailWidth": img_data.get("thumbnailWidth", 0),
                    "thumbnailHeight": img_data.get("thumbnailHeight", 0),
                    "sourceUrl": img_data.get("link", ""),
                    "domain": img_data.get("domain", ""),
                    "position": img_data.get("position", 0)
                }
                images.append(image_info)

        return json.dumps({
            "success": True,
            "query": query,
            "total_results": len(images),
            "images": images
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Image search failed: {str(e)}",
            "query": query,
            "images": []
        }, ensure_ascii=False)


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode a local image file to base64 string.

    Args:
        image_path: Path to the local image file

    Returns:
        Base64 encoded string of the image

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the file is not a valid image
    """
    import base64
    import mimetypes
    import os

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Check if it's a valid image file
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith('image/'):
        raise ValueError(f"Invalid image file: {image_path}")

    # Read and encode the image
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    return image_data


@mcp.tool()
async def visual_search(
    reference_id: str,
    image_url: str = "",
    region_of_interest: str = "[0.0, 0.0, 1.0, 1.0]",
    thumbnail_size: str = "medium",
) -> str:
    """
    Perform visual search (image-to-image search) using SerpAPI Google Lens.

    This tool takes an image reference and searches for similar images across the web.
    Supports both URLs and local file paths.

    Args:
        reference_id: Image reference identifier (format: '<image: 0>', '<image: 1>', etc.)
        image_url: Direct image URL or local file path to search (if provided, overrides reference_id)
        region_of_interest: Region of interest as normalized coordinates [min_x, min_y, max_x, max_y]
        thumbnail_size: Size of returned thumbnails (tiny/small/medium/large)

    Returns:
        JSON string containing visual search results with similar images
    """
    if not SERPAPI_API_KEY:
        return json.dumps({
            "success": False,
            "error": "SERPAPI_API_KEY environment variable not set",
            "images": []
        }, ensure_ascii=False)

    # Parse image reference
    image_index = parse_image_reference(reference_id)
    if image_index is None and not image_url:
        return json.dumps({
            "success": False,
            "error": f"Invalid reference_id format: {reference_id}. Expected format: '<image: N>' or provide image_url",
            "images": []
        }, ensure_ascii=False)

    # Parse region of interest
    roi_coords = parse_region_of_interest(region_of_interest)
    if roi_coords is None:
        return json.dumps({
            "success": False,
            "error": f"Invalid region_of_interest format: {region_of_interest}. Expected format: '[x1,y1,x2,y2]' with values 0-1",
            "images": []
        }, ensure_ascii=False)

    try:
        # If image_url is provided, use it directly
        # Otherwise, image data should be available in the conversation context
        if not image_url:
            return json.dumps({
                "success": False,
                "error": "Image URL must be provided for visual search. Use image_url parameter or ensure image is in conversation context.",
                "reference_id": reference_id,
                "images": []
            }, ensure_ascii=False)

        # Prepare SerpAPI Google Lens request
        params = {
            "api_key": SERPAPI_API_KEY,
            "engine": "google_lens",
        }

        # Check if image_url is a local file path or URL
        import os
        if os.path.exists(image_url):
            # It's a local file - encode to base64
            try:
                image_base64 = encode_image_to_base64(image_url)
                params["image"] = image_base64
            except Exception as e:
                return json.dumps({
                    "success": False,
                    "error": f"Failed to encode local image file: {str(e)}",
                    "reference_id": reference_id,
                    "images": []
                }, ensure_ascii=False)
        else:
            # It's a URL
            params["url"] = image_url

        # Make the API request
        response = make_serpapi_request(params)
        data = response.json()

        # Process visual search results
        visual_matches = []
        
        # Extract visual matches from Google Lens results
        if "visual_matches" in data:
            for idx, match in enumerate(data["visual_matches"][:20]):
                match_info = {
                    "index": idx + 1,
                    "title": match.get("title", ""),
                    "link": match.get("link", ""),
                    "source": match.get("source", ""),
                    "thumbnail": match.get("thumbnail", ""),
                    "position": match.get("position", 0)
                }
                visual_matches.append(match_info)

        # Extract knowledge graph info if available
        knowledge_graph = None
        if "knowledge_graph" in data:
            kg = data["knowledge_graph"]
            knowledge_graph = {
                "title": kg.get("title", ""),
                "description": kg.get("description", ""),
                "source": kg.get("source", {}).get("name", "")
            }

        return json.dumps({
            "success": True,
            "reference_id": reference_id,
            "image_url": image_url,
            "region_of_interest": roi_coords,
            "thumbnail_size": thumbnail_size,
            "total_results": len(visual_matches),
            "knowledge_graph": knowledge_graph,
            "visual_matches": visual_matches
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Visual search failed: {str(e)}",
            "reference_id": reference_id,
            "image_url": image_url,
            "images": []
        }, ensure_ascii=False)


if __name__ == "__main__":
    # Run the MCP server
    import asyncio
    asyncio.run(mcp.run())
