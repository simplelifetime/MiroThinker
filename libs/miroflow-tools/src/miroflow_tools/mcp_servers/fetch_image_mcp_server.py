# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Fetch Image MCP Server

This server provides a tool to download images from URLs and load them into
the agent's context in a multi-modal format compatible with OpenAI's API.
"""

import base64
import mimetypes
from urllib.parse import urlparse

from fastmcp import FastMCP
import requests


def is_valid_image(content: bytes) -> bool:
    """
    Check if the content is a valid image by checking magic bytes.

    Args:
        content: The downloaded content bytes

    Returns:
        True if content starts with valid image magic bytes, False otherwise
    """
    if not content or len(content) < 8:
        return False

    # Common image file signatures (magic bytes)
    image_signatures = [
        b'\xFF\xD8\xFF',  # JPEG
        b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A',  # PNG
        b'GIF87a',  # GIF87a
        b'GIF89a',  # GIF89a
        b'BM',  # BMP
        b'II*\x00',  # TIFF (little-endian)
        b'MM\x00*',  # TIFF (big-endian)
        b'\x00\x00\x01\x00',  # ICO
        b'\x52\x49\x46\x46',  # WEBP (RIFF)
    ]

    return any(content.startswith(sig) for sig in image_signatures)

# Initialize FastMCP server
mcp = FastMCP("fetch-image-mcp-server")


def get_mime_type_from_url(url: str) -> str:
    """
    Guess MIME type from URL.

    Args:
        url: The URL to analyze

    Returns:
        MIME type string (defaults to image/jpeg if unknown)
    """
    # Try to get MIME type from URL extension
    mime_type, _ = mimetypes.guess_type(url)

    # Default to image/jpeg for unknown types
    if mime_type is None or not mime_type.startswith('image/'):
        return 'image/jpeg'

    return mime_type


def download_image_from_url(image_url: str, timeout: int = 30) -> tuple[bytes, str, str]:
    """
    Download image from URL and return bytes, MIME type, and error message.

    Args:
        image_url: URL of the image to download
        timeout: Request timeout in seconds

    Returns:
        Tuple of (image_bytes, mime_type, error_message)
        - If successful: (bytes, mime_type, "")
        - If failed: (None, None, error_message)
    """
    try:
        # Validate URL format
        parsed_url = urlparse(image_url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            return None, None, f"Invalid URL format: {image_url}"

        if parsed_url.scheme not in ['http', 'https']:
            return None, None, f"Unsupported URL scheme: {parsed_url.scheme}. Only http and https are supported."

        # Download image
        response = requests.get(image_url, timeout=timeout, stream=True)
        response.raise_for_status()

        # Get MIME type
        mime_type = get_mime_type_from_url(image_url)

        # Optionally, check Content-Type header
        content_type = response.headers.get('Content-Type', '')
        if content_type.startswith('image/'):
            mime_type = content_type

        # Validate that the content is actually an image
        if not is_valid_image(response.content):
            # Provide helpful error message
            content_preview = response.content[:200].decode('utf-8', errors='ignore')
            if '<html' in content_preview.lower() or '<!doctype' in content_preview.lower():
                return None, None, f"The URL does not point to a valid image. The response appears to be an HTML page (possibly a redirect page or error page). URL: {image_url}"
            else:
                return None, None, f"The URL does not point to a valid image. Content-Type: {content_type}, URL: {image_url}"

        return response.content, mime_type, ""

    except requests.exceptions.Timeout:
        return None, None, f"Download timeout after {timeout} seconds"
    except requests.exceptions.RequestException as e:
        return None, None, f"Download failed: {str(e)}"
    except Exception as e:
        return None, None, f"Unexpected error: {str(e)}"


@mcp.tool()
async def fetch_image(url: str) -> str:
    """Download an image from a URL and load it into the agent's context.

    This tool downloads an image from the provided URL and converts it to base64 format,
    then returns it in a multi-modal format that can be directly processed by vision-capable
    LLMs. The image will be included in the conversation history for visual analysis.

    Args:
        url: The URL of the image to download. Must start with http:// or https://

    Returns:
        A JSON-formatted string containing the image data in multi-modal format.
        The format is a list with two elements:
        1. A text description of the image source
        2. The image data in base64 format with appropriate MIME type

    Example:
        Image downloaded from https://example.com/image.jpg will be returned as:
        [
            {"type": "text", "text": "Image downloaded from: https://example.com/image.jpg"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]

    Note:
        - Supported image formats: jpg, jpeg, png, gif, webp, bmp, etc.
        - Maximum download timeout: 30 seconds
        - The image is automatically encoded to base64 for inclusion in the context
        - Vision-capable models will be able to directly analyze the downloaded image
    """
    # Download image
    image_bytes, mime_type, error_message = download_image_from_url(url)

    if error_message:
        # Return error message in JSON format
        return f'{{"error": "{error_message}"}}'

    # Encode to base64
    try:
        base64_data = base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        return f'{{"error": "Failed to encode image to base64: {str(e)}"}}'

    # Create data URL
    data_url = f"data:{mime_type};base64,{base64_data}"

    # Return multi-modal format as JSON string
    # This format is compatible with OpenAI's multi-modal input
    import json
    result = [
        {
            "type": "text",
            "text": f"Image downloaded from: {url}"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": data_url
            }
        }
    ]

    return json.dumps(result)


if __name__ == "__main__":
    mcp.run(transport="stdio")
