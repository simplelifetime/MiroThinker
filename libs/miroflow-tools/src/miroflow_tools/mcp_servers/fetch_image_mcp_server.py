# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Fetch Image MCP Server

This server provides a tool to download images from URLs and load them into
the agent's context in a multi-modal format compatible with OpenAI's API.
"""

import base64
import io
import mimetypes
import os
from urllib.parse import urlparse

from fastmcp import FastMCP
import requests
from PIL import Image
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Load environment variables for pixel limits
# Parse environment variables - treat "0" or empty string as disabled (None)
def parse_pixel_limit(env_value: str) -> int | None:
    """Parse pixel limit from environment variable string."""
    if not env_value or env_value.strip() == "" or env_value.strip() == "0":
        return None
    try:
        value = int(env_value.strip())
        return value if value > 0 else None
    except (ValueError, AttributeError):
        return None

MIN_PIXELS = parse_pixel_limit(os.getenv("IMAGE_MIN_PIXELS", ""))
MAX_PIXELS = parse_pixel_limit(os.getenv("IMAGE_MAX_PIXELS", ""))


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

# Image format magic bytes (file signatures)
IMAGE_SIGNATURES = {
    b'\x89PNG\r\n\x1a\n': 'image/png',
    b'\xff\xd8\xff': 'image/jpeg',
    b'GIF87a': 'image/gif',
    b'GIF89a': 'image/gif',
    b'RIFF': 'image/webp',  # WebP starts with RIFF, need to check further
    b'BM': 'image/bmp',
    b'\x00\x00\x01\x00': 'image/x-icon',  # ICO
    b'\x00\x00\x02\x00': 'image/x-icon',  # CUR
}


def validate_image_content(image_bytes: bytes) -> tuple[bool, str, str]:
    """
    Validate that the downloaded content is actually an image by checking magic bytes.
    
    Args:
        image_bytes: The raw bytes of the downloaded content
        
    Returns:
        Tuple of (is_valid, detected_mime_type, error_message)
        - If valid image: (True, mime_type, "")
        - If invalid: (False, "", error_message)
    """
    if not image_bytes or len(image_bytes) < 8:
        return False, "", "Downloaded content is empty or too small to be a valid image"
    
    # Check for HTML content (common when server returns error page)
    # HTML typically starts with <!DOCTYPE, <html, or whitespace followed by these
    content_start = image_bytes[:100].strip().lower()
    if content_start.startswith(b'<!doctype') or content_start.startswith(b'<html') or content_start.startswith(b'<head') or content_start.startswith(b'<?xml'):
        return False, "", "Downloaded content is HTML/XML, not an image. The URL may have returned an error page or redirect."
    
    # Check magic bytes for known image formats
    for signature, mime_type in IMAGE_SIGNATURES.items():
        if image_bytes.startswith(signature):
            # Special case for WebP: need to verify "WEBP" at offset 8
            if signature == b'RIFF' and len(image_bytes) >= 12:
                if image_bytes[8:12] != b'WEBP':
                    continue  # Not a WebP file, might be other RIFF format
            return True, mime_type, ""
    
    # If no known signature matched, try to use PIL to validate
    try:
        from PIL import Image
        from io import BytesIO
        img = Image.open(BytesIO(image_bytes))
        img.verify()  # Verify it's a valid image
        mime_type = f"image/{img.format.lower()}" if img.format else "image/unknown"
        return True, mime_type, ""
    except ImportError:
        # PIL not available, accept unknown format with warning
        return True, "image/unknown", ""
    except Exception as e:
        return False, "", f"Downloaded content is not a valid image: {str(e)}"


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


def resize_image_with_pixel_limits(
    image: Image.Image,
    min_pixels: int = None,
    max_pixels: int = None,
) -> tuple[Image.Image, bool]:
    """
    Resize image to fit within pixel limits while maintaining aspect ratio.

    If the image is too small (width * height < min_pixels), it will be scaled up.
    If the image is too large (width * height > max_pixels), it will be scaled down.

    Args:
        image: PIL Image object
        min_pixels: Minimum total pixels (width * height). If None, no minimum limit.
        max_pixels: Maximum total pixels (width * height). If None, no maximum limit.

    Returns:
        Tuple of (resized_image, was_resized)
        - resized_image: The resized PIL Image (or original if no resize needed)
        - was_resized: True if image was resized, False otherwise
    """
    if min_pixels is None and max_pixels is None:
        return image, False

    width, height = image.size
    current_pixels = width * height

    # Check if resize is needed
    needs_resize = False
    target_pixels = None

    if min_pixels is not None and current_pixels < min_pixels:
        needs_resize = True
        target_pixels = min_pixels
    elif max_pixels is not None and current_pixels > max_pixels:
        needs_resize = True
        target_pixels = max_pixels

    if not needs_resize or target_pixels is None:
        return image, False

    # Calculate scale factor to reach target pixels while maintaining aspect ratio
    scale_factor = (target_pixels / current_pixels) ** 0.5

    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Resize using high-quality resampling
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    return resized_image, True


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

        # Download image with User-Agent header to avoid 403 errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(image_url, timeout=timeout, stream=True, headers=headers)
        response.raise_for_status()

        # Check Content-Type header first - reject if it's HTML/text
        content_type = response.headers.get('Content-Type', '')
        if content_type.startswith('text/html') or content_type.startswith('text/plain'):
            return None, None, f"URL returned {content_type} instead of an image. The server may have returned an error page or redirect."

        # Get the content
        image_bytes = response.content
        
        # Validate the content is actually an image (check magic bytes)
        is_valid, detected_mime_type, validation_error = validate_image_content(image_bytes)
        if not is_valid:
            return None, None, validation_error
        
        # Determine final MIME type
        # Priority: detected from content > Content-Type header > guessed from URL
        if detected_mime_type and detected_mime_type != "image/unknown":
            mime_type = detected_mime_type
        elif content_type.startswith('image/'):
            mime_type = content_type.split(';')[0]  # Remove charset if present
        else:
            mime_type = get_mime_type_from_url(image_url)

        return image_bytes, mime_type, ""

    except requests.exceptions.Timeout:
        return None, None, f"Download timeout after {timeout} seconds"
    except requests.exceptions.RequestException as e:
        return None, None, f"Download failed: {str(e)}"
    except Exception as e:
        return None, None, f"Unexpected error: {str(e)}"


@mcp.tool()
async def fetch_image(url: str) -> str:
    """Download an image from a URL and load it into the agent's context.

    This tool downloads an image and converts it to base64 format for vision-capable LLMs.
    Automatically resizes the image if it exceeds configured pixel limits.

    Args:
        url: The URL of the image to download. Must start with http:// or https://

    Returns:
        JSON-formatted multi-modal content with text description and base64-encoded image

    Note:
        - Supported formats: jpg, png, gif, webp, bmp, etc.
        - Maximum timeout: 30 seconds
        - Image will be resized if outside MIN_PIXELS and MAX_PIXELS limits (if configured)
    """
    # Download image
    image_bytes, mime_type, error_message = download_image_from_url(url)

    if error_message:
        # Return error message in JSON format
        return f'{{"error": "{error_message}"}}'

    # Check if we need to resize the image based on pixel limits
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))

        # Get original dimensions
        orig_width, orig_height = image.size
        orig_pixels = orig_width * orig_height

        # Apply pixel limits if configured
        resized_image, was_resized = resize_image_with_pixel_limits(
            image, MIN_PIXELS, MAX_PIXELS
        )

        if was_resized:
            new_width, new_height = resized_image.size
            new_pixels = new_width * new_height

            # Convert resized image back to bytes
            buffer = io.BytesIO()
            # Determine format from mime_type
            format_map = {
                'image/jpeg': 'JPEG',
                'image/jpg': 'JPEG',
                'image/png': 'PNG',
                'image/gif': 'GIF',
                'image/webp': 'WEBP',
                'image/bmp': 'BMP',
            }
            pil_format = format_map.get(mime_type, 'JPEG')

            resized_image.save(buffer, format=pil_format)
            image_bytes = buffer.getvalue()

            # Update mime_type if needed (JPEG is safest for resizing)
            if pil_format == 'JPEG':
                mime_type = 'image/jpeg'

        # Encode to base64
        base64_data = base64.b64encode(image_bytes).decode('utf-8')

    except Exception as e:
        # If image processing fails, try to encode the original bytes
        try:
            base64_data = base64.b64encode(image_bytes).decode('utf-8')
            was_resized = False
            orig_width, orig_height = 0, 0
            orig_pixels = 0
        except Exception as e2:
            return f'{{"error": "Failed to process image: {str(e2)}"}}'

    # Create data URL
    data_url = f"data:{mime_type};base64,{base64_data}"

    # Build description text
    if was_resized:
        description = (f"Image downloaded from: {url} (resized from {orig_width}x{orig_height} "
                      f"({orig_pixels} pixels) to {new_width}x{new_height} ({new_pixels} pixels))")
    else:
        description = f"Image downloaded from: {url}"

    # Return multi-modal format as JSON string
    # This format is compatible with OpenAI's multi-modal input
    import json
    result = [
        {
            "type": "text",
            "text": description
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
