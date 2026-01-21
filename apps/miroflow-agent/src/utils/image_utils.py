# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Image processing utilities for multi-modal support.

This module provides functions for:
- Uploading images to Aliyun OSS
- Downloading and encoding images
- Generating image descriptions for multi-modal context
"""

import base64
import os
import random
import string
from io import BytesIO
from typing import Optional, Tuple

import requests
from dotenv import load_dotenv

# Ensure .env file is loaded
load_dotenv()


class OSSUploader:
    """Handler for uploading images to Aliyun OSS."""

    def __init__(self):
        """Initialize OSS uploader with credentials from environment variables."""
        self.access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
        self.access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
        self.bucket_name = os.getenv("OSS_BUCKET_NAME", "mitalinlp")
        self.endpoint = os.getenv("OSS_ENDPOINT", "http://oss-cn-hangzhou.aliyuncs.com")

        # Import oss2 only when needed to avoid dependency issues
        try:
            import oss2

            self.oss2 = oss2
        except ImportError:
            print("Warning: oss2 not installed. OSS upload will be disabled.")
            self.oss2 = None

    def generate_random_string(self, length=32):
        """
        Generate a random string for file naming.

        Args:
            length: Length of the random string

        Returns:
            Random alphanumeric string
        """
        letters = string.ascii_letters + string.digits
        return "".join(random.choice(letters) for _ in range(length))

    def upload(self, image, byte=False) -> Optional[str]:
        """
        Upload image to Aliyun OSS.

        Args:
            image: Image data (bytes or file path)
            byte: Whether the input is in byte format

        Returns:
            Signed OSS URL (valid for 100 hours), or None if upload fails
        """
        if not self.oss2:
            print("Error: oss2 not available. Cannot upload to OSS.")
            return None

        if not self.access_key_id or not self.access_key_secret:
            print("Error: OSS credentials not configured.")
            return None

        try:
            image_name = f"{self.generate_random_string()}.jpeg"
            target_path = f"zhili.zl/qwenvl_rft/image/{image_name}"

            # Check file size
            if byte:
                image_bytes = BytesIO(image)
                image_size = image_bytes.getbuffer().nbytes
            else:
                image_size = os.path.getsize(image)

            # Skip small files (< 1KB)
            if image_size <= 1024:
                print("Info: Image size too small (< 1KB), skipping upload.")
                return None

            # Authenticate and create bucket
            auth = self.oss2.Auth(self.access_key_id, self.access_key_secret)
            bucket = self.oss2.Bucket(auth, self.endpoint, self.bucket_name)

            # Upload
            if byte:
                bucket.put_object(target_path, image_bytes.getvalue())
            else:
                bucket.put_object_from_file(target_path, image)

            # Generate signed URL (valid for 100 hours)
            file_url = bucket.sign_url("GET", target_path, 360000)
            return file_url

        except Exception as e:
            print(f"Error: Failed to upload image to OSS: {str(e)}")
            return None


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Encode an image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64-encoded image string, or None if encoding fails
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error: Failed to encode image to base64: {str(e)}")
        return None


def get_image_mime_type(image_path: str) -> str:
    """
    Get MIME type for an image file based on its extension.

    Args:
        image_path: Path to the image file

    Returns:
        MIME type string
    """
    _, ext = os.path.splitext(image_path)
    ext = ext.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/jpeg")


def download_image_from_url(image_url: str, timeout=10) -> Optional[bytes]:
    """
    Download an image from URL.

    Args:
        image_url: URL of the image
        timeout: Request timeout in seconds

    Returns:
        Image bytes, or None if download fails
    """
    try:
        response = requests.get(image_url, timeout=timeout, stream=True)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error: Failed to download image from {image_url}: {str(e)}")
        return None


def encode_image_bytes_to_base64(image_bytes: bytes) -> str:
    """
    Encode image bytes to base64 string.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Base64-encoded string
    """
    return base64.b64encode(image_bytes).decode("utf-8")


def generate_simple_image_caption(image_path: str, task_description: str = "") -> str:
    """
    Generate a simple caption for an image (brief description only).

    Unlike the detailed caption generation, this creates a minimal caption
    for multi-modal context.

    Args:
        image_path: Path to the image file
        task_description: Optional task context

    Returns:
        Simple caption string
    """
    # For initial input images, we provide a minimal description
    # The actual visual understanding will be done by the multi-modal model
    return f"[Original input image: {os.path.basename(image_path)}]"


def format_image_for_context(
    image_base64: str,
    image_url: Optional[str],
    description: str,
    webpage_url: Optional[str] = None,
) -> Tuple[dict, str]:
    """
    Format image data for inclusion in LLM context.

    Args:
        image_base64: Base64-encoded image data
        image_url: URL of the image (can be None)
        description: Text description of the image
        webpage_url: URL of the webpage containing the image (optional)

    Returns:
        Tuple of (image_content_dict, text_description)
    """
    # Create image content dict for OpenAI API format
    image_content = {
        "type": "image_url",
        "image_url": {"url": image_base64},
    }

    # Create text description with metadata
    text_parts = [f"Image URL: {image_url or 'N/A'}", f"Description: {description}"]

    if webpage_url:
        text_parts.append(f"Webpage URL: {webpage_url}")

    text_description = ", ".join(text_parts)

    return image_content, text_description
