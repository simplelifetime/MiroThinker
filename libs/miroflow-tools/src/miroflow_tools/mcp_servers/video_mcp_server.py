# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Video processing MCP server for analyzing video content.

This server provides tools to:
- Download videos from YouTube URLs using yt-dlp
- Download videos from direct URLs
- Analyze video content using LLM APIs with video support
- Return structured responses compatible with other MiroFlow tools
"""

import asyncio
import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import requests
import yt_dlp
from fastmcp import FastMCP
from openai import OpenAI

# Check if socks proxy support is available
try:
    import socks
    SOCKS_SUPPORT = True
except ImportError:
    SOCKS_SUPPORT = False

# Environment variables for video API
VIDEO_API_KEY = os.environ.get("VIDEO_API_KEY", "")
VIDEO_BASE_URL = os.environ.get("VIDEO_BASE_URL", "")
VIDEO_MODEL_NAME = os.environ.get("VIDEO_MODEL_NAME", "")

# Fall back to OpenAI credentials for compatibility
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Maximum video size (50MB)
MAX_VIDEO_SIZE = 50 * 1024 * 1024

# Default video save directory
DEFAULT_VIDEO_SAVE_DIR = "/media/liuzikang/save_videos"

# Proxy configuration (from environment variable)
VIDEO_PROXY = os.environ.get("VIDEO_PROXY", "")

# Initialize FastMCP server
mcp = FastMCP("video-mcp-server")


def is_youtube_url(url: str) -> bool:
    """
    Check if the URL is a YouTube URL.

    Args:
        url: The URL to check

    Returns:
        True if it's a YouTube URL, False otherwise
    """
    youtube_domains = [
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "www.youtu.be",
    ]
    return any(domain in url for domain in youtube_domains)


def download_video_with_ytdlp(url: str, save_dir: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
    """
    Download video using yt-dlp.

    Args:
        url: The video URL (supports YouTube and other platforms)
        save_dir: Directory to save the video. If None, uses default directory.
                  If default directory doesn't exist, uses temp directory.

    Returns:
        Tuple of (success, message, video_file_path)
        - success: True if download succeeded
        - message: Error message if failed
        - video_file_path: Path to downloaded video file
    """
    temp_file = None
    video_path = None

    try:
        # Determine save directory
        if save_dir is None:
            save_dir = DEFAULT_VIDEO_SAVE_DIR

        # Create directory if it doesn't exist for default path
        use_temp = False
        if save_dir == DEFAULT_VIDEO_SAVE_DIR:
            try:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                # Fall back to temp directory if default path is not accessible
                use_temp = True

        # Determine output directory and template
        if use_temp or (save_dir != DEFAULT_VIDEO_SAVE_DIR and not os.path.exists(save_dir)):
            # Use temp directory with a unique template
            import uuid
            temp_dir = tempfile.gettempdir()
            video_template = os.path.join(temp_dir, f"video_{uuid.uuid4().hex[:8]}")
            download_dir = temp_dir
        else:
            # Use persistent storage directory
            import uuid
            video_template = os.path.join(save_dir, f"video_{uuid.uuid4().hex[:8]}")
            download_dir = save_dir

        # Configure yt-dlp options
        ydl_opts = {
            # Format selection (strict order):
            # 1. Best video with height<=540 + best audio (MP4 preferred)
            # 2. Best single file with height<=540
            # 3. Fallback to lower quality if needed
            "format": "bestvideo[height<=540][ext=mp4]+bestaudio/bestvideo[height<=540]+bestaudio/best[height<=540][ext=mp4]/best[height<=540]",
            "outtmpl": video_template,  # Let yt-dlp handle the extension
            "quiet": False,  # Show errors
            "no_warnings": False,
            "max_filesize": MAX_VIDEO_SIZE,
            "socket_timeout": 30,  # Add socket timeout
            "progress_hooks": [],  # Add progress hooks
            # Add retries for transient errors
            "retries": 3,
            "fragment_retries": 3,
        }

        # Add proxy if configured
        if VIDEO_PROXY:
            # Check if using SOCKS proxy
            if VIDEO_PROXY.startswith("socks"):
                if not SOCKS_SUPPORT:
                    return False, "SOCKS proxy requested but PySocks is not installed. Install with: pip install PySocks", None
            ydl_opts["proxy"] = VIDEO_PROXY
            print(f"Using proxy: {VIDEO_PROXY}")

        # Download video
        print(f"Starting download from: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"Download completed")

        # Find the actual file created (yt-dlp may add extension)
        # First, list all files in download directory and find matching ones
        possible_extensions = ['.mp4', '.webm', '.mkv', '.avi', '.mov', '.m4a']
        video_path = None

        if os.path.exists(download_dir):
            files = os.listdir(download_dir)
            for f in files:
                if f.startswith(os.path.basename(video_template)):
                    potential_path = os.path.join(download_dir, f)
                    if os.path.isfile(potential_path):
                        video_path = potential_path
                        break

        # If still not found, try the old method with extensions
        if video_path is None:
            for ext in possible_extensions:
                potential_path = video_template + ext
                if os.path.exists(potential_path):
                    video_path = potential_path
                    break

        # If not found with extension, check if template exists as-is
        if video_path is None and os.path.exists(video_template):
            video_path = video_template

        # Verify file exists and has content
        if not video_path or not os.path.exists(video_path):
            # List what files are actually in the directory
            files_list = os.listdir(download_dir) if os.path.exists(download_dir) else []
            return False, f"Download failed: file not created. Directory contents: {files_list}. This may be due to network connectivity issues or YouTube access restrictions.", None

        file_size = os.path.getsize(video_path)
        if file_size == 0:
            if os.path.exists(video_path):
                os.unlink(video_path)
            return False, "Download failed: empty file", None

        if file_size > MAX_VIDEO_SIZE:
            if os.path.exists(video_path):
                os.unlink(video_path)
            return (
                False,
                f"Video size ({file_size / (1024 * 1024):.2f}MB) exceeds maximum allowed size ({MAX_VIDEO_SIZE / (1024 * 1024):.0f}MB)",
                None,
            )

        return True, "", video_path

    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        # Check if it's a 503 or network error that might be transient
        if "503" in error_msg or "Service Unavailable" in error_msg:
            return False, f"Download failed: YouTube service unavailable (HTTP 503). This may be due to rate limiting or the specific video format being blocked. Try again later or use a different video.", None

        # Clean up any partial files
        if os.path.exists(download_dir):
            try:
                files = os.listdir(download_dir)
                for f in files:
                    if f.startswith(os.path.basename(video_template)):
                        try:
                            os.unlink(os.path.join(download_dir, f))
                        except Exception:
                            pass
            except Exception:
                pass
        return False, f"Download error: {error_msg}", None
    except Exception as e:
        # Clean up downloaded file on any error
        if video_path and os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except Exception:
                pass
        return False, f"Unexpected error during download: {str(e)}", None


def download_video_direct(url: str, save_dir: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
    """
    Download video from direct URL.

    Args:
        url: The direct video URL
        save_dir: Directory to save the video. If None, uses default directory.
                  If default directory doesn't exist, uses temp directory.

    Returns:
        Tuple of (success, message, video_file_path)
        - success: True if download succeeded
        - message: Error message if failed
        - video_file_path: Path to downloaded video file
    """
    temp_file = None
    try:
        # Download video
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        # Check content length if available
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > MAX_VIDEO_SIZE:
            return (
                False,
                f"Video size ({int(content_length) / (1024 * 1024):.2f}MB) exceeds maximum allowed size ({MAX_VIDEO_SIZE / (1024 * 1024):.0f}MB)",
                None,
            )

        # Determine save directory
        if save_dir is None:
            save_dir = DEFAULT_VIDEO_SAVE_DIR

        # Create directory if it doesn't exist for default path
        use_temp = False
        if save_dir == DEFAULT_VIDEO_SAVE_DIR:
            try:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                # Fall back to temp directory if default path is not accessible
                use_temp = True

        # Create file path
        if use_temp or (save_dir != DEFAULT_VIDEO_SAVE_DIR and not os.path.exists(save_dir)):
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            video_path = temp_file.name
        else:
            # Use persistent storage
            import uuid
            video_filename = f"video_{uuid.uuid4().hex[:8]}.mp4"
            video_path = os.path.join(save_dir, video_filename)
            temp_file = open(video_path, 'wb')

        # Write video data
        downloaded_size = 0
        for chunk in response.iter_content(chunk_size=8192):
            downloaded_size += len(chunk)
            if downloaded_size > MAX_VIDEO_SIZE:
                temp_file.close()
                if os.path.exists(video_path):
                    os.unlink(video_path)
                return (
                    False,
                    f"Video size exceeds maximum allowed size ({MAX_VIDEO_SIZE / (1024 * 1024):.0f}MB)",
                    None,
                )
            temp_file.write(chunk)

        temp_file.close()

        # Verify file has content
        if os.path.getsize(video_path) == 0:
            if os.path.exists(video_path):
                os.unlink(video_path)
            return False, "Download failed: empty file", None

        return True, "", video_path

    except requests.exceptions.Timeout:
        if temp_file:
            temp_file.close()
        return False, "Download timeout: request took too long", None
    except requests.exceptions.RequestException as e:
        if temp_file:
            temp_file.close()
        return False, f"Download error: {str(e)}", None
    except Exception as e:
        if temp_file:
            temp_file.close()
        return False, f"Unexpected error during download: {str(e)}", None


def analyze_video_with_llm(
    video_path: str, query: str, api_key: str, base_url: str, model_name: str
) -> Tuple[bool, str]:
    """
    Analyze video content using LLM API.

    Args:
        video_path: Path to the video file
        query: The question/query about the video
        api_key: API key for the LLM service
        base_url: Base URL for the LLM API
        model_name: Model name to use

    Returns:
        Tuple of (success, response_or_error)
    """
    try:
        # Read and encode video file
        with open(video_path, "rb") as video_file:
            video_data = base64.b64encode(video_file.read()).decode("utf-8")

        # Create OpenAI client
        client = OpenAI(api_key=api_key, base_url=base_url)

        # Prepare messages with video content
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{video_data}"},
                    },
                ],
            }
        ]

        # Call LLM API
        response = client.chat.completions.create(
            model=model_name, messages=messages, max_tokens=4096
        )

        # Extract response
        if response.choices and len(response.choices) > 0:
            result = response.choices[0].message.content
            return True, result
        else:
            return False, "Received empty response from API"

    except Exception as e:
        return False, f"Video analysis failed: {str(e)}"


@mcp.tool()
async def video_qa(url: str, query: str) -> str:
    """
    Analyze video content from a URL (YouTube or direct video URL) and answer questions about it.

    This tool can:
    - Download videos from YouTube URLs (using yt-dlp)
    - Download videos from direct URLs
    - Analyze video content using LLM APIs
    - Answer questions about the video

    Args:
        url: The video URL (supports YouTube, youtu.be, and direct video URLs)
        query: The question or query about the video content

    Returns:
        JSON-formatted response with the analysis result:
        {
            "success": true/false,
            "error": "error message if failed",
            "result": "analysis result",
            "metadata": {"source": "youtube/direct", "url": "original_url"}
        }
    """
    video_file_path = None

    try:
        # Determine which download method to use
        if is_youtube_url(url):
            source = "youtube"
            success, message, video_file_path = download_video_with_ytdlp(url)
        else:
            source = "direct"
            success, message, video_file_path = download_video_direct(url)

        if not success:
            return json.dumps(
                {
                    "success": False,
                    "error": message,
                    "result": None,
                    "metadata": {"source": source, "url": url},
                },
                ensure_ascii=False,
            )

        # Determine which API credentials to use
        # Prefer VIDEO_API_* if available, fall back to OPENAI_API_*
        api_key = VIDEO_API_KEY or OPENAI_API_KEY
        base_url = VIDEO_BASE_URL or OPENAI_BASE_URL
        model_name = VIDEO_MODEL_NAME or "gpt-4o"  # Default to gpt-4o

        if not api_key:
            # Clean up video file only if it's in temp directory
            if video_file_path and os.path.exists(video_file_path):
                if not video_file_path.startswith(DEFAULT_VIDEO_SAVE_DIR):
                    os.unlink(video_file_path)
            return json.dumps(
                {
                    "success": False,
                    "error": "No API key configured. Set VIDEO_API_KEY or OPENAI_API_KEY environment variable.",
                    "result": None,
                    "metadata": {"source": source, "url": url},
                },
                ensure_ascii=False,
            )

        # Analyze video
        success, result_or_error = analyze_video_with_llm(
            video_file_path, query, api_key, base_url, model_name
        )

        # Clean up video file only if it's in temp directory (not in default save directory)
        if video_file_path and os.path.exists(video_file_path):
            if not video_file_path.startswith(DEFAULT_VIDEO_SAVE_DIR):
                os.unlink(video_file_path)

        if success:
            return json.dumps(
                {
                    "success": True,
                    "error": None,
                    "result": result_or_error,
                    "metadata": {"source": source, "url": url},
                },
                ensure_ascii=False,
            )
        else:
            return json.dumps(
                {
                    "success": False,
                    "error": result_or_error,
                    "result": None,
                    "metadata": {"source": source, "url": url},
                },
                ensure_ascii=False,
            )

    except Exception as e:
        # Clean up video file on any error (only if it's in temp directory)
        if video_file_path and os.path.exists(video_file_path):
            if not video_file_path.startswith(DEFAULT_VIDEO_SAVE_DIR):
                os.unlink(video_file_path)

        return json.dumps(
            {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "result": None,
                "metadata": {"source": "unknown", "url": url},
            },
            ensure_ascii=False,
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")
