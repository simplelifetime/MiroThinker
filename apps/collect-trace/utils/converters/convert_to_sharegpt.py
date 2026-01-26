# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Convert log files to ShareGPT format for multi-modal training.

This module extracts message history from logs and converts them to ShareGPT format,
which is suitable for training multi-modal LLMs. It handles both text and image content.
"""

import json
import os
import shutil
import base64
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime


def decode_base64_image(base64_string: str, output_path: Path) -> bool:
    """
    Decode a base64 image string and save it as a file.

    Args:
        base64_string: Base64-encoded image string (with or without data:image/...;base64, prefix)
        output_path: Path where the image file should be saved

    Returns:
        True if successful, False otherwise
    """
    try:
        # Remove data URL prefix if present
        if "," in base64_string:
            base64_data = base64_string.split(",", 1)[1]
        else:
            base64_data = base64_string

        # Decode base64
        image_data = base64.b64decode(base64_data)

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(image_data)

        return True
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return False


def get_mime_type_from_base64(base64_string: str) -> str:
    """
    Extract MIME type from base64 data URL.

    Args:
        base64_string: Base64 string with potential data URL prefix

    Returns:
        MIME type string (e.g., "image/jpeg") or default "image/jpeg"
    """
    if base64_string.startswith("data:image/"):
        # Extract MIME type from data URL
        mime_part = base64_string.split(";")[0]
        return mime_part.replace("data:", "")
    return "image/jpeg"


def get_file_extension_from_mime(mime_type: str) -> str:
    """
    Convert MIME type to file extension.

    Args:
        mime_type: MIME type string (e.g., "image/jpeg")

    Returns:
        File extension (e.g., ".jpg")
    """
    mime_to_ext = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
    }
    return mime_to_ext.get(mime_type, ".jpg")


def process_content_with_images(
    content: Any,
    images_dir: Path,
    task_id: str,
    msg_idx: int,
    image_counter: List[int],
    existing_images: List[str] = None
) -> Tuple[str, List[str]]:
    """
    Process message content, extract and save images, update content with <image> markers.

    Args:
        content: Message content (string, list, or dict)
        images_dir: Directory to save extracted images
        task_id: Task identifier for naming images
        msg_idx: Message index for naming images
        image_counter: Mutable list to track image count [current_count]
        existing_images: List of existing image file paths (if already saved)

    Returns:
        Tuple of (processed_content_with_markers, list_of_image_paths)
    """
    image_paths = []

    if isinstance(content, list):
        # Multi-modal content (list of content items)
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    # Check if we have existing images to use
                    if existing_images and image_counter[0] < len(existing_images):
                        # Use existing image
                        image_paths.append(existing_images[image_counter[0]])
                        image_counter[0] += 1
                    else:
                        # Try to extract from base64
                        image_url = item.get("image_url", {}).get("url", "")
                        if image_url and image_url != "<image>" and image_url.startswith("data:image"):
                            # Save image to file
                            mime_type = get_mime_type_from_base64(image_url)
                            ext = get_file_extension_from_mime(mime_type)
                            image_filename = f"{task_id}_msg{msg_idx}_img{image_counter[0]}{ext}"
                            image_path = images_dir / image_filename

                            if decode_base64_image(image_url, image_path):
                                image_paths.append(str(image_path))
                                image_counter[0] += 1
                            else:
                                print(f"Warning: Failed to decode and save image {image_filename}")

                    # Always add <image> marker
                    text_parts.append("<image>")
                elif isinstance(item, str):
                    text_parts.append(item)
                else:
                    text_parts.append(str(item))
            elif isinstance(item, str):
                text_parts.append(item)
            else:
                text_parts.append(str(item))

        processed_content = " ".join(text_parts) if text_parts else ""

    elif isinstance(content, str):
        # Text content - check if it contains base64 image
        if content.startswith("data:image") and ";base64," in content:
            # Check if we have existing images to use
            if existing_images and image_counter[0] < len(existing_images):
                # Use existing image
                image_paths.append(existing_images[image_counter[0]])
                image_counter[0] += 1
                processed_content = "<image>"
            else:
                # Save image to file
                mime_type = get_mime_type_from_base64(content)
                ext = get_file_extension_from_mime(mime_type)
                image_filename = f"{task_id}_msg{msg_idx}_img{image_counter[0]}{ext}"
                image_path = images_dir / image_filename

                if decode_base64_image(content, image_path):
                    image_paths.append(str(image_path))
                    image_counter[0] += 1
                else:
                    print(f"Warning: Failed to decode and save image {image_filename}")

                processed_content = "<image>"
        else:
            processed_content = content
    else:
        # Other formats - convert to string
        processed_content = str(content)

    return processed_content, image_paths


def convert_messages_to_sharegpt(
    messages: List[Dict[str, Any]], images_dir: Path, task_id: str, existing_images: List[str] = None
) -> Dict[str, Any]:
    """
    Convert message list to ShareGPT format.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        images_dir: Directory to save extracted images
        task_id: Task identifier for naming images
        existing_images: List of existing image file paths (if already saved)

    Returns:
        ShareGPT-formatted dictionary with 'messages' and 'images' keys
    """
    sharegpt_messages = []
    all_image_paths = []
    image_counter = [0]  # Use list to allow mutation in nested function

    for msg_idx, message in enumerate(messages):
        role = message.get("role", "")
        content = message.get("content", "")

        # Skip certain message types
        if role == "tool" or role == "system":
            continue

        # Process content and extract images
        processed_content, image_paths = process_content_with_images(
            content, images_dir, task_id, msg_idx, image_counter, existing_images
        )

        # Add to messages if content is not empty
        if processed_content:
            sharegpt_messages.append({
                "content": processed_content,
                "role": role
            })

        # Collect all image paths
        all_image_paths.extend(image_paths)

    return {
        "messages": sharegpt_messages,
        "images": all_image_paths
    }


def extract_and_save_sharegpt(
    log_data: Dict[str, Any],
    output_dir: Path,
    input_filename: str,
    original_log_path: str = None,
):
    """
    Extract message history from log data and save as ShareGPT format.

    Args:
        log_data: Log data dictionary
        output_dir: Output directory for ShareGPT JSON files
        input_filename: Input filename (without extension)
        original_log_path: Path to original log file (for loading existing images)
    """
    # Create images directory
    images_dir = output_dir / f"{input_filename}_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Try to find and use existing images directory
    existing_images = []

    if original_log_path:
        original_path = Path(original_log_path)
        original_dir = original_path.parent

        # Try multiple locations for images directory (in order of priority)
        possible_locations = [
            # 1. Same directory as the JSON file (for successful_logs case)
            original_dir / f"{input_filename}_images",
            # 2. New format: save_images/xxx_images (in same dir)
            original_dir / "save_images" / f"{input_filename}_images",
            # 3. Old format: xxx_images (in parent dir)
            original_dir.parent / f"{input_filename}_images",
            # 4. save_images in parent dir
            original_dir.parent / "save_images" / f"{input_filename}_images",
        ]

        for images_dir in possible_locations:
            if images_dir.exists() and images_dir.is_dir():
                # Load images from this directory
                for img_file in sorted(images_dir.iterdir()):
                    if img_file.is_file():
                        existing_images.append(str(img_file))
                if existing_images:
                    print(f"✓ Found {len(existing_images)} existing image(s) in: {images_dir}")
                    break

    # 1. Extract main_agent_message_history
    main_agent_history = log_data.get("main_agent_message_history", {})
    if main_agent_history and "message_history" in main_agent_history:
        main_messages = main_agent_history["message_history"]
        if main_messages:
            sharegpt_data = convert_messages_to_sharegpt(
                main_messages, images_dir, input_filename, existing_images
            )

            # Save main agent ShareGPT records
            main_output_file = output_dir / f"{input_filename}_main_agent_sharegpt.json"
            with open(main_output_file, "w", encoding="utf-8") as f:
                json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)

            print(f"✓ Saved main agent ShareGPT: {main_output_file}")
            print(f"  - Messages: {len(sharegpt_data['messages'])}")
            print(f"  - Images: {len(sharegpt_data['images'])}")

    # 2. Extract sub_agent_message_history_sessions
    sub_agent_sessions = log_data.get("sub_agent_message_history_sessions", {})
    if sub_agent_sessions:
        for session_name, session_data in sub_agent_sessions.items():
            if "message_history" in session_data:
                sub_messages = session_data["message_history"]
                if sub_messages:
                    # Create separate images directory for each sub-agent
                    sub_images_dir = images_dir / f"{input_filename}_{session_name}_images"

                    sharegpt_data = convert_messages_to_sharegpt(
                        sub_messages, sub_images_dir, f"{input_filename}_{session_name}", None
                    )

                    # Save sub agent ShareGPT records
                    sub_output_file = (
                        output_dir / f"{input_filename}_{session_name}_sharegpt.json"
                    )
                    with open(sub_output_file, "w", encoding="utf-8") as f:
                        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)

                    print(f"✓ Saved sub agent {session_name} ShareGPT: {sub_output_file}")
                    print(f"  - Messages: {len(sharegpt_data['messages'])}")
                    print(f"  - Images: {len(sharegpt_data['images'])}")


def main():
    """Main function"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python convert_to_sharegpt.py <log_file_path> [output_dir]")
        print(
            "Example: python convert_to_sharegpt.py logs/debug_logs/task_1.json"
        )
        print(
            "Example: python convert_to_sharegpt.py logs/debug_logs/task_1.json ./sharegpt_output"
        )
        sys.exit(1)

    log_file_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("sharegpt_output")

    # Check if input file exists
    if not log_file_path.exists():
        print(f"Error: Log file does not exist: {log_file_path}")
        sys.exit(1)

    try:
        # Read log file
        print(f"Reading log file: {log_file_path}")
        with open(log_file_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)

        # Extract input filename (without extension)
        input_filename = log_file_path.stem

        # Extract and save ShareGPT format
        print(f"Converting to ShareGPT format to: {output_dir}")
        extract_and_save_sharegpt(log_data, output_dir, input_filename, str(log_file_path))

        print("\n✓ ShareGPT conversion completed!")
        print(f"Output directory: {output_dir.absolute()}")

    except json.JSONDecodeError as e:
        print(f"Error: Cannot parse JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
