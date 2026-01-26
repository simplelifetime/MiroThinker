# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""Output formatting utilities for agent responses."""

import json
import re
from typing import Tuple, Union

from ..utils.prompt_utils import FORMAT_ERROR_MESSAGE

# Maximum length for tool results before truncation (100k chars ≈ 25k tokens)
TOOL_RESULT_MAX_LENGTH = 100_000


class OutputFormatter:
    """Formatter for processing and formatting agent outputs."""

    def _extract_boxed_content(self, text: str) -> str:
        r"""
        Extract the content of the last \boxed{...} occurrence in the given text.

        Supports:
          - Arbitrary levels of nested braces
          - Escaped braces (\{ and \})
          - Whitespace between \boxed and the opening brace
          - Empty content inside braces
          - Incomplete boxed expressions (extracts to end of string as fallback)

        Args:
            text: Input text that may contain \boxed{...} expressions

        Returns:
            The extracted boxed content, or empty string if no match is found.
        """
        if not text:
            return ""

        _BOXED_RE = re.compile(r"\\boxed\b", re.DOTALL)

        last_result = None  # Track the last boxed content (complete or incomplete)
        i = 0
        n = len(text)

        while True:
            # Find the next \boxed occurrence
            m = _BOXED_RE.search(text, i)
            if not m:
                break
            j = m.end()

            # Skip any whitespace after \boxed
            while j < n and text[j].isspace():
                j += 1

            # Require that the next character is '{'
            if j >= n or text[j] != "{":
                i = j
                continue

            # Parse the brace content manually to handle nesting and escapes
            depth = 0
            k = j
            escaped = False
            found_closing = False
            while k < n:
                ch = text[k]
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    # When depth returns to zero, the boxed content ends
                    if depth == 0:
                        last_result = text[j + 1 : k]
                        i = k + 1
                        found_closing = True
                        break
                k += 1

            # If we didn't find a closing brace, this is an incomplete boxed
            # Store it as the last result (will be overwritten if we find more boxed later)
            if not found_closing and depth > 0:
                last_result = text[j + 1 : n]
                i = k  # Continue from where we stopped
            elif not found_closing:
                i = j + 1  # Move past this invalid boxed

        # Return the last boxed content found (complete or incomplete)
        black_list = ["?", "??", "???", "？", "……", "…", "...", "unknown", None]
        return last_result.strip() if last_result not in black_list else ""

    def format_tool_result_for_user(
        self, tool_call_execution_result: dict
    ) -> Union[dict, list]:
        """
        Format tool execution results to be fed back to LLM as user messages.

        Only includes necessary information (results or errors). Long results
        are truncated to TOOL_RESULT_MAX_LENGTH to prevent context overflow.

        For image search results, returns a multi-modal format with base64 images.

        Args:
            tool_call_execution_result: Dict containing server_name, tool_name,
                and either 'result' or 'error'.

        Returns:
            Dict with 'type' and 'text' keys, or a list containing mixed
            content (text + images) for multi-modal models.
        """
        server_name = tool_call_execution_result["server_name"]
        tool_name = tool_call_execution_result["tool_name"]

        if "error" in tool_call_execution_result:
            # Provide concise error information to LLM
            content = f"Tool call to {tool_name} on {server_name} failed. Error: {tool_call_execution_result['error']}"
            return {"type": "text", "text": content}
        elif "result" in tool_call_execution_result:
            result = tool_call_execution_result["result"]

            # Check if this is a fetch_image result (multi-modal format)
            if tool_name == "fetch_image":
                try:
                    # Parse JSON result
                    if isinstance(result, str):
                        data = json.loads(result)
                    else:
                        data = result

                    # Check if this contains an error
                    if isinstance(data, dict) and "error" in data:
                        return {"type": "text", "text": f"Image download failed: {data['error']}"}

                    # Check if this is a multi-modal content list
                    if isinstance(data, list):
                        # Validate the structure
                        valid_content = True
                        for item in data:
                            if not isinstance(item, dict) or "type" not in item:
                                valid_content = False
                                break

                        if valid_content:
                            # Return the multi-modal content list directly
                            return data
                except (json.JSONDecodeError, KeyError, TypeError):
                    # If parsing fails, treat as regular text
                    pass

            # Check if this is an image search result with base64 data
            if tool_name in ["image_search", "visual_search"]:
                try:
                    # Try to parse as JSON
                    if isinstance(result, str):
                        data = json.loads(result)
                    else:
                        data = result

                    # Check if this contains images with base64 data
                    if "images" in data and isinstance(data["images"], list):
                        return self._format_image_search_result(
                            data, tool_name, server_name
                        )
                except (json.JSONDecodeError, KeyError):
                    pass  # Fall through to regular text formatting

            # Check if this is an image processing tool result (multi-modal format)
            if tool_name in ["zoom_in", "rotate", "flip", "put_box"]:
                try:
                    # Parse JSON result
                    if isinstance(result, str):
                        data = json.loads(result)
                    else:
                        data = result

                    # Check if this contains an error
                    if isinstance(data, list) and len(data) > 0:
                        # Validate the structure as multi-modal content
                        valid_content = True
                        for item in data:
                            if not isinstance(item, dict) or "type" not in item:
                                valid_content = False
                                break

                        if valid_content:
                            # Return the multi-modal content list directly
                            return data
                except (json.JSONDecodeError, KeyError, TypeError):
                    # If parsing fails, treat as regular text
                    pass

            # Provide the original output result of the tool
            content = result
            # Truncate overly long results to prevent context overflow
            if len(content) > TOOL_RESULT_MAX_LENGTH:
                content = content[:TOOL_RESULT_MAX_LENGTH] + "\n... [Result truncated]"
            return {"type": "text", "text": content}
        else:
            content = f"Tool call to {tool_name} on {server_name} completed, but produced no specific output or result."
            return {"type": "text", "text": content}

    def _format_image_search_result(
        self, data: dict, tool_name: str, server_name: str
    ) -> list:
        """
        Format image search results as multi-modal content.

        Creates a list with text description and base64-encoded images
        for direct visual processing by multi-modal models.

        Args:
            data: Parsed JSON response from image search
            tool_name: Name of the tool that was called
            server_name: Name of the MCP server

        Returns:
            List of content items (text + images) in OpenAI API format
        """
        content_items = []
        images = data.get("images", [])

        # Add text summary
        search_type = (
            "Visual search" if tool_name == "visual_search" else "Image search"
        )
        text_summary = f"{search_type} completed on {server_name}. Found {len(images)} images.\n\n"

        # Add text descriptions for images (limited to avoid context explosion)
        for idx, img in enumerate(images[:10]):  # Limit text descriptions to 10
            title = img.get("title", "")
            link = img.get("link", "")
            image_url = img.get("imageUrl", "")

            parts = [f"{idx + 1}. "]
            if title:
                parts.append(f"Title: {title}")
            if image_url:
                parts.append(f"Image URL: {image_url}")
            if link:
                parts.append(f"Source: {link}")

            text_summary += " | ".join(parts) + "\n"

        if len(images) > 10:
            text_summary += f"\n... and {len(images) - 10} more images.\n"

        # Add note about which images have base64 data
        base64_count = sum(1 for img in images if "base64_data" in img)
        if base64_count > 0:
            text_summary += (
                f"\nNote: The first {base64_count} images are included below "
                "for direct visual analysis."
            )

        content_items.append({"type": "text", "text": text_summary})

        # Add base64 images (first 5)
        for idx, img in enumerate(images[:5]):
            if "base64_data" in img:
                # Create image content with metadata description
                image_url = img.get("imageUrl", "N/A")
                title = img.get("title", "")
                link = img.get("link", "")

                # Build text description
                desc_parts = [f"Image {idx + 1}"]
                if image_url and image_url != "N/A":
                    desc_parts.append(f"Image URL: {image_url}")
                if title:
                    desc_parts.append(f"Title: {title}")
                if link:
                    desc_parts.append(f"Webpage URL: {link}")

                text_description = ", ".join(desc_parts)

                # Add image with text description
                content_items.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": img["base64_data"]},
                    }
                )
                content_items.append({"type": "text", "text": text_description})

        return content_items

    def format_final_summary_and_log(
        self, final_answer_text: str, client=None
    ) -> Tuple[str, str, str]:
        """
        Format final summary information, including answers and token statistics.

        Args:
            final_answer_text: The final answer text from the agent
            client: Optional LLM client for token usage statistics

        Returns:
            Tuple of (summary_text, boxed_result, usage_log)
        """
        summary_lines = []
        summary_lines.append("\n" + "=" * 30 + " Final Answer " + "=" * 30)
        summary_lines.append(final_answer_text)

        # Extract boxed result - find the last match using safer regex patterns
        boxed_result = self._extract_boxed_content(final_answer_text)

        # Add extracted result section
        summary_lines.append("\n" + "-" * 20 + " Extracted Result " + "-" * 20)

        if boxed_result:
            summary_lines.append(boxed_result)
        elif final_answer_text:
            summary_lines.append("No \\boxed{} content found.")
            boxed_result = FORMAT_ERROR_MESSAGE

        # Token usage statistics and cost estimation - use client method
        if client and hasattr(client, "format_token_usage_summary"):
            token_summary_lines, log_string = client.format_token_usage_summary()
            summary_lines.extend(token_summary_lines)
        else:
            # If no client or client doesn't support it, use default format
            summary_lines.append("\n" + "-" * 20 + " Token Usage & Cost " + "-" * 20)
            summary_lines.append("Token usage information not available.")
            summary_lines.append("-" * (40 + len(" Token Usage & Cost ")))
            log_string = "Token usage information not available."

        return "\n".join(summary_lines), boxed_result, log_string
