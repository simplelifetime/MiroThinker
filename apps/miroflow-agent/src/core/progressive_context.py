# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Progressive Context Management Module.

This module provides a pluggable context management system that maintains
a compressed, evolving summary of the research trajectory without linear
growth in context size. It tracks:
1. Text evolution: Research progress, findings, and verification status
2. Multimodal evolution: Image URLs and their relationships/transformations

The system is designed to be:
- Pluggable: Can be enabled/disabled via config without affecting existing code
- Compressed: Summary grows sub-linearly with search steps
- Comprehensive: Captures key findings and multimodal relationships
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ImageRecord:
    """Record of an image encountered during search."""
    url: str
    source: str  # "input", "search_result", "fetch_image", "image_processing"
    turn: int
    description: str = ""
    parent_url: Optional[str] = None  # For transformed images (e.g., zoom_in)
    transformation: Optional[str] = None  # e.g., "zoom_in", "rotate"
    used_in_planning: bool = False  # Whether this image was actually used by model


@dataclass
class MultimodalEvolution:
    """
    Multimodal evolution table tracking images and their relationships.

    Design principles:
    - Only keep top 3 images from search results
    - Keep all images that were actually used (fetch_image, image_processing)
    - Track image transformations (zoom_in, rotate, etc.)
    - Keep the table from excessive growth
    """
    images: List[ImageRecord] = field(default_factory=list)
    image_urls_seen: Set[str] = field(default_factory=set)

    def add_image(
        self,
        url: str,
        source: str,
        turn: int,
        description: str = "",
        parent_url: Optional[str] = None,
        transformation: Optional[str] = None,
        used_in_planning: bool = False,
    ) -> bool:
        """
        Add an image to the evolution table.

        Returns:
            True if image was added, False if already exists or filtered
        """
        if url in self.image_urls_seen:
            return False

        # For search results, only keep if it will be used
        if source == "search_result" and not used_in_planning:
            return False

        record = ImageRecord(
            url=url,
            source=source,
            turn=turn,
            description=description,
            parent_url=parent_url,
            transformation=transformation,
            used_in_planning=used_in_planning,
        )
        self.images.append(record)
        self.image_urls_seen.add(url)
        return True

    def mark_image_used(self, url: str):
        """Mark an image as used in planning."""
        for img in self.images:
            if img.url == url:
                img.used_in_planning = True
                return

    def to_context_string(self) -> str:
        """
        Convert to a compact string for context injection.
        Format is designed to be informative but not verbose.
        """
        if not self.images:
            return ""

        lines = ["## Images Encountered"]

        for i, img in enumerate(self.images, 1):
            if img.parent_url:
                # Transformed image
                lines.append(
                    f"[Image {i}] {img.transformation} from previous image - {img.description}"
                )
            else:
                lines.append(
                    f"[Image {i}] {img.source} - {img.description}"
                )

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "images": [
                {
                    "url": img.url,
                    "source": img.source,
                    "turn": img.turn,
                    "description": img.description,
                    "parent_url": img.parent_url,
                    "transformation": img.transformation,
                    "used_in_planning": img.used_in_planning,
                }
                for img in self.images
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MultimodalEvolution":
        """Deserialize from dictionary."""
        evolution = cls()
        for img_data in data.get("images", []):
            record = ImageRecord(
                url=img_data["url"],
                source=img_data["source"],
                turn=img_data["turn"],
                description=img_data.get("description", ""),
                parent_url=img_data.get("parent_url"),
                transformation=img_data.get("transformation"),
                used_in_planning=img_data.get("used_in_planning", False),
            )
            evolution.images.append(record)
            evolution.image_urls_seen.add(record.url)
        return evolution


# =============================================================================
# Progressive Context Prompt Templates
# =============================================================================

PROGRESSIVE_CONTEXT_SYSTEM_PROMPT = """You are a specialized Sub-Agent for Research State Management. Your purpose is to synthesize search trajectories and findings into a rigorous, evolving research report. You act as the "memory and analytical core" that ensures the research context is preserved and logically progressed.

CORE OBJECTIVE:
Analyze the transition from the Last Status Report to the Latest Action and Response, and generate a consolidated, high-fidelity documentation of the current research state.

OUTPUT FORMAT REQUIREMENTS:
You MUST output your entire response within <report> tags. No other tags or conversational filler are allowed.

Mandatory Structure:

1. Evolutionary Trace
   - Previous Action Review: Briefly state the specific intent and action of the last tool call.
   - Current State Evolution: Describe how the latest observation has shifted the research. (e.g., "Hypothesis A confirmed," "New branch opened regarding Variable B," or "Previous source found to be outdated.")

2. Consolidated Findings & Evidence
   Strict Requirement: Directly record actual data, facts, and figures. Do not use summaries like "information about X was found." Instead, list "X is [Value] according to [Source]."
   - Confirmed Fact A: [Details + Source]
   - Confirmed Fact B: [Details + Source]
   ...

3. Verification & Credibility Matrix
   | Key Finding | Status | Evidence Strength |
   |-------------|--------|-------------------|
   | [Finding 1] | [Confirmed/Inference/Hypothesis] | [High/Med/Low] |

4. Remaining Information Gaps
   - Specific questions that remain unanswered.
   - Potential contradictions or dead ends identified in the latest step.

GUIDING PRINCIPLES:
- Lossless Documentation: Ensure no critical data (names, dates, metrics) from previous reports or new observations is lost.
- Context Inheritance: A third party should be able to read this report and fully understand the current progress without seeing the raw logs.
- Rigorous Distinction: Clearly separate what is "proven fact" from what is "highly likely inference."
- Conciseness: Keep the report focused and avoid redundancy."""

PROGRESSIVE_CONTEXT_USER_PROMPT = """INPUT DATA:
Question: {question}
Last Status Report: {last_report}
Last Tool Call: {action}
Last Tool Response: {observation}

Now, synthesize the evolution of this research and provide the updated report."""


# =============================================================================
# Progressive Context Manager
# =============================================================================

class ProgressiveContext:
    """
    Main class for managing progressive context during agent execution.

    This class maintains:
    1. A text evolution report that summarizes research progress
    2. A multimodal evolution table that tracks images and transformations

    Key features:
    - Sub-linear growth: Report is compressed each turn
    - Pluggable: Can be enabled/disabled without affecting main flow
    - Non-invasive: Does not modify existing message history
    """

    def __init__(
        self,
        task_description: str,
        summary_llm_api_key: Optional[str] = None,
        summary_llm_base_url: Optional[str] = None,
        summary_llm_model_name: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Initialize the progressive context manager.

        Args:
            task_description: The original task/question
            summary_llm_api_key: API key for summary LLM
            summary_llm_base_url: Base URL for summary LLM
            summary_llm_model_name: Model name for summary LLM
            enabled: Whether progressive context is enabled
        """
        self.task_description = task_description
        self.enabled = enabled

        # Initialize LLM client for summary generation
        self.summary_llm_api_key = summary_llm_api_key or os.environ.get("SUMMARY_LLM_API_KEY")
        self.summary_llm_base_url = summary_llm_base_url or os.environ.get("SUMMARY_LLM_BASE_URL")
        self.summary_llm_model_name = summary_llm_model_name or os.environ.get("SUMMARY_LLM_MODEL_NAME")

        # State
        self.text_report = ""  # Current text evolution report
        self.multimodal_evolution = MultimodalEvolution()
        self.turn_count = 0

        # Pending images from search results (to be confirmed if used)
        self._pending_search_images: List[Tuple[str, int]] = []

        # LLM client (lazy initialization)
        self._llm_client = None

    def _get_llm_client(self):
        """Lazy initialization of LLM client."""
        if self._llm_client is None and self.summary_llm_api_key:
            try:
                from openai import AsyncOpenAI
                # Handle base_url that might already contain /chat/completions or /v1/chat/completions
                base_url = self.summary_llm_base_url
                if base_url:
                    # Remove trailing /chat/completions if present (OpenAI client adds /chat/completions)
                    # This handles both /v1/chat/completions and /chat/completions
                    if '/chat/completions' in base_url:
                        base_url = base_url.split('/chat/completions')[0]
                    logger.debug(f"ProgressiveContext: Adjusted base_url to {base_url}")
                self._llm_client = AsyncOpenAI(
                    api_key=self.summary_llm_api_key,
                    base_url=base_url,
                )
                logger.info(f"ProgressiveContext: LLM client initialized with base_url={base_url}")
            except ImportError:
                logger.warning("ProgressiveContext: OpenAI package not available")
            except Exception as e:
                logger.warning(f"ProgressiveContext: Failed to initialize LLM client: {e}")
        return self._llm_client

    async def update(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: Any,
    ) -> str:
        """
        Update the progressive context after a tool execution.

        Args:
            tool_name: Name of the tool that was executed
            tool_args: Arguments passed to the tool
            tool_result: Result from the tool execution

        Returns:
            Updated context string for injection (empty if disabled)
        """
        if not self.enabled:
            return ""

        self.turn_count += 1

        # Extract images from tool result
        self._extract_images_from_result(tool_name, tool_args, tool_result)

        # Update text report using LLM
        await self._update_text_report(tool_name, tool_args, tool_result)

        return self.get_context_string()

    def _extract_images_from_result(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: Any,
    ):
        """Extract and track images from tool results."""

        # Handle fetch_image tool
        if "fetch_image" in tool_name.lower():
            if isinstance(tool_result, dict):
                result = tool_result.get("result", "")
                # Extract URLs from result
                urls = self._extract_urls_from_text(str(result))
                for url in urls:
                    self.multimodal_evolution.add_image(
                        url=url,
                        source="fetch_image",
                        turn=self.turn_count,
                        description=f"Fetched via {tool_name}",
                        used_in_planning=True,
                    )

        # Handle image processing tools
        elif "image_processing" in tool_name.lower() or "zoom" in tool_name.lower():
            if isinstance(tool_result, dict):
                result = tool_result.get("result", "")
                urls = self._extract_urls_from_text(str(result))
                # Get the transformation type from tool name or args
                transformation = tool_name
                if "zoom_in" in str(tool_args):
                    transformation = "zoom_in"
                elif "rotate" in str(tool_args):
                    transformation = "rotate"
                elif "flip" in str(tool_args):
                    transformation = "flip"

                # Find parent URL from args
                parent_url = None
                if isinstance(tool_args, dict):
                    parent_url = tool_args.get("image_url") or tool_args.get("url")

                for url in urls:
                    self.multimodal_evolution.add_image(
                        url=url,
                        source="image_processing",
                        turn=self.turn_count,
                        description=f"Transformed via {transformation}",
                        parent_url=parent_url,
                        transformation=transformation,
                        used_in_planning=True,
                    )

        # Handle search results with images
        elif "search" in tool_name.lower():
            if isinstance(tool_result, dict):
                result = tool_result.get("result", "")
                # Extract potential image URLs from search results
                urls = self._extract_urls_from_text(str(result))
                image_urls = [u for u in urls if self._is_image_url(u)]
                # Only track these - they'll be marked as used if fetch_image is called
                for url in image_urls[:3]:  # Only top 3
                    self._pending_search_images.append((url, self.turn_count))

    def mark_image_fetched(self, url: str):
        """Mark a previously seen search image as fetched/used."""
        # Remove from pending and add as used
        self._pending_search_images = [
            (u, t) for u, t in self._pending_search_images if u != url
        ]
        self.multimodal_evolution.mark_image_used(url)

    def _extract_urls_from_text(self, text: str) -> List[str]:
        """Extract URLs from text."""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        return re.findall(url_pattern, text)

    def _is_image_url(self, url: str) -> bool:
        """Check if URL points to an image."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
        url_lower = url.lower()
        return any(ext in url_lower for ext in image_extensions)

    async def _update_text_report(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: Any,
    ):
        """Update the text evolution report using LLM."""
        client = self._get_llm_client()
        if not client:
            # Fallback: simple concatenation
            self._fallback_update_report(tool_name, tool_args, tool_result)
            logger.info(f"ProgressiveContext: Fallback update for '{tool_name}', report_len={len(self.text_report)}")
            return

        try:
            # Format the action and observation
            action_str = f"Tool: {tool_name}\nArgs: {json.dumps(tool_args, ensure_ascii=False)[:500]}"
            observation_str = str(tool_result.get("result", tool_result))[:2000] if isinstance(tool_result, dict) else str(tool_result)[:2000]

            prompt = PROGRESSIVE_CONTEXT_USER_PROMPT.format(
                question=self.task_description,
                last_report=self.text_report or "(Initial state - no previous report)",
                action=action_str,
                observation=observation_str,
            )

            response = await client.chat.completions.create(
                model=self.summary_llm_model_name,
                messages=[
                    {"role": "system", "content": PROGRESSIVE_CONTEXT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=4096,
            )

            content = response.choices[0].message.content
            if not content or not content.strip():
                self._fallback_update_report(tool_name, tool_args, tool_result)
                logger.info(f"ProgressiveContext: Empty response (model={self.summary_llm_model_name}), fallback for '{tool_name}', report_len={len(self.text_report)}")
                return
            # Extract report from <report> tags
            match = re.search(r'<report>(.*?)</report>', content, re.DOTALL)
            if match:
                self.text_report = match.group(1).strip()
            else:
                self.text_report = content.strip()
            logger.info(f"ProgressiveContext: LLM update for '{tool_name}', report_len={len(self.text_report)}")
        except Exception as e:
            self._fallback_update_report(tool_name, tool_args, tool_result)
            logger.info(f"ProgressiveContext: Error ({e}), fallback for '{tool_name}', report_len={len(self.text_report)}")

    def _fallback_update_report(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: Any,
    ):
        """Fallback method when LLM is not available."""
        action_str = f"[Turn {self.turn_count}] Called {tool_name}"
        result_str = str(tool_result.get("result", tool_result))[:500] if isinstance(tool_result, dict) else str(tool_result)[:500]

        if self.text_report:
            self.text_report = f"{self.text_report}\n\n{action_str}\nResult: {result_str}"
        else:
            self.text_report = f"{action_str}\nResult: {result_str}"

    def get_context_string(self) -> str:
        """
        Get the current context string for injection into prompts.

        Returns:
            Formatted context string (empty if disabled)
        """
        if not self.enabled:
            return ""

        parts = []

        if self.text_report:
            parts.append("## Research Progress Report\n" + self.text_report)

        multimodal_str = self.multimodal_evolution.to_context_string()
        if multimodal_str:
            parts.append(multimodal_str)

        return "\n\n".join(parts)

    def to_dict(self) -> Dict:
        """Serialize state to dictionary."""
        return {
            "enabled": self.enabled,
            "task_description": self.task_description,
            "text_report": self.text_report,
            "multimodal_evolution": self.multimodal_evolution.to_dict(),
            "turn_count": self.turn_count,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ProgressiveContext":
        """Deserialize from dictionary."""
        instance = cls(
            task_description=data.get("task_description", ""),
            enabled=data.get("enabled", True),
        )
        instance.text_report = data.get("text_report", "")
        instance.multimodal_evolution = MultimodalEvolution.from_dict(
            data.get("multimodal_evolution", {})
        )
        instance.turn_count = data.get("turn_count", 0)
        return instance


# =============================================================================
# Helper function for integration
# =============================================================================

def create_progressive_context(
    cfg,
    task_description: str,
) -> Optional[ProgressiveContext]:
    """
    Factory function to create a ProgressiveContext instance based on config.

    Args:
        cfg: Hydra configuration object
        task_description: The task description

    Returns:
        ProgressiveContext instance or None if disabled
    """
    enabled = cfg.agent.get("use_progressive_context", False)

    if not enabled:
        return None

    return ProgressiveContext(
        task_description=task_description,
        enabled=True,
    )
