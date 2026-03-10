# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Base client module for LLM providers.

This module defines the abstract base class and common utilities for LLM clients,
supporting both OpenAI and Anthropic API formats.
"""

import asyncio
import dataclasses
from abc import ABC
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
)

from omegaconf import DictConfig

from ..logging.task_logger import TaskLog
from .util import with_timeout

# Default timeout for LLM API calls (10 minutes)
DEFAULT_LLM_TIMEOUT_SECONDS = 600


class TokenUsage(TypedDict, total=True):
    """
    Unified token usage tracking across different LLM providers.

    We unify OpenAI and Anthropic formats. There are four usage types:
    - input/output tokens: Standard input and output token counts
    - cache write/read tokens: Tokens involved in caching operations

    Provider-specific notes:
    - OpenAI: Cache write is free, cache read is cheaper
    - Anthropic: Cache write has a small cost, cache read is cheaper
    """

    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_input_tokens: int
    total_cache_write_input_tokens: int


@dataclasses.dataclass
class BaseClient(ABC):
    """
    Abstract base class for LLM provider clients.

    This class provides the common interface and utilities for interacting with
    different LLM providers (OpenAI, Anthropic, etc.). Concrete implementations
    should override _create_client() and provider-specific methods.

    Attributes:
        task_id: Unique identifier for the current task (used for tracking)
        cfg: Hydra configuration containing LLM settings
        task_log: Optional logger for recording task execution details
    """

    # Required arguments (no default value)
    task_id: str
    cfg: DictConfig

    # Optional arguments (with default value)
    task_log: Optional["TaskLog"] = None

    # Initialized in __post_init__
    client: Any = dataclasses.field(init=False)
    token_usage: TokenUsage = dataclasses.field(init=False)
    last_call_tokens: Dict[str, int] = dataclasses.field(init=False)

    def __post_init__(self):
        # Initialize last_call_tokens before other operations
        self.last_call_tokens: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

        # Explicitly assign from cfg object
        self.provider: str = self.cfg.llm.provider
        self.model_name: str = self.cfg.llm.model_name
        self.temperature: float = self.cfg.llm.temperature
        self.top_p: float = self.cfg.llm.top_p
        self.min_p: float = self.cfg.llm.min_p
        self.top_k: int = self.cfg.llm.top_k
        self.max_context_length: int = self.cfg.llm.max_context_length
        self.max_tokens: int = self.cfg.llm.max_tokens
        self.async_client: bool = self.cfg.llm.async_client
        self.keep_tool_result: float = self.cfg.agent.keep_tool_result
        self.token_threshold: float = self.cfg.agent.get("token_threshold", 0.8)
        self.api_key: Optional[str] = self.cfg.llm.get("api_key")
        self.base_url: Optional[str] = self.cfg.llm.get("base_url")
        self.use_tool_calls: Optional[bool] = self.cfg.llm.get("use_tool_calls")
        self.repetition_penalty: float = self.cfg.llm.get("repetition_penalty", 1.0)

        self.token_usage = self._reset_token_usage()
        self.client = self._create_client()

        # Track indices of messages that have been permanently omitted to save tokens
        # This ensures compressed messages stay compressed across turns
        self.omitted_message_indices: set = set()

        # Track the first tool result index to keep (for iterative discard)
        # When set, all tool results before this index should be discarded
        self.first_kept_tool_result_idx: Optional[int] = None

        self.task_log.log_step(
            "info",
            "LLM | Initialization",
            f"LLMClient {self.provider} {self.model_name} initialization completed.",
        )

    def _reset_token_usage(self) -> TokenUsage:
        """
        Reset token usage counter to zero.

        Returns:
            A new TokenUsage dict with all counters set to zero.
        """
        return TokenUsage(
            total_input_tokens=0,
            total_output_tokens=0,
            total_cache_write_input_tokens=0,
            total_cache_read_input_tokens=0,
        )

    def _remove_tool_result_from_messages(
        self, messages, keep_tool_result
    ) -> List[Dict]:
        """Remove tool results from messages with iterative discard support

        This method implements iterative context discard:
        1. First discard (e.g., at turn 12): Keep last 20% of tool results [9, 10, 11, 12]
        2. Subsequent turns (13-15): Messages [1-8] stay omitted, [9-12] contain placeholder text
        3. Second discard (e.g., at turn 20): Keep last 20% of tool results [15, 16, 17, 18, 19, 20]
       The key is that we maintain a continuous window of recent messages, discarding everything
       before the new boundary to avoid "islands" of omitted messages.

        Args:
            messages: List of message dictionaries
            keep_tool_result: Fraction of tool results to keep (0.2 = keep last 20%).
                             Set to -1 to keep all (except previously omitted ones).

        Returns:
            List of messages with tool results filtered according to keep_tool_result
            and permanently omitted messages preserved.
        """
        messages_copy = [m.copy() for m in messages]

        # Find indices of all user/tool messages (these are tool results)
        user_indices = [
            i
            for i, msg in enumerate(messages_copy)
            if msg.get("role") == "user" or msg.get("role") == "tool"
        ]

        if len(user_indices) == 0:
            # No user/tool messages found
            self.task_log.log_step(
                "info",
                "LLM | Message Retention",
                "No user/tool messages found in the history.",
            )
            return messages_copy

        # The first user message is the initial task, not a tool result
        # Tool results start from the second user message onwards
        if len(user_indices) == 1:
            # Only one user message (the initial task), no tool results to filter
            self.task_log.log_step(
                "info",
                "LLM | Message Retention",
                "Only 1 user message found (initial task). Keeping it as is.",
            )
            return messages_copy

        # Tool result indices (excluding the first user message which is the initial task)
        tool_result_indices = user_indices[1:]
        first_user_idx = user_indices[
            0
        ]  # Always keep the first user message (initial task)

        # Start with the first user message that must always be kept
        indices_to_keep = [first_user_idx]

        # Apply iterative discard logic
        if keep_tool_result != -1:
            # Calculate how many tool results to keep from the end
            # IMPORTANT: Always calculate based on TOTAL tool results to maintain consistent boundary
            if keep_tool_result == 0:
                # Keep 0 tool results, only keep the initial task
                num_tool_results_to_keep = 0
            elif 0 < keep_tool_result < 1:
                # keep_tool_result is a fraction (e.g., 0.2 = keep last 20%)
                num_tool_results_to_keep = int(
                    len(tool_result_indices) * keep_tool_result
                )
            else:
                # keep_tool_result is an absolute number (backward compatibility)
                num_tool_results_to_keep = min(
                    int(keep_tool_result), len(tool_result_indices)
                )

            # Get indices of tool results to keep from the end
            # We always select from the END of ALL tool results
            tool_result_indices_to_keep = (
                tool_result_indices[-num_tool_results_to_keep:]
                if num_tool_results_to_keep > 0
                else []
            )

            # Update the boundary: the first tool result we're keeping
            if tool_result_indices_to_keep:
                new_boundary = tool_result_indices_to_keep[0]
                if self.first_kept_tool_result_idx is None or new_boundary > self.first_kept_tool_result_idx:
                    self.first_kept_tool_result_idx = new_boundary

            # Combine first message (initial task) and tool results to keep
            indices_to_keep = [first_user_idx] + tool_result_indices_to_keep

            # Calculate which indices should be omitted
            # For iterative discard: omit everything before the new boundary
            if self.first_kept_tool_result_idx is not None:
                indices_to_omit = {
                    idx for idx in tool_result_indices
                    if idx < self.first_kept_tool_result_idx
                }
            else:
                indices_to_omit = set(tool_result_indices) - set(tool_result_indices_to_keep)

            self.omitted_message_indices.update(indices_to_omit)

            self.task_log.log_step(
                "info",
                "LLM | Message Retention",
                f"Message retention summary: Total user/tool messages: {len(user_indices)}, "
                f"Initial task at index: {first_user_idx}, "
                f"Total tool results: {len(tool_result_indices)}, "
                f"Keeping last {num_tool_results_to_keep} tool results ({keep_tool_result:.1%}) at indices: {tool_result_indices_to_keep}, "
                f"Discard boundary at index: {self.first_kept_tool_result_idx}, "
                f"Omitting {len(indices_to_omit)} messages at indices: {sorted(indices_to_omit)}, "
                f"Total messages to keep: {len(indices_to_keep)}, "
                f"Total permanently omitted: {len(self.omitted_message_indices)}",
            )
        else:
            # keep_tool_result == -1, but we still need to respect previously omitted messages
            # Only keep messages that are not in the omitted set
            indices_to_keep = [
                idx for idx in user_indices
                if idx not in self.omitted_message_indices
            ]

            self.task_log.log_step(
                "info",
                "LLM | Message Retention",
                f"Message retention summary: Total user/tool messages: {len(user_indices)}, "
                f"Keeping {len(indices_to_keep)} messages (excluding {len(self.omitted_message_indices)} permanently omitted), "
                f"keep_tool_result=-1 (preserving previous omissions), "
                f"Discard boundary at index: {self.first_kept_tool_result_idx}",
            )

        # Replace content of tool results that should be omitted
        for i, msg in enumerate(messages_copy):
            if (
                msg.get("role") == "user" or msg.get("role") == "tool"
            ) and i not in indices_to_keep:
                # Preserve the message structure but replace content
                if isinstance(msg.get("content"), list):
                    # For Anthropic format
                    msg["content"] = [
                        {
                            "type": "text",
                            "text": "Tool result is omitted to save tokens.",
                        }
                    ]
                else:
                    # For OpenAI format
                    msg["content"] = "Tool result is omitted to save tokens."

        return messages_copy

    @with_timeout(DEFAULT_LLM_TIMEOUT_SECONDS)
    async def create_message(
        self,
        system_prompt: str,
        message_history: List[Dict],
        tool_definitions: List[Dict],
        keep_tool_result: int = -1,
        step_id: int = 1,
        task_log: Optional["TaskLog"] = None,
        agent_type: str = "main",
    ) -> Tuple[Any, List[Dict]]:
        """
        Call LLM to generate a response with optional tool call support.

        This is the main entry point for LLM interactions. It handles:
        - Message history management
        - Dynamic tool result filtering based on token threshold
        - Error handling and logging

        Args:
            system_prompt: System prompt to guide the LLM's behavior
            message_history: List of previous messages in the conversation
            tool_definitions: List of available tool definitions
            keep_tool_result: Fraction of recent tool results to keep (-1 = keep all)
            step_id: Current step identifier for logging
            task_log: Optional logger for task execution
            agent_type: Type of agent making the call ("main" or sub-agent name)

        Returns:
            Tuple of (response, updated_message_history)
        """
        # Dynamic context management: enable keep_tool_result when token usage reaches threshold
        effective_keep_tool_result = keep_tool_result
        if self.token_threshold > 0 and keep_tool_result != -1:
            # Use the prompt tokens from the most recent API call as the current context size.
            # This reflects how full the context window actually is right now, and stays bounded
            # within [0, max_context_length] — unlike cumulative session totals which grow without bound.
            # OpenAI uses "prompt_tokens"; Anthropic uses "input_tokens". Fall back to 0 before the first call.
            current_context_tokens = self.last_call_tokens.get(
                "prompt_tokens", None
            ) or self.last_call_tokens.get("input_tokens", 0)

            token_usage_ratio = current_context_tokens / self.max_context_length if self.max_context_length > 0 else 0

            if token_usage_ratio >= self.token_threshold:
                # Token usage reached threshold, enable keep_tool_result
                effective_keep_tool_result = self.keep_tool_result
                self.task_log.log_step(
                    "info",
                    f"LLM | Dynamic Context Management",
                    f"Context usage {token_usage_ratio:.1%} ({current_context_tokens}/{self.max_context_length}) "
                    f"reached threshold {self.token_threshold:.1%}. "
                    f"Enabling keep_tool_result={self.keep_tool_result:.1%} to reduce context size.",
                )
            else:
                # Token usage below threshold, keep all tool results
                effective_keep_tool_result = -1
                self.task_log.log_step(
                    "info",
                    f"LLM | Dynamic Context Management",
                    f"Context usage {token_usage_ratio:.1%} ({current_context_tokens}/{self.max_context_length}) "
                    f"below threshold {self.token_threshold:.1%}. "
                    f"Keeping all tool results (keep_tool_result=-1).",
                )

        # Unified LLM call processing
        try:
            response, message_history = await self._create_message(
                system_prompt,
                message_history,
                tool_definitions,
                keep_tool_result=effective_keep_tool_result,
            )

        except Exception as e:
            self.task_log.log_step(
                "error",
                f"FATAL ERROR | {agent_type} | LLM Call ERROR",
                f"{agent_type} failed: {str(e)}",
            )
            response = None

        return response, message_history

    @staticmethod
    def convert_tool_definition_to_openai_format(tools_definitions):
        """
        Convert MCP tool definitions to OpenAI function call format.

        Transforms the internal tool definition format used by MCP servers into
        the format expected by OpenAI's function calling API.

        Args:
            tools_definitions: List of server definitions, each containing a 'name'
                and 'tools' list with tool specifications.

        Returns:
            List of tool definitions in OpenAI function call format, where each
            tool name is prefixed with its server name (e.g., "server-name-tool-name").
        """
        tool_list = []
        for server in tools_definitions:
            if "tools" in server and len(server["tools"]) > 0:
                for tool in server["tools"]:
                    # Skip tools that failed to load (they only have 'error' key)
                    if "error" in tool and "name" not in tool:
                        continue
                    tool_def = dict(
                        type="function",
                        function=dict(
                            name=f"{server['name']}-{tool['name']}",
                            description=tool["description"],
                            parameters=tool["schema"],
                        ),
                    )
                    tool_list.append(tool_def)
        return tool_list

    def close(self):
        """Close client connection.

        Note: For async clients (AsyncOpenAI, AsyncAnthropic), the connection
        will be closed when the client object is garbage collected.
        For proper async cleanup, use `await client.aclose()` in an async context.
        """
        if hasattr(self.client, "close"):
            if asyncio.iscoroutinefunction(self.client.close):
                # For async clients, we cannot call close() synchronously.
                # The async HTTP client will be closed when garbage collected.
                # For explicit async cleanup, call aclose() from an async context.
                if hasattr(self.client, "_client"):
                    # Try to close the underlying httpx client if available
                    try:
                        self.client._client.close()
                    except Exception:
                        pass  # Ignore errors during cleanup
            else:
                self.client.close()
        elif hasattr(self.client, "_client") and hasattr(self.client._client, "close"):
            # Some clients may have internal _client attribute
            self.client._client.close()

    def _format_response_for_log(self, response) -> Dict:
        """Format response for logging"""
        if not response:
            return {}

        # Basic response information
        formatted = {
            "response_type": type(response).__name__,
        }

        # Anthropic response
        if hasattr(response, "content"):
            formatted["content"] = []
            for block in response.content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        formatted["content"].append(
                            {
                                "type": "text",
                                "text": block.text[:500] + "..."
                                if len(block.text) > 500
                                else block.text,
                            }
                        )
                    elif block.type == "tool_use":
                        formatted["content"].append(
                            {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": str(block.input)[:200] + "..."
                                if len(str(block.input)) > 200
                                else str(block.input),
                            }
                        )

        # OpenAI response
        if hasattr(response, "choices"):
            formatted["choices"] = []
            for choice in response.choices:
                choice_data = {"finish_reason": choice.finish_reason}
                if hasattr(choice, "message"):
                    message = choice.message
                    choice_data["message"] = {
                        "role": message.role,
                        "content": message.content[:500] + "..."
                        if message.content and len(message.content) > 500
                        else message.content,
                    }
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        choice_data["message"]["tool_calls_count"] = len(
                            message.tool_calls
                        )
                formatted["choices"].append(choice_data)

        return formatted
