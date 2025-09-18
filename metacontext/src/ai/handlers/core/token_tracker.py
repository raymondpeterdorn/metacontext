"""Token usage tracking for LLM providers.

This module implements a dedicated token tracking class to eliminate redundant
token tracking logic across provider implementations.
"""

from copy import deepcopy
from typing import Any, ClassVar

from src.schemas.core.core import TokenUsage


class TokenTracker:
    """Tracks token usage across LLM calls.

    This class centralizes token tracking logic to avoid duplicating the same
    dictionary manipulation code across multiple provider implementations.
    """

    # Default values for token tracking
    DEFAULT_VALUES: ClassVar[dict[str, Any]] = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_api_calls": 0,
        "provider": None,
        "model": None,
    }

    def __init__(self, provider_name: str, model_name: str) -> None:
        """Initialize token tracker with provider and model information.

        Args:
            provider_name: Name of the LLM provider
            model_name: Name of the model being used

        """
        self._usage = self.DEFAULT_VALUES.copy()
        self._usage["provider"] = provider_name
        self._usage["model"] = model_name

    def track_api_call(self) -> None:
        """Increment the API call counter."""
        self._usage["total_api_calls"] += 1

    def add_prompt_tokens(self, count: int) -> None:
        """Add prompt tokens to the tracker.

        Args:
            count: Number of tokens to add

        """
        if count <= 0:
            return

        self._usage["prompt_tokens"] += count
        self._usage["total_tokens"] += count

    def add_completion_tokens(self, count: int) -> None:
        """Add completion tokens to the tracker.

        Args:
            count: Number of tokens to add

        """
        if count <= 0:
            return

        self._usage["completion_tokens"] += count
        self._usage["total_tokens"] += count

    def add_total_tokens(self, count: int) -> None:
        """Add to the total token count directly.

        This is useful when a provider only reports total tokens without
        breaking them down into prompt and completion tokens.

        Args:
            count: Number of tokens to add

        """
        if count <= 0:
            return

        self._usage["total_tokens"] += count

    def track_response(
        self,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        """Track tokens from an API response.

        This convenience method handles the common case of updating token
        counts from an API response.

        Args:
            prompt_tokens: Number of prompt tokens, if provided
            completion_tokens: Number of completion tokens, if provided
            total_tokens: Total token count, if provided

        """
        self.track_api_call()

        if prompt_tokens is not None:
            self.add_prompt_tokens(prompt_tokens)

        if completion_tokens is not None:
            self.add_completion_tokens(completion_tokens)

        if total_tokens is not None:
            # If we have detailed counts, only use total_tokens as a fallback
            if prompt_tokens is None and completion_tokens is None:
                self.add_total_tokens(total_tokens)
            # Otherwise validate that our detailed counts match the total
            elif (prompt_tokens or 0) + (completion_tokens or 0) != total_tokens:
                # If they don't match, trust the total and adjust
                self._usage["total_tokens"] = total_tokens

    def get_usage(self) -> dict[str, Any]:
        """Get a copy of the current token usage data.

        Returns:
            Dictionary with token usage statistics

        """
        return deepcopy(self._usage)

    def to_schema(self) -> TokenUsage:
        """Convert token usage to a TokenUsage schema object.

        Returns:
            TokenUsage schema instance

        """
        return TokenUsage(**self.get_usage())
