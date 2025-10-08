"""Factory for creating companion provider instances with detection logic."""

import logging
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from metacontext.ai.handlers.companions.companion_provider import (
        BaseCompanionProvider,
    )

logger = logging.getLogger(__name__)


class CompanionProviderFactory:
    """Factory for creating and detecting companion provider instances."""

    # Registry of available companion providers in priority order
    _companion_providers: ClassVar[list[type["BaseCompanionProvider"]]] = []

    @classmethod
    def _ensure_providers_registered(cls) -> None:
        """Ensure companion providers are registered (lazy loading)."""
        if not cls._companion_providers:
            # Import here to avoid circular imports
            from metacontext.ai.handlers.companions.copilot_provider import (
                GitHubCopilotProvider,
            )

            cls._companion_providers = [
                GitHubCopilotProvider,
                # Future companion providers can be added here
                # CursorProvider,
                # CodeiumProvider,
                # TabNineProvider,
            ]

    @classmethod
    def detect_available_companion(cls) -> "BaseCompanionProvider | None":
        """Detect the first available companion provider.

        Returns:
            First available companion provider instance, or None if none available

        """
        cls._ensure_providers_registered()
        for provider_cls in cls._companion_providers:
            try:
                provider = provider_cls()
                if provider.is_available():
                    logger.info(
                        "Detected available companion: %s",
                        provider.companion_type,
                    )
                    return provider
            except Exception as e:
                logger.debug(
                    "Failed to create companion %s: %s",
                    provider_cls.__name__,
                    e,
                )
                continue

        logger.info("No companion providers available")
        return None

    @classmethod
    def create_companion(cls, companion_type: str) -> "BaseCompanionProvider | None":
        """Create a specific companion provider by type.

        Args:
            companion_type: Type of companion (e.g., "github_copilot")

        Returns:
            Companion provider instance, or None if not available

        """
        cls._ensure_providers_registered()
        for provider_cls in cls._companion_providers:
            try:
                provider = provider_cls()
                if provider.companion_type == companion_type:
                    if provider.is_available():
                        logger.info("Created companion: %s", companion_type)
                        return provider
                    logger.warning(
                        "Companion %s not available on this system",
                        companion_type,
                    )
                    return None
            except Exception as e:
                logger.debug(
                    "Failed to create companion %s: %s",
                    provider_cls.__name__,
                    e,
                )
                continue

        logger.warning("Unknown companion type: %s", companion_type)
        return None

    @classmethod
    def list_available_companions(cls) -> list[dict[str, Any]]:
        """List all available companion providers with their info.

        Returns:
            List of companion info dictionaries

        """
        cls._ensure_providers_registered()
        available_companions = []

        for provider_cls in cls._companion_providers:
            try:
                provider = provider_cls()
                companion_info = provider.get_companion_info()
                available_companions.append(companion_info)
            except Exception as e:
                logger.debug(
                    "Failed to check companion %s: %s",
                    provider_cls.__name__,
                    e,
                )
                # Still add it to the list but mark as unavailable
                available_companions.append(
                    {
                        "type": getattr(provider_cls, "companion_type", "unknown"),
                        "available": False,
                        "error": str(e),
                    },
                )

        return available_companions

    @classmethod
    def is_companion_available(cls, companion_type: str | None = None) -> bool:
        """Check if any companion or a specific companion is available.

        Args:
            companion_type: Specific companion type to check, or None for any

        Returns:
            True if companion(s) available, False otherwise

        """
        if companion_type is None:
            # Check if any companion is available
            return cls.detect_available_companion() is not None

        # Check specific companion type
        companion = cls.create_companion(companion_type)
        return companion is not None

    @classmethod
    def register_companion(cls, provider_cls: type["BaseCompanionProvider"]) -> None:
        """Register a new companion provider.

        Args:
            provider_cls: Companion provider class to register

        """
        if provider_cls not in cls._companion_providers:
            cls._companion_providers.append(provider_cls)
            logger.info("Registered companion provider: %s", provider_cls.__name__)
        else:
            logger.warning(
                "Companion provider already registered: %s",
                provider_cls.__name__,
            )

    @classmethod
    def get_companion_detection_info(cls) -> dict[str, Any]:
        """Get detailed detection information for all companions.

        Returns:
            Dictionary with detection details for debugging

        """
        detection_info = {
            "total_companions": len(cls._companion_providers),
            "companions": [],
        }

        for provider_cls in cls._companion_providers:
            try:
                provider = provider_cls()
                companion_info = provider.get_companion_info()

                # Add detection details if available
                if hasattr(provider, "get_detection_info"):
                    companion_info["detection_details"] = provider.get_detection_info()

                detection_info["companions"].append(companion_info)
            except Exception as e:
                logger.debug(
                    "Failed to get detection info for %s: %s",
                    provider_cls.__name__,
                    e,
                )
                detection_info["companions"].append(
                    {
                        "type": getattr(provider_cls, "companion_type", "unknown"),
                        "available": False,
                        "error": str(e),
                    },
                )

        return detection_info
