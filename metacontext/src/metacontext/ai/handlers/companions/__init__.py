"""Code companion providers package."""

from metacontext.ai.handlers.companions.companion_factory import (
    CompanionProviderFactory,
)
from metacontext.ai.handlers.companions.companion_provider import (
    BaseCompanionProvider,
    TemplateContext,
)
from metacontext.ai.handlers.companions.copilot_provider import GitHubCopilotProvider

__all__ = [
    "BaseCompanionProvider",
    "CompanionProviderFactory",
    "GitHubCopilotProvider",
    "TemplateContext",
]
