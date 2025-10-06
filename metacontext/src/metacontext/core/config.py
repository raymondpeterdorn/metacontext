"""Configuration management system for Metacontext.

This module provides a centralized configuration system that follows a hierarchical approach:
1. Default configurations defined in this module
2. Environment variables (override defaults)
3. User configuration files (override environment variables)
4. Runtime configuration (overrides everything)

This eliminates scattered configuration values across multiple modules and provides
a consistent interface for accessing configuration values.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import ClassVar

import yaml


class ConfigSource(str, Enum):
    """Source of a configuration value."""

    DEFAULT = "default"
    ENV_VAR = "environment_variable"
    CONFIG_FILE = "config_file"
    RUNTIME = "runtime"


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    provider: str = "gemini"
    model: str | None = None  # Will be dynamically determined by provider
    api_key: str | None = None
    temperature: float = 0.1
    max_tokens: int = 4000
    max_retries: int = 3

    sources: dict[str, ConfigSource] = field(default_factory=dict)


@dataclass
class AppConfig:
    """Main application configuration."""

    # LLM configuration
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Output configuration
    output_format: str = "yaml"
    verbosity: bool = False

    # Feature flags
    include_llm_analysis: bool = True
    scan_codebase: bool = True
    scan_depth: int = 3  # How deep to scan the codebase

    # Paths
    prompt_template_path: str = "prompts/templates"

    # Internal tracking
    sources: dict[str, ConfigSource] = field(default_factory=dict)


class ConfigManager:
    """Configuration manager for the application.

    This class handles loading configuration from various sources and provides
    a unified interface for accessing configuration values.
    """

    # Configuration file names to search for (in order)
    CONFIG_FILES: ClassVar[list[str]] = [
        ".metacontextrc",
        ".metacontext.yaml",
        ".metacontext.yml",
    ]

    # Environment variable prefixes
    ENV_PREFIX: ClassVar[str] = "METACONTEXT_"

    # Provider-specific environment variables for API keys
    PROVIDER_ENV_VARS: ClassVar[dict[str, str]] = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "cohere": "COHERE_API_KEY",
        "azure": "AZURE_OPENAI_API_KEY",
    }

    # Global instance
    _instance: ClassVar["ConfigManager | None"] = None

    def __init__(self) -> None:
        """Initialize a new configuration manager."""
        self.config = AppConfig()
        self._load_defaults()

    @classmethod
    def get_instance(cls) -> "ConfigManager":
        """Get the global configuration manager instance.

        Returns:
            The global configuration manager instance

        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load_config()
        return cls._instance

    def _load_defaults(self) -> None:
        """Load default configuration values."""
        # All defaults are already set in the dataclass defaults
        self.config.sources["default"] = ConfigSource.DEFAULT
        self.config.llm.sources["default"] = ConfigSource.DEFAULT

    def _load_env_vars(self) -> None:
        """Load configuration from environment variables."""
        # LLM provider selection
        if provider := os.getenv(f"{self.ENV_PREFIX}LLM_PROVIDER"):
            self.config.llm.provider = provider
            self.config.llm.sources["provider"] = ConfigSource.ENV_VAR

        # LLM model selection
        if model := os.getenv(f"{self.ENV_PREFIX}LLM_MODEL"):
            self.config.llm.model = model
            self.config.llm.sources["model"] = ConfigSource.ENV_VAR

        # LLM API key - try provider-specific first, then generic
        provider = self.config.llm.provider
        if (provider in self.PROVIDER_ENV_VARS and
                (api_key := os.getenv(self.PROVIDER_ENV_VARS[provider]))):
            self.config.llm.api_key = api_key
            self.config.llm.sources["api_key"] = ConfigSource.ENV_VAR

        # Generic API key fallback
        if (not self.config.llm.api_key and
                (api_key := os.getenv(f"{self.ENV_PREFIX}LLM_API_KEY"))):
            self.config.llm.api_key = api_key
            self.config.llm.sources["api_key"] = ConfigSource.ENV_VAR

        # Other LLM parameters
        if temp := os.getenv(f"{self.ENV_PREFIX}LLM_TEMPERATURE"):
            self.config.llm.temperature = float(temp)
            self.config.llm.sources["temperature"] = ConfigSource.ENV_VAR

        if max_tokens := os.getenv(f"{self.ENV_PREFIX}LLM_MAX_TOKENS"):
            self.config.llm.max_tokens = int(max_tokens)
            self.config.llm.sources["max_tokens"] = ConfigSource.ENV_VAR

        if max_retries := os.getenv(f"{self.ENV_PREFIX}LLM_MAX_RETRIES"):
            self.config.llm.max_retries = int(max_retries)
            self.config.llm.sources["max_retries"] = ConfigSource.ENV_VAR

        # Feature flags
        if include_llm := os.getenv(f"{self.ENV_PREFIX}INCLUDE_LLM_ANALYSIS"):
            self.config.include_llm_analysis = include_llm.lower() in ("true", "1", "yes")
            self.config.sources["include_llm_analysis"] = ConfigSource.ENV_VAR

        if scan_codebase := os.getenv(f"{self.ENV_PREFIX}SCAN_CODEBASE"):
            self.config.scan_codebase = scan_codebase.lower() in ("true", "1", "yes")
            self.config.sources["scan_codebase"] = ConfigSource.ENV_VAR

        # Output format
        if output_format := os.getenv(f"{self.ENV_PREFIX}OUTPUT_FORMAT"):
            self.config.output_format = output_format
            self.config.sources["output_format"] = ConfigSource.ENV_VAR

        # Verbosity
        if verbosity := os.getenv(f"{self.ENV_PREFIX}VERBOSITY"):
            self.config.verbosity = verbosity.lower() in ("true", "1", "yes")
            self.config.sources["verbosity"] = ConfigSource.ENV_VAR

        # Scan depth
        if scan_depth := os.getenv(f"{self.ENV_PREFIX}SCAN_DEPTH"):
            self.config.scan_depth = int(scan_depth)
            self.config.sources["scan_depth"] = ConfigSource.ENV_VAR

    def _find_config_file(self) -> Path | None:
        """Find a configuration file in standard locations.

        Returns:
            Path to the configuration file if found, None otherwise

        """
        # Check current directory
        for filename in self.CONFIG_FILES:
            config_path = Path.cwd() / filename
            if config_path.exists():
                return config_path

        # Check user home directory
        for filename in self.CONFIG_FILES:
            config_path = Path.home() / filename
            if config_path.exists():
                return config_path

        return None

    def _load_config_file(self, config_path: Path) -> None:
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the configuration file

        """
        try:
            with config_path.open() as f:
                file_config = yaml.safe_load(f)

            if not isinstance(file_config, dict):
                return

            # LLM configuration
            if (llm_config := file_config.get("llm")) and isinstance(llm_config, dict):
                for key, value in llm_config.items():
                    if hasattr(self.config.llm, key):
                        setattr(self.config.llm, key, value)
                        self.config.llm.sources[key] = ConfigSource.CONFIG_FILE

            # Other configuration keys
            for key, value in file_config.items():
                if key != "llm" and hasattr(self.config, key):
                    setattr(self.config, key, value)
                    self.config.sources[key] = ConfigSource.CONFIG_FILE

        except (yaml.YAMLError, OSError):
            # Ignore errors - configuration files are optional
            pass

    def load_config(self) -> None:
        """Load configuration from all sources."""
        # Load in order of precedence (lowest to highest)
        self._load_defaults()
        self._load_env_vars()

        if config_path := self._find_config_file():
            self._load_config_file(config_path)

    def update_config(self, **kwargs: dict[str, object]) -> None:
        """Update configuration with runtime values.

        Args:
            **kwargs: Configuration values to update

        """
        # Update LLM configuration
        if (llm_config := kwargs.pop("llm", None)) and isinstance(llm_config, dict):
            for key, value in llm_config.items():
                if hasattr(self.config.llm, key):
                    setattr(self.config.llm, key, value)
                    self.config.llm.sources[key] = ConfigSource.RUNTIME

        # Update other configuration
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.config.sources[key] = ConfigSource.RUNTIME

    def get_config(self) -> AppConfig:
        """Get the current configuration.

        Returns:
            The current application configuration

        """
        return self.config


# Convenience function to get the global configuration
def get_config() -> AppConfig:
    """Get the current application configuration.

    Returns:
        The current application configuration

    """
    return ConfigManager.get_instance().get_config()
