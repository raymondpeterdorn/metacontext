"""Tests for the LLM handler module (metacontext.llm_handler)."""

from unittest.mock import Mock, patch

from metacontext.ai.handlers import LLMHandler


class LLMTestError(Exception):
    """Custom exception for LLM tests."""


def test_llm_handler_invalid_provider():
    """Test LLM handler with invalid provider."""
    with patch("builtins.print") as mock_print:
        handler = LLMHandler({"provider": "invalid_provider"})
        assert handler.client is None
        mock_print.assert_called()


@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
@patch("openai.OpenAI")
def test_llm_handler_openai_success(mock_openai):
    """Test successful OpenAI initialization and usage."""
    mock_client = Mock()
    mock_openai.return_value = mock_client

    # Mock successful API response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[
        0
    ].message.content = '{"analysis": "success", "summary": "test"}'
    mock_client.chat.completions.create.return_value = mock_response

    handler = LLMHandler({"provider": "openai", "api_key": "test-key"})
    assert handler.client is not None
    assert handler.provider == "openai"

    # Test context generation
    result = handler.generate_context({"test": "data"})
    assert result == {"analysis": "success", "summary": "test"}


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
@patch("anthropic.Anthropic")
def test_llm_handler_anthropic_success(mock_anthropic):
    """Test successful Anthropic initialization and usage."""
    mock_client = Mock()
    mock_anthropic.return_value = mock_client

    # Mock successful response
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = '{"analysis": "anthropic success"}'
    mock_client.messages.create.return_value = mock_response

    handler = LLMHandler({"provider": "anthropic", "api_key": "test-key"})
    assert handler.client is not None
    assert handler.provider == "anthropic"

    result = handler.generate_context({"test": "data"})
    assert result == {"analysis": "anthropic success"}


@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
@patch("openai.OpenAI")
def test_llm_handler_api_error(mock_openai):
    """Test LLM handler API error handling."""
    mock_client = Mock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.side_effect = ValueError("API Error")

    handler = LLMHandler({"provider": "openai", "api_key": "test-key"})
    result = handler.generate_context({"test": "data"})

    assert "error" in result
    assert "API Error" in result["error"]


@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
@patch("openai.OpenAI")
def test_llm_handler_invalid_json(mock_openai):
    """Test LLM handler with invalid JSON response."""
    mock_client = Mock()
    mock_openai.return_value = mock_client

    # Mock invalid JSON response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "This is not valid JSON"
    mock_client.chat.completions.create.return_value = mock_response

    handler = LLMHandler({"provider": "openai", "api_key": "test-key"})
    result = handler.generate_context({"test": "data"})

    assert "error" in result


def test_llm_handler_no_client():
    """Test LLM handler behavior when client is None."""
    handler = LLMHandler.__new__(LLMHandler)
    handler.client = None

    result = handler.generate_context({"test": "data"})
    assert result == {"error": "LLM client not initialized"}


def test_llm_handler_unsupported_provider_in_config():
    """Test providing an unsupported provider in config."""
    handler = LLMHandler({"provider": "unsupported_provider"})
    assert handler.client is None


def test_llm_handler_missing_api_key():
    """Test LLM handler with missing API key."""
    with patch.dict("os.environ", {}, clear=True):
        handler = LLMHandler({"provider": "openai"})
        assert handler.client is None


def test_llm_handler_generate_context_with_no_client():
    """Test generate_context when client is not initialized."""
    handler = LLMHandler({"provider": "unsupported"})
    result = handler.generate_context({})
    assert "error" in result
    assert "not initialized" in result["error"]


def test_llm_response_parsing():
    """Test LLM response parsing methods."""
    handler = LLMHandler.__new__(LLMHandler)

    # Test valid JSON
    result = handler._parse_response('{"key": "value", "number": 42}')
    assert result == {"key": "value", "number": 42}

    # Test invalid JSON (current behavior)
    result = handler._parse_response("Not valid JSON at all")
    assert "error" in result
    assert result["error"] == "Response was not valid JSON"
    assert result["parsed"] is False
    assert result["raw_response"] == "Not valid JSON at all"


def test_llm_handler_import_errors():
    """Test LLM handler with import errors."""
    # Test OpenAI import error
    with patch("builtins.__import__") as mock_import:

        def mock_import_func(name, *args, **kwargs):
            if name == "openai":
                msg = "OpenAI not installed"
                raise ImportError(msg)
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = mock_import_func

        with patch("builtins.print") as mock_print:
            handler = LLMHandler({"provider": "openai"})
            assert handler.client is None
            mock_print.assert_called_with(
                "OpenAI package not installed. Install with: pip install openai",
            )


def test_llm_handler_format_sample_files():
    """Test sample files formatting method."""
    handler = LLMHandler.__new__(LLMHandler)

    # Test with empty files
    result = handler.format_sample_files([])
    assert result == "No files analyzed."

    # Test with sample files
    files = [
        {
            "path": "test.py",
            "language": "python",
            "lines": 10,
            "content": "def hello():\n    return 'world'",
        },
    ]
    result = handler.format_sample_files(files)
    assert "test.py" in result
    assert "python" in result
    assert "10" in result


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
def test_llm_handler_anthropic_config():
    """Test Anthropic configuration."""
    config = {
        "provider": "anthropic",
        "model": "claude-3-sonnet-20240229",
        "api_key": "test-key",
    }

    with patch("anthropic.Anthropic") as mock_anthropic:
        handler = LLMHandler(config)
        assert handler.provider == "anthropic"
        assert handler.model == "claude-3-sonnet-20240229"
        mock_anthropic.assert_called_with(api_key="test-key")


def test_anthropic_json_parsing():
    """Test that Anthropic responses are correctly parsed."""
    # Test case where the response is a simple JSON string
    text1 = '{"key": "value"}'
    assert LLMHandler.parse_anthropic_response(text1) == {"key": "value"}

    # Test case with code block markers
    text2 = '```json\n{"key": "value"}\n```'
    assert LLMHandler.parse_anthropic_response(text2) == {"key": "value"}

    # Test with leading/trailing whitespace
    text3 = '  {"key": "value"}  '
    assert LLMHandler.parse_anthropic_response(text3) == {"key": "value"}

    # Test invalid JSON
    text4 = "not json"
    parsed = LLMHandler.parse_anthropic_response(text4)
    assert "error" in parsed
    assert "Failed to decode" in parsed["error"]


def test_init_client_without_provider():
    """Test that _init_client handles no provider gracefully."""
    handler = LLMHandler({})
    assert handler.client is None
    assert handler.provider is None


def test_init_client_with_empty_provider():
    """Test that _init_client handles an empty provider string."""
    handler = LLMHandler({"provider": ""})
    assert handler.client is None
    assert handler.provider == ""
