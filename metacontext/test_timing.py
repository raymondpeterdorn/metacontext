#!/usr/bin/env python3
"""Quick test script to verify timing instrumentation in tabular handler."""

import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from metacontext.handlers.tabular import TabularHandler

# Set up logging to see timing output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


async def test_timing():
    """Test the timing instrumentation with a sample CSV."""
    # Use the bird demo CSV as a test file
    test_file = Path(
        "/Users/bigdawg/Documents/repos/metacontext/bird_demo/bird_demo/data/birdos.csv",
    )

    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return None

    handler = TabularHandler()

    # Mock LLM handler for testing (to avoid actual API calls)
    class MockLLMHandler:
        async def query_gemini(self, prompt, **kwargs):
            # Simulate some delay to test timing
            await asyncio.sleep(0.1)  # 100ms delay
            return {
                "analysis": "Mock analysis result",
                "insights": ["Mock insight 1", "Mock insight 2"],
            }

    mock_llm = MockLLMHandler()

    # Test generate_context which includes the timing instrumentation
    print(f"Testing timing instrumentation with file: {test_file}")
    result = await handler.generate_context(test_file, {}, mock_llm)

    print("Test completed!")
    return result


if __name__ == "__main__":
    asyncio.run(test_timing())
