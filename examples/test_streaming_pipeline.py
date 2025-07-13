#!/usr/bin/env python3
"""
Test script to demonstrate the streaming LLM → Sentencizer → TTS pipeline
"""

import sys
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock

# Add src to path
# fmt: off
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from macecho.sentencizer import LLMSentencizer
from macecho.llm.base import LLMStreamChunk


class MockMLXQwenChat:
    """Mock MLX Qwen chat for testing"""

    def __init__(self, **kwargs):
        self.model_name = "mock-model"

    def _generate_response(self, messages, max_tokens=1000, temperature=0.7, stream=True):
        """Simulate streaming LLM response"""
        if not stream:
            return LLMStreamChunk(content="Complete response here.")

        # Simulate streaming chunks that will form complete sentences
        chunks = [
            LLMStreamChunk(id="test", model="mock", role="assistant"),
            LLMStreamChunk(id="test", model="mock", content="Hello there!"),
            LLMStreamChunk(id="test", model="mock", content=" How can I"),
            LLMStreamChunk(id="test", model="mock", content=" help you today?\n"),
            LLMStreamChunk(id="test", model="mock", content=" I'm an AI assistant"),
            LLMStreamChunk(id="test", model="mock", content=" designed to be helpful.\n"),
            LLMStreamChunk(id="test", model="mock", content=" What would you"),
            LLMStreamChunk(id="test", model="mock", content=" like to know?\n"),
            LLMStreamChunk(id="test", model="mock", finish_reason="stop")
        ]

        for chunk in chunks:
            yield chunk


async def test_streaming_pipeline():
    """Test the streaming LLM → Sentencizer → TTS pipeline"""
    print("=== Testing Streaming LLM → Sentencizer → TTS Pipeline ===\n")

    # Initialize components
    mock_llm = MockMLXQwenChat()
    sentencizer = LLMSentencizer(newline_as_separator=False, strip_newlines=True)

    print("📝 Components initialized")
    print(f"   - LLM: {type(mock_llm).__name__}")
    print(f"   - Sentencizer: {type(sentencizer).__name__} (punctuation-based)")

    # Simulate the streaming process
    messages = [{"role": "user", "content": "Hello, can you help me?"}]

    print(f"\n🔄 Starting streaming for: {messages[0]['content']}")

    stream_generator = mock_llm._generate_response(
        messages=messages,
        max_tokens=1000,
        temperature=0.7,
        stream=True
    )

    sentence_count = 0
    tts_tasks = []

    print("\n📡 Processing streaming chunks:")

    for chunk in stream_generator:
        if hasattr(chunk, 'content') and chunk.content:
            print(f"   Received chunk: {repr(chunk.content)}")

            # Process through sentencizer
            sentences = sentencizer.process_chunk(chunk.content)

            # Handle complete sentences
            for sentence in sentences:
                if sentence.strip():
                    sentence_count += 1
                    print(f"   ✅ Complete sentence {sentence_count}: '{sentence}'")

                    # Simulate TTS processing
                    print(f"   🔊 → Sending to TTS: '{sentence[:30]}...'")
                    tts_tasks.append(f"TTS_{sentence_count}")

    # Process remaining content
    remaining_sentences = sentencizer.finish()
    for sentence in remaining_sentences:
        if sentence.strip():
            sentence_count += 1
            print(f"   ✅ Final sentence {sentence_count}: '{sentence}'")
            print(f"   🔊 → Sending to TTS: '{sentence[:30]}...'")
            tts_tasks.append(f"TTS_{sentence_count}")

    print(f"\n📊 Pipeline Results:")
    print(f"   - Total sentences processed: {sentence_count}")
    print(f"   - TTS tasks created: {len(tts_tasks)}")
    print(f"   - Sentences: {tts_tasks}")

    print("\n✅ Streaming pipeline test completed successfully!")


async def test_different_sentence_patterns():
    """Test sentencizer with different sentence patterns"""
    print("\n=== Testing Different Sentence Patterns ===\n")

    test_cases = [
        {
            'name': 'Basic punctuation',
            'chunks': ['Hello!', ' How are you?', ' I am fine.'],
            'expected_sentences': 3
        },
        {
            'name': 'Mixed punctuation',
            'chunks': ['Great question!', ' Let me think...', ' Actually, yes.'],
            'expected_sentences': 3
        },
        {
            'name': 'No punctuation until end',
            'chunks': ['This is a long', ' sentence that continues', ' for a while.'],
            'expected_sentences': 1
        }
    ]

    for case in test_cases:
        print(f"🧪 Test case: {case['name']}")
        sentencizer = LLMSentencizer(newline_as_separator=False, strip_newlines=True)

        sentences_found = []
        for chunk in case['chunks']:
            sentences = sentencizer.process_chunk(chunk)
            sentences_found.extend(sentences)
            print(f"   Chunk: {repr(chunk)} → {len(sentences)} sentences")

        # Get remaining
        remaining = sentencizer.finish()
        sentences_found.extend(remaining)

        print(f"   📋 Total sentences: {len(sentences_found)}")
        for i, sentence in enumerate(sentences_found, 1):
            print(f"      {i}: '{sentence}'")

        status = "✅ PASS" if len(sentences_found) == case['expected_sentences'] else "❌ FAIL"
        print(f"   {status} (expected {case['expected_sentences']}, got {len(sentences_found)})\n")


async def main():
    """Run all tests"""
    await test_streaming_pipeline()
    await test_different_sentence_patterns()
    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())