#!/usr/bin/env python3
"""
Test script to demonstrate the updated streaming pipeline using chat_with_context
"""

import sys
import os
import asyncio

# Add src to path
# fmt: off
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from macecho.sentencizer import LLMSentencizer


class MockMLXQwenChatWithContext:
    """Mock MLX Qwen chat with context support for testing"""
    
    def __init__(self, **kwargs):
        self.model_name = "mock-model"
        self.context_enabled = True
    
    def chat_with_context(self, user_message, max_tokens=1000, temperature=0.7, stream=True):
        """Simulate chat_with_context streaming response"""
        if not stream:
            return {"choices": [{"message": {"content": "Complete response here."}}]}
        
        # Simulate streaming chunks in dictionary format (OpenAI-style)
        chunks = [
            {
                "choices": [{"delta": {"role": "assistant"}}],
                "model": self.model_name
            },
            {
                "choices": [{"delta": {"content": "Hello there!"}}],
                "model": self.model_name
            },
            {
                "choices": [{"delta": {"content": " How can I help you today?"}}],
                "model": self.model_name
            },
            {
                "choices": [{"delta": {"content": " I'm here to assist with your questions."}}],
                "model": self.model_name
            },
            {
                "choices": [{"delta": {"content": " What would you like to know?"}}],
                "model": self.model_name
            },
            {
                "choices": [{"finish_reason": "stop"}],
                "model": self.model_name
            }
        ]
        
        for chunk in chunks:
            yield chunk


async def test_chat_with_context_pipeline():
    """Test the updated streaming pipeline using chat_with_context"""
    print("=== Testing chat_with_context Integration ===\n")
    
    # Initialize components
    mock_llm = MockMLXQwenChatWithContext()
    sentencizer = LLMSentencizer(newline_as_separator=False, strip_newlines=True)
    
    print("📝 Components initialized")
    print(f"   - LLM: {type(mock_llm).__name__}")
    print(f"   - Context enabled: {mock_llm.context_enabled}")
    print(f"   - Sentencizer: {type(sentencizer).__name__} (punctuation-based)")
    
    # Test chat_with_context streaming
    user_message = "Hello, can you help me?"
    print(f"\n🔄 Starting chat_with_context for: {user_message}")
    
    stream_generator = mock_llm.chat_with_context(
        user_message=user_message,
        max_tokens=1000,
        temperature=0.7,
        stream=True
    )
    
    sentence_count = 0
    tts_tasks = []
    
    print("\n📡 Processing streaming chunks:")
    
    for chunk_dict in stream_generator:
        print(f"   Received chunk: {chunk_dict}")
        
        # Extract content from the chunk dictionary (same as agent implementation)
        if chunk_dict and chunk_dict.get("choices"):
            delta = chunk_dict["choices"][0].get("delta", {})
            content = delta.get("content")
            
            if content:
                print(f"   Content: {repr(content)}")
                
                # Process through sentencizer
                sentences = sentencizer.process_chunk(content)
                
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
    print(f"   - TTS tasks: {tts_tasks}")
    
    print("\n✅ chat_with_context integration test completed successfully!")


async def main():
    """Run the test"""
    await test_chat_with_context_pipeline()
    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())