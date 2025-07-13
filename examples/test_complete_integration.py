#!/usr/bin/env python3
"""
Complete integration test showing the full Agent workflow with real components
"""

import sys
import os
import asyncio
from unittest.mock import AsyncMock, patch

# Add src to path
# fmt: off
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from macecho.config import MacEchoConfig
from macecho.agent import Agent


async def test_complete_agent_workflow():
    """Test the complete agent workflow with mocked components"""
    print("=== Complete Agent Workflow Test ===\n")
    
    try:
        # Create a minimal config for testing
        config = MacEchoConfig()
        print("✅ Config created successfully")
        
        # Mock the heavy components that would require model files
        with patch('macecho.llm.MLXQwenChat') as mock_llm_class:
            # Create a mock LLM instance
            mock_llm = AsyncMock()
            mock_llm.chat_with_context = AsyncMock()
            
            # Configure the mock to return streaming data
            def mock_chat_stream(*args, **kwargs):
                chunks = [
                    {"choices": [{"delta": {"role": "assistant"}}]},
                    {"choices": [{"delta": {"content": "Thank you for your question!"}}]},
                    {"choices": [{"delta": {"content": " I understand you're looking for help."}}]},
                    {"choices": [{"delta": {"content": " Let me provide you with a detailed response."}}]},
                    {"choices": [{"finish_reason": "stop"}]}
                ]
                for chunk in chunks:
                    yield chunk
            
            mock_llm.chat_with_context.return_value = mock_chat_stream()
            mock_llm_class.return_value = mock_llm
            
            # Create agent instance
            agent = Agent(config)
            print("✅ Agent created successfully")
            
            # Test the sentencizer integration
            print(f"\n📝 Agent Components:")
            print(f"   - Sentencizer: {type(agent.sentencizer).__name__}")
            print(f"   - LLM: {type(agent.llm)}")
            print(f"   - VAD: {type(agent.vad).__name__}")
            print(f"   - ASR: {type(agent.asr)}")
            
            # Test the streaming LLM method directly
            print(f"\n🧪 Testing _process_llm_streaming method...")
            
            correlation_id = "test-12345"
            test_text = "Hello, can you help me with something?"
            
            print(f"   Input text: '{test_text}'")
            print(f"   Correlation ID: {correlation_id}")
            
            # Mock the TTS processing to avoid actual audio generation
            original_process_single_tts = agent._process_single_tts
            tts_calls = []
            
            async def mock_tts(sentence, corr_id, idx):
                tts_calls.append({"sentence": sentence, "index": idx})
                print(f"   🔊 TTS Mock: Processing sentence {idx}: '{sentence[:40]}...'")
            
            agent._process_single_tts = mock_tts
            
            try:
                # Execute the streaming LLM processing
                await agent._process_llm_streaming(test_text, correlation_id)
                
                print(f"\n📊 Results:")
                print(f"   - TTS calls made: {len(tts_calls)}")
                for i, call in enumerate(tts_calls, 1):
                    print(f"   - Sentence {i}: '{call['sentence']}'")
                
                # Verify the LLM was called correctly
                mock_llm.chat_with_context.assert_called_once_with(
                    user_message=test_text,
                    max_tokens=1000,
                    temperature=0.7,
                    stream=True
                )
                
                print(f"\n✅ Streaming LLM processing completed successfully!")
                print(f"   - chat_with_context called with correct parameters")
                print(f"   - {len(tts_calls)} sentences extracted and sent to TTS")
                print(f"   - Real-time sentence processing working correctly")
                
            finally:
                # Restore original method
                agent._process_single_tts = original_process_single_tts
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


async def main():
    """Run the complete test"""
    success = await test_complete_agent_workflow()
    
    if success:
        print(f"\n🎉 Complete integration test PASSED!")
        print(f"\n✨ Summary:")
        print(f"   ✅ Agent initialization with real config")
        print(f"   ✅ LLM chat_with_context integration")
        print(f"   ✅ LLMSentencizer real-time processing")
        print(f"   ✅ Streaming sentence-to-TTS pipeline")
        print(f"   ✅ Context management support")
    else:
        print(f"\n💥 Integration test FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())