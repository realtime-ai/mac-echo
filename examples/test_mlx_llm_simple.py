#!/usr/bin/env python3
"""
Simple test script for MLX LLM
"""

import asyncio
import sys
import os
import time
from typing import Dict, Any

# Add src to path for imports
# fmt: off
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from macecho.llm.factory import LLMFactory
from macecho.llm.base import LLMResponse

async def test_mlx_llm_basic():
    """Test basic MLX LLM functionality"""
    print("🤖 MLX LLM Basic Test")
    print("=" * 40)
    
    try:
        # Create LLM config
        llm_config = {
            "provider": "mlx",
            "model_name": "mlx-community/Qwen3-4B-8bit",
            "warmup_enabled": True,
            "context_enabled": True,
            "max_context_rounds": 5,
            "system_prompt": "You are a helpful AI assistant."
        }
        
        # Initialize LLM
        print("Initializing MLX LLM...")
        llm = LLMFactory.create_llm(llm_config)
        print(f"✅ LLM initialized: {type(llm).__name__}")
        print(f"   Model: {llm.model_name}")
        
        # Get model info
        model_info = llm.get_model_info()
        print(f"   Info: {model_info}")
        
        # Test prompt
        test_prompt = "Hello! Can you explain what an LLM is in one sentence?"
        print(f"\n📝 Test prompt: {test_prompt}")
        
        # Generate response
        print("🔄 Generating response...")
        start_time = time.time()
        
        # Use chat_completions method
        messages = [{"role": "user", "content": test_prompt}]
        response_dict = await asyncio.to_thread(
            llm.chat_completions, 
            messages,
            max_tokens=100
        )
        
        generation_time = time.time() - start_time
        
        if response_dict:
            print(f"✅ Generation successful!")
            print(f"   Time: {generation_time:.2f} seconds")
            response_content = response_dict['choices'][0]['message']['content']
            print(f"   Response: {response_content}")
            if 'usage' in response_dict:
                print(f"   Token count: {response_dict['usage'].get('total_tokens', 'N/A')}")
            print(f"   Finish reason: {response_dict['choices'][0]['finish_reason']}")
        else:
            print("❌ Generation failed - no response returned")
            
    except Exception as e:
        print(f"❌ Error in basic test: {e}")
        import traceback
        traceback.print_exc()


async def test_mlx_llm_streaming():
    """Test streaming MLX LLM functionality"""
    print("\n\n🌊 MLX LLM Streaming Test")
    print("=" * 40)
    
    try:
        # Create LLM config
        llm_config = {
            "provider": "mlx",
            "model_name": "mlx-community/Qwen3-4B-8bit",
            "warmup_enabled": False,  # Skip warmup for faster test
            "context_enabled": False,
            "system_prompt": "You are a helpful AI assistant."
        }
        
        # Initialize LLM
        print("Initializing MLX LLM for streaming...")
        llm = LLMFactory.create_llm(llm_config)
        print(f"✅ LLM initialized: {type(llm).__name__}")
        
        # Test prompt
        test_prompt = "Write a haiku about artificial intelligence."
        print(f"\n📝 Test prompt: {test_prompt}")
        
        # Stream response
        print("🔄 Streaming response:")
        start_time = time.time()
        full_response = ""
        token_count = 0
        
        # Use stream chat_completions method
        messages = [{"role": "user", "content": test_prompt}]
        stream_gen = await asyncio.to_thread(
            llm.chat_completions,
            messages,
            max_tokens=100,
            stream=True
        )
        
        for chunk in stream_gen:
            if chunk and chunk.get('choices'):
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    print(delta['content'], end='', flush=True)
                    full_response += delta['content']
                if chunk.get('usage'):
                    token_count = chunk['usage'].get('total_tokens', 0)
        
        generation_time = time.time() - start_time
        
        print(f"\n\n✅ Streaming completed!")
        print(f"   Time: {generation_time:.2f} seconds")
        print(f"   Total tokens: {token_count}")
        print(f"   Characters: {len(full_response)}")
        
    except Exception as e:
        print(f"\n❌ Error in streaming test: {e}")
        import traceback
        traceback.print_exc()


async def test_mlx_llm_context():
    """Test MLX LLM with context management"""
    print("\n\n💬 MLX LLM Context Test")
    print("=" * 40)
    
    try:
        # Create LLM config with context enabled
        llm_config = {
            "provider": "mlx",
            "model_name": "mlx-community/Qwen3-4B-8bit",
            "warmup_enabled": False,
            "context_enabled": True,
            "max_context_rounds": 3,
            "context_window_size": 2000,
            "system_prompt": "You are a helpful AI assistant with a good memory."
        }
        
        # Initialize LLM
        print("Initializing MLX LLM with context...")
        llm = LLMFactory.create_llm(llm_config)
        print(f"✅ LLM initialized with context management")
        
        # First message
        prompt1 = "My name is Alice and I love programming in Python."
        print(f"\n📝 Message 1: {prompt1}")
        response1_dict = await asyncio.to_thread(
            llm.chat_with_context,
            prompt1,
            max_tokens=500
        )
        response1_content = response1_dict['choices'][0]['message']['content']
        print(f"🤖 Response 1: {response1_content}")
        
        # Second message (should remember context)
        prompt2 = "What's my name and what do I love?"
        print(f"\n📝 Message 2: {prompt2}")
        response2_dict = await asyncio.to_thread(
            llm.chat_with_context,
            prompt2,
            max_tokens=500
        )
        response2_content = response2_dict['choices'][0]['message']['content']
        print(f"🤖 Response 2: {response2_content}")
        
        # Clear context
        llm.clear_conversation_history()
        print("\n🧹 Context cleared")
        
        # Third message (should not remember)
        prompt3 = "Do you remember my name?"
        print(f"\n📝 Message 3: {prompt3}")
        response3_dict = await asyncio.to_thread(
            llm.chat_with_context,
            prompt3,
            max_tokens=500
        )
        response3_content = response3_dict['choices'][0]['message']['content']
        print(f"🤖 Response 3: {response3_content}")
        
        print("\n✅ Context test completed!")
        
    except Exception as e:
        print(f"\n❌ Error in context test: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all MLX LLM tests"""
    print("🚀 Starting MLX LLM Tests\n")
    
    # Run basic test
    await test_mlx_llm_basic()
    
    # Run streaming test
    await test_mlx_llm_streaming()
    
    # Run context test
    await test_mlx_llm_context()
    
    print("\n\n🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())