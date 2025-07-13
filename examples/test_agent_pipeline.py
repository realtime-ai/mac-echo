#!/usr/bin/env python3
"""
Test the complete Agent processing pipeline with interruption handling
"""

import sys
import asyncio
from pathlib import Path
import numpy as np
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from macecho.agent import Agent
from macecho.config import MacEchoConfig
from macecho.vad.interface import VadState


async def simulate_speech_processing():
    """Simulate the complete speech processing pipeline with interruption"""
    
    print("🎬 Testing Complete Agent Pipeline")
    print("=" * 60)
    
    # Create config
    config = MacEchoConfig(debug=True)
    agent = Agent(config)
    
    print(f"Agent initialized with ASR: {agent.asr is not None and agent.asr.is_ready()}")
    
    # Create synthetic speech segments
    sample_rate = config.audio_recording.sample_rate
    duration = 2.0  # 2 seconds
    samples = int(sample_rate * duration)
    
    # First speech segment
    print("\n📢 Processing first speech segment...")
    t1 = np.linspace(0, duration, samples)
    speech1 = (0.3 * np.sin(2 * np.pi * 440 * t1)).astype(np.float32)
    
    # Start processing first segment
    task1 = asyncio.create_task(agent._process_speech_segment(speech1))
    
    # Let it run for a bit
    await asyncio.sleep(1.5)
    
    # Second speech segment (should interrupt the first)
    print("\n📢 Processing second speech segment (should interrupt first)...")
    t2 = np.linspace(0, duration, samples)
    speech2 = (0.3 * np.sin(2 * np.pi * 880 * t2)).astype(np.float32)  # Higher frequency
    
    # Start processing second segment (this should cancel the first)
    task2 = asyncio.create_task(agent._process_speech_segment(speech2))
    
    # Wait for both tasks to complete (first should be cancelled)
    try:
        await asyncio.gather(task1, task2, return_exceptions=True)
    except Exception as e:
        print(f"Task exception: {e}")
    
    # Let TTS finish
    await asyncio.sleep(5)
    
    print("\n✅ Pipeline test completed!")
    
    # Clean up
    await agent.stop()


async def test_message_queue():
    """Test the message queue functionality"""
    
    print("\n🔍 Testing Message Queue")
    print("=" * 40)
    
    config = MacEchoConfig(debug=True)
    agent = Agent(config)
    
    # Monitor message queue
    async def message_monitor():
        while True:
            try:
                message = await asyncio.wait_for(agent.message_queue.get(), timeout=0.1)
                print(f"📬 Message: {message.message_type.value} - {message}")
            except asyncio.TimeoutError:
                break
            except Exception as e:
                print(f"Message monitor error: {e}")
                break
    
    # Create a synthetic speech segment  
    sample_rate = config.audio_recording.sample_rate
    duration = 1.0
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples)
    speech = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    # Start message monitoring
    monitor_task = asyncio.create_task(message_monitor())
    
    # Process speech segment
    process_task = asyncio.create_task(agent._process_speech_segment(speech))
    
    # Wait for processing to complete
    await process_task
    
    # Give message monitor a chance to collect messages
    await asyncio.sleep(0.5)
    
    # Cancel monitor
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass
    
    print("✅ Message queue test completed!")
    
    # Clean up
    await agent.stop()


async def test_interruption_timing():
    """Test the timing of interruption handling"""
    
    print("\n⚡ Testing Interruption Timing")
    print("=" * 45)
    
    config = MacEchoConfig(debug=True)
    agent = Agent(config)
    
    # Create speech segments
    sample_rate = config.audio_recording.sample_rate
    duration = 1.0
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples)
    speech = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    print("Starting first processing...")
    start_time = time.time()
    
    # Start first processing
    task1 = asyncio.create_task(agent._process_speech_segment(speech))
    
    # Wait until LLM should be running
    await asyncio.sleep(1.2)
    
    interrupt_time = time.time()
    print(f"Interrupting after {interrupt_time - start_time:.2f}s...")
    
    # Start second processing (should interrupt first)
    task2 = asyncio.create_task(agent._process_speech_segment(speech))
    
    # Wait for completion
    results = await asyncio.gather(task1, task2, return_exceptions=True)
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f}s")
    
    # Analyze results
    for i, result in enumerate(results):
        if isinstance(result, asyncio.CancelledError):
            print(f"Task {i+1}: Cancelled successfully")
        elif isinstance(result, Exception):
            print(f"Task {i+1}: Exception - {result}")
        else:
            print(f"Task {i+1}: Completed successfully")
    
    print("✅ Interruption timing test completed!")
    
    # Clean up
    await agent.stop()


async def main():
    """Run all tests"""
    try:
        await simulate_speech_processing()
        await asyncio.sleep(1)
        
        await test_message_queue()
        await asyncio.sleep(1)
        
        await test_interruption_timing()
        
        print("\n🎉 All tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())