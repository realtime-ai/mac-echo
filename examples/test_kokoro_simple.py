#!/usr/bin/env python3
"""
Simple test script for Kokoro TTS
"""


import asyncio
import sys
import os
import time
import wave
from pathlib import Path

# Add src to path for imports
# fmt: off
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from macecho.tts import KokoroTTS

async def simple_kokoro_test():
    """Simple Kokoro TTS test"""
    print("🎤 Simple Kokoro TTS Test")
    print("=" * 40)

    try:
        # Initialize TTS
        print("Initializing Kokoro TTS...")
        tts = KokoroTTS(
            voice="zf_001",  # Default voice for Kokoro
            sample_rate=24000
        )
        print(f"✅ TTS initialized with voice: {tts.voice}")

        # Test text (Chinese)
        test_text = "你好，这是一个文本转语音合成的测试。"
        print(f"\n📝 Test text: {test_text}")

        # Synthesize
        print("🔄 Synthesizing audio...")
        start_time = time.time()

        audio_data = await asyncio.to_thread(tts.synthesize, test_text)

        synthesis_time = time.time() - start_time

        if audio_data:
            print(f"✅ Synthesis successful!")
            print(f"   Audio size: {len(audio_data)} bytes")
            print(f"   Time: {synthesis_time:.2f} seconds")

            # Save to file
            output_file = Path("test_output") / "kokoro_simple_test.wav"
            output_file.parent.mkdir(exist_ok=True)

            with wave.open(str(output_file), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(24000)
                wav_file.writeframes(audio_data)

            # Calculate duration
            duration = len(audio_data) / (24000 * 2)  # 16-bit = 2 bytes
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Saved to: {output_file}")

            print(f"\n🎉 Test completed successfully!")

        else:
            print("❌ Synthesis failed - no audio data returned")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(simple_kokoro_test())