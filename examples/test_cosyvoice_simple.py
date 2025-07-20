#!/usr/bin/env python3
"""
Simple test script for CosyVoice TTS
"""


import asyncio
import sys
import os
import time
import wave
from pathlib import Path
import dotenv

dotenv.load_dotenv()

# Add src to path for imports
# fmt: off
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from macecho.tts import CosyVoiceTTS

async def simple_cosyvoice_test():
    """Simple CosyVoice TTS stream synthesis test"""
    print("🎤 Simple CosyVoice TTS Test")
    print("=" * 40)

    try:
        # Initialize TTS
        print("Initializing CosyVoice TTS...")
        tts = CosyVoiceTTS(
            voice="david",
            sample_rate=24000,
            api_key=os.environ.get('SILICONFLOW_API_KEY'),
            base_url=os.environ.get('SILICONFLOW_BASE_URL'),
        )
        print(f"✅ TTS initialized with voice: {tts.voice}")

        # Test text
        test_text = "Hello, this is a test of CosyVoice text-to-speech synthesis."
        print(f"\n📝 Test text: {test_text}")

        # Non-streaming synthesis
        print("\n🔊 Non-streaming synthesis...")
        start_time = time.time()
        
        audio_data_non_stream = await asyncio.to_thread(tts.synthesize, test_text)
        
        non_stream_time = time.time() - start_time
        
        if audio_data_non_stream:
            print(f"✅ Non-streaming synthesis successful!")
            print(f"   Audio size: {len(audio_data_non_stream)} bytes")
            print(f"   Time: {non_stream_time:.2f} seconds")
            
            # Save non-streaming output
            output_file_non_stream = Path("test_output") / "cosyvoice_non_stream_test.wav"
            output_file_non_stream.parent.mkdir(exist_ok=True)
            
            with wave.open(str(output_file_non_stream), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(24000)
                wav_file.writeframes(audio_data_non_stream)
            
            duration_non_stream = len(audio_data_non_stream) / (24000 * 2)  # 16-bit = 2 bytes
            print(f"   Duration: {duration_non_stream:.2f} seconds")
            print(f"   Saved to: {output_file_non_stream}")
        else:
            print("❌ Non-streaming synthesis failed - no audio data returned")

        # Stream Synthesize
        print("\n🔄 Streaming synthesis...")
        start_time = time.time()
        
        # Collect audio chunks from streaming
        audio_chunks = []
        stream_generator = await asyncio.to_thread(tts.stream_synthesize, test_text)
        
        chunk_count = 0
        for audio_chunk in stream_generator:
            if audio_chunk:
                audio_chunks.append(audio_chunk)
                chunk_count += 1
                print(f"   Received chunk {chunk_count} ({len(audio_chunk)} bytes)")
        
        # Combine all chunks
        audio_data = b''.join(audio_chunks) if audio_chunks else None
        
        synthesis_time = time.time() - start_time

        if audio_data:
            print(f"✅ Stream synthesis successful!")
            print(f"   Total chunks: {chunk_count}")
            print(f"   Audio size: {len(audio_data)} bytes")
            print(f"   Time: {synthesis_time:.2f} seconds")

            # Save to file
            output_file = Path("test_output") / "cosyvoice_simple_test.wav"
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

            print(f"\n📊 Comparison:")
            print(f"   Non-streaming: {non_stream_time:.2f}s")
            print(f"   Streaming: {synthesis_time:.2f}s (first chunk latency)")
            print(f"   Speed improvement: {non_stream_time/synthesis_time:.1f}x")
            
            print(f"\n🎉 Test completed successfully!")

        else:
            print("❌ Synthesis failed - no audio data returned")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":    
    
    asyncio.run(simple_cosyvoice_test())
   