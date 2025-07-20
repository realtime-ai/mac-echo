#!/usr/bin/env python3
"""
Example usage of the Agent class with proper exception handling.
"""

import os
import asyncio
import signal
import sys
from pathlib import Path



# fmt: off
# Add src to path for imports BEFORE importing macecho modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# fmt: on

## disable auto format
from macecho.agent import Agent
from macecho.config import MacEchoConfig

import dotenv
dotenv.load_dotenv()


async def main():
    """Main example function demonstrating different usage patterns"""

    # 创建配置
    config = MacEchoConfig(
        audio_recording={
            "sample_rate": 16000,
            "format": "int16",
            "channels": 1
        },
        audio_player={
            "sample_rate": 24000,
            "format": "int16",
            "channels": 1
        },
        tts={
            "model_name": "FunAudioLLM/CosyVoice2-0.5B",
            "voice_id": "david",
            "speed": 1.0,
            "api_key": os.environ.get('SILICONFLOW_API_KEY'),
            "base_url": os.environ.get('SILICONFLOW_BASE_URL'),
        },
        debug=True
    )

    print("=== Agent Usage Example ===")
    print()

    # 方法 1: 使用 try-except-finally 手动管理
    print("1. Manual resource management:")
    agent = Agent(config)

    try:
        print("Starting agent...")
        await asyncio.wait_for(agent.start(), timeout=20.0)
    except asyncio.TimeoutError:
        print("Agent stopped after timeout")
    except KeyboardInterrupt:
        print("Agent interrupted by user")
    except Exception as e:
        print(f"Agent failed with error: {e}")
    finally:
        print("Manually cleaning up agent...")
        await agent.cleanup()


async def signal_handler():
    """Handle shutdown signals gracefully"""
    print("\nReceived shutdown signal, cleaning up...")
    # 在实际应用中，这里可以设置一个全局标志来停止 agent
    return


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    if sys.platform != "win32":
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig, lambda: asyncio.create_task(signal_handler()))


async def long_running_example():
    """Example of a long-running agent with signal handling"""
    print("=== Long-running Agent Example ===")
    print("Press Ctrl+C to stop...")

    config = MacEchoConfig()

    try:

        agent = Agent(config)
        # 设置信号处理
        setup_signal_handlers()

        # 运行 agent
        await agent.start()

    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"Agent failed: {e}")
    finally:
        print("Agent shutdown complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent usage examples")
    parser.add_argument("--mode", choices=["demo", "run"], default="demo",
                        help="Mode: 'demo' for short examples, 'run' for long-running")

    args = parser.parse_args()

    if args.mode == "demo":
        asyncio.run(main())
    else:
        asyncio.run(long_running_example())
