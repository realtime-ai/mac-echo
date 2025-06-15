
# 定义一个agent类

import asyncio
from macecho.config import MacEchoConfig
from macecho.device import device
from macecho.device.device import AudioPlayer, AudioRecorder
from macecho.utils.queue import QueueIterator


class Agent:
    def __init__(self, config: MacEchoConfig):
        self.config = config
        self.asr = None
        self.tts = None
        self.vad = None
        self.audio_player = None
        self.audio_recorder = None
        self.audio_recorder = AudioRecorder(
            device=config.audio_recording.device_index,
            channels=config.audio_recording.channels,
            samplerate=config.audio_recording.sample_rate,
            dtype=config.audio_recording.numpy_dtype,
            blocksize=config.audio_recording.chunk_size)

        self.audio_player = AudioPlayer(
            device=config.audio_player.device_index,
            channels=config.audio_player.channels,
            samplerate=config.audio_player.sample_rate,
            dtype=config.audio_player.numpy_dtype,
            blocksize=config.audio_player.chunk_size)

        self.audio_player_queue = asyncio.Queue()
        self.audio_player_queue_iterator = QueueIterator(
            self.audio_player_queue)

        self.running = False

    async def start(self):
        """Start the audio processing agent with proper exception handling"""
        # 打印所有的设备
        device.list_devices()

        self.running = True
        play_task = None

        try:
            # 创建播放任务
            play_task = asyncio.create_task(
                self.audio_player.play(self.audio_player_queue_iterator))

            # 开始音频录制和处理
            async for audio_data in self.audio_recorder.start():
                if not self.running:
                    break

                try:
                    await self.process_audio(audio_data)
                except Exception as e:
                    print(f"Error processing audio data: {e}")
                    # 继续处理下一个音频块，不中断整个流程
                    continue

        except asyncio.CancelledError:
            print("Audio processing was cancelled")
            raise
        except Exception as e:
            print(f"Critical error in audio processing: {e}")
            raise
        finally:
            # 确保清理资源
            await self.cleanup(play_task)

    async def process_audio(self, audio_data: bytes):
        """Process incoming audio data"""
        pass

    async def stop(self):
        """Stop the audio processing agent gracefully"""
        print("Stopping audio processing agent...")
        self.running = False

        try:
            # 停止音频录制
            if self.audio_recorder:
                await self.audio_recorder.stop()

            # 停止音频播放队列
            await self.audio_player_queue.put(None)  # 发送停止信号

        except Exception as e:
            print(f"Error during agent stop: {e}")

    async def cleanup(self, play_task=None):
        """Clean up resources"""
        print("Cleaning up audio agent resources...")

        try:
            # 设置停止标志
            self.running = False

            # 取消播放任务
            if play_task and not play_task.done():
                play_task.cancel()
                try:
                    await play_task
                except asyncio.CancelledError:
                    print("Play task cancelled successfully")
                except Exception as e:
                    print(f"Error cancelling play task: {e}")

            # 清理音频录制器
            if self.audio_recorder:
                try:
                    await self.audio_recorder.stop()
                except Exception as e:
                    print(f"Error stopping audio recorder: {e}")

            # 清理音频播放器
            if self.audio_player:
                try:
                    await self.audio_player.stop()
                except Exception as e:
                    print(f"Error stopping audio player: {e}")

            # 清空播放队列
            if hasattr(self, 'audio_player_queue'):
                try:
                    # 清空队列中的所有项目
                    while not self.audio_player_queue.empty():
                        try:
                            self.audio_player_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                except Exception as e:
                    print(f"Error clearing audio player queue: {e}")

        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            print("Cleanup completed")
