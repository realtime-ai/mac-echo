# 定义一个agent类

import asyncio
import time
import os
import wave
from datetime import datetime
from pathlib import Path
from macecho.config import MacEchoConfig
from macecho.device import device
from macecho.device.device import AudioPlayer, AudioRecorder
from macecho.utils.queue import QueueIterator
import numpy as np
import collections

from macecho.vad.vad import VadProcessor
from macecho.vad.interface import VadState
from macecho.asr.sencevoice.model import SenceVoiceASR


class Agent:
    def __init__(self, config: MacEchoConfig):
        self.config = config
        self.asr = None
        self.tts = None
        self.vad = None
        self.audio_player = None
        self.audio_recorder = None

        # Initialize VAD processor with mapped config
        from macecho.vad.vad import VadConfig
        vad_config = VadConfig(
            threshold=config.vad.threshold,
            sampling_rate=config.audio_recording.sample_rate,
            padding_duration=config.vad.padding_duration,
            min_speech_duration=config.vad.min_speech_duration,
            silence_duration=config.vad.silence_duration,
            per_frame_duration=config.vad.per_frame_duration,
            model_path=config.vad.model_path
        )
        self.vad = VadProcessor(vad_config)

        # Initialize ASR processor
        try:
            self.asr = SenceVoiceASR(
                language=config.asr.language,
                sample_rate=config.audio_recording.sample_rate,
                model_dir=config.asr.model_name,
                device=config.asr.device,
                auto_initialize=True
            )
            print(f"ASR initialized: {type(self.asr)}")
        except Exception as e:
            print(f"Warning: Failed to initialize ASR: {e}")
            self.asr = None

        # Create debug audio output directory if in debug mode
        if config.debug:
            self.debug_audio_dir = Path("debug_audio")
            self.debug_audio_dir.mkdir(exist_ok=True)
            print(
                f"Debug mode: Audio files will be saved to {self.debug_audio_dir}")
        else:
            self.debug_audio_dir = None
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

        # 音频缓冲和分帧相关
        self.frame_duration_ms = 32.0  # 32ms帧长
        self.frame_size = int(self.frame_duration_ms *
                              config.audio_recording.sample_rate / 1000)  # 每帧样本数
        self.audio_buffer = collections.deque(
            maxlen=self.frame_size * 10)  # 缓冲区大小设为10帧
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
        """
        处理输入的音频数据：
        1. 将字节数据转换为numpy数组
        2. 将数据添加到缓冲区
        3. 从缓冲区提取完整的帧进行处理
        4. 对每一帧进行VAD等后续处理

        Args:
            audio_data: 输入的音频数据（字节格式）
        """
        try:
            # 1. 将字节数据转换为numpy数组
            audio_array = np.frombuffer(
                audio_data, dtype=self.config.audio_recording.numpy_dtype)

            # 2. 将数据添加到缓冲区
            self.audio_buffer.extend(audio_array)

            # 3. 从缓冲区提取完整的帧进行处理
            while len(self.audio_buffer) >= self.frame_size:
                # 提取一帧数据
                frame = np.array([self.audio_buffer.popleft()
                                 for _ in range(self.frame_size)])

                # 4. 对帧进行后续处理
                await self._process_frame(frame)

        except Exception as e:
            print(f"Error in process_audio: {e}")
            # 清空缓冲区，避免数据堆积
            self.audio_buffer.clear()

    async def _process_frame(self, frame: np.ndarray):
        """
        处理单个音频帧

        Args:
            frame: 音频帧数据（numpy数组）
        """
        try:
            # Use the initialized VAD processor
            if self.vad:
                speech_segment, state = self.vad.process_audio(frame)

                if state == VadState.SPEECH_START:
                    print(f'VAD: Speech start detected at {time.time():.3f}')

                if state == VadState.SPEECH_END and speech_segment is not None:
                    print(
                        f'VAD: Speech end detected at {time.time():.3f}, segment duration: {len(speech_segment)/self.config.audio_recording.sample_rate:.2f}s')

                    # Save audio in debug mode
                    if self.config.debug and self.debug_audio_dir:
                        await self._save_debug_audio(speech_segment)

                    # Here you can process the complete speech segment
                    # For example: send to ASR, then to LLM, then to TTS
                    await self._process_speech_segment(speech_segment)
            else:
                print('Warning: VAD processor not initialized')

        except Exception as e:
            print(f"Error in _process_frame: {e}")

    async def _process_speech_segment(self, speech_segment: np.ndarray):
        """
        Process a complete speech segment through the pipeline

        Args:
            speech_segment: Complete speech segment detected by VAD
        """
        try:
            print(
                f"Processing speech segment of {len(speech_segment)} samples ({len(speech_segment)/self.config.audio_recording.sample_rate:.2f}s)")

            # ASR processing
            transcribed_text = ""
            if self.asr and self.asr.is_ready():
                try:
                    print("ASR: Starting async transcription...")
                    # Convert numpy array to bytes for ASR processing
                    if speech_segment.dtype == np.float32:
                        # Convert float32 to int16 for ASR
                        audio_int16 = (speech_segment * 32767).astype(np.int16)
                        audio_bytes = audio_int16.tobytes()
                    else:
                        audio_bytes = speech_segment.tobytes()

                    # Use async transcribe to avoid blocking the main thread
                    transcribed_text = await self.asr.transcribe(audio_bytes)
                    print(f"ASR: {transcribed_text}")
                except Exception as e:
                    print(f"ASR Error: {e}")
                    transcribed_text = ""
            else:
                print("ASR: Not available or not ready")

            # TODO: Add LLM processing
            # if self.llm and transcribed_text:
            #     response = await self.llm.generate(transcribed_text)
            #     print(f"LLM: {response}")

            # TODO: Add TTS processing
            # if self.tts and response:
            #     audio_response = await self.tts.synthesize(response)
            #     await self.audio_player_queue.put(audio_response)

        except Exception as e:
            print(f"Error processing speech segment: {e}")

    async def _save_debug_audio(self, speech_segment: np.ndarray):
        """
        Save speech segment to WAV file in debug mode

        Args:
            speech_segment: Audio data to save
        """
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
                :-3]  # Include milliseconds
            filename = f"speech_{timestamp}.wav"
            filepath = self.debug_audio_dir / filename

            # Convert float32 audio to int16 for WAV format
            if speech_segment.dtype == np.float32:
                # Scale from [-1, 1] to [-32768, 32767]
                audio_int16 = (speech_segment * 32767).astype(np.int16)
            elif speech_segment.dtype == np.int16:
                audio_int16 = speech_segment
            else:
                # Convert other types to float32 first, then to int16
                audio_float = speech_segment.astype(np.float32)
                if audio_float.max() > 1.0 or audio_float.min() < -1.0:
                    # Normalize if values are outside [-1, 1] range
                    audio_float = audio_float / np.max(np.abs(audio_float))
                audio_int16 = (audio_float * 32767).astype(np.int16)

            # Save as WAV file
            with wave.open(str(filepath), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.config.audio_recording.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

            duration = len(speech_segment) / \
                self.config.audio_recording.sample_rate
            print(
                f"Debug: Saved speech segment to {filepath} (duration: {duration:.2f}s, {len(speech_segment)} samples)")

        except Exception as e:
            print(f"Error saving debug audio: {e}")

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

            # 清空音频缓冲区
            self.audio_buffer.clear()

            # Reset VAD if initialized
            if self.vad:
                self.vad.reset()

            # Release ASR resources if initialized
            if self.asr:
                self.asr.release()

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
