import collections
from enum import Enum
import numpy as np
import onnxruntime as ort
import os
import urllib.request
from typing import Optional, Tuple, Protocol, runtime_checkable
import logging
import wave

from pydantic import BaseModel, Field, field_validator
from .interface import VADInterface, VadState


logger = logging.getLogger(__name__)


class VadConfig(BaseModel):
    """VAD配置类"""
    threshold: float = Field(default=0.7, ge=0.0, le=1.0,
                             description="VAD 检测阈值，范围 0-1")
    sampling_rate: int = Field(default=16000, description="音频采样率")
    padding_duration: float = Field(
        default=0.2, ge=0.0, description="语音开始前的填充时长，单位秒")
    min_speech_duration: float = Field(
        default=0.1, ge=0.0, description="最小语音段长度，单位秒")
    silence_duration: float = Field(
        default=0.5, ge=0.0, description="静音判断时长，单位秒")
    per_frame_duration: float = Field(
        default=0.032, gt=0.0, description="每帧音频的时长，单位秒 (16kHz: 0.032s=512样本, 8kHz: 0.032s=256样本)")
    model_path: str = Field(default="silero_vad.onnx", description="VAD模型路径")

    @field_validator('sampling_rate')
    @classmethod
    def validate_sampling_rate(cls, v):
        # Silero VAD 支持的采样率
        if v not in [8000, 16000]:
            raise ValueError(
                'Silero VAD sampling rate must be 8000 or 16000')
        return v


class VadProcessor(VADInterface):
    """改进的VAD处理器，修复了原有的逻辑问题"""

    def __init__(self, config: Optional[VadConfig] = None, **kwargs):
        """
        初始化 VAD 处理器

        Args:
            config: VadConfig配置对象，如果为None则使用kwargs创建
            **kwargs: 直接传递的配置参数（向后兼容）
        """
        # 处理配置
        if config is None:
            # 向后兼容，使用传统参数
            config = VadConfig(**kwargs)
        self.config = config

        # 下载并加载 ONNX 模型
        self._load_model()

        # 初始化状态
        self._reset_internal_state()

        # 计算相关参数
        self._calculate_parameters()

        logger.info(f"VAD 处理器初始化完成:")
        logger.info(f"- 采样率: {self.config.sampling_rate}Hz")
        logger.info(f"- 检测阈值: {self.config.threshold}")
        logger.info(
            f"- 最小语音段长度: {self.config.min_speech_duration}秒 ({self.min_speech_samples}样本)")
        logger.info(
            f"- 静音判断时长: {self.config.silence_duration}秒 ({self.silence_samples}样本)")
        logger.info(
            f"- 语音前填充时长: {self.config.padding_duration}秒 ({self.padding_frames}帧)")

    def _load_model(self):
        """加载VAD模型"""
        try:
            if not os.path.exists(self.config.model_path):
                logger.info(f"下载 Silero VAD 模型到: {self.config.model_path}")
                urllib.request.urlretrieve(
                    "https://github.com/snakers4/silero-vad/raw/refs/heads/master/src/silero_vad/data/silero_vad.onnx",
                    self.config.model_path
                )

            # 初始化 ONNX Runtime
            self.session = ort.InferenceSession(self.config.model_path)

            # 检查模型输入输出规格
            self.input_names = [
                input.name for input in self.session.get_inputs()]
            self.output_names = [
                output.name for output in self.session.get_outputs()]

            logger.info("VAD 模型加载成功")
            logger.info(f"模型输入: {self.input_names}")
            logger.info(f"模型输出: {self.output_names}")

            # 检测模型类型
            if 'state' in self.input_names:
                self.model_type = 'new'  # 新版本格式
                logger.info("检测到新版本Silero VAD模型格式")
            elif 'h' in self.input_names and 'c' in self.input_names:
                self.model_type = 'old'  # 旧版本格式
                logger.info("检测到旧版本Silero VAD模型格式")
            else:
                self.model_type = 'simple'  # 简单格式
                logger.info("检测到简单Silero VAD模型格式")

        except Exception as e:
            logger.error(f"VAD 模型加载失败: {e}")
            raise

    def _reset_internal_state(self):
        """重置内部状态"""
        # ONNX模型的隐藏状态 - 参考silero.py的实现
        self._state = np.zeros((2, 1, 128), dtype='float32')
        self._context = np.zeros((1, 0), dtype='float32')
        self._last_sr = 0
        self._last_batch_size = 0

        # VAD状态
        self.current_state = VadState.IDLE
        self.is_recording = False

        # 音频缓冲
        self.audio_buffer = []
        self.silence_counter = 0

        # 填充缓冲区 - 修复：正确计算maxlen
        self.padding_buffer = None  # 将在_calculate_parameters中初始化

    def _reset_states(self, batch_size=1):
        """重置ONNX模型状态 - 参考silero.py"""
        self._state = np.zeros((2, batch_size, 128), dtype="float32")
        self._context = np.zeros((batch_size, 0), dtype="float32")
        self._last_sr = 0
        self._last_batch_size = 0

    def _calculate_parameters(self):
        """计算各种参数"""
        # 每帧样本数
        self.frame_samples = int(
            self.config.per_frame_duration * self.config.sampling_rate)

        # 最小语音样本数
        self.min_speech_samples = int(
            self.config.min_speech_duration * self.config.sampling_rate)

        # 静音样本数
        self.silence_samples = int(
            self.config.silence_duration * self.config.sampling_rate)

        # 填充帧数 - 修复：基于帧数而不是样本数
        self.padding_frames = int(
            self.config.padding_duration / self.config.per_frame_duration)

        # 初始化填充缓冲区 - 修复：使用正确的maxlen
        if self.padding_frames > 0:
            self.padding_buffer = collections.deque(maxlen=self.padding_frames)
        else:
            self.padding_buffer = collections.deque(maxlen=1)  # 至少保持1帧

    def _validate_input(self, audio_chunk: np.ndarray) -> np.ndarray:
        """验证和标准化输入音频格式"""
        if audio_chunk is None:
            raise ValueError("音频数据不能为None")

        if not isinstance(audio_chunk, np.ndarray):
            audio_chunk = np.array(audio_chunk)

        # 转换为float32
        if audio_chunk.dtype != np.float32:
            if audio_chunk.dtype in [np.int16, np.int32]:
                # 整数类型需要归一化
                max_val = np.iinfo(audio_chunk.dtype).max
                audio_chunk = audio_chunk.astype(np.float32) / max_val
            else:
                audio_chunk = audio_chunk.astype(np.float32)

        # 处理多声道音频
        if audio_chunk.ndim == 2:
            if audio_chunk.shape[1] > 1:
                # 多声道转单声道
                audio_chunk = np.mean(audio_chunk, axis=1)
            else:
                audio_chunk = audio_chunk.flatten()
        elif audio_chunk.ndim > 2:
            raise ValueError(f"不支持的音频维度: {audio_chunk.ndim}")

        return audio_chunk

    def is_speech(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """
        检测音频片段是否包含语音

        Returns:
            Tuple[bool, float]: (是否包含语音, 语音概率)
        """
        try:
            audio_chunk = self._validate_input(audio_chunk)

            # 确保音频长度正确
            expected_length = int(
                self.config.per_frame_duration * self.config.sampling_rate)
            if len(audio_chunk) != expected_length:
                # 填充或截断音频
                if len(audio_chunk) < expected_length:
                    audio_chunk = np.pad(
                        audio_chunk, (0, expected_length - len(audio_chunk)))
                else:
                    audio_chunk = audio_chunk[:expected_length]

            # 参考silero.py: 确定正确的帧大小
            num_samples = 512 if self.config.sampling_rate == 16000 else 256

            # 重新调整音频长度为标准帧大小
            if len(audio_chunk) != num_samples:
                if len(audio_chunk) < num_samples:
                    audio_chunk = np.pad(
                        audio_chunk, (0, num_samples - len(audio_chunk)))
                else:
                    audio_chunk = audio_chunk[:num_samples]

            # 参考silero.py: 添加批次维度
            if np.ndim(audio_chunk) == 1:
                audio_chunk = np.expand_dims(audio_chunk, 0)

            batch_size = np.shape(audio_chunk)[0]
            context_size = 64 if self.config.sampling_rate == 16000 else 32

            # 参考silero.py: 状态重置逻辑
            if not self._last_batch_size:
                self._reset_states(batch_size)
            if self._last_sr and self._last_sr != self.config.sampling_rate:
                self._reset_states(batch_size)
            if self._last_batch_size and self._last_batch_size != batch_size:
                self._reset_states(batch_size)

            # 参考silero.py: 上下文管理
            if not np.shape(self._context)[1]:
                self._context = np.zeros(
                    (batch_size, context_size), dtype='float32')

            # 参考silero.py: 拼接上下文和当前音频
            x = np.concatenate((self._context, audio_chunk), axis=1)

            # 参考silero.py: 准备ONNX输入
            ort_inputs = {
                "input": x,
                "state": self._state,
                "sr": np.array(self.config.sampling_rate, dtype="int64")
            }

            # 运行推理
            ort_outs = self.session.run(None, ort_inputs)
            out, state = ort_outs

            # 更新状态
            self._state = state
            self._context = x[..., -context_size:]
            self._last_sr = self.config.sampling_rate
            self._last_batch_size = batch_size

            # 获取语音概率
            speech_prob = float(out[0])

            return speech_prob > self.config.threshold, speech_prob

        except Exception as e:
            logger.error(f"VAD推理错误: {e}")
            return False, 0.0

    def process_audio(self, audio_chunk: np.ndarray) -> Tuple[Optional[np.ndarray], VadState]:
        """
        处理音频片段，返回完整的语音段和当前状态

        Args:
            audio_chunk: 音频数据

        Returns:
            Tuple[Optional[np.ndarray], VadState]: (完整语音段或None, 当前VAD状态)
        """
        try:
            audio_chunk = self._validate_input(audio_chunk)

            # 将当前音频片段添加到填充缓冲区
            if self.config.padding_duration > 0:
                self.padding_buffer.append(audio_chunk.copy())

            # 检测是否包含语音
            is_speech_frame, speech_prob = self.is_speech(audio_chunk)

            if is_speech_frame:
                return self._handle_speech_frame(audio_chunk, speech_prob)
            else:
                return self._handle_silence_frame(audio_chunk, speech_prob)

        except Exception as e:
            logger.error(f"音频处理错误: {e}")
            return None, self.current_state

    def _handle_speech_frame(self, audio_chunk: np.ndarray, speech_prob: float) -> Tuple[Optional[np.ndarray], VadState]:
        """处理检测到语音的帧"""
        # 重置静音计数
        self.silence_counter = 0

        if self.current_state == VadState.IDLE:
            # 语音开始
            self.current_state = VadState.SPEECH_START
            self.is_recording = True

            # 添加填充音频
            if self.config.padding_duration > 0:
                for padding_chunk in self.padding_buffer:
                    self.audio_buffer.append(padding_chunk.copy())

            logger.debug(f"语音开始，概率: {speech_prob:.3f}")

        elif self.current_state in [VadState.SPEECH_START, VadState.SPEECH_CONTINUE]:
            # 语音继续
            self.current_state = VadState.SPEECH_CONTINUE

        # 添加当前音频块
        self.audio_buffer.append(audio_chunk.copy())

        return None, self.current_state

    def _handle_silence_frame(self, audio_chunk: np.ndarray, speech_prob: float) -> Tuple[Optional[np.ndarray], VadState]:
        """处理静音帧"""
        if self.is_recording:
            # 增加静音计数
            self.silence_counter += len(audio_chunk)

            # 继续添加音频（可能是语音间的短暂停顿）
            self.audio_buffer.append(audio_chunk.copy())

            # 检查是否达到静音阈值
            if self.silence_counter >= self.silence_samples:
                return self._finalize_speech_segment()

        return None, self.current_state

    def _finalize_speech_segment(self) -> Tuple[Optional[np.ndarray], VadState]:
        """完成语音段的处理"""
        self.current_state = VadState.SPEECH_END
        self.is_recording = False

        # 检查语音段是否足够长
        total_samples = sum(len(chunk) for chunk in self.audio_buffer)

        if total_samples >= self.min_speech_samples:
            # 语音段有效，返回完整音频
            speech_segment = np.concatenate(self.audio_buffer)
            logger.debug(
                f"检测到有效语音段，长度: {len(speech_segment) / self.config.sampling_rate:.2f}秒")

            # 重置缓冲区
            self.audio_buffer = []
            self.silence_counter = 0
            self.current_state = VadState.IDLE

            return speech_segment, VadState.SPEECH_END
        else:
            # 语音段太短，丢弃
            logger.debug(
                f"语音段太短，丢弃: {total_samples / self.config.sampling_rate:.2f}秒")
            self.audio_buffer = []
            self.silence_counter = 0
            self.current_state = VadState.IDLE

            return None, VadState.IDLE

    def reset(self):
        """重置 VAD 状态"""
        logger.debug("重置VAD状态")
        self._reset_internal_state()
        self._calculate_parameters()

    def get_stats(self) -> dict:
        """获取VAD统计信息"""
        return {
            "current_state": self.current_state.value,
            "is_recording": self.is_recording,
            "buffer_length": len(self.audio_buffer),
            "buffer_duration": sum(len(chunk) for chunk in self.audio_buffer) / self.config.sampling_rate,
            "silence_counter": self.silence_counter,
            "silence_duration": self.silence_counter / self.config.sampling_rate,
            "padding_buffer_size": len(self.padding_buffer) if self.padding_buffer else 0
        }


# 向后兼容的工厂函数
def create_vad_processor(threshold=0.7, sampling_rate=16000, padding_duration=0.2,
                         min_speech_duration=0.1, silence_duration=0.5,
                         per_frame_duration=0.032) -> VadProcessor:
    """创建VAD处理器（向后兼容）"""
    config = VadConfig(
        threshold=threshold,
        sampling_rate=sampling_rate,
        padding_duration=padding_duration,
        min_speech_duration=min_speech_duration,
        silence_duration=silence_duration,
        per_frame_duration=per_frame_duration
    )
    return VadProcessor(config)


if __name__ == "__main__":
    # 真实音频文件测试
    import logging
    import os
    import wave
    import struct

    logging.basicConfig(level=logging.INFO)

    # 检查en.wav文件是否存在
    audio_file_path = os.path.join(os.path.dirname(__file__), "en.wav")
    if not os.path.exists(audio_file_path):
        print(f"❌ 音频文件不存在: {audio_file_path}")
        print("请确保en.wav文件在相同目录下")
        exit(1)

    print(f"📁 加载音频文件: {audio_file_path}")

    # 读取WAV文件
    try:
        with wave.open(audio_file_path, 'rb') as wav_file:
            # 获取音频参数
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            duration = frames / sample_rate

            print(f"🎵 音频信息:")
            print(f"  - 采样率: {sample_rate}Hz")
            print(f"  - 声道数: {channels}")
            print(f"  - 采样宽度: {sample_width}字节")
            print(f"  - 总帧数: {frames}")
            print(f"  - 总时长: {duration:.2f}秒")

            # 读取所有音频数据
            raw_audio = wav_file.readframes(frames)

            if sample_width == 2:
                # 16位有符号
                audio_data = np.frombuffer(raw_audio, dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0
            else:
                raise ValueError(f"不支持的采样宽度: {sample_width}")

            # 处理多声道音频
            if channels > 1:
                raise ValueError(f"不支持的多声道音频: {channels}")

    except Exception as e:
        print(f"❌ 读取音频文件失败: {e}")
        exit(1)

    # 创建VAD处理器
    config = VadConfig(
        threshold=0.5,
        sampling_rate=sample_rate if sample_rate in [8000, 16000] else 16000,
        padding_duration=0.3,
        min_speech_duration=0.1,
        silence_duration=0.8
    )

    vad = VadProcessor(config)
    print(f"✅ VAD处理器创建成功")

    # 分帧处理音频
    frame_duration = 0.032  # 32ms帧
    frame_size = int(frame_duration * sample_rate)  # 每帧样本数
    total_frames = len(audio_data) // frame_size

    print(f"\n🎬 开始处理音频:")
    print(f"  - 帧大小: {frame_size}样本 ({frame_duration*1000:.0f}ms)")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 处理时长: {total_frames * frame_duration:.2f}秒")
    print("-" * 50)

    speech_segments = []
    frame_results = []

    for frame_idx in range(total_frames):
        # 提取当前帧
        start_sample = frame_idx * frame_size
        end_sample = start_sample + frame_size
        audio_frame = audio_data[start_sample:end_sample]

        # 确保帧长度正确
        if len(audio_frame) < frame_size:
            audio_frame = np.pad(
                audio_frame, (0, frame_size - len(audio_frame)))

        # VAD处理
        speech_segment, state = vad.process_audio(audio_frame)
        is_speech, prob = vad.is_speech(audio_frame)

        frame_time = frame_idx * frame_duration
        frame_results.append({
            'frame': frame_idx,
            'time': frame_time,
            'state': state.value,
            'probability': prob,
            'is_speech': is_speech
        })

        if state == VadState.SPEECH_START:
            print(f"🗣️  帧{frame_idx:4d} ({frame_time:5.2f}s): 检测到语音段")

        if state == VadState.SPEECH_END:
            print(f"🗣️  帧{frame_idx:4d} ({frame_time:5.2f}s): 检测到语音段结束")

        # 检测到语音段
        if speech_segment is not None:
            segment_duration = len(speech_segment) / sample_rate
            speech_segments.append({
                'frame': frame_idx,
                'time': frame_time,
                'duration': segment_duration,
                'samples': len(speech_segment)
            })
            print(
                f"🗣️  帧{frame_idx:4d} ({frame_time:5.2f}s): 检测到语音段 {segment_duration:.2f}秒")

            # 保存语音段为WAV文件
            output_filename = f"speech_segment_{frame_time:5.2f}.wav"
            with wave.open(output_filename, 'wb') as wav_file:
                wav_file.setnchannels(1)  # 单声道
                wav_file.setsampwidth(2)  # 16位
                wav_file.setframerate(sample_rate)
                # 将float32转换回int16
                audio_int16 = (speech_segment * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
            print(f"💾 已保存语音段到: {output_filename}")

        # 每50帧打印一次状态
        if frame_idx % 50 == 0:
            stats = vad.get_stats()
            print(f"📊 帧{frame_idx:4d} ({frame_time:5.2f}s): "
                  f"状态={state.value:15s}, 概率={prob:.3f}, "
                  f"缓冲={stats['buffer_duration']:.2f}s")

    # 最终统计
    print("\n" + "=" * 50)
    print("📈 处理结果统计:")
    print(f"  - 总处理帧数: {len(frame_results)}")
    print(f"  - 检测到语音段数: {len(speech_segments)}")

    # 语音/静音帧统计
    speech_frames = sum(1 for r in frame_results if r['is_speech'])
    silence_frames = len(frame_results) - speech_frames
    speech_duration = speech_frames * frame_duration
    silence_duration = silence_frames * frame_duration

    print(
        f"  - 语音帧数: {speech_frames} ({speech_duration:.2f}s, {speech_duration/duration*100:.1f}%)")
    print(
        f"  - 静音帧数: {silence_frames} ({silence_duration:.2f}s, {silence_duration/duration*100:.1f}%)")

    # 显示检测到的语音段
    if speech_segments:
        print(f"\n🎯 检测到的语音段:")
        total_speech_time = sum(seg['duration'] for seg in speech_segments)
        for i, seg in enumerate(speech_segments, 1):
            print(f"  {i}. 时间: {seg['time']:.2f}s, 时长: {seg['duration']:.2f}s")
        print(f"  总语音时长: {total_speech_time:.2f}s")
    else:
        print("⚠️  未检测到任何语音段")

    print(f"\n✅ 音频处理完成！")
