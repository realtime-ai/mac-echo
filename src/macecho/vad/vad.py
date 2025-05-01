import collections
from enum import Enum
import numpy as np
import onnxruntime as ort
import os
import urllib.request


# 增加vad的几种状态
class VadState(Enum):
    SpeechStart = 0
    SpeechEnd = 1
    SpeechSpeaking = 2


class VadProcessor:
    def __init__(self, threshold=0.7,
                 sampling_rate=16000,
                 padding_duration=0.3,
                 min_speech_duration=0.1,
                 silence_duration=0.5,
                 per_frame_duration=0.032):
        """
        初始化 VAD 处理器

        Args:
            threshold (float): VAD 检测阈值，范围 0-1
            sampling_rate (int): 音频采样率
            min_speech_duration (float): 最小语音段长度，单位秒
            silence_duration (float): 静音判断时长，单位秒
        """
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        # 下载并加载 ONNX 模型
        model_path = "silero_vad.onnx"
        if not os.path.exists(model_path):
            print("下载 Silero VAD 模型...")
            urllib.request.urlretrieve(
                "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx",
                model_path
            )

        # 初始化 ONNX Runtime
        self.session = ort.InferenceSession(model_path)
        self.h = np.zeros((2, 1, 64)).astype('float32')
        self.c = np.zeros((2, 1, 64)).astype('float32')

        # 用于存储音频缓冲
        self.audio_buffer = []

        # 用于存储填充的音频缓冲
        self.padding_audio_buffer = collections.deque(
            maxlen=int(padding_duration / per_frame_duration))

        # 用于存储临时音频缓冲
        self.tmp_audio_buffer = bytearray()

        self.frame_size = int((sampling_rate / 1000) *
                              per_frame_duration * 2)  # int16 should *2

        # 配置参数
        self.min_speech_samples = int(min_speech_duration * sampling_rate)
        self.silence_samples = int(silence_duration * sampling_rate)
        self.silence_counter = 0

        print(f"VAD 配置:")
        print(
            f"- 最小语音段长度: {min_speech_duration}秒 ({self.min_speech_samples}样本)")
        print(f"- 静音判断时长: {silence_duration}秒 ({self.silence_samples}样本)")
        print(f"- 检测阈值: {threshold}")

    def _validate_input(self, audio_chunk):
        """验证输入音频格式"""
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        if audio_chunk.ndim == 2:
            audio_chunk = audio_chunk.flatten()
        return audio_chunk

    def is_speech(self, audio_chunk):
        """检测音频片段是否包含语音"""
        audio_chunk = self._validate_input(audio_chunk)

        # 准备输入
        input_data = {
            'input': audio_chunk.reshape(1, -1),
            'sr': np.array(self.sampling_rate, dtype=np.int64),
            'h': self.h,
            'c': self.c
        }

        # 运行推理
        out, self.h, self.c = self.session.run(
            ['output', 'hn', 'cn'], input_data)
        speech_prob = out[0][-1]

        return speech_prob > self.threshold, speech_prob

    def process_audio(self, audio_chunk):
        """处理音频片段，返回有效的语音段"""
        is_speech_frame, prob = self.is_speech(audio_chunk)

        if is_speech_frame:
            # 检测到语音，重置静音计数器
            self.silence_counter = 0
            self.audio_buffer.append(audio_chunk)
            return None
        else:
            # 未检测到语音，增加静音计数
            self.silence_counter += len(audio_chunk)

            if len(self.audio_buffer) > 0:
                # 如果静音时长超过阈值，且有足够的语音数据
                if self.silence_counter >= self.silence_samples:
                    if len(self.audio_buffer) * len(audio_chunk) >= self.min_speech_samples:
                        speech_segment = np.concatenate(self.audio_buffer)
                        self.audio_buffer = []
                        self.silence_counter = 0
                        return speech_segment
                    else:
                        # 语音段太短，丢弃
                        self.audio_buffer = []
                        self.silence_counter = 0
                else:
                    # 静音未达到阈值，继续缓存音频
                    self.audio_buffer.append(audio_chunk)

        return None

    def reset(self):
        """重置 VAD 状态"""
        self.audio_buffer = []
        self.silence_counter = 0
        self.h = np.zeros((2, 1, 64)).astype('float32')
        self.c = np.zeros((2, 1, 64)).astype('float32')
