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
                 padding_duration=0.0,
                 min_speech_duration=0.1,
                 silence_duration=0.5,
                 per_frame_duration=0.032):
        """
        初始化 VAD 处理器

        Args:
            threshold (float): VAD 检测阈值，范围 0-1
            sampling_rate (int): 音频采样率
            padding_duration (float): 语音开始前的填充时长，单位秒
            min_speech_duration (float): 最小语音段长度，单位秒
            silence_duration (float): 静音判断时长，单位秒
            per_frame_duration (float): 每帧音频的时长，单位秒
        """
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.padding_duration = padding_duration
        self.per_frame_duration = per_frame_duration

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
        self.padding_buffer = collections.deque(maxlen=int(padding_duration * sampling_rate / len(
            self._validate_input(np.zeros(int(per_frame_duration * sampling_rate), dtype=np.float32)))))

        # 是否正在录制语音
        self.is_recording = False

        # 用于存储临时音频缓冲
        self.tmp_audio_buffer = bytearray()

        # 每帧的字节数：采样率 * 单帧时长 * 2（int16）
        # 原实现使用 ``sampling_rate / 1000`` 导致结果始终为 1，
        # 无法正确分割音频帧
        self.frame_size = int(sampling_rate * per_frame_duration * 2)

        # 配置参数
        self.min_speech_samples = int(min_speech_duration * sampling_rate)
        self.silence_samples = int(silence_duration * sampling_rate)
        self.silence_counter = 0

        print(f"VAD 配置:")
        print(
            f"- 最小语音段长度: {min_speech_duration}秒 ({self.min_speech_samples}样本)")
        print(f"- 静音判断时长: {silence_duration}秒 ({self.silence_samples}样本)")
        print(f"- 语音前填充时长: {padding_duration}秒")
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
        audio_chunk = self._validate_input(audio_chunk)

        # 将当前音频片段添加到填充缓冲区
        if self.padding_duration > 0:
            self.padding_buffer.append(audio_chunk)

        is_speech_frame, prob = self.is_speech(audio_chunk)

        if is_speech_frame:
            # 检测到语音
            self.silence_counter = 0

            # 如果之前没有检测到语音，这是新语音的开始
            if not self.is_recording and self.padding_duration > 0:
                # 将填充缓冲区中的音频添加到主缓冲区
                for padding_chunk in self.padding_buffer:
                    self.audio_buffer.append(padding_chunk)
                # 标记为正在录制
                self.is_recording = True

            # 添加当前音频块
            self.audio_buffer.append(audio_chunk)
            return None
        else:
            # 未检测到语音，增加静音计数
            self.silence_counter += len(audio_chunk)

            if len(self.audio_buffer) > 0:
                # 如果静音时长超过阈值，且有足够的语音数据
                if self.silence_counter >= self.silence_samples:
                    # 重置录制状态
                    self.is_recording = False

                    if sum(len(chunk) for chunk in self.audio_buffer) >= self.min_speech_samples:
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
            elif self.padding_duration > 0:
                # 没有检测到语音时，继续更新填充缓冲区但不做其他处理
                pass

        return None

    def reset(self):
        """重置 VAD 状态"""
        self.audio_buffer = []
        self.silence_counter = 0
        self.is_recording = False
        # 清空填充缓冲区但保持其容量不变
        self.padding_buffer.clear()
        self.h = np.zeros((2, 1, 64)).astype('float32')
        self.c = np.zeros((2, 1, 64)).astype('float32')


# 测试代码
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    import wave
    import struct

    def generate_test_audio(duration, sample_rate, has_speech=False):
        """生成测试音频"""
        samples = int(duration * sample_rate)
        # 基础噪音
        noise_level = 0.05
        audio = np.random.normal(0, noise_level, samples).astype(np.float32)

        # 如果需要添加语音，生成一个简单的正弦波
        if has_speech:
            # 生成一个1秒的440Hz正弦波作为"语音"
            speech_samples = min(samples, int(1.0 * sample_rate))
            speech_start = int((samples - speech_samples) / 2)  # 放在中间
            t = np.arange(speech_samples) / sample_rate
            speech = 0.5 * np.sin(2 * np.pi * 440 * t)
            audio[speech_start:speech_start+speech_samples] += speech

        return audio

    # 测试不同的填充时长
    padding_durations = [0.0, 0.2, 0.5]

    for padding_duration in padding_durations:
        print(f"\n=== 测试填充时长: {padding_duration}秒 ===")

        # 创建VAD处理器
        vad = VadProcessor(threshold=0.5,
                           padding_duration=padding_duration,
                           silence_duration=0.3,
                           min_speech_duration=0.2)

        # 生成测试音频：先2秒静音，然后1秒语音，再1秒静音
        audio_length = 4.0  # 总长4秒
        sample_rate = 16000

        # 生成静音段
        silence1 = generate_test_audio(2.0, sample_rate, has_speech=False)
        # 生成语音段
        speech = generate_test_audio(1.0, sample_rate, has_speech=True)
        # 生成另一段静音
        silence2 = generate_test_audio(1.0, sample_rate, has_speech=False)

        # 组合音频
        full_audio = np.concatenate([silence1, speech, silence2])

        # 处理音频块
        frame_size = int(0.032 * sample_rate)  # 32ms帧
        results = []

        print("处理音频...")
        for i in range(0, len(full_audio), frame_size):
            frame = full_audio[i:i+frame_size]
            if len(frame) == frame_size:  # 确保帧完整
                result = vad.process_audio(frame)
                if result is not None:
                    results.append(result)
                    print(f"检测到语音段，长度: {len(result) / sample_rate:.2f}秒")

        # 如果有检测结果，分析第一个结果
        if results:
            result_audio = results[0]
            duration = len(result_audio) / sample_rate
            expected_min_duration = 1.0  # 预期的语音部分
            expected_padding = padding_duration
            expected_total = expected_min_duration + expected_padding

            print(f"检测到的语音段长度: {duration:.2f}秒")
            print(
                f"预期语音段长度: 约 {expected_total:.2f}秒 (语音1秒 + 填充{padding_duration:.2f}秒)")

            if duration >= expected_min_duration:
                print("✓ 成功捕获了语音部分")
            else:
                print("✗ 未能完全捕获语音部分")

            if padding_duration > 0:
                if duration >= expected_total - 0.1:  # 允许0.1秒误差
                    print(f"✓ 成功捕获了填充部分 ({padding_duration:.2f}秒)")
                else:
                    print(f"✗ 未能正确捕获填充部分 (预期{padding_duration:.2f}秒)")

            # 保存检测到的音频到WAV文件
            filename = f"detected_speech_padding_{padding_duration:.1f}.wav"
            with wave.open(filename, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                # 转换为16位整数
                audio_int16 = (result_audio * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
            print(f"已保存检测到的语音段到: {filename}")
        else:
            print("未检测到语音段")

        # 重置VAD状态
        vad.reset()
