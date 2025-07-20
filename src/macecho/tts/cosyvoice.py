import time
from pathlib import Path
import os
from openai import OpenAI
import io
from typing import Generator, Optional

from .base import BaseTTS
from .factory import register_tts


# CosyVoice TTS 使用云端的版本
@register_tts("cosyvoice")
class CosyVoiceTTS(BaseTTS):
    """
    基于CosyVoice的语音合成实现类，继承自BaseTTS
    使用硅云API作为后端
    """

    def __init__(self, voice: str = "david", sample_rate: int = 24000, **kwargs):
        """
        初始化CosyVoice TTS服务

        Args:
            voice: 声音名称，默认为"david"
            sample_rate: 采样率，默认为24000Hz
            **kwargs: 其他参数，可包括：
                - api_key: 硅云API密钥
                - base_url: 硅云API基础URL
                - model: 模型名称，默认为"FunAudioLLM/CosyVoice2-0.5B"
                - output_dir: 输出目录，默认为当前脚本所在目录
        """
        super().__init__(voice, sample_rate, **kwargs)

        # 设置API密钥和URL
        self.api_key = self.get_config(
            "api_key", os.getenv("SILICONFLOW_API_KEY"))
        self.base_url = self.get_config(
            "base_url", "https://api.siliconflow.cn/v1")

        print("=" * 40)
        print(f"API密钥: {self.api_key}")
        print(f"Base URL: {self.base_url}")

        # 设置模型和输出目录
        self.model = self.get_config("model", "FunAudioLLM/CosyVoice2-0.5B")
        self.output_dir = self.get_config(
            "output_dir", str(Path(__file__).parent))

        # 初始化客户端
        self.client = None
        self.initialize()

    def initialize(self) -> bool:
        """
        初始化OpenAI客户端

        Returns:
            bool: 初始化是否成功
        """
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            return True
        except Exception as e:
            print(f"初始化CosyVoice TTS失败: {e}")
            return False

    def synthesize(self, text: str) -> bytes:
        """
        同步方式合成整段语音

        Args:
            text: 要合成的文本

        Returns:
            bytes: 合成后的音频数据
        """
        if not self.client:
            self.initialize()

        try:
            # 非流式请求
            response = self.client.audio.speech.create(
                model=self.model,
                voice=f"{self.model}:{self.voice}",
                input=text,
                response_format="pcm",
            )
            print(f"response: {response}")
            return response.content
        except Exception as e:
            print(f"同步合成失败: {e}")
            return bytes()

    def stream_synthesize(self, text: str) -> Generator[bytes, None, None]:
        """
        流式方式合成语音

        Args:
            text: 要合成的文本

        Returns:
            Generator[bytes, None, None]: 音频块生成器
        """
        if not self.client:
            self.initialize()

        try:
            # 记录开始时间
            start_time = time.time()
            first_chunk_time = None
            chunk_count = 0

            # 流式请求
            with self.client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=f"{self.model}:{self.voice}",
                input=text,
                response_format="pcm",
            ) as response:
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue

                    # 记录第一个块的时间
                    if chunk_count == 0:
                        first_chunk_time = time.time() - start_time
                        print(f"首个音频块生成耗时: {first_chunk_time:.4f}秒")

                    chunk_count += 1
                    yield chunk

            # 记录结束时间
            total_time = time.time() - start_time
            print(f"流式合成完成，总耗时: {total_time:.4f}秒，块数量: {chunk_count}")

        except Exception as e:
            print(f"流式合成失败: {e}")
            yield bytes()
