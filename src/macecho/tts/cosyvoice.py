import time
from pathlib import Path
import os
from openai import OpenAI
import io
from typing import Generator, Optional

from .base import BaseTTS


# CosyVoice TTS 使用云端的版本
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
            "api_key", os.getenv("COSYVOICE_API_KEY"))
        self.base_url = self.get_config(
            "base_url", "http://api.siliconflow.cn/v1")

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
                response_format="wav",
            )
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
                response_format="wav",
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

    def save_to_file(self, text: str, file_path: str) -> None:
        """
        将文本转换为语音并保存为文件

        Args:
            text: 要合成的文本
            file_path: 保存路径
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(
                os.path.abspath(file_path)), exist_ok=True)

            # 使用流式合成，边合成边写入文件
            with open(file_path, "wb") as f:
                for chunk in self.stream_synthesize(text):
                    f.write(chunk)

            print(f"已保存音频文件到: {file_path}")
        except Exception as e:
            print(f"保存到文件失败: {e}")


# 示例用法
if __name__ == "__main__":
    # 创建TTS实例
    tts = CosyVoiceTTS(voice="david")

    # 要合成的文本
    text = "今天真是太开心了，马上要放假了！I'm so happy, Spring Festival is coming!"

    # 示例1: 保存到文件
    output_path = Path(__file__).parent / "cosyvoice-output.wav"
    print(f"\n示例1：合成并保存到文件")
    tts.save_to_file(text, str(output_path))

    # 示例2: 流式合成
    print(f"\n示例2：流式合成并统计")
    start_time = time.time()
    chunk_count = 0
    total_bytes = 0

    print("开始流式合成...")
    for i, chunk in enumerate(tts.stream_synthesize(text)):
        chunk_count += 1
        total_bytes += len(chunk)
        print(f"接收第 {chunk_count} 个音频块, 大小: {len(chunk)} 字节")

    total_time = time.time() - start_time
    print(f"流式合成统计：")
    print(f"- 总耗时: {total_time:.4f}秒")
    print(f"- 块数量: {chunk_count}")
    print(f"- 总大小: {total_bytes} 字节")
