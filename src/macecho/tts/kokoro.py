# 使用 Kokoro 的 TTS 服务

import os
import io
import numpy as np
from misaki import zh
from kokoro_onnx import Kokoro
from typing import Generator

from .base import BaseTTS
from .factory import register_tts


@register_tts("kokoro")
class KokoroTTS(BaseTTS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_dir = os.path.join(os.path.dirname(__file__), "kokoro")
        self.kokoro = None
        self.g2p = None
        self.g2p_en = None
        self._initialize_models()

    def _initialize_models(self):
        """初始化 Kokoro 模型和 G2P 处理器"""
        model_path = os.path.join(self.model_dir, "kokoro-v1.1-zh.onnx")
        voices_path = os.path.join(self.model_dir, "voices-v1.1-zh.bin")
        config_path = os.path.join(self.model_dir, "config.json")
        
        if not all(os.path.exists(p) for p in [model_path, voices_path, config_path]):
            raise FileNotFoundError(f"Kokoro model files not found in {self.model_dir}")
        
        
        self.kokoro = Kokoro(model_path, voices_path, vocab_config=config_path)
        self.g2p = zh.ZHG2P(version="1.1")
        

    def synthesize(self, text: str) -> bytes:
        """同步合成整段语音"""
        if not self.kokoro or not self.g2p:
            raise RuntimeError("Kokoro models not initialized")
        
        # 获取音素
        phonemes, _ = self.g2p(text)
        
        # 生成音频
        samples, sample_rate = self.kokoro.create(
            phonemes, 
            voice=self.voice, 
            speed=self.get_config("speed", 1.0), 
            is_phonemes=True
        )
        
        # 将浮点音频转换为 int16 格式的原始 PCM 字节
        # 假设 samples 是 [-1, 1] 范围的浮点数
        if samples.dtype == np.float32 or samples.dtype == np.float64:
            samples = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
        elif samples.dtype != np.int16:
            samples = samples.astype(np.int16)
        
        # 返回原始 PCM 字节
        return samples.tobytes()

    def stream_synthesize(self, text: str) -> Generator[bytes, None, None]:
        """流式合成语音，使用 create_stream 方法"""
        import asyncio
        import inspect
        
        if not self.kokoro or not self.g2p:
            raise RuntimeError("Kokoro models not initialized")
        
        # 获取音素
        phonemes, _ = self.g2p(text)
        
        # 使用 create_stream 进行流式生成
        stream_result = self.kokoro.create_stream(
            phonemes,
            voice=self.voice,
            speed=self.get_config("speed", 1.0),
            is_phonemes=True
        )
        
        # 处理 async generator
        if inspect.isasyncgen(stream_result):
            # 获取事件循环并运行 async generator
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            async def _run_async_generator():
                chunks = []
                async for audio_chunk,sample_rate in stream_result:
                    # 将音频块转换为 int16 格式的原始 PCM 字节
                    if audio_chunk.dtype == np.float32 or audio_chunk.dtype == np.float64:
                        audio_chunk = np.clip(audio_chunk * 32767, -32768, 32767).astype(np.int16)
                    elif audio_chunk.dtype != np.int16:
                        audio_chunk = audio_chunk.astype(np.int16)
                    chunks.append(audio_chunk.tobytes())
                return chunks
            
            chunks = loop.run_until_complete(_run_async_generator())
            for chunk in chunks:
                yield chunk
        else:
            # 同步 generator
            for audio_chunk,sample_rate in stream_result:
                # 将音频块转换为 int16 格式的原始 PCM 字节
                if audio_chunk.dtype == np.float32 or audio_chunk.dtype == np.float64:
                    audio_chunk = np.clip(audio_chunk * 32767, -32768, 32767).astype(np.int16)
                elif audio_chunk.dtype != np.int16:
                    audio_chunk = audio_chunk.astype(np.int16)
                yield audio_chunk.tobytes()
