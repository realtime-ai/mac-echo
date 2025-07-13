# 使用 Koroto 的 TTS 服务

from .base import BaseTTS
from .factory import register_tts


@register_tts("koroto")
class KorotoTTS(BaseTTS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def synthesize(self, text: str) -> bytes:
        return b''
    
    def stream_synthesize(self, text: str):
        yield b''
