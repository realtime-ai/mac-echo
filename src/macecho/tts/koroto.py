# 使用 Koroto 的 TTS 服务

from .base import BaseTTS


class KorotoTTS(BaseTTS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def synthesize(self, text: str) -> bytes:
        pass
