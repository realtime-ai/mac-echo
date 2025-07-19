from .base import BaseTTS
from .cosyvoice import CosyVoiceTTS
from .kokoro import KokoroTTS
from .factory import TTSFactory, register_tts

__all__ = [
    'BaseTTS',
    'CosyVoiceTTS',
    'KokoroTTS',
    'TTSFactory',
    'register_tts',
]