from abc import ABC, abstractmethod
from typing import Generator, Any, Dict


class BaseTTS(ABC):
    """
    抽象的 TTS 基类，支持同步与流式语音合成，并支持自定义参数。
    """

    def __init__(self, voice: str = "default", sample_rate: int = 22050, **kwargs):
        """
        初始化 TTS 配置，包括语音、采样率和额外自定义参数。
        """
        self.voice = voice
        self.sample_rate = sample_rate
        self.config: Dict[str, Any] = kwargs  # 存储 api_key、url 等额外配置

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """
        同步方式合成整段语音。
        """
        pass

    @abstractmethod
    def stream_synthesize(self, text: str) -> Generator[bytes, None, None]:
        """
        流式方式合成语音。
        返回音频块的生成器，支持边合成边播放。
        """
        pass

    def set_voice(self, voice: str):
        self.voice = voice

    def set_sample_rate(self, sample_rate: int):
        self.sample_rate = sample_rate

    def set_config(self, **kwargs):
        """
        批量更新配置参数。
        """
        self.config.update(kwargs)

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        获取某个配置项。
        """
        return self.config.get(key, default)
