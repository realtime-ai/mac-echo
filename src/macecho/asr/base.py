from abc import ABC, abstractmethod
from typing import Generator, Any, Dict, Union


class BaseASR(ABC):
    """
    抽象的 ASR 基类，支持同步和流式语音识别，并支持自定义参数配置。
    """

    def __init__(self, language: str = "zh", sample_rate: int = 16000, **kwargs):
        """
        初始化 ASR 配置参数。

        :param language: 语种代码，如 'zh', 'en'
        :param sample_rate: 音频采样率
        :param kwargs: 其他自定义配置参数，例如 api_key、模型路径等
        """
        self.language = language
        self.sample_rate = sample_rate
        self.config: Dict[str, Any] = kwargs

    @abstractmethod
    def transcribe(self, audio: Union[bytes, str]) -> str:
        """
        同步识别一段音频。

        :param audio: 音频数据（bytes）或文件路径（str）
        :return: 识别文本
        """
        pass

    @abstractmethod
    def stream_transcribe(self, audio_stream: Generator[bytes, None, None]) -> Generator[str, None, None]:
        """
        流式识别，将音频块转换为逐步输出的文本结果。

        :param audio_stream: 音频块生成器
        :return: 文本结果生成器
        """
        pass

    def set_language(self, language: str):
        self.language = language

    def set_sample_rate(self, sample_rate: int):
        self.sample_rate = sample_rate

    def set_config(self, **kwargs):
        self.config.update(kwargs)

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
