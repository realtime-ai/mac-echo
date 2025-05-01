from abc import ABC, abstractmethod


class ASR(ABC):
    """
    ASR 基类，定义了语音识别系统的基本接口
    """

    @abstractmethod
    def initialize(self):
        """
        初始化 ASR 系统，例如加载模型、配置文件等
        """
        pass

    @abstractmethod
    def recognize(self, audio_data: bytes) -> str:
        """
        对音频数据进行识别，返回识别结果的文本
        :param audio_data: 音频数据
        :return: 识别后的文本
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """
        检查 ASR 系统是否已准备好进行语音识别
        :return: True 如果准备好，否则 False
        """
        pass

    @abstractmethod
    def release(self):
        """
        释放资源，停止 ASR 系统
        """
        pass
