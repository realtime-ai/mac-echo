from enum import Enum
from typing import Optional, Tuple, Protocol, runtime_checkable
import numpy as np


class VadState(Enum):
    """VAD状态枚举"""
    IDLE = "idle"              # 空闲状态，未检测到语音
    SPEECH_START = "speech_start"   # 语音开始
    SPEECH_CONTINUE = "speech_continue"  # 语音持续中
    SPEECH_END = "speech_end"      # 语音结束


@runtime_checkable
class VADInterface(Protocol):
    """VAD接口定义"""

    def process_audio(self, audio_chunk: np.ndarray) -> Tuple[Optional[np.ndarray], VadState]:
        """
        处理音频片段，返回完整的语音段和当前状态

        Args:
            audio_chunk: 音频数据，numpy数组

        Returns:
            Tuple[Optional[np.ndarray], VadState]: (完整语音段或None, 当前VAD状态)
        """
        ...

    def is_speech(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """
        检测音频片段是否包含语音

        Args:
            audio_chunk: 音频数据，numpy数组

        Returns:
            Tuple[bool, float]: (是否包含语音, 语音概率)
        """
        ...

    def reset(self) -> None:
        """重置 VAD 状态"""
        ...

    def get_stats(self) -> dict:
        """
        获取VAD统计信息

        Returns:
            dict: 包含VAD当前状态的统计信息
        """
        ...
