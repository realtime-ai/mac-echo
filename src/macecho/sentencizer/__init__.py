from abc import ABC, abstractmethod
from typing import Generator, List, Set, Optional


class BaseSentencizer(ABC):
    """
    流式句子分割器基类

    用于将大语言模型的流式输出文本实时分割成句子，
    当检测到完整句子时立即返回，而不必等待整个响应完成。
    """

    def __init__(self,
                 end_punctuations: Optional[Set[str]] = None,
                 pause_punctuations: Optional[Set[str]] = None):
        """
        初始化句子分割器

        Args:
            end_punctuations: 结束标点符号集合，表示句子结束
            pause_punctuations: 暂停标点符号集合，表示句子可能结束但需要更多上下文确认
        """
        # 默认的结束标点
        self.end_punctuations = end_punctuations or {
            '.', '!', '?', '。', '！', '？', '…'}

        # 默认的暂停标点（需要更多上下文确认句子是否结束）
        self.pause_punctuations = pause_punctuations or {
            ',', ';', ':', '，', '；', '：', '、'}

        # 当前缓冲区文本
        self.buffer = ""

    def reset(self):
        """重置分割器状态"""
        self.buffer = ""

    @abstractmethod
    def _is_sentence_complete(self, text: str) -> bool:
        """
        判断给定文本是否构成一个完整的句子

        Args:
            text: 要检查的文本

        Returns:
            bool: 是否是完整句子
        """
        pass

    @abstractmethod
    def _extract_sentences(self, text: str) -> List[str]:
        """
        从文本中提取所有完整的句子

        Args:
            text: 要处理的文本

        Returns:
            List[str]: 提取出的句子列表
        """
        pass

    def process_chunk(self, text_chunk: str) -> List[str]:
        """
        处理一个文本块，返回所有完整的句子

        Args:
            text_chunk: 新的文本块

        Returns:
            List[str]: 提取出的完整句子列表
        """
        # 添加新文本到缓冲区
        self.buffer += text_chunk

        # 提取完整句子
        sentences = self._extract_sentences(self.buffer)

        # 如果找到句子，更新缓冲区
        if sentences:
            # 计算已处理的文本长度
            processed_length = sum(len(s) for s in sentences)

            # 保留未处理完的部分在缓冲区
            self.buffer = self.buffer[processed_length:]

        return sentences

    def get_remaining(self) -> str:
        """
        获取缓冲区中剩余的未构成完整句子的文本

        Returns:
            str: 缓冲区中的剩余文本
        """
        return self.buffer

    def finish(self) -> List[str]:
        """
        结束处理，返回缓冲区中的所有内容作为最后的句子

        Returns:
            List[str]: 最后的句子列表，如果缓冲区不为空
        """
        remaining = self.buffer.strip()
        self.reset()
        return [remaining] if remaining else []

    def process_stream(self, text_stream: Generator[str, None, None]) -> Generator[str, None, None]:
        """
        处理文本流，返回句子生成器

        Args:
            text_stream: 输入文本流

        Returns:
            Generator[str, None, None]: 句子生成器
        """
        for chunk in text_stream:
            sentences = self.process_chunk(chunk)
            for sentence in sentences:
                yield sentence

        # 处理剩余内容
        for sentence in self.finish():
            yield sentence

    @staticmethod
    def simulate_llm_stream(text: str, chunk_size: int = 5) -> Generator[str, None, None]:
        """
        模拟大语言模型的流式输出，用于测试句子分割器

        Args:
            text: 要分割成流的完整文本
            chunk_size: 每个块的大小（字符数）

        Returns:
            Generator[str, None, None]: 文本块生成器
        """
        for i in range(0, len(text), chunk_size):
            yield text[i:i+chunk_size]



# 使用示例
