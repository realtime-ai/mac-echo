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


class SimpleSentencizer(BaseSentencizer):
    """
    简单的句子分割器实现

    基于标点符号的基本句子分割，适用于大多数语言。
    """

    def _is_sentence_complete(self, text: str) -> bool:
        """判断文本是否是一个完整的句子"""
        if not text:
            return False

        # 检查是否以结束标点结尾
        return text[-1] in self.end_punctuations

    def _extract_sentences(self, text: str) -> List[str]:
        """从文本中提取完整的句子"""
        sentences = []
        start_pos = 0

        for i, char in enumerate(text):
            # 检查是否找到句子结束标点
            if char in self.end_punctuations:
                # 提取句子（包括结束标点）
                sentence = text[start_pos:i+1].strip()
                if sentence:
                    sentences.append(sentence)
                start_pos = i + 1

        return sentences


class MultilingualSentencizer(BaseSentencizer):
    """
    多语言句子分割器

    能够处理多种语言的文本，包括中英文混合文本。
    处理更复杂的句子结构和边界情况。
    """

    def __init__(self,
                 end_punctuations: Optional[Set[str]] = None,
                 pause_punctuations: Optional[Set[str]] = None,
                 quote_pairs: Optional[dict] = None):
        """
        初始化多语言句子分割器

        Args:
            end_punctuations: 结束标点符号集合
            pause_punctuations: 暂停标点符号集合
            quote_pairs: 引号对字典
        """
        super().__init__(end_punctuations, pause_punctuations)

        # 默认引号对
        default_pairs = {
            '(': ')',
            '[': ']',
            '{': '}',
            '"': '"',
            "'": "'"
        }

        # 中文引号对 - 使用Unicode编码
        default_pairs.update({
            '\u201c': '\u201d',  # 中文双引号
            '\u2018': '\u2019',  # 中文单引号
            '\u300c': '\u300d',  # 中文直角引号
            '\u300e': '\u300f',  # 中文书名号
            '\u300a': '\u300b'   # 中文书名号
        })

        self.quote_pairs = quote_pairs if quote_pairs is not None else default_pairs

        # 引号堆栈，用于追踪未闭合的引号
        self.quote_stack = []

    def reset(self):
        """重置分割器状态"""
        super().reset()
        self.quote_stack = []

    def _is_sentence_complete(self, text: str) -> bool:
        """
        判断文本是否构成完整句子，考虑引号配对
        """
        if not text:
            return False

        # 检查引号是否配对
        if self.quote_stack:
            return False

        # 检查是否以结束标点结尾
        return text[-1] in self.end_punctuations

    def _extract_sentences(self, text: str) -> List[str]:
        """从文本中提取完整的句子，处理引号和特殊情况"""
        sentences = []
        start_pos = 0

        i = 0
        while i < len(text):
            char = text[i]

            # 处理引号开始
            if char in self.quote_pairs:
                self.quote_stack.append(char)

            # 处理引号结束
            elif self.quote_stack and char == self.quote_pairs.get(self.quote_stack[-1]):
                self.quote_stack.pop()

            # 检查句子结束条件（在没有未闭合引号的情况下）
            if not self.quote_stack and char in self.end_punctuations:
                # 提取句子（包括结束标点）
                sentence = text[start_pos:i+1].strip()
                if sentence:
                    sentences.append(sentence)
                start_pos = i + 1

            i += 1

        return sentences


# 使用示例
if __name__ == "__main__":
    # 创建一个简单的分割器
    simple_sentencizer = SimpleSentencizer()

    # 创建一个多语言分割器
    multilingual_sentencizer = MultilingualSentencizer()

    # 要处理的测试文本
    test_text = "你好，世界！这是一个测试。This is another test. 引号「测试文本」和(括号测试)都应该正确处理。"

    print("=== 测试文本流模拟 ===")
    # 模拟流式输入
    text_stream = BaseSentencizer.simulate_llm_stream(test_text, chunk_size=8)

    print("\n=== 简单分割器测试 ===")
    # 使用简单分割器处理流
    for i, sentence in enumerate(simple_sentencizer.process_stream(
            BaseSentencizer.simulate_llm_stream(test_text, chunk_size=8))):
        print(f"句子 {i+1}: {sentence}")

    print("\n=== 多语言分割器测试 ===")
    # 使用多语言分割器处理流
    for i, sentence in enumerate(multilingual_sentencizer.process_stream(
            BaseSentencizer.simulate_llm_stream(test_text, chunk_size=8))):
        print(f"句子 {i+1}: {sentence}")

    # 测试引号嵌套
    nested_text = "他说：「这是『引用』内部的引用」，然后继续说。"

    print("\n=== 测试引号嵌套 ===")
    multilingual_sentencizer.reset()
    for i, sentence in enumerate(multilingual_sentencizer.process_stream(
            BaseSentencizer.simulate_llm_stream(nested_text, chunk_size=5))):
        print(f"句子 {i+1}: {sentence}")
