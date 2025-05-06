from typing import Optional, Set, List, Generator

from macecho.sentencizer import BaseSentencizer


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


# 测试代码
if __name__ == "__main__":
    # 创建多语言分割器实例
    sentencizer = MultilingualSentencizer()

    print("=== 多语言分割器测试 ===")

    # 基本测试文本
    test_texts = [
        "你好！我是一个测试。这是第三句话？",
        "Hello world. This is a test!",
        "混合语言测试。English and Chinese混合。",
        "带有逗号，分号；冒号：的句子。"
    ]

    print("\n=== 基本句子分割测试 ===")
    for i, text in enumerate(test_texts):
        sentencizer.reset()
        print(f"\n测试 {i+1}: {text}")
        sentences = sentencizer.process_chunk(text)
        print(f"分割结果: {sentences}")
        print(f"缓冲区剩余: {sentencizer.get_remaining()}")

    print("\n=== 引号处理测试 ===")
    quote_tests = [
        "他说：\"这是一个测试\"，然后继续。",
        "带有(括号内容)的句子。",
        "带有[中括号内容]的句子！",
        "带有{大括号内容}的句子？",
        "混合「引号『嵌套』引号」的句子。",
        "\"未闭合的引号",
        "他说：\"这是一段话。这还是同一段话！\""
    ]

    for i, text in enumerate(quote_tests):
        sentencizer.reset()
        print(f"\n引号测试 {i+1}: {text}")
        sentences = sentencizer.process_chunk(text)
        print(f"分割结果: {sentences}")
        print(f"缓冲区剩余: {sentencizer.get_remaining()}")
        # 处理剩余内容
        remaining = sentencizer.finish()
        if remaining:
            print(f"结束处理后剩余: {remaining}")

    print("\n=== 流式处理测试 ===")
    # 流式测试文本，包含复杂引号结构
    stream_text = "这是一个包含「引号内容」的流式测试。这句话有\"英文引号\"！第三句带有(括号内容)？"
    print(f"原始文本: {stream_text}")

    # 使用不同块大小测试
    chunk_sizes = [4, 7, 10]

    for chunk_size in chunk_sizes:
        sentencizer.reset()
        print(f"\n块大小: {chunk_size}字符")

        # 收集结果
        results = []
        for i, sentence in enumerate(sentencizer.process_stream(
                BaseSentencizer.simulate_llm_stream(stream_text, chunk_size))):
            results.append(sentence)
            print(f"  句子 {i+1}: {sentence}")

        print(f"  总共提取: {len(results)}个句子")

    print("\n=== 复杂引号嵌套测试 ===")
    complex_text = "她说：「他告诉我『这是'引用中'的引用』，非常有趣」，然后笑了。"
    sentencizer.reset()

    # 一次处理少量字符，测试复杂引号的处理
    print("\n引号嵌套流式处理:")
    for i in range(0, len(complex_text), 3):
        chunk = complex_text[i:i+3]
        print(f"  输入块: '{chunk}'")
        result = sentencizer.process_chunk(chunk)
        if result:
            print(f"  提取句子: {result}")

    # 处理剩余内容
    remaining = sentencizer.finish()
    if remaining:
        print(f"  最终剩余: {remaining}")

    print("\n=== 引号不匹配测试 ===")
    mismatched_quotes = [
        "这是一个「未闭合的引号。",
        "这是一个(未闭合的括号。",
        "这个『引号』正确闭合，但这个「没有闭合。"
    ]

    for i, text in enumerate(mismatched_quotes):
        sentencizer.reset()
        print(f"\n不匹配测试 {i+1}: {text}")
        sentences = sentencizer.process_chunk(text)
        print(f"分割结果: {sentences}")
        print(f"缓冲区剩余: {sentencizer.get_remaining()}")
        # 结束处理
        remaining = sentencizer.finish()
        print(f"结束处理后: {remaining}")

    print("\n=== 自定义引号对测试 ===")
    # 创建自定义引号对的分割器
    custom_quote_pairs = {
        '<': '>',
        '#': '#',
        '~': '~'
    }
    custom_sentencizer = MultilingualSentencizer(
        quote_pairs=custom_quote_pairs)

    custom_tests = [
        "这是<自定义引号>测试。",
        "这是#单字符引号#测试！",
        "这是~另一种单字符引号~测试？"
    ]

    for i, text in enumerate(custom_tests):
        custom_sentencizer.reset()
        print(f"\n自定义测试 {i+1}: {text}")
        sentences = custom_sentencizer.process_chunk(text)
        print(f"分割结果: {sentences}")
        print(f"缓冲区剩余: {custom_sentencizer.get_remaining()}")
