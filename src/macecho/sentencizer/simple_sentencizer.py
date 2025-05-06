from macecho.sentencizer import BaseSentencizer
from typing import List, Generator


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


# 测试代码
if __name__ == "__main__":
    # 创建SimpleSentencizer实例
    sentencizer = SimpleSentencizer()

    # 测试文本
    test_texts = [
        "你好！我是一个测试。这是第三句话？",
        "Hello world. This is a test!",
        "混合语言测试。English and Chinese混合。",
        "不完整的句子",
        "句子1。句子2！句子3？",
        "带有逗号，分号；冒号：的句子。"
    ]

    print("=== 单块处理测试 ===")
    for i, text in enumerate(test_texts):
        print(f"\n测试 {i+1}: {text}")
        sentences = sentencizer.process_chunk(text)
        print(f"分割结果: {sentences}")
        print(f"缓冲区剩余: {sentencizer.get_remaining()}")
        # 清理缓冲区，准备下一个测试
        sentencizer.reset()

    print("\n=== 流式处理测试 ===")
    # 流式测试文本
    stream_text = "这是一个流式处理测试。我们将文本分成小块进行处理！看看效果如何？"
    print(f"原始文本: {stream_text}")

    # 使用较小的块大小来模拟流式输入
    chunk_sizes = [3, 5, 8]

    for chunk_size in chunk_sizes:
        sentencizer.reset()
        print(f"\n块大小: {chunk_size}字符")

        # 模拟流式输入
        text_stream = BaseSentencizer.simulate_llm_stream(
            stream_text, chunk_size)

        # 收集结果
        results = []
        for i, sentence in enumerate(sentencizer.process_stream(
                BaseSentencizer.simulate_llm_stream(stream_text, chunk_size))):
            results.append(sentence)
            print(f"  句子 {i+1}: {sentence}")

        print(f"  总共提取: {len(results)}个句子")

    print("\n=== 边界情况测试 ===")
    # 测试空输入
    sentencizer.reset()
    print("空字符串测试:", sentencizer.process_chunk(""))

    # 测试只有标点符号
    sentencizer.reset()
    print("只有标点符号:", sentencizer.process_chunk(".!?"))

    # 测试长文本的分段能力
    long_text = "这是第一句。这是第二句！这是第三句？这是第四句。"
    sentencizer.reset()

    # 一次处理两个字符，模拟慢速输入
    print("\n慢速流式输入测试:")
    sentences = []
    for i in range(0, len(long_text), 2):
        chunk = long_text[i:i+2]
        print(f"  输入块: '{chunk}'")
        result = sentencizer.process_chunk(chunk)
        if result:
            sentences.extend(result)
            print(f"  提取句子: {result}")

    # 处理可能剩余的内容
    remaining = sentencizer.finish()
    if remaining:
        sentences.extend(remaining)
        print(f"  剩余内容: {remaining}")

    print(f"  最终结果: {sentences}")
