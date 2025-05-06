from mlx_lm import load, stream_generate, generate
import time
import torch
from typing import List, Dict, Optional, Union, Generator, Tuple
import uuid
import json


class MLXQwenChat:
    # 类级变量，用于存储已加载的模型实例
    _loaded_models = {}

    @staticmethod
    def load_model(model_repo: str = "mlx-community/Qwen3-4B-8bit") -> Tuple:
        """
        静态方法：加载模型并返回模型实例和分词器

        Args:
            model_repo: Hugging Face上的模型仓库名称或本地路径

        Returns:
            Tuple: (model, tokenizer)
        """
        # 检查是否已加载
        if model_repo in MLXQwenChat._loaded_models:
            print(f"使用已加载的模型: {model_repo}")
            return MLXQwenChat._loaded_models[model_repo]

        # 加载新模型
        print(f"正在加载模型: {model_repo}...")
        start_load_time = time.time()

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"使用设备: {device}")

        model, tokenizer = load(model_repo)
        load_time = time.time() - start_load_time
        print(f"模型加载完成，耗时: {load_time:.2f}秒")

        # 存储已加载的模型
        MLXQwenChat._loaded_models[model_repo] = (model, tokenizer)

        return model, tokenizer

    @staticmethod
    def warmup_model(model, tokenizer, prompts: List[str] = None):
        """
        静态方法：预热模型，通过运行一些简单的推理来提高后续请求的响应速度

        Args:
            model: 加载的模型实例
            tokenizer: 分词器实例
            prompts: 预热用的提示语列表，如果为None则使用默认提示语
        """
        if prompts is None:
            prompts = [
                "你好，请介绍一下你自己",
            ]

        print("开始模型预热...")
        start_time = time.time()

        for i, prompt in enumerate(prompts):
            print(f"预热请求 {i+1}/{len(prompts)}...")
            messages = [{"role": "user", "content": prompt}]
            chat_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

            # 只生成少量token，目的是预热
            _ = generate(model, tokenizer, chat_prompt, max_tokens=10)

        warmup_time = time.time() - start_time
        print(f"模型预热完成，耗时: {warmup_time:.2f}秒")

    def __init__(self, model_repo: str = "mlx-community/Qwen3-4B-8bit", do_warmup: bool = True):
        """
        初始化MLX Qwen聊天模型

        Args:
            model_repo: Hugging Face上的模型仓库名称或本地路径
            do_warmup: 是否在加载后进行模型预热
        """
        self.model_repo = model_repo

        # 使用静态方法加载模型
        self.model, self.tokenizer = self.load_model(model_repo)

        # 如果需要，进行模型预热
        if do_warmup:
            self.warmup_model(self.model, self.tokenizer)

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        将消息列表格式化为模型输入格式
        """
        # 确保消息列表不为空且最后一条消息是 user
        if not messages or messages[-1]['role'] != 'user':
            raise ValueError("消息列表不能为空，且最后一条消息的角色必须是 'user'")

        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False  # 返回字符串格式的prompt
        )

    def chat_completions(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        stream: bool = True,
        top_p: float = 0.9,
    ) -> Union[Dict, Generator[Dict, None, None]]:
        """
        生成聊天回复，模拟OpenAI的 chat.completions.create 接口

        Args:
            messages: 消息列表，格式如 [{"role": "user", "content": "你好"}]
            max_tokens: 生成的最大token数
            stream: 是否以流式返回
            top_p: 核心采样概率

        Returns:
            如果stream=False，返回一个包含回复的字典 (类似ChatCompletion)
            如果stream=True，返回一个生成器，逐块产生回复 (类似ChatCompletionChunk)
        """
        request_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())

        try:
            prompt = self._format_messages(messages)
        except ValueError as e:
            # 可以返回一个表示错误的结构，或者直接抛出异常
            raise e

        # --- 流式处理 ---
        if stream:
            def stream_generator() -> Generator[Dict, None, None]:
                start_time = time.time()
                first_token_time = None
                accumulated_text = ""

                streamer = stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )

                # 1. 发送角色块
                yield {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": self.model_repo,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": None},
                        "finish_reason": None
                    }]
                }

                # 2. 逐个发送内容块
                for i, chunk in enumerate(streamer):
                    token_text = chunk  # stream_generate 直接返回文本
                    accumulated_text += token_text

                    if i == 0:
                        first_token_time = time.time() - start_time

                    yield {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,  # 保持一致
                        "model": self.model_repo,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token_text},
                            "finish_reason": None  # 中间块没有finish_reason
                        }]
                    }

                # 3. 发送结束块
                yield {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": self.model_repo,
                    "choices": [{
                        "index": 0,
                        "delta": {},  # 结束块delta为空
                        "finish_reason": "stop"  # 或 "length" 如果达到max_tokens
                    }]
                }

                # 发送自定义的计时信息
                total_time = time.time() - start_time
                yield {
                    "id": request_id,
                    "_type": "custom_timing",  # 使用下划线避免与OpenAI字段冲突
                    "first_token_time": first_token_time,
                    "total_time": total_time
                }

            return stream_generator()

        # --- 非流式处理 ---
        else:
            start_time = time.time()
            first_token_time = None
            response_text = ""

            # 使用stream_generate迭代构建完整响应，并获取first_token_time
            for i, chunk in enumerate(stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                top_p=top_p,
            )):
                if i == 0:
                    first_token_time = time.time() - start_time
                response_text += chunk

            total_time = time.time() - start_time

            return {
                "id": request_id,
                "object": "chat.completion",
                "created": created_time,
                "model": self.model_repo,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text,
                        },
                        "finish_reason": "stop",  # 假设正常停止
                    }
                ],
                "usage": {  # 无法精确获取token计数
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                },
                "_timing": {  # 自定义计时信息
                    "first_token_time": first_token_time,
                    "total_time": total_time
                }
            }


# --- 使用示例 ---
if __name__ == "__main__":
    try:
        # 创建聊天实例 (模型会在初始化时加载并预热)
        chat_model = MLXQwenChat()

        # --- 非流式输出示例 ---
        print("\n--- 非流式请求 ---")
        messages_non_stream = [{"role": "user", "content": "你好，请用中文介绍一下你自己"}]
        completion = chat_model.chat_completions(
            messages_non_stream, stream=False, max_tokens=100)

        # 打印类似OpenAI的结构
        print(json.dumps(completion, indent=2, ensure_ascii=False))

        # 单独提取内容和计时
        if completion and completion.get("choices"):
            content = completion["choices"][0]["message"]["content"]
            timing = completion.get("_timing", {})
            print(f"\n回复内容:\n{content}")
            print(f"\n计时: "
                  f"首Token: {timing.get('first_token_time', 'N/A'):.2f}秒, "
                  f"总耗时: {timing.get('total_time', 'N/A'):.2f}秒")
        else:
            print("未能获取有效回复。")

        # --- 流式输出示例 ---
        print("\n--- 流式请求 ---")
        messages_stream = [{"role": "user", "content": "给我讲一个关于太空旅行的短故事"}]
        stream = chat_model.chat_completions(
            messages_stream, stream=True, max_tokens=150)

        full_streamed_content = ""
        stream_timing = {}

        print("回复内容:")
        for chunk in stream:
            # 处理自定义计时块
            if chunk.get("_type") == "custom_timing":
                stream_timing = chunk
                continue  # 不打印计时块本身

            # 处理OpenAI格式的块
            if chunk and chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                content_piece = delta.get("content")
                if content_piece:
                    print(content_piece, end="", flush=True)
                    full_streamed_content += content_piece

        print()  # 换行

        # 打印流式计信息息
        print(f"\n流式计时: "
              f"首Token: {stream_timing.get('first_token_time', 'N/A'):.2f}秒, "
              f"总耗时: {stream_timing.get('total_time', 'N/A'):.2f}秒")

    except Exception as e:
        print(f"\n发生错误: {e}")
