from mlx_lm import load, stream_generate, generate
from mlx_lm.sample_utils import make_sampler
import time
import torch
from typing import List, Dict, Optional, Union, Generator, Tuple, Any
import uuid
from .base import BaseLLM, LLMProvider, LLMResponse, LLMStreamChunk
from .context_manager import ConversationContextManager


class MLXQwenChat(BaseLLM):
    """基于MLX的Qwen聊天模型实现"""
    
    # 类级变量，用于存储已加载的模型实例
    _loaded_models = {}

    def __init__(self, 
                 model_name: str = "mlx-community/Qwen3-4B-8bit",
                 do_warmup: bool = True,
                 device: str = "auto",
                 **kwargs):
        """
        初始化MLX Qwen聊天模型

        Args:
            model_name: Hugging Face上的模型仓库名称或本地路径
            do_warmup: 是否在加载后进行模型预热
            device: 设备选择 (auto, cpu, mps)
            **kwargs: 传递给基类的其他参数
        """
        super().__init__(model_name=model_name, **kwargs)
        
        self.device = device
        self.do_warmup = do_warmup
        
        # 加载模型
        self.model, self.tokenizer = self._load_model()
        
        # 如果需要，进行模型预热
        if self.do_warmup:
            self._warmup_model()

    @property
    def provider(self) -> LLMProvider:
        """返回LLM提供商类型"""
        return LLMProvider.MLX

    def _load_model(self) -> Tuple:
        """加载模型的具体实现"""
        return self.load_model(self.model_name)

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

    def _warmup_model(self, prompts: Optional[List[str]] = None):
        """
        预热模型，通过运行一些简单的推理来提高后续请求的响应速度

        Args:
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
            chat_prompt = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False,
                enable_thinking=False
            )

            # 只生成少量token，目的是预热
            _ = generate(self.model, self.tokenizer, chat_prompt, max_tokens=10)

        warmup_time = time.time() - start_time
        print(f"模型预热完成，耗时: {warmup_time:.2f}秒")

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
            enable_thinking=False,
            tokenize=False  # 返回字符串格式的prompt
        )

    def _generate_response(self,
                          messages: List[Dict[str, str]],
                          max_tokens: int = 1000,
                          stream: bool = False,
                          temperature: float = 0.7,
                          **kwargs) -> Union[LLMResponse, Generator[LLMStreamChunk, None, None]]:
        """生成响应的具体实现"""
        if not self.validate_messages(messages):
            raise ValueError("Invalid message format")
        
        request_id = f"mlx-{uuid.uuid4()}"
        created_time = int(time.time())

        try:
            prompt = self._format_messages(messages)
        except ValueError as e:
            raise e

        if stream:
            # 流式响应
            return self._generate_stream(request_id, prompt, max_tokens, temperature, **kwargs)
        else:
            # 非流式响应
            return self._generate_complete(request_id, prompt, max_tokens, temperature, **kwargs)

    def _generate_stream(self, request_id: str, prompt: str, max_tokens: int, 
                        temperature: float, **kwargs) -> Generator[LLMStreamChunk, None, None]:
        """生成流式响应"""
        start_time = time.time()
        first_token_time = None

        # Create sampler with temperature
        sampler = make_sampler(temp=temperature)

        streamer = stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )

        # 1. 发送角色块
        yield LLMStreamChunk(
            id=request_id,
            model=self.model_name,
            role="assistant"
        )

        # 2. 逐个发送内容块
        for i, response in enumerate(streamer):
            if i == 0:
                first_token_time = time.time() - start_time

            yield LLMStreamChunk(
                id=request_id,
                model=self.model_name,
                content=response.text
            )

        # 3. 发送结束块
        total_time = time.time() - start_time
        yield LLMStreamChunk(
            id=request_id,
            model=self.model_name,
            finish_reason="stop",
            timing={
                "first_token_time": first_token_time,
                "total_time": total_time
            }
        )

    def _generate_complete(self, request_id: str, prompt: str, max_tokens: int,
                          temperature: float, **kwargs) -> LLMResponse:
        """生成完整响应"""
        start_time = time.time()
        first_token_time = None
        response_text = ""

        # Create sampler with temperature
        sampler = make_sampler(temp=temperature)

        # 使用stream_generate迭代构建完整响应，并获取first_token_time
        for i, response in enumerate(stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )):
            if i == 0:
                first_token_time = time.time() - start_time
            response_text += response.text

        total_time = time.time() - start_time

        return LLMResponse(
            id=request_id,
            model=self.model_name,
            content=response_text,
            finish_reason="stop",
            timing={
                "first_token_time": first_token_time,
                "total_time": total_time
            }
        )

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "provider": self.provider.value,
            "model_name": self.model_name,
            "device": self.device,
            "context_enabled": self.context_enabled,
            "loaded": self.model_name in self._loaded_models,
            "warmup_enabled": self.do_warmup
        }


# --- 使用示例 ---
if __name__ == "__main__":
    try:
        # 创建聊天实例 (启用上下文管理)
        chat_model = MLXQwenChat(
            model_name="mlx-community/Qwen3-4B-8bit",
            context_enabled=True,
            max_context_rounds=5,
            context_window_size=2000,
            system_prompt="你是一个有用的AI助手，请用中文回答问题。"
        )

        print("=== 基于BaseLLM的上下文管理功能演示 ===")
        print(f"模型信息: {chat_model.get_model_info()}")
        
        # 多轮对话演示
        conversation_turns = [
            "你好，我叫小明，今年25岁",
            "我的爱好是什么？",  # 这里AI应该无法回答，因为没有相关信息
            "我喜欢踢足球和看电影",
            "现在你知道我的爱好了吗？",  # 现在AI应该能回答
            "我多大年龄？"  # 测试前面提到的年龄信息
        ]
        
        for i, user_message in enumerate(conversation_turns, 1):
            print(f"\n--- 第{i}轮对话 ---")
            print(f"用户: {user_message}")
            
            # 使用带上下文的对话方法
            response = chat_model.chat_with_context(
                user_message=user_message,
                stream=False,
                max_tokens=200
            )
            
            if response and response.get("choices"):
                assistant_reply = response["choices"][0]["message"]["content"]
                print(f"助手: {assistant_reply}")
            
            # 显示当前上下文状态
            context_summary = chat_model.get_context_summary()
            print(f"上下文状态: {context_summary['total_turns']}轮对话, "
                  f"约{context_summary['estimated_tokens']}个token")
        
        # 最终上下文状态
        final_summary = chat_model.get_context_summary()
        print(f"\n最终上下文状态: {final_summary}")

    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
