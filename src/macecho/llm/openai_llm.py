import os
import time
from typing import List, Dict, Optional, Union, Generator, Any
import uuid
from .base import BaseLLM, LLMProvider, LLMResponse, LLMStreamChunk

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class OpenAILLM(BaseLLM):
    """基于OpenAI API的LLM实现"""
    
    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 organization: Optional[str] = None,
                 **kwargs):
        """
        初始化OpenAI LLM
        
        Args:
            model_name: OpenAI模型名称
            api_key: OpenAI API密钥
            base_url: 自定义API基础URL
            organization: OpenAI组织ID
            **kwargs: 传递给基类的其他参数
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required but not installed. Please install it with: pip install openai")
        
        super().__init__(model_name=model_name, **kwargs)
        
        # API配置
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.organization = organization or os.getenv("OPENAI_ORGANIZATION")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # 初始化OpenAI客户端
        self.client = self._load_model()
    
    @property
    def provider(self) -> LLMProvider:
        """返回LLM提供商类型"""
        return LLMProvider.OPENAI
    
    def _load_model(self) -> OpenAI:
        """加载模型的具体实现（对于OpenAI来说就是初始化客户端）"""
        client_kwargs = {
            "api_key": self.api_key
        }
        
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.organization:
            client_kwargs["organization"] = self.organization
            
        return OpenAI(**client_kwargs)
    
    def _generate_response(self,
                          messages: List[Dict[str, str]],
                          max_tokens: int = 1000,
                          temperature: float = 0.7,
                          stream: bool = False,
                          top_p: float = 0.9,
                          **kwargs) -> Union[LLMResponse, Generator[LLMStreamChunk, None, None]]:
        """生成响应的具体实现"""
        if not self.validate_messages(messages):
            raise ValueError("Invalid message format")
        
        request_id = f"openai-{uuid.uuid4()}"
        
        # 准备OpenAI API调用参数
        api_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            **kwargs
        }
        
        start_time = time.time()
        
        try:
            if stream:
                return self._generate_stream_response(request_id, api_kwargs, start_time)
            else:
                return self._generate_complete_response(request_id, api_kwargs, start_time)
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}") from e
    
    def _generate_stream_response(self, request_id: str, api_kwargs: dict, start_time: float) -> Generator[LLMStreamChunk, None, None]:
        """生成流式响应"""
        first_token_time = None
        
        try:
            response_stream = self.client.chat.completions.create(**api_kwargs)
            
            for i, chunk in enumerate(response_stream):
                if i == 0:
                    first_token_time = time.time() - start_time
                
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    # 发送角色信息（第一个chunk通常包含role）
                    if delta.role:
                        yield LLMStreamChunk(
                            id=request_id,
                            model=self.model_name,
                            role=delta.role
                        )
                    
                    # 发送内容
                    if delta.content:
                        yield LLMStreamChunk(
                            id=request_id,
                            model=self.model_name,
                            content=delta.content
                        )
                    
                    # 发送结束标志
                    if choice.finish_reason:
                        total_time = time.time() - start_time
                        yield LLMStreamChunk(
                            id=request_id,
                            model=self.model_name,
                            finish_reason=choice.finish_reason,
                            timing={
                                "first_token_time": first_token_time,
                                "total_time": total_time
                            }
                        )
                        
        except Exception as e:
            # 在流式生成中出错时，发送错误信息
            total_time = time.time() - start_time
            yield LLMStreamChunk(
                id=request_id,
                model=self.model_name,
                finish_reason="error",
                error=str(e),
                timing={"total_time": total_time}
            )
    
    def _generate_complete_response(self, request_id: str, api_kwargs: dict, start_time: float) -> LLMResponse:
        """生成完整响应"""
        response = self.client.chat.completions.create(**api_kwargs)
        total_time = time.time() - start_time
        
        if not response.choices or len(response.choices) == 0:
            raise RuntimeError("OpenAI API returned no choices")
        
        choice = response.choices[0]
        content = choice.message.content or ""
        
        # 构建使用情况信息
        usage = {}
        if hasattr(response, 'usage') and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        
        return LLMResponse(
            id=request_id,
            model=self.model_name,
            content=content,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            timing={
                "total_time": total_time,
                "first_token_time": None  # OpenAI API不提供首token时间
            }
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "provider": self.provider.value,
            "model_name": self.model_name,
            "api_key_set": bool(self.api_key),
            "base_url": self.base_url,
            "organization": self.organization,
            "context_enabled": self.context_enabled,
            "client_initialized": bool(self.client)
        }
    
    def estimate_token_count(self, text: str) -> int:
        """估算文本的token数量（OpenAI特定的估算）"""
        # 对于OpenAI模型，英文大约4个字符1个token，中文大约1.5个字符1个token
        # 这是一个粗略估算，实际token数可能有差异
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        other_chars = len(text) - chinese_chars
        
        estimated_tokens = chinese_chars // 1.5 + other_chars // 4
        return max(1, int(estimated_tokens))


# --- 使用示例 ---
if __name__ == "__main__":
    try:
        # 创建OpenAI聊天实例
        openai_llm = OpenAILLM(
            model_name="gpt-3.5-turbo",
            context_enabled=True,
            max_context_rounds=5,
            system_prompt="You are a helpful assistant."
        )
        
        print("=== OpenAI LLM 功能演示 ===")
        print(f"模型信息: {openai_llm.get_model_info()}")
        
        # 单轮对话测试
        print("\n--- 单轮对话测试 ---")
        response = openai_llm.chat_completions(
            messages=[{"role": "user", "content": "Hello, tell me a joke."}],
            max_tokens=100,
            stream=False
        )
        
        print(f"响应: {response}")
        
        # 多轮上下文对话测试
        print("\n--- 多轮上下文对话测试 ---")
        conversation_turns = [
            "My name is Alice and I'm 25 years old.",
            "What's my name?",
            "How old am I?",
            "What do you know about me?"
        ]
        
        for i, user_message in enumerate(conversation_turns, 1):
            print(f"\n第{i}轮对话:")
            print(f"User: {user_message}")
            
            response = openai_llm.chat_with_context(
                user_message=user_message,
                stream=False,
                max_tokens=100
            )
            
            if response and response.get("choices"):
                assistant_reply = response["choices"][0]["message"]["content"]
                print(f"Assistant: {assistant_reply}")
            
            # 显示上下文状态
            context_summary = openai_llm.get_context_summary()
            print(f"Context: {context_summary['total_turns']} turns, ~{context_summary['estimated_tokens']} tokens")
        
        # 流式对话测试
        print("\n--- 流式对话测试 ---")
        print("User: Tell me a short story about friendship.")
        print("Assistant: ", end="", flush=True)
        
        stream = openai_llm.chat_with_context(
            user_message="Tell me a short story about friendship.",
            stream=True,
            max_tokens=200
        )
        
        for chunk in stream:
            if chunk and chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                content_piece = delta.get("content")
                if content_piece:
                    print(content_piece, end="", flush=True)
        
        print()  # 换行
        
        # 最终状态
        final_summary = openai_llm.get_context_summary()
        print(f"\n最终上下文状态: {final_summary}")
        
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()