from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Generator, Any, Tuple
import time
import uuid
from enum import Enum
from .context_manager import ConversationContextManager


class LLMProvider(Enum):
    """LLM提供商枚举"""
    MLX = "mlx"
    OPENAI = "openai"
    CUSTOM = "custom"

    
class LLMResponse:
    """标准化的LLM响应格式"""
    
    def __init__(self,
                 id: str,
                 model: str,
                 content: str,
                 finish_reason: str = "stop",
                 usage: Optional[Dict[str, int]] = None,
                 created: Optional[int] = None,
                 timing: Optional[Dict[str, float]] = None,
                 **kwargs):
        self.id = id
        self.model = model
        self.content = content
        self.finish_reason = finish_reason
        self.usage = usage or {}
        self.created = created or int(time.time())
        self.timing = timing or {}
        self.extra = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": self.content
                },
                "finish_reason": self.finish_reason
            }],
            "usage": self.usage,
            "timing": self.timing,
            **self.extra
        }


class LLMStreamChunk:
    """流式响应的数据块"""
    
    def __init__(self,
                 id: str,
                 model: str,
                 content: Optional[str] = None,
                 role: Optional[str] = None,
                 finish_reason: Optional[str] = None,
                 created: Optional[int] = None,
                 **kwargs):
        self.id = id
        self.model = model
        self.content = content
        self.role = role
        self.finish_reason = finish_reason
        self.created = created or int(time.time())
        self.extra = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        delta = {}
        if self.role:
            delta["role"] = self.role
        if self.content is not None:
            delta["content"] = self.content
        
        return {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": self.finish_reason
            }],
            **self.extra
        }


class BaseLLM(ABC):
    """LLM基类，定义统一的接口"""
    
    def __init__(self,
                 model_name: str,
                 context_enabled: bool = True,
                 max_context_rounds: int = 10,
                 context_window_size: int = 4000,
                 auto_truncate_context: bool = True,
                 system_prompt: str = "",
                 **kwargs):
        """
        初始化LLM基类
        
        Args:
            model_name: 模型名称
            context_enabled: 是否启用上下文管理
            max_context_rounds: 最大上下文轮数
            context_window_size: 上下文窗口大小
            auto_truncate_context: 是否自动截断超长上下文
            system_prompt: 系统提示词
            **kwargs: 其他参数
        """
        self.model_name = model_name
        self.context_enabled = context_enabled
        self.system_prompt = system_prompt
        
        # 初始化上下文管理器
        if self.context_enabled:
            self.context_manager = ConversationContextManager(
                max_rounds=max_context_rounds,
                max_context_tokens=context_window_size,
                auto_truncate=auto_truncate_context,
                system_prompt=system_prompt
            )
        else:
            self.context_manager = None
    
    @property
    @abstractmethod
    def provider(self) -> LLMProvider:
        """返回LLM提供商类型"""
        pass
    
    @abstractmethod
    def _load_model(self) -> Any:
        """加载模型的具体实现"""
        pass
    
    @abstractmethod
    def _generate_response(self,
                          messages: List[Dict[str, str]],
                          max_tokens: int = 1000,
                          temperature: float = 0.7,
                          stream: bool = False,
                          **kwargs) -> Union[LLMResponse, Generator[LLMStreamChunk, None, None]]:
        """生成响应的具体实现"""
        pass
    
    def chat_completions(self,
                        messages: List[Dict[str, str]],
                        max_tokens: int = 1000,
                        temperature: float = 0.7,
                        stream: bool = False,
                        **kwargs) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        标准的聊天完成接口
        
        Args:
            messages: 消息列表
            max_tokens: 最大生成token数
            temperature: 采样温度
            stream: 是否流式返回
            **kwargs: 其他参数
            
        Returns:
            响应字典或流式生成器
        """
        result = self._generate_response(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            **kwargs
        )
        
        if stream:
            # 流式响应：转换每个chunk为字典格式
            def stream_wrapper():
                for chunk in result:
                    yield chunk.to_dict()
            return stream_wrapper()
        else:
            # 非流式响应：转换为字典格式
            return result.to_dict()
    
    def chat_with_context(self,
                         user_message: str,
                         max_tokens: int = 1000,
                         temperature: float = 0.7,
                         stream: bool = False,
                         **kwargs) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        使用上下文进行对话
        
        Args:
            user_message: 用户消息
            max_tokens: 最大生成token数
            temperature: 采样温度
            stream: 是否流式返回
            **kwargs: 其他参数
            
        Returns:
            响应字典或流式生成器
        """
        if not self.context_enabled or not self.context_manager:
            # 如果未启用上下文，回退到普通对话
            messages = [{"role": "user", "content": user_message}]
            return self.chat_completions(messages, max_tokens, temperature, stream, **kwargs)
        
        # 获取包含历史上下文的完整消息列表
        messages = self.context_manager.get_context_for_new_message(user_message)
        
        # 调用底层的chat_completions方法
        result = self.chat_completions(messages, max_tokens, temperature, stream, **kwargs)
        
        if stream:
            return self._handle_stream_with_context(user_message, result)
        else:
            # 非流式：提取回复并更新上下文
            if isinstance(result, dict) and result.get("choices"):
                assistant_message = result["choices"][0]["message"]["content"]
                self.context_manager.add_conversation_turn(user_message, assistant_message)
            return result
    
    def _handle_stream_with_context(self, user_message: str, stream_generator: Generator) -> Generator:
        """处理流式响应并更新上下文"""
        accumulated_response = ""
        
        for chunk in stream_generator:
            # 累积assistant的回复内容
            if chunk and chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                content_piece = delta.get("content")
                if content_piece:
                    accumulated_response += content_piece
            
            yield chunk
        
        # 流式结束后，将完整的对话轮次添加到上下文
        if accumulated_response.strip():
            self.context_manager.add_conversation_turn(user_message, accumulated_response.strip())
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """获取对话历史"""
        if not self.context_enabled or not self.context_manager:
            return []
        return self.context_manager.export_history()
    
    def clear_conversation_history(self):
        """清空对话历史"""
        if self.context_enabled and self.context_manager:
            self.context_manager.clear_history()
    
    def get_context_summary(self) -> Dict[str, Any]:
        """获取上下文摘要信息"""
        if not self.context_enabled or not self.context_manager:
            return {"context_enabled": False}
        
        summary = self.context_manager.get_history_summary()
        summary["context_enabled"] = True
        summary["max_rounds"] = self.context_manager.max_rounds
        summary["max_context_tokens"] = self.context_manager.max_context_tokens
        summary["auto_truncate"] = self.context_manager.auto_truncate
        summary["provider"] = self.provider.value
        return summary
    
    def update_system_prompt(self, new_system_prompt: str):
        """更新系统提示词"""
        self.system_prompt = new_system_prompt
        if self.context_enabled and self.context_manager:
            self.context_manager.update_system_prompt(new_system_prompt)
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        pass
    
    def validate_messages(self, messages: List[Dict[str, str]]) -> bool:
        """验证消息格式"""
        if not messages:
            return False
        
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg or "content" not in msg:
                return False
            if msg["role"] not in ["system", "user", "assistant"]:
                return False
        
        return True
    
    def estimate_token_count(self, text: str) -> int:
        """估算文本的token数量（粗略估算）"""
        # 简单的token估算：中英文混合文本平均每2个字符约1个token
        return len(text) // 2
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, provider={self.provider.value})"
    
    def __repr__(self) -> str:
        return self.__str__()