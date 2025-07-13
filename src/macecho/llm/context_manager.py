from typing import List, Dict, Any, Optional
from collections import deque
import time
import logging
from dataclasses import dataclass
from .prompt import SYSTEM_FORMAT_PROMPT


@dataclass
class ConversationTurn:
    """单轮对话记录"""
    user_message: str
    assistant_message: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'user_message': self.user_message,
            'assistant_message': self.assistant_message,
            'timestamp': self.timestamp,
            'metadata': self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """从字典创建对话轮次"""
        return cls(
            user_message=data['user_message'],
            assistant_message=data['assistant_message'],
            timestamp=data['timestamp'],
            metadata=data.get('metadata')
        )


class ConversationContextManager:
    """对话上下文管理器

    负责管理多轮对话的上下文，包括：
    - 对话历史存储
    - 上下文长度控制
    - 自动截断策略
    """

    def __init__(self,
                 max_rounds: int = 20,
                 max_context_tokens: int = 4000,
                 auto_truncate: bool = True,
                 system_prompt: str = ""):
        """
        初始化上下文管理器

        Args:
            max_rounds: 最大保留的对话轮数
            max_context_tokens: 最大上下文token数量(近似)
            auto_truncate: 是否自动截断超长上下文
            system_prompt: 系统提示词
        """
        self.max_rounds = max_rounds
        self.max_context_tokens = max_context_tokens
        self.auto_truncate = auto_truncate
        self.system_prompt = system_prompt

        # 使用deque实现固定长度的对话历史
        self.conversation_history: deque = deque(maxlen=max_rounds)
        self.logger = logging.getLogger(__name__)

    def add_conversation_turn(self, user_message: str, assistant_message: str, metadata: Optional[Dict[str, Any]] = None):
        """添加一轮对话到历史记录"""
        turn = ConversationTurn(
            user_message=user_message,
            assistant_message=assistant_message,
            timestamp=time.time(),
            metadata=metadata
        )

        self.conversation_history.append(turn)
        self.logger.debug(
            f"Added conversation turn. Total turns: {len(self.conversation_history)}")

        # 如果启用自动截断，检查并处理超长上下文
        if self.auto_truncate:
            self._truncate_if_needed()

    def get_conversation_messages(self) -> List[Dict[str, str]]:
        """获取格式化的对话消息列表，用于LLM输入

        Returns:
            格式化的消息列表，包含system、user、assistant消息
        """
        messages = []

        # 添加系统提示词
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
            messages.append(
                {"role": "system", "content": SYSTEM_FORMAT_PROMPT})

        # 添加历史对话
        for turn in self.conversation_history:
            messages.append({"role": "user", "content": turn.user_message})
            messages.append(
                {"role": "assistant", "content": turn.assistant_message})

        return messages

    def get_context_for_new_message(self, new_user_message: str) -> List[Dict[str, str]]:
        """为新的用户消息准备上下文

        Args:
            new_user_message: 新的用户消息

        Returns:
            包含历史上下文和新消息的完整消息列表
        """
        messages = self.get_conversation_messages()
        messages.append({"role": "user", "content": new_user_message})
        return messages

    def _truncate_if_needed(self):
        """根据token限制截断上下文"""
        if not self.conversation_history:
            return

        # 简单的token估算：中文字符*1.5 + 英文单词*1
        # 这是一个粗略估算，实际token数可能有差异
        total_tokens = self._estimate_total_tokens()

        while total_tokens > self.max_context_tokens and len(self.conversation_history) > 0:
            # 移除最早的对话轮次
            removed_turn = self.conversation_history.popleft()
            total_tokens = self._estimate_total_tokens()
            self.logger.info(f"Truncated conversation turn from {removed_turn.timestamp}. "
                             f"Remaining turns: {len(self.conversation_history)}, "
                             f"Estimated tokens: {total_tokens}")

    def _estimate_total_tokens(self) -> int:
        """估算当前上下文的总token数"""
        total_chars = len(self.system_prompt)

        for turn in self.conversation_history:
            total_chars += len(turn.user_message) + len(turn.assistant_message)

        # 粗略估算：平均每2个字符约等于1个token
        # 对于中英文混合文本，这是一个合理的估算
        estimated_tokens = total_chars // 2
        return estimated_tokens

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history.clear()
        self.logger.info("Conversation history cleared")

    def get_history_summary(self) -> Dict[str, Any]:
        """获取历史记录摘要信息"""
        if not self.conversation_history:
            return {
                'total_turns': 0,
                'estimated_tokens': len(self.system_prompt) // 2,
                'oldest_timestamp': None,
                'latest_timestamp': None
            }

        return {
            'total_turns': len(self.conversation_history),
            'estimated_tokens': self._estimate_total_tokens(),
            'oldest_timestamp': self.conversation_history[0].timestamp,
            'latest_timestamp': self.conversation_history[-1].timestamp
        }

    def export_history(self) -> List[Dict[str, Any]]:
        """导出对话历史为字典列表"""
        return [turn.to_dict() for turn in self.conversation_history]

    def import_history(self, history_data: List[Dict[str, Any]]):
        """从字典列表导入对话历史"""
        self.conversation_history.clear()
        for turn_data in history_data:
            turn = ConversationTurn.from_dict(turn_data)
            self.conversation_history.append(turn)

        self.logger.info(f"Imported {len(history_data)} conversation turns")

        # 导入后可能需要截断
        if self.auto_truncate:
            self._truncate_if_needed()

    def update_system_prompt(self, new_system_prompt: str):
        """更新系统提示词"""
        self.system_prompt = new_system_prompt
        self.logger.info("System prompt updated")

        # 更新后重新检查是否需要截断
        if self.auto_truncate:
            self._truncate_if_needed()

    def get_last_n_turns(self, n: int) -> List[ConversationTurn]:
        """获取最近N轮对话"""
        return list(self.conversation_history)[-n:] if n > 0 else []

    def __len__(self) -> int:
        """返回当前对话轮数"""
        return len(self.conversation_history)

    def __str__(self) -> str:
        """字符串表示"""
        summary = self.get_history_summary()
        return f"ConversationContextManager(turns={summary['total_turns']}, tokens≈{summary['estimated_tokens']})"
