from abc import ABC, abstractmethod
import asyncio
from enum import Enum
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import uuid
import json
import numpy as np


class MessageType(Enum):
    """消息类型枚举"""
    # 系统控制消息
    INTERRUPT = "interrupt"           # 打断消息
    SHUTDOWN = "shutdown"            # 关闭消息
    HEARTBEAT = "heartbeat"          # 心跳消息

    # 音频处理消息
    VAD_START = "vad_start"          # VAD检测到语音开始
    VAD_END = "vad_end"              # VAD检测到语音结束
    ASR_REQUEST = "asr_request"      # ASR识别请求
    ASR_RESPONSE = "asr_response"    # ASR识别结果

    # LLM处理消息
    LLM_REQUEST = "llm_request"      # LLM生成请求
    LLM_RESPONSE = "llm_response"    # LLM生成结果
    LLM_STREAM = "llm_stream"        # LLM流式生成

    # TTS处理消息
    TTS_REQUEST = "tts_request"      # TTS合成请求
    TTS_RESPONSE = "tts_response"    # TTS合成结果

    # 音频播放消息
    AUDIO_PLAY = "audio_play"        # 音频播放请求
    AUDIO_STOP = "audio_stop"        # 音频停止请求

    # 状态消息
    STATUS_UPDATE = "status_update"  # 状态更新
    ERROR = "error"                  # 错误消息

    # 用户交互消息
    USER_INPUT = "user_input"        # 用户输入
    SYSTEM_OUTPUT = "system_output"  # 系统输出


class MessagePriority(Enum):
    """消息优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class BaseMessage(ABC):
    """消息基类"""

    def __init__(self,
                 message_type: MessageType,
                 priority: MessagePriority = MessagePriority.NORMAL,
                 message_id: Optional[str] = None,
                 correlation_id: Optional[str] = None,
                 timestamp: Optional[datetime] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        初始化消息基类

        Args:
            message_type: 消息类型
            priority: 消息优先级
            message_id: 消息唯一ID
            correlation_id: 关联ID（用于跟踪相关消息）
            timestamp: 时间戳
            metadata: 额外元数据
        """
        self.message_type = message_type
        self.priority = priority
        self.message_id = message_id or str(uuid.uuid4())
        self.correlation_id = correlation_id
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}

    @abstractmethod
    def get_payload(self) -> Dict[str, Any]:
        """获取消息载荷"""
        pass

    @abstractmethod
    def from_dict(self, data: Dict[str, Any]) -> 'BaseMessage':
        """从字典创建消息对象"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'priority': self.priority.value,
            'correlation_id': self.correlation_id,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'payload': self.get_payload()
        }

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(type={self.message_type.value}, id={self.message_id[:8]}...)"

    def __repr__(self) -> str:
        return self.__str__()


class InterruptMessage(BaseMessage):
    """打断消息"""

    def __init__(self,
                 reason: str = "user_interrupt",
                 source: str = "user",
                 **kwargs):
        super().__init__(MessageType.INTERRUPT, MessagePriority.CRITICAL, **kwargs)
        self.reason = reason  # 打断原因
        self.source = source  # 打断来源

    def get_payload(self) -> Dict[str, Any]:
        return {
            'reason': self.reason,
            'source': self.source
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InterruptMessage':
        payload = data['payload']
        return cls(
            reason=payload['reason'],
            source=payload['source'],
            message_id=data['message_id'],
            correlation_id=data.get('correlation_id'),
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )


class ASRMessage(BaseMessage):
    """ASR消息"""

    def __init__(self,
                 text: str = "",
                 confidence: float = 0.0,
                 language: str = "auto",
                 audio_duration: float = 0.0,
                 processing_time: float = 0.0,
                 is_request: bool = False,
                 audio_data: Optional[bytes] = None,
                 **kwargs):
        message_type = MessageType.ASR_REQUEST if is_request else MessageType.ASR_RESPONSE
        super().__init__(message_type, **kwargs)
        self.text = text
        self.confidence = confidence
        self.language = language
        self.audio_duration = audio_duration
        self.processing_time = processing_time
        self.audio_data = audio_data  # 音频数据（请求时使用）

    def get_payload(self) -> Dict[str, Any]:
        payload = {
            'text': self.text,
            'confidence': self.confidence,
            'language': self.language,
            'audio_duration': self.audio_duration,
            'processing_time': self.processing_time
        }
        # 不在payload中包含音频数据，避免序列化问题
        if self.audio_data:
            payload['has_audio_data'] = True
            payload['audio_data_size'] = len(self.audio_data)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ASRMessage':
        payload = data['payload']
        return cls(
            text=payload.get('text', ''),
            confidence=payload.get('confidence', 0.0),
            language=payload.get('language', 'auto'),
            audio_duration=payload.get('audio_duration', 0.0),
            processing_time=payload.get('processing_time', 0.0),
            is_request=data['message_type'] == MessageType.ASR_REQUEST.value,
            message_id=data['message_id'],
            correlation_id=data.get('correlation_id'),
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )


class LLMMessage(BaseMessage):
    """LLM消息"""

    def __init__(self,
                 text: str = "",
                 prompt: str = "",
                 model_name: str = "",
                 temperature: float = 0.7,
                 max_tokens: int = 1000,
                 processing_time: float = 0.0,
                 token_count: int = 0,
                 is_request: bool = False,
                 is_stream: bool = False,
                 is_complete: bool = True,
                 # Context-related fields
                 conversation_history: Optional[List[Dict[str, str]]] = None,
                 context_enabled: bool = False,
                 context_rounds_used: int = 0,
                 **kwargs):
        if is_stream:
            message_type = MessageType.LLM_STREAM
        elif is_request:
            message_type = MessageType.LLM_REQUEST
        else:
            message_type = MessageType.LLM_RESPONSE

        super().__init__(message_type, **kwargs)
        self.text = text
        self.prompt = prompt
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.processing_time = processing_time
        self.token_count = token_count
        self.is_complete = is_complete  # 对于流式响应，标记是否完成
        
        # Context management fields
        self.conversation_history = conversation_history or []
        self.context_enabled = context_enabled
        self.context_rounds_used = context_rounds_used

    def get_payload(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'prompt': self.prompt,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'processing_time': self.processing_time,
            'token_count': self.token_count,
            'is_complete': self.is_complete,
            'context_enabled': self.context_enabled,
            'context_rounds_used': self.context_rounds_used,
            'conversation_history': self.conversation_history
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMMessage':
        payload = data['payload']
        message_type = MessageType(data['message_type'])

        return cls(
            text=payload.get('text', ''),
            prompt=payload.get('prompt', ''),
            model_name=payload.get('model_name', ''),
            temperature=payload.get('temperature', 0.7),
            max_tokens=payload.get('max_tokens', 1000),
            processing_time=payload.get('processing_time', 0.0),
            token_count=payload.get('token_count', 0),
            is_request=message_type == MessageType.LLM_REQUEST,
            is_stream=message_type == MessageType.LLM_STREAM,
            is_complete=payload.get('is_complete', True),
            conversation_history=payload.get('conversation_history', []),
            context_enabled=payload.get('context_enabled', False),
            context_rounds_used=payload.get('context_rounds_used', 0),
            message_id=data['message_id'],
            correlation_id=data.get('correlation_id'),
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )


class TTSMessage(BaseMessage):
    """TTS消息"""

    def __init__(self,
                 text: str = "",
                 voice_id: str = "",
                 speed: float = 1.0,
                 audio_format: str = "wav",
                 sample_rate: int = 24000,
                 processing_time: float = 0.0,
                 audio_duration: float = 0.0,
                 is_request: bool = False,
                 audio_data: Optional[bytes] = None,
                 **kwargs):
        message_type = MessageType.TTS_REQUEST if is_request else MessageType.TTS_RESPONSE
        super().__init__(message_type, **kwargs)
        self.text = text
        self.voice_id = voice_id
        self.speed = speed
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.processing_time = processing_time
        self.audio_duration = audio_duration
        self.audio_data = audio_data

    def get_payload(self) -> Dict[str, Any]:
        payload = {
            'text': self.text,
            'voice_id': self.voice_id,
            'speed': self.speed,
            'audio_format': self.audio_format,
            'sample_rate': self.sample_rate,
            'processing_time': self.processing_time,
            'audio_duration': self.audio_duration
        }
        # 不在payload中包含音频数据
        if self.audio_data:
            payload['has_audio_data'] = True
            payload['audio_data_size'] = len(self.audio_data)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TTSMessage':
        payload = data['payload']
        return cls(
            text=payload.get('text', ''),
            voice_id=payload.get('voice_id', ''),
            speed=payload.get('speed', 1.0),
            audio_format=payload.get('audio_format', 'wav'),
            sample_rate=payload.get('sample_rate', 24000),
            processing_time=payload.get('processing_time', 0.0),
            audio_duration=payload.get('audio_duration', 0.0),
            is_request=data['message_type'] == MessageType.TTS_REQUEST.value,
            message_id=data['message_id'],
            correlation_id=data.get('correlation_id'),
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )


class VADMessage(BaseMessage):
    """VAD消息"""

    def __init__(self,
                 is_speech_start: bool = True,
                 confidence: float = 0.0,
                 audio_duration: float = 0.0,
                 segment_samples: int = 0,
                 **kwargs):
        message_type = MessageType.VAD_START if is_speech_start else MessageType.VAD_END
        super().__init__(message_type, **kwargs)
        self.confidence = confidence
        self.audio_duration = audio_duration
        self.segment_samples = segment_samples

    def get_payload(self) -> Dict[str, Any]:
        return {
            'confidence': self.confidence,
            'audio_duration': self.audio_duration,
            'segment_samples': self.segment_samples
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VADMessage':
        payload = data['payload']
        return cls(
            is_speech_start=data['message_type'] == MessageType.VAD_START.value,
            confidence=payload.get('confidence', 0.0),
            audio_duration=payload.get('audio_duration', 0.0),
            segment_samples=payload.get('segment_samples', 0),
            message_id=data['message_id'],
            correlation_id=data.get('correlation_id'),
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )


class StatusMessage(BaseMessage):
    """状态消息"""

    def __init__(self,
                 component: str = "",
                 status: str = "",
                 details: str = "",
                 metrics: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__(MessageType.STATUS_UPDATE, **kwargs)
        self.component = component  # 组件名称（如 "asr", "llm", "tts"）
        self.status = status       # 状态（如 "ready", "processing", "error"）
        self.details = details     # 详细信息
        self.metrics = metrics or {}  # 性能指标

    def get_payload(self) -> Dict[str, Any]:
        return {
            'component': self.component,
            'status': self.status,
            'details': self.details,
            'metrics': self.metrics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StatusMessage':
        payload = data['payload']
        return cls(
            component=payload.get('component', ''),
            status=payload.get('status', ''),
            details=payload.get('details', ''),
            metrics=payload.get('metrics', {}),
            message_id=data['message_id'],
            correlation_id=data.get('correlation_id'),
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )


class ErrorMessage(BaseMessage):
    """错误消息"""

    def __init__(self,
                 error_type: str = "",
                 error_message: str = "",
                 component: str = "",
                 stack_trace: str = "",
                 recoverable: bool = True,
                 **kwargs):
        super().__init__(MessageType.ERROR, MessagePriority.HIGH, **kwargs)
        self.error_type = error_type
        self.error_message = error_message
        self.component = component
        self.stack_trace = stack_trace
        self.recoverable = recoverable

    def get_payload(self) -> Dict[str, Any]:
        return {
            'error_type': self.error_type,
            'error_message': self.error_message,
            'component': self.component,
            'stack_trace': self.stack_trace,
            'recoverable': self.recoverable
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorMessage':
        payload = data['payload']
        return cls(
            error_type=payload.get('error_type', ''),
            error_message=payload.get('error_message', ''),
            component=payload.get('component', ''),
            stack_trace=payload.get('stack_trace', ''),
            recoverable=payload.get('recoverable', True),
            message_id=data['message_id'],
            correlation_id=data.get('correlation_id'),
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )


class MessageFactory:
    """消息工厂类"""

    _message_classes = {
        MessageType.INTERRUPT: InterruptMessage,
        MessageType.ASR_REQUEST: ASRMessage,
        MessageType.ASR_RESPONSE: ASRMessage,
        MessageType.LLM_REQUEST: LLMMessage,
        MessageType.LLM_RESPONSE: LLMMessage,
        MessageType.LLM_STREAM: LLMMessage,
        MessageType.TTS_REQUEST: TTSMessage,
        MessageType.TTS_RESPONSE: TTSMessage,
        MessageType.VAD_START: VADMessage,
        MessageType.VAD_END: VADMessage,
        MessageType.STATUS_UPDATE: StatusMessage,
        MessageType.ERROR: ErrorMessage,
    }

    @classmethod
    def create_from_dict(cls, data: Dict[str, Any]) -> BaseMessage:
        """从字典创建消息对象"""
        message_type = MessageType(data['message_type'])
        message_class = cls._message_classes.get(message_type)

        if not message_class:
            raise ValueError(f"Unsupported message type: {message_type}")

        return message_class.from_dict(data)

    @classmethod
    def create_from_json(cls, json_str: str) -> BaseMessage:
        """从JSON字符串创建消息对象"""
        data = json.loads(json_str)
        return cls.create_from_dict(data)


# 便利函数
def create_interrupt_message(reason: str = "user_interrupt", source: str = "user") -> InterruptMessage:
    """创建打断消息"""
    return InterruptMessage(reason=reason, source=source)


def create_asr_request(audio_data: bytes, language: str = "auto", correlation_id: str = None) -> ASRMessage:
    """创建ASR请求消息"""
    return ASRMessage(
        audio_data=audio_data,
        language=language,
        is_request=True,
        correlation_id=correlation_id
    )


def create_asr_response(text: str, confidence: float, processing_time: float, correlation_id: str = None) -> ASRMessage:
    """创建ASR响应消息"""
    return ASRMessage(
        text=text,
        confidence=confidence,
        processing_time=processing_time,
        is_request=False,
        correlation_id=correlation_id
    )


def create_llm_request(prompt: str, 
                      model_name: str = "", 
                      correlation_id: str = None,
                      conversation_history: Optional[List[Dict[str, str]]] = None,
                      context_enabled: bool = False) -> LLMMessage:
    """创建LLM请求消息"""
    return LLMMessage(
        prompt=prompt,
        model_name=model_name,
        is_request=True,
        correlation_id=correlation_id,
        conversation_history=conversation_history,
        context_enabled=context_enabled
    )


def create_llm_response(text: str, 
                       processing_time: float, 
                       token_count: int = 0, 
                       correlation_id: str = None,
                       context_rounds_used: int = 0,
                       context_enabled: bool = False) -> LLMMessage:
    """创建LLM响应消息"""
    return LLMMessage(
        text=text,
        processing_time=processing_time,
        token_count=token_count,
        is_request=False,
        correlation_id=correlation_id,
        context_rounds_used=context_rounds_used,
        context_enabled=context_enabled
    )


def create_tts_request(text: str, voice_id: str = "", correlation_id: str = None) -> TTSMessage:
    """创建TTS请求消息"""
    return TTSMessage(
        text=text,
        voice_id=voice_id,
        is_request=True,
        correlation_id=correlation_id
    )


def create_tts_response(audio_data: bytes, audio_duration: float, processing_time: float, correlation_id: str = None) -> TTSMessage:
    """创建TTS响应消息"""
    return TTSMessage(
        audio_data=audio_data,
        audio_duration=audio_duration,
        processing_time=processing_time,
        is_request=False,
        correlation_id=correlation_id
    )


def create_vad_message(is_speech_start: bool, confidence: float = 0.0, audio_duration: float = 0.0) -> VADMessage:
    """创建VAD消息"""
    return VADMessage(
        is_speech_start=is_speech_start,
        confidence=confidence,
        audio_duration=audio_duration
    )


def create_status_message(component: str, status: str, details: str = "", metrics: Dict[str, Any] = None) -> StatusMessage:
    """创建状态消息"""
    return StatusMessage(
        component=component,
        status=status,
        details=details,
        metrics=metrics
    )


def create_error_message(error_type: str, error_message: str, component: str = "", recoverable: bool = True) -> ErrorMessage:
    """创建错误消息"""
    return ErrorMessage(
        error_type=error_type,
        error_message=error_message,
        component=component,
        recoverable=recoverable
    )


class MessageQueue:
    """简单的消息队列实现"""

    def __init__(self):
        self.queue = asyncio.Queue()
        self.subscribers = {}

    async def publish(self, message):
        """发布消息"""
        await self.queue.put(message)
        print(f"📨 Published: {message}")

    async def subscribe(self, message_type: MessageType, callback):
        """订阅特定类型的消息"""
        if message_type not in self.subscribers:
            self.subscribers[message_type] = []
        self.subscribers[message_type].append(callback)

    async def process_messages(self):
        """处理消息队列"""
        while True:
            try:
                message = await self.queue.get()

                # 根据优先级处理
                if message.priority == MessagePriority.CRITICAL:
                    print(f"🚨 CRITICAL: {message}")

                # 调用订阅者
                subscribers = self.subscribers.get(message.message_type, [])
                for callback in subscribers:
                    try:
                        await callback(message)
                    except Exception as e:
                        print(f"❌ Error in subscriber: {e}")

                self.queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Error processing message: {e}")
