from typing import Dict, Type, Any
from .base import BaseTTS


class TTSFactory:
    """
    TTS工厂类，用于创建和管理不同类型的TTS实现
    """
    
    _tts_classes: Dict[str, Type[BaseTTS]] = {}
    
    @classmethod
    def register(cls, name: str, tts_class: Type[BaseTTS]):
        """
        注册TTS实现类
        
        Args:
            name: TTS实现的名称
            tts_class: TTS实现类
        """
        cls._tts_classes[name] = tts_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseTTS:
        """
        创建TTS实例
        
        Args:
            name: TTS实现的名称
            **kwargs: 传递给TTS构造函数的参数
            
        Returns:
            BaseTTS: TTS实例
            
        Raises:
            ValueError: 当指定的TTS实现不存在时
        """
        if name not in cls._tts_classes:
            available = list(cls._tts_classes.keys())
            raise ValueError(f"TTS实现 '{name}' 不存在。可用的实现: {available}")
        
        tts_class = cls._tts_classes[name]
        return tts_class(**kwargs)
    
    @classmethod
    def get_available_tts(cls) -> list:
        """
        获取所有可用的TTS实现名称
        
        Returns:
            list: 可用的TTS实现名称列表
        """
        return list(cls._tts_classes.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        检查TTS实现是否已注册
        
        Args:
            name: TTS实现的名称
            
        Returns:
            bool: 是否已注册
        """
        return name in cls._tts_classes


def register_tts(name: str):
    """
    装饰器用于自动注册TTS实现
    
    Args:
        name: TTS实现的名称
    """
    def decorator(tts_class: Type[BaseTTS]):
        TTSFactory.register(name, tts_class)
        return tts_class
    return decorator