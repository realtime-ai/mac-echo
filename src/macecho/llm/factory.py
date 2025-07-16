from typing import Dict, Any, List
from .base import BaseLLM, LLMProvider
from .mlx_qwen import MLXQwenChat
from .openai_llm import OpenAILLM


class LLMFactory:
    """LLM工厂类，用于根据配置创建不同的LLM实例"""

    @staticmethod
    def create_llm(config: Dict[str, Any]) -> BaseLLM:
        """
        根据配置创建LLM实例

        Args:
            config: LLM配置字典，包含provider和其他参数

        Returns:
            BaseLLM实例

        Raises:
            ValueError: 不支持的provider
            ImportError: 缺少必要的依赖
        """
        provider = config.get("provider", "mlx").lower()

        # 通用参数
        common_params = {
            "model_name": config.get("model_name", ""),
            "context_enabled": config.get("context_enabled", True),
            "max_context_rounds": config.get("max_context_rounds", 10),
            "context_window_size": config.get("context_window_size", 4000),
            "auto_truncate_context": config.get("auto_truncate_context", True),
            "system_prompt": config.get("system_prompt", "")
        }

        if provider == "mlx":
            return LLMFactory._create_mlx_llm(config, common_params)
        elif provider == "openai":
            return LLMFactory._create_openai_llm(config, common_params)
        elif provider == "custom":
            return LLMFactory._create_openai_llm(config, common_params)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def _create_mlx_llm(config: Dict[str, Any], common_params: Dict[str, Any]) -> MLXQwenChat:
        """创建MLX LLM实例"""
        mlx_params = {
            **common_params,
            "do_warmup": config.get("warmup_enabled", True),
            "device": config.get("mlx_device", "auto")
        }

        # 设置默认模型名称
        if not mlx_params["model_name"]:
            mlx_params["model_name"] = "mlx-community/Qwen2.5-7B-Instruct-4bit"

        return MLXQwenChat(**mlx_params)

    @staticmethod
    def _create_openai_llm(config: Dict[str, Any], common_params: Dict[str, Any]) -> OpenAILLM:
        """创建OpenAI LLM实例"""
        openai_params = {
            **common_params,
            "api_key": config.get("openai_api_key"),
            "base_url": config.get("openai_base_url"),
            "organization": config.get("openai_organization")
        }

        # 设置默认模型名称
        if not openai_params["model_name"]:
            openai_params["model_name"] = "gpt-3.5-turbo"

        return OpenAILLM(**openai_params)

    @staticmethod
    def get_supported_providers() -> List[str]:
        """获取支持的LLM提供商列表"""
        return ["mlx", "openai"]  # "anthropic", "custom" 还未实现

    @staticmethod
    def create_llm_from_config_object(config_obj) -> BaseLLM:
        """
        从配置对象创建LLM实例

        Args:
            config_obj: MacEchoConfig对象或LLMConfig对象

        Returns:
            BaseLLM实例
        """
        # 检查是否是MacEchoConfig对象
        if hasattr(config_obj, 'llm'):
            llm_config = config_obj.llm
        else:
            # 假设是LLMConfig对象
            llm_config = config_obj

        # 转换为字典
        config_dict = {}
        for field_name in llm_config.__fields__:
            config_dict[field_name] = getattr(llm_config, field_name)

        return LLMFactory.create_llm(config_dict)


def create_llm_from_config(config) -> BaseLLM:
    """
    便利函数：从配置创建LLM实例

    Args:
        config: 可以是字典、LLMConfig对象或MacEchoConfig对象

    Returns:
        BaseLLM实例
    """
    if isinstance(config, dict):
        return LLMFactory.create_llm(config)
    else:
        return LLMFactory.create_llm_from_config_object(config)


# --- 使用示例 ---
if __name__ == "__main__":
    # 示例1：使用字典配置创建MLX LLM
    mlx_config = {
        "provider": "mlx",
        "model_name": "mlx-community/Qwen3-4B-8bit",
        "context_enabled": True,
        "max_context_rounds": 5,
        "system_prompt": "你是一个有用的AI助手。"
    }

    try:
        print("=== 创建MLX LLM ===")
        mlx_llm = LLMFactory.create_llm(mlx_config)
        print(f"创建成功: {mlx_llm}")
        print(f"模型信息: {mlx_llm.get_model_info()}")
    except Exception as e:
        print(f"创建MLX LLM失败: {e}")

    # 示例2：使用字典配置创建OpenAI LLM
    openai_config = {
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
        "context_enabled": True,
        "max_context_rounds": 10,
        "system_prompt": "You are a helpful assistant.",
        "openai_api_key": "your-api-key-here"  # 在实际使用中应该从环境变量获取
    }

    try:
        print("\n=== 创建OpenAI LLM ===")
        openai_llm = LLMFactory.create_llm(openai_config)
        print(f"创建成功: {openai_llm}")
        print(f"模型信息: {openai_llm.get_model_info()}")
    except Exception as e:
        print(f"创建OpenAI LLM失败: {e}")

    # 示例3：从MacEcho配置创建LLM
    try:
        from ..config import MacEchoConfig

        print("\n=== 从MacEcho配置创建LLM ===")
        config = MacEchoConfig()
        llm = create_llm_from_config(config)
        print(f"创建成功: {llm}")
        print(f"模型信息: {llm.get_model_info()}")
    except Exception as e:
        print(f"从配置创建LLM失败: {e}")

    print(f"\n支持的提供商: {LLMFactory.get_supported_providers()}")
