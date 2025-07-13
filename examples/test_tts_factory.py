#!/usr/bin/env python3

"""
TTS Factory Pattern 测试示例
"""

import sys
import os

# Add src to path for imports
# fmt: off
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from macecho.tts import TTSFactory, CosyVoiceTTS, KorotoTTS


def test_factory_basic():
    """测试基本工厂功能"""
    print("=== 测试TTS工厂基本功能 ===")

    # 查看可用的TTS实现
    print(f"可用的TTS实现: {TTSFactory.get_available_tts()}")

    # 检查注册状态
    print(f"cosyvoice是否已注册: {TTSFactory.is_registered('cosyvoice')}")
    print(f"koroto是否已注册: {TTSFactory.is_registered('koroto')}")
    print(f"unknown是否已注册: {TTSFactory.is_registered('unknown')}")

def test_factory_create_cosyvoice():
    """测试创建CosyVoice TTS实例"""
    print("\n=== 测试创建CosyVoice TTS实例 ===")

    try:
        # 通过工厂创建实例
        tts = TTSFactory.create("cosyvoice", voice="alice", sample_rate=24000)
        print(f"创建成功: {type(tts).__name__}")
        print(f"语音: {tts.voice}")
        print(f"采样率: {tts.sample_rate}")

        # 直接创建实例进行对比
        direct_tts = CosyVoiceTTS(voice="alice", sample_rate=24000)
        print(f"直接创建: {type(direct_tts).__name__}")

        # 验证类型一致
        assert type(tts) == type(direct_tts)
        print("✓ 工厂创建的实例类型正确")

    except Exception as e:
        print(f"创建CosyVoice TTS失败: {e}")

def test_factory_create_koroto():
    """测试创建Koroto TTS实例"""
    print("\n=== 测试创建Koroto TTS实例 ===")

    try:
        # 通过工厂创建实例
        tts = TTSFactory.create("koroto", voice="default")
        print(f"创建成功: {type(tts).__name__}")
        print(f"语音: {tts.voice}")

        # 直接创建实例进行对比
        direct_tts = KorotoTTS(voice="default")
        print(f"直接创建: {type(direct_tts).__name__}")

        # 验证类型一致
        assert type(tts) == type(direct_tts)
        print("✓ 工厂创建的实例类型正确")

    except Exception as e:
        print(f"创建Koroto TTS失败: {e}")

def test_factory_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")

    try:
        # 尝试创建不存在的TTS实现
        tts = TTSFactory.create("nonexistent")
        print("❌ 应该抛出异常但没有")
    except ValueError as e:
        print(f"✓ 正确捕获异常: {e}")
    except Exception as e:
        print(f"❌ 捕获了意外的异常类型: {e}")

def test_tts_functionality():
    """测试TTS基本功能"""
    print("\n=== 测试TTS基本功能 ===")

    try:
        # 创建CosyVoice实例
        tts = TTSFactory.create("cosyvoice", voice="david")

        # 测试配置方法
        tts.set_voice("alice")
        tts.set_sample_rate(24000)
        tts.set_config(api_key="test_key", model="test_model")

        print(f"✓ 语音设置: {tts.voice}")
        print(f"✓ 采样率设置: {tts.sample_rate}")
        print(f"✓ API密钥配置: {tts.get_config('api_key')}")
        print(f"✓ 模型配置: {tts.get_config('model')}")

    except Exception as e:
        print(f"TTS功能测试失败: {e}")

if __name__ == "__main__":
    test_factory_basic()
    test_factory_create_cosyvoice()
    test_factory_create_koroto()
    test_factory_error_handling()
    test_tts_functionality()

    print("\n=== 所有测试完成 ===")