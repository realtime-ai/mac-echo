import time
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import os
import logging
from typing import Generator, Union, Any, Dict, Optional, Tuple
import asyncio

from macecho.asr.base import BaseASR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MacEcho.ASR.SenceVoice")

# 测试代码放到 if __name__ == "__main__" 块中，避免在导入时执行
# 删除这里的全局变量和测试代码


class SenceVoiceASR(BaseASR):
    """SenceVoice语音识别实现"""

    # 类级变量用于缓存已加载的模型
    _loaded_models = {}

    @staticmethod
    def load_model(model_dir: str, device: str = "mps", batch_size: int = 10,
                   disable_update: bool = True) -> Tuple[Any, bool]:
        """
        静态方法：加载SenceVoice模型并缓存

        Args:
            model_dir: 模型目录
            device: 设备类型，如 'mps', 'cpu', 'cuda'
            batch_size: 批处理大小
            disable_update: 是否禁用更新

        Returns:
            Tuple[Any, bool]: (模型实例, 是否成功加载)
        """
        # 生成模型的唯一缓存键
        cache_key = f"{model_dir}_{device}_{batch_size}_{disable_update}"

        # 检查缓存中是否已有该模型
        if cache_key in SenceVoiceASR._loaded_models:
            logger.info(f"使用缓存的模型: {model_dir}")
            return SenceVoiceASR._loaded_models[cache_key], True

        # 加载新模型
        try:
            logger.info(f"正在加载模型: {model_dir}, 设备: {device}")
            start_time = time.time()

            model = AutoModel(
                model=model_dir,
                device=device,
                batch_size=batch_size,
                disable_update=disable_update
            )

            load_time = time.time() - start_time
            logger.info(f"模型加载完成，耗时: {load_time:.2f}秒")

            # 缓存模型
            SenceVoiceASR._loaded_models[cache_key] = model

            return model, True

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return None, False

    def __init__(self, language: str = "zh", sample_rate: int = 16000, **kwargs):
        """
        初始化SenceVoice ASR

        Args:
            language: 语种代码，如 'zh', 'en'
            sample_rate: 音频采样率
            **kwargs: 其他配置参数，如：
                - model_dir: 模型目录，默认为"iic/SenseVoiceSmall"
                - device: 设备，默认为"mps"
                - batch_size: 批处理大小，默认为10
                - disable_update: 是否禁用更新，默认为True
                - use_itn: 是否使用ITN，默认为True
        """
        # 调用父类初始化方法
        super().__init__(language, sample_rate, **kwargs)

        # 设置特定于SenceVoice的配置
        self.model_dir = self.get_config("model_dir", "iic/SenseVoiceSmall")
        self.device = self.get_config("device", "mps")
        self.batch_size = self.get_config("batch_size", 10)
        self.disable_update = self.get_config("disable_update", True)
        self.use_itn = self.get_config("use_itn", True)

        # 设置状态变量
        self.model = None
        self.ready = False
        self.logger = logger

        # 自动初始化模型
        if self.get_config("auto_initialize", True):
            self.initialize()

    def initialize(self) -> bool:
        """
        初始化ASR模型

        Returns:
            bool: 初始化是否成功
        """
        try:
            self.logger.info(f"正在初始化SenceVoice ASR，模型路径：{self.model_dir}")

            # 使用静态方法加载模型
            self.model, success = self.load_model(
                model_dir=self.model_dir,
                device=self.device,
                batch_size=self.batch_size,
                disable_update=self.disable_update
            )

            if success:
                self.ready = True
                return True
            else:
                self.logger.error(f"使用静态方法加载模型失败")
                self.ready = False
                return False

        except Exception as e:
            self.logger.error(f"SenceVoice ASR初始化失败: {e}")
            self.ready = False
            return False

    async def transcribe(self, audio: Union[bytes, str]) -> str:
        """
        异步识别一段音频。

        Args:
            audio: 音频数据（bytes）或文件路径（str）

        Returns:
            str: 识别文本

        Raises:
            RuntimeError: 如果模型未初始化
            ValueError: 如果音频格式不正确
        """
        if not self.is_ready():
            raise RuntimeError("SenceVoice ASR模型未初始化")

        try:
            # 处理输入
            audio_data = self._prepare_audio(audio)

            # 记录开始时间
            self.logger.debug(f"开始异步识别，音频数据大小: {len(audio_data)} 字节")
            start_time = time.time()

            # 在线程池中异步执行识别，避免阻塞主线程
            def _sync_transcribe():
                res = self.model.generate(
                    input=audio_data,
                    language=self.language,
                    use_itn=self.use_itn,
                )
                return rich_transcription_postprocess(res[0]["text"])

            # 使用 asyncio.to_thread 在线程池中执行同步代码
            text = await asyncio.to_thread(_sync_transcribe)

            # 记录结果
            process_time = time.time() - start_time
            self.logger.debug(f"异步识别完成，耗时: {process_time:.2f}秒，结果: {text}")

            return text
        except Exception as e:
            self.logger.error(f"异步识别失败: {e}")
            return ""

    def stream_transcribe(self, audio_stream: Generator[bytes, None, None]) -> Generator[str, None, None]:
        """
        流式识别，将音频块转换为逐步输出的文本结果。

        Args:
            audio_stream: 音频块生成器

        Returns:
            Generator[str, None, None]: 文本结果生成器

        Raises:
            RuntimeError: 如果模型未初始化
        """
        if not self.is_ready():
            raise RuntimeError("SenceVoice ASR模型未初始化")

        # 由于SenceVoice不直接支持流式识别，我们将缓冲音频数据，达到一定长度后进行识别
        buffer = bytearray()
        buffer_threshold = self.get_config(
            "buffer_threshold", 4000)  # 默认约250ms@16kHz

        for audio_chunk in audio_stream:
            if not audio_chunk:
                continue

            # 添加到缓冲区
            buffer.extend(audio_chunk)

            # 当缓冲区达到阈值时进行识别
            if len(buffer) >= buffer_threshold:
                try:
                    # 识别当前缓冲区
                    res = self.model.generate(
                        input=bytes(buffer),
                        language=self.language,
                        use_itn=self.use_itn,
                    )

                    # 处理结果
                    text = rich_transcription_postprocess(res[0]["text"])
                    if text.strip():
                        yield text

                    # 清空缓冲区
                    buffer = bytearray()
                except Exception as e:
                    self.logger.error(f"流式识别出错: {e}")
                    buffer = bytearray()  # 出错时清空缓冲区

        # 处理剩余的音频数据
        if buffer:
            try:
                res = self.model.generate(
                    input=bytes(buffer),
                    language=self.language,
                    use_itn=self.use_itn,
                )
                text = rich_transcription_postprocess(res[0]["text"])
                if text.strip():
                    yield text
            except Exception as e:
                self.logger.error(f"处理剩余音频数据失败: {e}")

    def is_ready(self) -> bool:
        """
        检查ASR模型是否已准备好

        Returns:
            bool: 模型是否已加载并准备好
        """
        return self.ready and self.model is not None

    def release(self) -> bool:
        """
        释放模型资源

        Returns:
            bool: 释放是否成功
        """
        try:
            self.logger.info("正在释放SenceVoice ASR资源")
            # 注意：不会从缓存中删除模型，只是将当前实例的引用置为None
            self.model = None
            self.ready = False
            return True
        except Exception as e:
            self.logger.error(f"释放资源失败: {e}")
            return False

    @classmethod
    def clear_model_cache(cls) -> bool:
        """
        清除模型缓存

        Returns:
            bool: 清除是否成功
        """
        try:
            logger.info("正在清除模型缓存")
            cls._loaded_models.clear()
            return True
        except Exception as e:
            logger.error(f"清除模型缓存失败: {e}")
            return False

    def _prepare_audio(self, audio: Union[bytes, str]) -> bytes:
        """
        准备音频数据

        Args:
            audio: 音频数据（bytes）或文件路径（str）

        Returns:
            bytes: 处理后的音频数据

        Raises:
            ValueError: 如果音频格式不正确
            FileNotFoundError: 如果音频文件不存在
        """
        if isinstance(audio, bytes):
            return audio
        elif isinstance(audio, str):
            # 假设是文件路径
            if not os.path.exists(audio):
                raise FileNotFoundError(f"音频文件不存在: {audio}")
            with open(audio, 'rb') as f:
                return f.read()
        else:
            raise ValueError(f"不支持的音频格式: {type(audio)}")

    # 兼容旧接口，重命名为transcribe（已弃用，请使用async transcribe）
    async def recognize(self, audio_data: bytes) -> str:
        """
        异步识别接口（为兼容旧代码保留，建议使用transcribe）

        Args:
            audio_data: 原始音频数据（字节格式）

        Returns:
            str: 识别结果文本
        """
        return await self.transcribe(audio_data)


# 创建默认实例用于测试
if __name__ == "__main__":
    # 测试静态模型加载
    print("\n--- 测试静态模型加载 ---")
    model, success = SenceVoiceASR.load_model("iic/SenseVoiceSmall")
    if success:
        print("静态模型加载成功")
    else:
        print("静态模型加载失败")
        exit(1)

    # 创建ASR实例
    print("\n--- 创建ASR实例 ---")
    asr = SenceVoiceASR()

    # 测试文件路径
    testfile = "test_recording.wav"

    # 确保文件存在
    if not os.path.exists(testfile):
        print(f"测试文件不存在: {testfile}")
        exit(1)

    # 测试同步识别 -> 异步识别
    print("\n--- 测试异步识别 ---")
    
    async def test_async_transcribe():
        start_time = time.time()
        text = await asr.transcribe(testfile)
        process_time = time.time() - start_time
        print(f"异步识别完成，耗时: {process_time:.2f}秒")
        print(f"识别结果: {text}")
    
    asyncio.run(test_async_transcribe())

    # 测试流式识别
    print("\n--- 测试流式识别 ---")

    def simulate_audio_stream(file_path, chunk_size=1600):
        """模拟音频流，按块读取文件"""
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    # 创建模拟的音频流
    stream = simulate_audio_stream(testfile)

    # 测试流式识别
    start_time = time.time()
    results = []
    for text in asr.stream_transcribe(stream):
        results.append(text)
        print(f"流式部分结果: {text}")

    process_time = time.time() - start_time
    print(f"流式识别完成，耗时: {process_time:.2f}秒")
    print(f"最终结果: {''.join(results)}")

    # 释放资源
    asr.release()

    # 清除模型缓存
    print("\n--- 清除模型缓存 ---")
    if SenceVoiceASR.clear_model_cache():
        print("模型缓存清除成功")
