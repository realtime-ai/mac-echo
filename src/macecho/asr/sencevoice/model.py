import time
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import os

from macecho.asr.base import ASR

model_dir = "iic/SenseVoiceSmall"

model = AutoModel(
    model=model_dir,
    device="mps",
    batch_size=10,
    disable_update=True
)
# 设置测试文件路径

testfile = "test_recording.wav"

# 确保文件存在
if not os.path.exists(testfile):
    raise FileNotFoundError(f"测试文件不存在: {testfile}")

# 读取音频文件
with open(testfile, 'rb') as f:
    audio_data = f.read()


print(time.time())
# 使用音频数据进行识别
res = model.generate(
    input=audio_data,  # 直接使用音频数据
    language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
)


print(time.time())

print("识别结果:")
print(res)

# 后处理文本
text = rich_transcription_postprocess(res[0]["text"])
print("\n处理后的文本:")
print(text)


class SenceVoiceASR(ASR):

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.device = config.get("device", "mps")
        self.batch_size = config.get("batch_size", 10)
        self.disable_update = config.get("disable_update", True)
        self.language = config.get("language", "auto")
        self.use_itn = config.get("use_itn", True)

    def initialize(self):
        self.model = AutoModel(
            model=self.model_dir,
            device=self.device,
            batch_size=self.batch_size,
            disable_update=self.disable_update
        )

    # todo make this async
    def recognize(self, audio_data: bytes) -> str:
        res = self.model.generate(
            input=audio_data,  # 直接使用音频数据
            language=self.language,  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=self.use_itn,
        )
        text = rich_transcription_postprocess(res[0]["text"])
        return text

    def is_ready(self) -> bool:
        return self.model is not None

    def release(self):
        self.model = None
