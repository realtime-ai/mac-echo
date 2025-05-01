
# 定义一个agent类
class Agent:
    def __init__(self, config: dict):
        self.config = config
        self.asr = None
        self.tts = None
