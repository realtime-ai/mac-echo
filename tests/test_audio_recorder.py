import pytest
import numpy as np
import asyncio
import soundfile as sf
import os
import shutil
from unittest.mock import Mock, patch
from src.macecho.device.device import AudioRecorder


@pytest.mark.asyncio
async def test_recorder_save_to_wav():
    """测试录音并保存为WAV文件"""
    # 设置测试目录和文件路径
    test_dir = "test_output"
    output_file = os.path.join(test_dir, "test_recording.wav")

    # 如果目录存在，先删除它
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    # 创建新的测试目录
    os.makedirs(test_dir)

    # 设置录音参数
    duration = 10.0  # 10秒
    recorder = AudioRecorder()

    # 收集录制的音频数据
    recorded_data = []
    start_time = asyncio.get_event_loop().time()

    # 启动录音
    audio_iterator = recorder.start()

    # 模拟接收音频数据
    async for data in audio_iterator:
        recorded_data.append(np.frombuffer(data, dtype=np.int16))
        # 检查是否达到5秒
        if asyncio.get_event_loop().time() - start_time >= duration:
            break

    # 停止录音
    await recorder.stop()

    # 合并录制的数据
    recorded_audio = np.concatenate(recorded_data)

    # 如果文件已存在，先删除它
    if os.path.exists(output_file):
        os.remove(output_file)

    # 保存为WAV文件
    sf.write(output_file, recorded_audio, recorder.samplerate)

    # 验证文件是否创建成功
    assert os.path.exists(output_file)

    # 读取保存的文件并验证数据
    saved_audio, saved_samplerate = sf.read(output_file)
    assert saved_samplerate == recorder.samplerate
