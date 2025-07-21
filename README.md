# MacEcho

🎙️ A high-performance, privacy-focused voice assistant that runs entirely on your Mac, optimized for Apple Silicon.

<p align="center">
  <strong>✨ Real-time Voice AI • 🔒 100% Local • ⚡ Apple Silicon Optimized</strong>
</p>

## Overview

MacEcho is a sophisticated voice assistant built from the ground up for macOS, leveraging the power of Apple Silicon's Neural Engine and unified memory architecture. It provides real-time voice interaction with complete privacy - all processing happens locally on your device.

## 🚀 Quick Start

### Prerequisites

- macOS 12.0 or later (Apple Silicon recommended)
- Python 3.9+
- Homebrew (for audio dependencies)

### Installation

```bash
# Install audio dependencies
brew install portaudio

# Clone the repository
git clone https://github.com/realtime-ai/mac-echo.git
cd mac-echo

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Run Your First Assistant

```bash
# Basic voice assistant (runs continuously)
python examples/agent_usage_example.py --mode=run

# Demo mode (30-second demo)
python examples/agent_usage_example.py --mode=demo

# Custom configuration
MACECHO_LLM__MODEL_NAME="mlx-community/Qwen2.5-7B-Instruct-4bit" python examples/agent_usage_example.py
```

## ✨ Key Features

### 🎯 Core Capabilities
- **100% Local Processing** - Complete privacy with no cloud dependencies
- **Apple Silicon Optimized** - Leverages MLX framework for maximum performance on M-series chips
- **Real-time Streaming** - Sub-second response times with streaming audio pipeline
- **Multilingual Support** - Automatic language detection (English, Chinese, Japanese, Korean)
- **Context-Aware Conversations** - Maintains conversation history across interactions

### 🔊 Voice Processing Pipeline
- **Voice Activity Detection (VAD)** - Silero VAD for accurate speech detection with configurable thresholds
- **Speech Recognition (ASR)** - SenseVoice model with excellent accuracy and language auto-detection
- **Neural Language Models** - Qwen model family via MLX with 4-bit quantization support
- **Text-to-Speech (TTS)** - CosyVoice for natural-sounding speech synthesis with multiple voices

### ⚡ Advanced Architecture
- **Event-Driven Messaging** - Asynchronous message passing with priority queues
- **Modular Pipeline Design** - Easily swap or extend components
- **Streaming Sentencizer** - Real-time sentence boundary detection for immediate TTS
- **Interrupt Handling** - Graceful interruption during speech generation
- **Frame-Based Processing** - 32ms audio frames for ultra-low latency
- **Comprehensive Configuration** - Flexible Pydantic-based settings management

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Audio In  │ ──> │     VAD     │ ──> │     ASR     │ ──> │     LLM     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                     │
                                                                     v
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Audio Out  │ <── │     TTS     │ <── │ Sentencizer │ <── │   Response  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘

                          ▲                       │
                          └───── Message Bus ─────┘
                               (Event-Driven)
```

## ⚙️ Configuration

MacEcho uses a hierarchical configuration system with multiple sources:

### Configuration Priority (highest to lowest)
1. **Command-line arguments**
2. **Environment variables** (MACECHO_ prefix)
3. **`.env` file**
4. **Configuration files** (JSON/YAML)
5. **Default values**

### Quick Configuration

Create a `.env` file in the project root:

```bash
# Core Settings
MACECHO_DEBUG=false
MACECHO_APP_NAME="My Assistant"

# Audio Configuration
MACECHO_AUDIO_RECORDING__SAMPLE_RATE=16000
MACECHO_AUDIO_RECORDING__CHANNELS=1

# Model Selection
MACECHO_LLM__MODEL_NAME="mlx-community/Qwen2.5-7B-Instruct-4bit"
MACECHO_LLM__MAX_TOKENS=1000
MACECHO_LLM__TEMPERATURE=0.7

# Voice Settings
MACECHO_VAD__THRESHOLD=0.7
MACECHO_TTS__VOICE_ID="中文女"
```

See [Configuration Guide](docs/configuration.md) for detailed options.

## 🎮 Usage Examples

### Basic Voice Assistant
```python
from macecho.agent import Agent
from macecho.config import get_config

# Load configuration
config = get_config()

# Create and run agent
agent = Agent(config)
agent.run()
```

### Custom LLM Configuration
```python
from macecho.llm import MLXQwenChat

# Create chat model with context
chat = MLXQwenChat(
    model_name="mlx-community/Qwen2.5-14B-Instruct-4bit",
    context_enabled=True,
    max_context_rounds=10,
    system_prompt="You are a helpful coding assistant."
)

# Stream response
response = chat.chat_with_context(
    user_message="How do I sort a list in Python?",
    stream=True
)
```

### Message System Example
```python
from macecho.message import MessageQueue, MessageType, MessagePriority

# Create message queue
queue = MessageQueue()

# Subscribe to ASR messages
@queue.subscribe(MessageType.ASR)
async def handle_transcription(message):
    print(f"Transcribed: {message.data['text']}")

# Send high-priority message
await queue.send_message(
    MessageType.LLM,
    {"text": "Process this urgently"},
    priority=MessagePriority.HIGH
)
```

## 📊 Performance

On Apple Silicon (M1/M2/M3):
- **First response**: < 1 second
- **VAD latency**: < 50ms per frame
- **ASR processing**: ~200ms for 3-second audio
- **LLM token generation**: 20-50 tokens/second (model dependent)
- **TTS synthesis**: Real-time factor < 0.3



## 🔧 Troubleshooting

### Common Issues

**Audio Input Not Working**
```bash
# List audio devices
python -c "import pyaudio; p=pyaudio.PyAudio(); print([p.get_device_info_by_index(i)['name'] for i in range(p.get_device_count())])"

# Set specific device
export MACECHO_AUDIO_RECORDING__DEVICE_INDEX=1
```

**Model Download Issues**
```bash
# Models are cached in ~/.cache/modelscope/
# Clear cache if corrupted
rm -rf ~/.cache/modelscope/hub/iic/SenseVoiceSmall

# Set custom model directory
export MACECHO_STORAGE__MODELS_DIR=/path/to/models
```

**Memory Issues**
- Use smaller quantized models (4-bit recommended)
- Reduce context window size
- Disable model warmup for testing

## 🤝 Contributing

We welcome contributions! Areas of interest:

- 🎯 Additional language model support
- 🌍 More languages for ASR/TTS
- 🔊 Alternative TTS engines
- 🧪 Test coverage improvements
- 📚 Documentation enhancements

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

MacEcho builds upon excellent open-source projects:
- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) - Speech recognition
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - Text-to-speech
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice activity detection

---

<p align="center">
  Made with ❤️ for the Mac community
</p>