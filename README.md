# MacEcho
A voice assistant that runs completely on your Mac.

## Features

- **Voice Activity Detection (VAD)** - Automatically detect when you're speaking
- **Automatic Speech Recognition (ASR)** - Convert speech to text using SenseVoice
- **Large Language Model (LLM)** - Powered by Qwen models via MLX
- **Text-to-Speech (TTS)** - Convert responses back to speech using CosyVoice
- **Streaming Processing** - Real-time audio processing pipeline
- **Configurable** - Extensive configuration system using Pydantic Settings

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/mac-echo.git
cd mac-echo
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

MacEcho uses [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for configuration management, providing multiple ways to configure the application:

### Configuration Priority (highest to lowest)

1. **Explicit parameters** when creating config objects
2. **Environment variables** with `MACECHO_` prefix
3. **`.env` file** in the project root
4. **Configuration files** (JSON/YAML)
5. **Default values**

### Environment Variables

Set environment variables with the `MACECHO_` prefix. Use double underscores (`__`) to access nested settings:

```bash
# Application settings
export MACECHO_DEBUG=true
export MACECHO_APP_NAME="My Voice Assistant"

# Audio configuration
export MACECHO_AUDIO__SAMPLE_RATE=44100
export MACECHO_AUDIO__CHANNELS=1

# VAD settings
export MACECHO_VAD__THRESHOLD=0.8
export MACECHO_VAD__PADDING_DURATION=0.3

# LLM configuration
export MACECHO_LLM__MODEL_NAME="mlx-community/Qwen2.5-14B-Instruct-4bit"
export MACECHO_LLM__MAX_TOKENS=1500
export MACECHO_LLM__TEMPERATURE=0.7
```

### .env File

Create a `.env` file in the project root (see `examples/config_env_example.txt` for a complete template):

```env
# MacEcho Configuration
MACECHO_DEBUG=false
MACECHO_AUDIO__SAMPLE_RATE=16000
MACECHO_VAD__THRESHOLD=0.7
MACECHO_LLM__MODEL_NAME=mlx-community/Qwen2.5-7B-Instruct-4bit
```

### Configuration Files

#### JSON Configuration

```json
{
  "app_name": "MacEcho",
  "debug": false,
  "audio": {
    "sample_rate": 16000,
    "channels": 1
  },
  "vad": {
    "threshold": 0.7,
    "padding_duration": 0.2
  },
  "llm": {
    "model_name": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "max_tokens": 1000,
    "temperature": 0.7
  }
}
```

#### YAML Configuration

```yaml
app_name: MacEcho
debug: false
audio:
  sample_rate: 16000
  channels: 1
vad:
  threshold: 0.7
  padding_duration: 0.2
llm:
  model_name: mlx-community/Qwen2.5-7B-Instruct-4bit
  max_tokens: 1000
  temperature: 0.7
```

### Using Configuration in Code

```python
from macecho.config import get_config, MacEchoConfig

# Load configuration automatically (env vars + .env + defaults)
config = get_config()

# Or load from specific file
config = MacEchoConfig.from_file("my_config.json")

# Or create with specific settings
config = MacEchoConfig(
    debug=True,
    audio={"sample_rate": 44100},
    llm={"temperature": 0.5}
)

# Access configuration values
print(f"Sample rate: {config.audio.sample_rate}")
print(f"VAD threshold: {config.vad.threshold}")
print(f"LLM model: {config.llm.model_name}")

# Create required directories
config.create_directories()
```

## Configuration Sections

### Audio Settings
- `sample_rate`: Audio sample rate (8000, 16000, 22050, 44100, 48000)
- `channels`: Number of audio channels (1 or 2)
- `chunk_size`: Audio chunk size for streaming
- `device_index`: Audio device index (optional)

### VAD (Voice Activity Detection)
- `threshold`: Speech detection threshold (0.0-1.0)
- `padding_duration`: Pre-speech padding in seconds
- `min_speech_duration`: Minimum speech segment length
- `silence_duration`: Silence duration to end speech segment

### ASR (Automatic Speech Recognition)
- `model_name`: SenseVoice model name
- `language`: Target language ("auto", "en", "zh", etc.)
- `device`: Processing device ("cpu", "cuda", "mps")

### LLM (Large Language Model)
- `model_name`: Model name or path
- `max_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature (0.0-2.0)
- `system_prompt`: System prompt for the assistant

### TTS (Text-to-Speech)
- `model_name`: CosyVoice model name
- `voice_id`: Voice ID for synthesis
- `speed`: Speech speed multiplier
- `device`: Processing device

### Storage
- `data_dir`: Base data directory
- `models_dir`: Model cache directory
- `logs_dir`: Log files directory
- `temp_dir`: Temporary files directory

## Testing Configuration

Run the configuration test script:

```bash
python examples/test_config.py
```

This will test:
- Default configuration loading
- Environment variable overrides
- JSON file configuration
- Configuration validation
- Directory creation
- Model cache directories

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/ examples/
flake8 src/ tests/ examples/
```

## License

MIT License - see LICENSE file for details. 
