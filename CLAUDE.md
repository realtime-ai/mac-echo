# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

MacEcho is a voice assistant that runs entirely on Mac, built with a modular streaming audio processing pipeline:

### Core Components
- **Agent (`src/macecho/agent.py`)**: Main orchestrator that manages the audio processing pipeline
- **VAD (Voice Activity Detection)**: Uses Silero VAD model to detect speech segments
- **ASR (Automatic Speech Recognition)**: SenseVoice model for speech-to-text conversion  
- **LLM (Large Language Model)**: Qwen models via MLX for response generation
- **TTS (Text-to-Speech)**: CosyVoice for speech synthesis
- **Streaming Pipeline**: Real-time audio processing with frame-based processing (32ms frames)

### Configuration System
The project uses Pydantic Settings with hierarchical configuration:
1. Environment variables (MACECHO_ prefix)
2. .env file
3. JSON/YAML config files
4. Default values

All configuration is centralized in `src/macecho/config.py` with separate config classes for each component.

### Audio Processing Flow
1. Audio input → frame buffering (32ms chunks)
2. VAD processing per frame
3. Speech segment detection and accumulation
4. ASR processing on complete speech segments
5. LLM processing for response generation
6. TTS synthesis and audio playback

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_audio_recorder.py

# Test with async support
pytest -v tests/ --asyncio-mode=auto
```

### Code Quality
```bash
# Format code
black src/ tests/ examples/

# Lint code
flake8 src/ tests/ examples/
```

### Running the Application
```bash
# Run basic agent example
cd /path/to/mac-echo
python examples/agent_usage_example.py --mode=demo

# Run long-running agent
python examples/agent_usage_example.py --mode=run

# Test configuration loading
python examples/test_config.py

# Alternatively, install in development mode
pip install -e .
```

## Key Implementation Details

### Audio Frame Processing
- Frame duration: 32ms (configurable via `frame_duration_ms`)
- Frame size calculated as: `frame_duration_ms * sample_rate / 1000`
- Uses `collections.deque` for efficient audio buffering
- VAD processes each frame individually for real-time detection

### Configuration Usage
```python
from macecho.config import MacEchoConfig

# Load with defaults + env vars + .env
config = MacEchoConfig()

# Load from specific file
config = MacEchoConfig.from_file("config.json")

# Access nested settings
sample_rate = config.audio_recording.sample_rate
vad_threshold = config.vad.threshold
```

### Agent Lifecycle
- Agent initializes VAD and ASR processors automatically using config parameters
- VAD processor maps sampling rate from audio_recording config to VAD config
- ASR processor initialized with SenseVoice model and device configuration
- Supports both manual resource management and async context manager
- Proper exception handling for audio processing errors
- Graceful shutdown with cleanup of audio resources, VAD state reset, and ASR resource release
- Signal handling for long-running processes

### Device Support
- Optimized for Apple Silicon (MLX framework)
- CPU/CUDA/MPS device support for models
- PyAudio for cross-platform audio I/O
- Audio device enumeration and selection

## Model Integration Notes

### VAD (Voice Activity Detection)
- Uses `silero_vad.onnx` model in `src/macecho/vad/`
- Threshold-based speech detection with padding
- Frame-by-frame processing for low latency

### ASR (SenseVoice)
- Supports multilingual recognition (auto-detect, en, zh, jp, ko)
- **Async processing**: Uses `asyncio.to_thread` to avoid blocking main thread
- Model caching system to avoid reload overhead  
- Configurable device placement (CPU/GPU/MPS)
- FunASR-based implementation with post-processing
- Default model: `iic/SenseVoiceSmall` from ModelScope
- Automatic model download and caching in `~/.cache/modelscope/`

### LLM (MLX Integration) 
- Qwen model family support via MLX
- Quantized model support (4-bit by default)
- Configurable generation parameters (temperature, max_tokens)
- Model warmup for consistent performance

### TTS (CosyVoice)
- Multiple voice ID support
- Speed control for speech synthesis
- Streaming audio output via queue system