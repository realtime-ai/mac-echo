from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
import numpy as np


class AudioRecordingConfig(BaseModel):
    """Audio recording and playback configuration"""
    sample_rate: int = Field(
        default=16000, description="Audio sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")
    chunk_size: int = Field(
        default=1024, description="Audio chunk size for streaming")
    format: str = Field(default="int16", description="Audio format")
    device_index: Optional[int] = Field(
        default=None, description="Audio device index")

    @field_validator('sample_rate')
    @classmethod
    def validate_sample_rate(cls, v):
        if v not in [16000, 48000]:  # 24000 is not supported by pyaudio
            raise ValueError(
                'Sample rate must be one of: 16000, 48000')
        return v

    @field_validator('channels')
    @classmethod
    def validate_channels(cls, v):
        if v < 1 or v > 2:
            raise ValueError('Channels must be 1 (mono) or 2 (stereo)')
        return v

    @property
    def numpy_dtype(self) -> np.dtype:
        """Convert format string to numpy dtype"""
        format_mapping = {
            "int16": np.int16,
            "int32": np.int32,
            "float32": np.float32,
            "float64": np.float64,
            "uint8": np.uint8,
            "uint16": np.uint16,
        }
        if self.format not in format_mapping:
            raise ValueError(f"Unsupported audio format: {self.format}. "
                             f"Supported formats: {list(format_mapping.keys())}")
        return np.dtype(format_mapping[self.format])


class AudioPlayerConfig(BaseModel):
    """Audio recording and playback configuration"""
    sample_rate: int = Field(
        default=24000, description="Audio sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")
    chunk_size: int = Field(
        default=1024, description="Audio chunk size for streaming")
    format: str = Field(default="int16", description="Audio format")
    device_index: Optional[int] = Field(
        default=None, description="Audio device index")

    @field_validator('sample_rate')
    @classmethod
    def validate_sample_rate(cls, v):
        if v not in [16000, 24000, 48000]:
            raise ValueError(
                'Sample rate must be one of: 16000, 24000, 48000')
        return v

    @field_validator('channels')
    @classmethod
    def validate_channels(cls, v):
        if v < 1 or v > 2:
            raise ValueError('Channels must be 1 (mono) or 2 (stereo)')
        return v

    @property
    def numpy_dtype(self) -> np.dtype:
        """Convert dtype string to numpy dtype"""
        format_mapping = {
            "int16": np.int16,
            "int32": np.int32,
            "float32": np.float32,
            "float64": np.float64,
            "uint8": np.uint8,
            "uint16": np.uint16,
        }
        if self.format not in format_mapping:
            raise ValueError(f"Unsupported audio format: {self.format}. "
                             f"Supported formats: {list(format_mapping.keys())}")
        return np.dtype(format_mapping[self.format])


class VADConfig(BaseModel):
    """Voice Activity Detection configuration"""
    threshold: float = Field(default=0.7, ge=0.0, le=1.0,
                             description="VAD detection threshold")
    padding_duration: float = Field(
        default=0.2, ge=0.0, description="Speech padding duration in seconds")
    min_speech_duration: float = Field(
        default=0.1, ge=0.0, description="Minimum speech duration in seconds")
    silence_duration: float = Field(
        default=0.8, ge=0.5, description="Silence duration threshold in seconds")
    per_frame_duration: float = Field(
        default=0.032, gt=0.0, description="Per frame duration in seconds")
    model_path: str = Field(default="silero_vad.onnx",
                            description="Path to VAD model file")


class ASRConfig(BaseModel):
    """Automatic Speech Recognition configuration"""
    model_name: str = Field(default="iic/SenseVoiceSmall",
                            description="ASR model name")
    model_path: Optional[str] = Field(
        default=None, description="Custom model path")
    language: str = Field(
        default="auto", description="Target language for recognition")
    device: str = Field(
        default="cpu", description="Device to run ASR model on")
    max_cache_size: int = Field(
        default=1, ge=1, description="Maximum number of cached models")

    @field_validator('device')
    @classmethod
    def validate_device(cls, v):
        allowed_devices = ["cpu", "cuda", "mps"]
        if v not in allowed_devices:
            raise ValueError(f'Device must be one of: {allowed_devices}')
        return v


class LLMConfig(BaseModel):
    """Large Language Model configuration"""
    # Provider selection
    provider: str = Field(
        default="mlx", description="LLM provider (mlx, openai, anthropic, custom)")

    # Model configuration
    model_name: str = Field(
        default="mlx-community/Qwen3-4B-8bit", description="LLM model name")
    model_path: Optional[str] = Field(
        default=None, description="Custom model path")

    # Generation parameters
    max_tokens: int = Field(
        default=1000, ge=1, description="Maximum tokens to generate")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling parameter")

    # System configuration
    system_prompt: str = Field(
        default="You are a helpful voice assistant. Keep responses concise and conversational.",
        description="System prompt for the LLM"
    )
    
    warmup_enabled: bool = Field(
        default=True, description="Enable model warmup on load")

    # Context management settings
    context_enabled: bool = Field(
        default=True, description="Enable conversation context management")
    max_context_rounds: int = Field(
        default=10, ge=1, le=50, description="Maximum number of conversation rounds to keep in context")
    context_window_size: int = Field(
        default=10000, ge=500, description="Maximum context window size in tokens (approximate)")
    auto_truncate_context: bool = Field(
        default=True, description="Automatically truncate old context when window size is exceeded")

    # Provider-specific settings
    openai_api_key: Optional[str] = Field(
        default=None, description="OpenAI API key")
    openai_base_url: Optional[str] = Field(
        default=None, description="OpenAI API base URL (for custom endpoints)")
    openai_organization: Optional[str] = Field(
        default=None, description="OpenAI organization ID")

    anthropic_api_key: Optional[str] = Field(
        default=None, description="Anthropic API key")

    # MLX-specific settings
    mlx_device: str = Field(
        default="auto", description="Device for MLX models (auto, cpu, mps)")

    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v):
        allowed_providers = ["mlx", "openai", "anthropic", "custom"]
        if v not in allowed_providers:
            raise ValueError(f'Provider must be one of: {allowed_providers}')
        return v

    @field_validator('mlx_device')
    @classmethod
    def validate_mlx_device(cls, v):
        allowed_devices = ["auto", "cpu", "mps"]
        if v not in allowed_devices:
            raise ValueError(f'MLX device must be one of: {allowed_devices}')
        return v


class TTSConfig(BaseModel):
    """Text-to-Speech configuration"""
    model_name: str = Field(default="FunAudioLLM/CosyVoice2-0.5B",
                            description="TTS model name")
    model_path: Optional[str] = Field(
        default=None, description="Custom model path")
    voice_id: str = Field(default="david", description="Voice ID for synthesis")
    speed: float = Field(default=1.0, ge=0.8, le=2.0,
                         description="Speech speed multiplier")
    device: str = Field(
        default="cpu", description="Device to run TTS model on")
    api_key: Optional[str] = Field(
        default=None, description="API key for TTS model")
    base_url: Optional[str] = Field(
        default=None, description="Base URL for TTS model")

    @field_validator('device')
    @classmethod
    def validate_device(cls, v):
        allowed_devices = ["cpu", "cuda", "mps"]
        if v not in allowed_devices:
            raise ValueError(f'Device must be one of: {allowed_devices}')
        return v


class SentencizerConfig(BaseModel):
    """Sentence segmentation configuration"""
    type: str = Field(default="multilingual", description="Sentencizer type")
    languages: List[str] = Field(
        default=["en", "zh"], description="Supported languages")
    min_sentence_length: int = Field(
        default=5, ge=1, description="Minimum sentence length")
    buffer_size: int = Field(default=1000, ge=100,
                             description="Internal buffer size")

    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        allowed_types = ["simple", "multilingual"]
        if v not in allowed_types:
            raise ValueError(
                f'Sentencizer type must be one of: {allowed_types}')
        return v


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_file_size: int = Field(
        default=10485760, description="Maximum log file size in bytes (10MB)")
    backup_count: int = Field(
        default=5, description="Number of backup log files to keep")

    @field_validator('level')
    @classmethod
    def validate_level(cls, v):
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f'Log level must be one of: {allowed_levels}')
        return v.upper()


class MacEchoConfig(BaseSettings):
    """Main MacEcho configuration"""
    # Component configurations
    audio_recording: AudioRecordingConfig = Field(
        default_factory=AudioRecordingConfig)
    audio_player: AudioPlayerConfig = Field(default_factory=AudioPlayerConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    sentencizer: SentencizerConfig = Field(default_factory=SentencizerConfig)

    # System configurations
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Application settings
    app_name: str = Field(default="MacEcho", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")

    model_config = SettingsConfigDict(
        env_prefix="MACECHO_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @classmethod
    def from_env(cls) -> "MacEchoConfig":
        """Load configuration from environment variables"""
        return cls(_env_file=".env")


# Default configuration instance
default_config = MacEchoConfig()
