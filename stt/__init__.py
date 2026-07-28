"""Pluggable speech-to-text engines for WhisprFlow."""

from .base import TranscriptionEngine, TranscriptionResult, WordInfo, float_to_wav_bytes
from .assemblyai_client import AssemblyAIClient
from .local_client import LocalSherpaClient
from .router import EnginePolicy, EngineRouter
from .dictionary import UserDictionary, default_config_dir

__all__ = [
    "TranscriptionEngine",
    "TranscriptionResult",
    "WordInfo",
    "float_to_wav_bytes",
    "AssemblyAIClient",
    "LocalSherpaClient",
    "EngineRouter",
    "EnginePolicy",
    "UserDictionary",
    "default_config_dir",
]
