"""
Common interface for all speech-to-text engines.

Every engine returns a TranscriptionResult. Engines NEVER return a bare ""
to mean "something went wrong" -- that was the bug in the old SenseVoice
client (it discarded the user's speech silently). Empty text plus ok=True
means "the model genuinely heard nothing"; ok=False means "we failed, and
here is why" so the UI can say something useful and offer a retry.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

import numpy as np


@dataclass
class WordInfo:
    """A single recognised word with its confidence and timing (ms)."""
    text: str
    confidence: float = 1.0
    start: int = 0
    end: int = 0


@dataclass
class TranscriptionResult:
    """Uniform result from any engine."""

    text: str = ""
    confidence: float = 1.0
    words: List[WordInfo] = field(default_factory=list)
    language: Optional[str] = None

    engine: str = ""
    model: str = ""
    latency_ms: int = 0
    audio_duration_s: float = 0.0

    ok: bool = True
    error: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def low_confidence_words(self, threshold: float = 0.55) -> List[WordInfo]:
        """Words the engine was unsure about -- fed to the LLM corrector so it
        knows *where* to look instead of rewriting the whole sentence."""
        return [w for w in self.words if w.confidence < threshold]

    @classmethod
    def failure(cls, error: str, engine: str = "", model: str = "") -> "TranscriptionResult":
        return cls(ok=False, error=error, engine=engine, model=model)


class TranscriptionEngine(Protocol):
    """What main.py depends on. Implementations must be swappable."""

    name: str

    def preload(self) -> None:
        """Warm up in the background. Must never raise."""

    async def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        keyterms: Optional[List[str]] = None,
    ) -> TranscriptionResult:
        """Transcribe float32 mono samples in [-1, 1]."""

    async def close(self) -> None:
        """Release resources."""

    def get_info(self) -> dict:
        """Diagnostics for the UI."""


# ── shared helpers ────────────────────────────────────────────────────────


def float_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 mono [-1, 1] to an in-memory 16-bit PCM WAV.

    No temp files: the old code wrote NamedTemporaryFile(delete=False) and
    never cleaned it up, leaking WAVs into %TEMP% on every dictation.
    """
    samples = np.asarray(samples, dtype=np.float32).ravel()

    # Clip rather than wrap on overflow -- np.int16(x * 32767) on an
    # out-of-range value wraps around and produces loud noise bursts.
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()
