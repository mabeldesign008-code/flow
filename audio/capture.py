"""
Always-on audio capture with pre-roll.

Audit finding 1.7: the old recorder opened the PortAudio stream *after* a
blocking 100 ms winsound.Beep, so the mic went live 150-400 ms after the
user pressed the hotkey. Users start talking immediately, so every single
utterance lost its onset -- "delete" became "lete", "can't" became "ant".

Fix: the input stream stays open for the life of the app, continuously
filling a ring buffer. On hotkey we snapshot the last 500 ms of that ring
and prepend it to the recording. The onset is already captured before the
user's finger has left the key.

Also fixed here:
  1.10 captures at 16 kHz mono explicitly (was device default 44.1/48 kHz
       for a 16 kHz model, with a silent resample nobody measured)
  1.11 the audio callback does exactly one thing -- copy into the ring.
       The old one took a lock, sorted a 50-element list, computed a
       median and called logger.warning(), any of which can block and
       cause an xrun, which drops frames, which raises WER.
  1.15 the buffer is bounded in seconds, not queue entries. The old
       MAX_BUFFER_SIZE = 0 (infinite) with a 10-hour cap meant ~14 GB
       resident before the truncation check ever ran.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000       # what every ASR model wants
CHANNELS = 1
BLOCKSIZE = 512           # ~32 ms; also Silero VAD's window size
PREROLL_SECONDS = 0.5     # captured before the hotkey is even processed
MAX_RECORDING_SECONDS = 1800   # 30 min; ~115 MB at 16 kHz float32


@dataclass
class DeviceInfo:
    index: int
    name: str
    channels: int
    sample_rate: float
    is_default: bool


class AudioCapture:
    """Continuous 16 kHz mono capture with a pre-roll ring buffer."""

    def __init__(
        self,
        device: Optional[int] = None,
        preroll_seconds: float = PREROLL_SECONDS,
        max_seconds: float = MAX_RECORDING_SECONDS,
    ):
        self.device = device
        self.sample_rate = SAMPLE_RATE
        self.max_seconds = max_seconds

        # Ring buffer: pre-allocated, never grows, never allocates in the
        # callback. Writes wrap around; we only ever read the last N samples.
        self._ring_size = max(int(preroll_seconds * SAMPLE_RATE), BLOCKSIZE * 2)
        self._ring = np.zeros(self._ring_size, dtype=np.float32)
        self._ring_pos = 0
        self._ring_filled = False

        # Active recording buffer. Plain list of blocks; concatenated once
        # at stop. Bounded by _max_samples.
        self._blocks: List[np.ndarray] = []
        self._recorded_samples = 0
        self._max_samples = int(max_seconds * SAMPLE_RATE)

        self._is_recording = False
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.RLock()

        # Level metering for the UI. Written in the callback (a single
        # float store, which is atomic in CPython), read by the renderer.
        self._level = 0.0
        self._peak = 0.0

        self._overflows = 0
        self._truncated = False
        self._last_error: Optional[str] = None
        self._running = False

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start_stream(self) -> bool:
        """Open the mic and keep it open. Call once at app start."""
        with self._lock:
            if self._running:
                return True
            try:
                self._stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="float32",
                    device=self.device,
                    blocksize=BLOCKSIZE,
                    callback=self._callback,
                    latency="low",
                )
                self._stream.start()
                self._running = True
                self._last_error = None
                logger.info("Audio stream open: %d Hz mono, %d-frame blocks",
                            SAMPLE_RATE, BLOCKSIZE)
                return True
            except Exception as e:
                self._last_error = str(e)
                self._stream = None
                self._running = False
                logger.error("Failed to open audio stream: %s", e)
                return False

    def stop_stream(self) -> None:
        with self._lock:
            self._is_recording = False
            self._running = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.error("Error closing stream: %s", e)
                finally:
                    self._stream = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    # ── the callback: keep it trivial ─────────────────────────────────────

    def _callback(self, indata, frames, time_info, status):
        """Runs on the PortAudio thread. No locks, no logging, no allocation
        beyond one copy. Anything slower risks an xrun and dropped audio."""
        if status and status.input_overflow:
            self._overflows += 1

        mono = indata[:, 0] if indata.ndim > 1 else indata

        # Write into the ring (may wrap).
        n = len(mono)
        pos = self._ring_pos
        end = pos + n
        if end <= self._ring_size:
            self._ring[pos:end] = mono
        else:
            split = self._ring_size - pos
            self._ring[pos:] = mono[:split]
            self._ring[: end - self._ring_size] = mono[split:]
            self._ring_filled = True
        self._ring_pos = end % self._ring_size
        if end >= self._ring_size:
            self._ring_filled = True

        # If recording, also append to the take (bounded).
        if self._is_recording:
            if self._recorded_samples < self._max_samples:
                self._blocks.append(mono.copy())
                self._recorded_samples += n
            else:
                self._truncated = True

        # Cheap metering. abs().max() on 512 floats is ~1 microsecond.
        peak = float(np.abs(mono).max())
        self._level = peak
        if peak > self._peak:
            self._peak = peak

    # ── recording ─────────────────────────────────────────────────────────

    def begin(self) -> bool:
        """Start a take, seeded with the pre-roll. Returns False if the
        stream is not running."""
        with self._lock:
            if not self._running:
                return False
            if self._is_recording:
                return True

            self._blocks = []
            preroll = self._read_preroll()
            if len(preroll):
                self._blocks.append(preroll)
            self._recorded_samples = len(preroll)

            self._peak = 0.0
            self._truncated = False
            self._overflows = 0
            self._is_recording = True
            return True

    def _read_preroll(self) -> np.ndarray:
        """Copy the ring in chronological order."""
        pos = self._ring_pos
        if not self._ring_filled:
            return self._ring[:pos].copy()
        return np.concatenate((self._ring[pos:], self._ring[:pos]))

    def end(self) -> Optional[np.ndarray]:
        """Stop the take and return float32 mono @ 16 kHz, or None if empty."""
        with self._lock:
            if not self._is_recording:
                return None
            self._is_recording = False
            blocks, self._blocks = self._blocks, []

        if not blocks:
            return None
        audio = np.concatenate(blocks)
        return audio if len(audio) else None

    def tail_since(self, offset: int) -> Optional[np.ndarray]:
        """Return recorded samples past `offset`, for live streaming.

        Lets the streaming uploader drain the take while it is still being
        recorded, without disturbing the buffer the batch path will use.
        """
        with self._lock:
            if not self._blocks:
                return None
            total = sum(len(b) for b in self._blocks)
            if offset >= total:
                return None
            audio = np.concatenate(self._blocks) if len(self._blocks) > 1 else self._blocks[0]
            return audio[offset:].copy()

    def discard(self) -> None:
        """Abandon the take without returning anything."""
        with self._lock:
            self._is_recording = False
            self._blocks = []
            self._recorded_samples = 0

    # ── state for the UI ──────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def get_level(self) -> float:
        """Instantaneous peak in [0, 1]."""
        return self._level

    def get_stats(self) -> dict:
        return {
            "running": self._running,
            "recording": self._is_recording,
            "sample_rate": self.sample_rate,
            "device": self.device,
            "overflows": self._overflows,
            "truncated": self._truncated,
            "peak": self._peak,
            "seconds": self._recorded_samples / SAMPLE_RATE,
            "error": self._last_error,
        }

    # ── devices ───────────────────────────────────────────────────────────

    @staticmethod
    def list_devices() -> List[DeviceInfo]:
        out: List[DeviceInfo] = []
        try:
            default_in = sd.default.device[0]
            for idx, dev in enumerate(sd.query_devices()):
                if dev["max_input_channels"] > 0:
                    out.append(DeviceInfo(
                        index=idx,
                        name=dev["name"],
                        channels=dev["max_input_channels"],
                        sample_rate=dev["default_samplerate"],
                        is_default=(idx == default_in),
                    ))
        except Exception as e:
            logger.error("Failed to list devices: %s", e)
        return out

    def set_device(self, device: Optional[int]) -> bool:
        """Switch input device, reopening the stream."""
        was_running = self._running
        self.stop_stream()
        self.device = device
        return self.start_stream() if was_running else True

    def __del__(self):
        try:
            self.stop_stream()
        except Exception:
            pass
