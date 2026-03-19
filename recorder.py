import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import threading


class AudioRecorder:
    def __init__(self, sample_rate=16000, silence_threshold=0.01, min_duration=0.3):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.min_duration = min_duration
        self.recording = []
        self.stream = None
        self.current_level = 0.0
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            self.recording = []
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self.stream.start()

    def stop(self):
        """Stop recording. Returns (np.ndarray, sample_rate) or None."""
        with self._lock:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            if not self.recording:
                return None

            audio = np.concatenate(self.recording, axis=0).flatten()
            self.recording = []

        # Gate: reject recordings shorter than min_duration
        if len(audio) / self.sample_rate < self.min_duration:
            return None

        # Trim leading / trailing silence
        audio = self._trim_silence(audio)
        if audio is None or len(audio) < int(self.sample_rate * self.min_duration):
            return None

        # Peak-normalize to -1 dB headroom
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = audio * (0.9 / peak)

        return audio, self.sample_rate

    def stop_to_file(self):
        """Backward-compatible: stop and return a temp .wav path (used for retry cache)."""
        result = self.stop()
        if result is None:
            return None
        audio, sr = result
        audio_int16 = np.int16(audio * 32767)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.write(tmp.name, sr, audio_int16)
        return tmp.name

    def get_level(self):
        return self.current_level

    # ── internals ──────────────────────────────────────────────

    def _trim_silence(self, audio, frame_len=1600):
        """Cut silent frames from the start and end of the recording."""
        if len(audio) == 0:
            return None

        n_frames = len(audio) // frame_len
        if n_frames == 0:
            return audio

        energies = [
            np.sqrt(np.mean(audio[i * frame_len : (i + 1) * frame_len] ** 2))
            for i in range(n_frames)
        ]

        # All silence → discard
        if all(e <= self.silence_threshold for e in energies):
            return None

        start = next(
            (max(0, i - 1) for i, e in enumerate(energies) if e > self.silence_threshold),
            0,
        )
        end = next(
            (
                min(n_frames - 1, i + 1)
                for i in range(n_frames - 1, -1, -1)
                if energies[i] > self.silence_threshold
            ),
            n_frames - 1,
        )

        return audio[start * frame_len : min((end + 1) * frame_len, len(audio))]

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        self.recording.append(indata.copy())
        self.current_level = float(np.sqrt(np.mean(indata ** 2)))
