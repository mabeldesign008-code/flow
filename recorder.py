import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import tempfile
import threading
import queue
import logging
from collections import deque
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioDeviceInfo:
    """Information about an audio device."""
    index: int
    name: str
    channels: int
    sample_rate: float
    is_default: bool


class AudioRecorder:
    """
    Enhanced audio recorder with robust error handling, adaptive noise detection,
    device selection, and thread-safe operations.
    """
    
    # Constants
    MAX_RECORDING_DURATION = 36000  # 10 hours max
    MAX_BUFFER_SIZE = 0  # 0 means infinite size; prevents queue full errors on long recordings
    ADAPTIVE_WINDOW_SIZE = 50  # Frames for adaptive threshold calculation
    DC_OFFSET_FILTER_FREQ = 80  # Hz - high-pass filter cutoff
    CLIPPING_THRESHOLD = 0.95  # Detect clipping above this level
    
    def __init__(
        self,
        sample_rate: Optional[int] = None,
        silence_threshold: float = 0.01,
        min_duration: float = 0.3,
        device: Optional[int] = None,
        channels: int = 1,
        adaptive_threshold: bool = True,
        noise_reduction: bool = True,
    ):
        """
        Initialize the audio recorder.
        
        Args:
            sample_rate: Sample rate in Hz. If None, uses device default.
            silence_threshold: Initial silence threshold (RMS).
            min_duration: Minimum recording duration in seconds.
            device: Audio device index. If None, uses system default.
            channels: Number of audio channels (1=mono, 2=stereo).
            adaptive_threshold: Enable adaptive silence threshold.
            noise_reduction: Enable basic noise reduction (DC offset removal, high-pass).
        """
        self.device = device
        self.channels = channels
        self.adaptive_threshold_enabled = adaptive_threshold
        self.noise_reduction_enabled = noise_reduction
        self.silence_threshold = silence_threshold
        self.min_duration = min_duration
        
        # Validate and set sample rate
        self.sample_rate = self._validate_sample_rate(sample_rate)
        
        # Thread-safe state management
        self._lock = threading.RLock()  # Reentrant lock for nested locking
        self._callback_lock = threading.Lock()  # Separate lock for callback
        self._recording_queue = queue.Queue(maxsize=self.MAX_BUFFER_SIZE)
        self._is_recording = False
        self._stream = None
        self._current_level = 0.0
        self._error_count = 0
        self._last_error = None
        
        # Adaptive threshold tracking
        self._level_history = deque(maxlen=self.ADAPTIVE_WINDOW_SIZE)
        self._adaptive_threshold = silence_threshold
        
        # Quality monitoring
        self._clipping_detected = False
        self._buffer_overflow_count = 0
        self._total_frames_processed = 0
        
        # Validate device
        self._validate_device()
        
        logger.info(
            f"AudioRecorder initialized: device={self.device}, "
            f"sample_rate={self.sample_rate}, channels={self.channels}"
        )
    
    def _validate_sample_rate(self, sample_rate: Optional[int]) -> int:
        """Validate and return appropriate sample rate."""
        if sample_rate is not None:
            if sample_rate < 8000 or sample_rate > 48000:
                logger.warning(f"Unusual sample rate {sample_rate}, using 16000")
                return 16000
            return sample_rate
        
        # Try to get device default sample rate
        try:
            device_info = sd.query_devices(self.device, 'input')
            default_sr = int(device_info['default_samplerate'])
            logger.info(f"Using device default sample rate: {default_sr}")
            return default_sr
        except Exception as e:
            logger.warning(f"Could not query device sample rate: {e}, using 16000")
            return 16000
    
    def _validate_device(self):
        """Validate the selected audio device."""
        try:
            device_info = sd.query_devices(self.device, 'input')
            if device_info['max_input_channels'] < self.channels:
                logger.warning(
                    f"Device only supports {device_info['max_input_channels']} channels, "
                    f"requested {self.channels}. Adjusting to available channels."
                )
                self.channels = min(self.channels, device_info['max_input_channels'])
        except Exception as e:
            logger.error(f"Device validation failed: {e}")
            raise ValueError(f"Invalid audio device: {e}")
    
    @staticmethod
    def list_devices() -> List[AudioDeviceInfo]:
        """List all available input audio devices."""
        devices = []
        try:
            device_list = sd.query_devices()
            default_input = sd.default.device[0]
            
            for idx, dev in enumerate(device_list):
                if dev['max_input_channels'] > 0:
                    devices.append(AudioDeviceInfo(
                        index=idx,
                        name=dev['name'],
                        channels=dev['max_input_channels'],
                        sample_rate=dev['default_samplerate'],
                        is_default=(idx == default_input)
                    ))
        except Exception as e:
            logger.error(f"Failed to list devices: {e}")
        
        return devices
    
    def start(self) -> bool:
        """
        Start recording audio.
        
        Returns:
            True if recording started successfully, False otherwise.
        """
        with self._lock:
            if self._is_recording:
                logger.warning("Recording already in progress")
                return False
            
            try:
                # Clear previous state
                while not self._recording_queue.empty():
                    try:
                        self._recording_queue.get_nowait()
                    except queue.Empty:
                        break
                
                self._error_count = 0
                self._last_error = None
                self._clipping_detected = False
                self._buffer_overflow_count = 0
                self._total_frames_processed = 0
                self._level_history.clear()
                
                # Create stream
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype='float32',
                    device=self.device,
                    callback=self._callback,
                    blocksize=0,  # Use optimal block size
                )
                
                self._stream.start()
                self._is_recording = True
                logger.info("Recording started")
                return True
                
            except Exception as e:
                logger.error(f"Failed to start recording: {e}")
                self._last_error = str(e)
                self._stream = None
                return False
    
    def stop(self) -> Optional[Tuple[np.ndarray, int]]:
        """
        Stop recording and return the audio data.
        
        Returns:
            Tuple of (audio_array, sample_rate) or None if no valid audio.
        """
        with self._lock:
            if not self._is_recording:
                logger.warning("No recording in progress")
                return None
            
            self._is_recording = False
            
            # Stop and close stream
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.error(f"Error stopping stream: {e}")
                finally:
                    self._stream = None
            
            # Collect all audio chunks
            chunks = []
            while not self._recording_queue.empty():
                try:
                    chunk = self._recording_queue.get_nowait()
                    chunks.append(chunk)
                except queue.Empty:
                    break
            
            if not chunks:
                logger.warning("No audio data recorded")
                return None
            
            # Concatenate chunks
            try:
                audio = np.concatenate(chunks, axis=0)
                
                # Convert stereo to mono if needed
                if audio.ndim > 1 and audio.shape[1] > 1:
                    audio = np.mean(audio, axis=1)
                
                audio = audio.flatten()
                
            except Exception as e:
                logger.error(f"Failed to process audio chunks: {e}")
                return None
        
        # Process audio (outside lock to avoid blocking)
        return self._process_audio(audio)
    
    def _process_audio(self, audio: np.ndarray) -> Optional[Tuple[np.ndarray, int]]:
        """Process recorded audio with validation and enhancement."""
        
        # Duration check
        duration = len(audio) / self.sample_rate
        if duration < self.min_duration:
            logger.info(f"Recording too short: {duration:.2f}s < {self.min_duration}s")
            return None
        
        if duration > self.MAX_RECORDING_DURATION:
            logger.warning(f"Recording truncated from {duration:.2f}s to {self.MAX_RECORDING_DURATION}s")
            audio = audio[:int(self.MAX_RECORDING_DURATION * self.sample_rate)]
        
        # Apply noise reduction if enabled
        if self.noise_reduction_enabled:
            audio = self._apply_noise_reduction(audio)
        
        # Trim silence
        audio = self._trim_silence(audio)
        if audio is None or len(audio) < int(self.sample_rate * self.min_duration):
            logger.info("Audio is all silence after trimming")
            return None
        
        # Quality validation
        peak = np.max(np.abs(audio))
        if peak < 1e-6:
            logger.warning("Audio signal too weak")
            return None
        
        if peak > self.CLIPPING_THRESHOLD:
            logger.warning(f"Clipping detected (peak={peak:.3f})")
            self._clipping_detected = True
        
        # Normalize with headroom
        audio = audio * (0.9 / peak)
        
        logger.info(
            f"Recording processed: duration={len(audio)/self.sample_rate:.2f}s, "
            f"peak={peak:.3f}, clipping={self._clipping_detected}"
        )
        
        return audio, self.sample_rate
    
    def _apply_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """Apply basic noise reduction (DC offset removal and high-pass filter)."""
        try:
            # Remove DC offset
            audio = audio - np.mean(audio)
            
            # High-pass filter to remove low-frequency rumble
            nyquist = self.sample_rate / 2
            cutoff_normalized = self.DC_OFFSET_FILTER_FREQ / nyquist
            
            if cutoff_normalized < 1.0:
                b, a = signal.butter(2, cutoff_normalized, btype='high')
                audio = signal.filtfilt(b, a, audio)
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
        
        return audio
    
    def _trim_silence(self, audio: np.ndarray, frame_len: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Trim silence from start and end of recording.
        
        Uses adaptive threshold if enabled.
        """
        if len(audio) == 0:
            return None
        
        if frame_len is None:
            frame_len = int(self.sample_rate * 0.1)  # 100ms frames
        
        n_frames = len(audio) // frame_len
        if n_frames == 0:
            return audio
        
        # Calculate frame energies (RMS)
        energies = []
        for i in range(n_frames):
            frame = audio[i * frame_len : (i + 1) * frame_len]
            rms = np.sqrt(np.mean(frame ** 2))
            energies.append(rms)
        
        # Determine threshold
        if self.adaptive_threshold_enabled and len(energies) > 10:
            # Use median of lowest 30% as noise floor
            sorted_energies = sorted(energies)
            noise_floor = np.median(sorted_energies[:max(1, len(sorted_energies) // 3)])
            threshold = noise_floor * 3  # 3x noise floor
            threshold = max(threshold, self.silence_threshold)  # Don't go below minimum
        else:
            threshold = self.silence_threshold
        
        # Check if all silence
        if all(e <= threshold for e in energies):
            return None
        
        # Find speech boundaries (with 1-frame padding)
        start_frame = 0
        for i, e in enumerate(energies):
            if e > threshold:
                start_frame = max(0, i - 1)
                break
        
        end_frame = n_frames - 1
        for i in range(n_frames - 1, -1, -1):
            if energies[i] > threshold:
                end_frame = min(n_frames - 1, i + 1)
                break
        
        # Extract audio
        start_sample = start_frame * frame_len
        end_sample = min((end_frame + 1) * frame_len, len(audio))
        
        return audio[start_sample:end_sample]
    
    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        """
        Audio stream callback. Runs in separate thread.
        
        Must be fast and thread-safe.
        """
        # Log errors/warnings
        if status:
            with self._callback_lock:
                self._error_count += 1
                self._last_error = str(status)
            
            if status.input_overflow:
                self._buffer_overflow_count += 1
                logger.warning(f"Buffer overflow detected (count={self._buffer_overflow_count})")
            else:
                logger.warning(f"Audio callback status: {status}")
        
        # Store audio data
        try:
            self._recording_queue.put_nowait(indata.copy())
            self._total_frames_processed += frames
        except queue.Full:
            self._buffer_overflow_count += 1
            logger.error("Recording queue full - dropping audio!")
            return
        
        # Update level (thread-safe)
        try:
            rms = float(np.sqrt(np.mean(indata ** 2)))
            with self._callback_lock:
                self._current_level = rms
                self._level_history.append(rms)
                
                # Update adaptive threshold
                if self.adaptive_threshold_enabled and len(self._level_history) >= 10:
                    sorted_levels = sorted(self._level_history)
                    noise_floor = np.median(sorted_levels[:len(sorted_levels) // 3])
                    self._adaptive_threshold = max(noise_floor * 3, self.silence_threshold)
        except Exception as e:
            logger.error(f"Error in callback level calculation: {e}")
    
    def get_level(self) -> float:
        """Get current audio input level (RMS)."""
        with self._callback_lock:
            return self._current_level
    
    def get_adaptive_threshold(self) -> float:
        """Get current adaptive silence threshold."""
        with self._callback_lock:
            return self._adaptive_threshold
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        with self._lock:
            return self._is_recording
    
    def get_stats(self) -> Dict:
        """Get recording statistics and diagnostics."""
        with self._lock:
            with self._callback_lock:
                return {
                    'is_recording': self._is_recording,
                    'sample_rate': self.sample_rate,
                    'channels': self.channels,
                    'device': self.device,
                    'error_count': self._error_count,
                    'last_error': self._last_error,
                    'buffer_overflows': self._buffer_overflow_count,
                    'frames_processed': self._total_frames_processed,
                    'current_level': self._current_level,
                    'adaptive_threshold': self._adaptive_threshold,
                    'clipping_detected': self._clipping_detected,
                    'queue_size': self._recording_queue.qsize(),
                }
    
    def stop_to_file(self) -> Optional[str]:
        """
        Stop recording and save to temporary WAV file.
        
        Returns:
            Path to temporary WAV file or None.
        """
        result = self.stop()
        if result is None:
            return None
        
        audio, sr = result
        
        try:
            # Convert to int16
            audio_int16 = np.int16(audio * 32767)
            
            # Create temp file
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            wav.write(tmp.name, sr, audio_int16)
            
            logger.info(f"Audio saved to {tmp.name}")
            return tmp.name
            
        except Exception as e:
            logger.error(f"Failed to save audio file: {e}")
            return None
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            if self._is_recording:
                self.stop()
        except Exception:
            pass
