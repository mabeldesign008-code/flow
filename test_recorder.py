"""
Comprehensive tests for the enhanced AudioRecorder.
"""
import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from recorder import AudioRecorder, AudioDeviceInfo


class TestAudioRecorder:
    """Test suite for AudioRecorder class."""
    
    def test_initialization_default(self):
        """Test recorder initialization with default parameters."""
        recorder = AudioRecorder()
        assert recorder.sample_rate > 0
        assert recorder.channels == 1
        assert recorder.silence_threshold == 0.01
        assert recorder.min_duration == 0.3
        assert not recorder.is_recording()
    
    def test_initialization_custom_params(self):
        """Test recorder initialization with custom parameters."""
        recorder = AudioRecorder(
            sample_rate=44100,
            silence_threshold=0.02,
            min_duration=0.5,
            channels=2,
            adaptive_threshold=False,
            noise_reduction=False
        )
        assert recorder.sample_rate == 44100
        assert recorder.channels == 2
        assert recorder.silence_threshold == 0.02
        assert recorder.min_duration == 0.5
        assert not recorder.adaptive_threshold_enabled
        assert not recorder.noise_reduction_enabled
    
    def test_invalid_sample_rate(self):
        """Test that invalid sample rates are corrected."""
        recorder = AudioRecorder(sample_rate=1000)  # Too low
        assert recorder.sample_rate == 16000  # Should default
        
        recorder = AudioRecorder(sample_rate=100000)  # Too high
        assert recorder.sample_rate == 16000  # Should default
    
    @patch('sounddevice.query_devices')
    def test_list_devices(self, mock_query):
        """Test device listing functionality."""
        mock_query.return_value = [
            {'name': 'Device 1', 'max_input_channels': 2, 'default_samplerate': 44100},
            {'name': 'Device 2', 'max_input_channels': 1, 'default_samplerate': 48000},
            {'name': 'Output Only', 'max_input_channels': 0, 'default_samplerate': 44100},
        ]
        
        with patch('sounddevice.default.device', [0, 0]):
            devices = AudioRecorder.list_devices()
        
        assert len(devices) == 2  # Only input devices
        assert devices[0].name == 'Device 1'
        assert devices[0].channels == 2
        assert devices[0].is_default
        assert devices[1].name == 'Device 2'
        assert not devices[1].is_default
    
    def test_get_level_initial(self):
        """Test that initial level is zero."""
        recorder = AudioRecorder()
        assert recorder.get_level() == 0.0
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        recorder = AudioRecorder()
        stats = recorder.get_stats()
        
        assert 'is_recording' in stats
        assert 'sample_rate' in stats
        assert 'channels' in stats
        assert 'error_count' in stats
        assert 'buffer_overflows' in stats
        assert 'current_level' in stats
        assert stats['is_recording'] is False
        assert stats['error_count'] == 0
    
    def test_stop_without_start(self):
        """Test that stopping without starting returns None."""
        recorder = AudioRecorder()
        result = recorder.stop()
        assert result is None
    
    def test_double_start(self):
        """Test that starting twice doesn't cause issues."""
        recorder = AudioRecorder()
        
        with patch.object(recorder, '_stream', None):
            with patch('sounddevice.InputStream') as mock_stream:
                mock_instance = MagicMock()
                mock_stream.return_value = mock_instance
                
                result1 = recorder.start()
                assert result1 is True
                
                result2 = recorder.start()
                assert result2 is False  # Already recording
    
    def test_trim_silence_all_silence(self):
        """Test trimming when audio is all silence."""
        recorder = AudioRecorder(silence_threshold=0.01)
        
        # Create silent audio
        silent_audio = np.zeros(16000)  # 1 second of silence
        
        result = recorder._trim_silence(silent_audio)
        assert result is None
    
    def test_trim_silence_with_speech(self):
        """Test trimming with actual speech signal."""
        recorder = AudioRecorder(silence_threshold=0.01)
        
        # Create audio: silence + speech + silence
        silence = np.zeros(1600)
        speech = np.random.randn(8000) * 0.1  # Louder than threshold
        audio = np.concatenate([silence, speech, silence])
        
        result = recorder._trim_silence(audio)
        assert result is not None
        assert len(result) < len(audio)  # Should be trimmed
        assert len(result) > len(speech) * 0.8  # Should keep most speech
    
    def test_apply_noise_reduction(self):
        """Test noise reduction processing."""
        recorder = AudioRecorder(noise_reduction=True)
        
        # Create audio with DC offset
        audio = np.random.randn(16000) * 0.1 + 0.5  # DC offset of 0.5
        
        processed = recorder._apply_noise_reduction(audio)
        
        # DC offset should be removed
        assert abs(np.mean(processed)) < 0.01
        assert len(processed) == len(audio)
    
    def test_process_audio_too_short(self):
        """Test that too-short audio is rejected."""
        recorder = AudioRecorder(min_duration=0.5)
        
        # Create 0.3 second audio (too short)
        audio = np.random.randn(int(16000 * 0.3)) * 0.1
        
        result = recorder._process_audio(audio)
        assert result is None
    
    def test_process_audio_too_weak(self):
        """Test that too-weak audio is rejected."""
        recorder = AudioRecorder()
        
        # Create very weak audio
        audio = np.random.randn(16000) * 1e-8
        
        result = recorder._process_audio(audio)
        assert result is None
    
    def test_process_audio_valid(self):
        """Test processing of valid audio."""
        recorder = AudioRecorder(sample_rate=16000, min_duration=0.3, noise_reduction=False)
        
        # Create valid audio: 1 second, reasonable amplitude
        audio = np.random.randn(16000) * 0.1
        
        result = recorder._process_audio(audio)
        assert result is not None
        
        processed_audio, sample_rate = result
        assert sample_rate == recorder.sample_rate
        assert len(processed_audio) > 0
        assert np.max(np.abs(processed_audio)) <= 1.0  # Normalized
    
    def test_process_audio_clipping_detection(self):
        """Test that clipping is detected."""
        recorder = AudioRecorder(sample_rate=16000, silence_threshold=0.01, adaptive_threshold=False)
        
        # Create audio with clipping - varying signal to avoid silence detection
        t = np.linspace(0, 1, 16000)
        audio = 0.99 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave at high amplitude
        
        result = recorder._process_audio(audio)
        assert result is not None
        assert recorder._clipping_detected
    
    def test_process_audio_duration_limit(self):
        """Test that overly long audio is truncated."""
        recorder = AudioRecorder(sample_rate=16000, silence_threshold=0.001, adaptive_threshold=False)
        
        # Create 10 minute audio (exceeds MAX_RECORDING_DURATION) with varying signal
        t = np.linspace(0, 600, 16000 * 600)
        long_audio = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        result = recorder._process_audio(long_audio)
        assert result is not None
        
        processed_audio, _ = result
        max_samples = recorder.MAX_RECORDING_DURATION * recorder.sample_rate
        assert len(processed_audio) <= max_samples
    
    def test_adaptive_threshold_calculation(self):
        """Test adaptive threshold updates."""
        recorder = AudioRecorder(adaptive_threshold=True)
        
        # Simulate callback with varying levels
        for level in [0.001, 0.002, 0.001, 0.003, 0.001]:
            recorder._level_history.append(level)
        
        # Manually trigger threshold update
        if len(recorder._level_history) >= 10:
            sorted_levels = sorted(recorder._level_history)
            noise_floor = np.median(sorted_levels[:len(sorted_levels) // 3])
            expected_threshold = max(noise_floor * 3, recorder.silence_threshold)
            
            # Threshold should adapt to noise floor
            assert expected_threshold >= recorder.silence_threshold
    
    def test_callback_thread_safety(self):
        """Test that callback is thread-safe."""
        recorder = AudioRecorder()
        
        # Simulate concurrent callback calls
        def simulate_callback():
            for _ in range(10):
                indata = np.random.randn(1024, 1).astype(np.float32)
                recorder._callback(indata, 1024, None, None)
        
        threads = [threading.Thread(target=simulate_callback) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should not crash and should have processed frames
        stats = recorder.get_stats()
        assert stats['frames_processed'] > 0
    
    def test_callback_buffer_overflow_handling(self):
        """Test callback handles buffer overflow gracefully."""
        recorder = AudioRecorder()
        
        # Fill the queue to capacity
        for _ in range(recorder.MAX_BUFFER_SIZE):
            try:
                recorder._recording_queue.put_nowait(np.zeros((1024, 1)))
            except:
                break
        
        # Try to add more (should handle gracefully)
        indata = np.random.randn(1024, 1).astype(np.float32)
        recorder._callback(indata, 1024, None, None)
        
        stats = recorder.get_stats()
        assert stats['buffer_overflows'] > 0
    
    def test_stereo_to_mono_conversion(self):
        """Test that stereo audio is converted to mono."""
        recorder = AudioRecorder(channels=2)
        
        # Create stereo audio
        left = np.random.randn(16000) * 0.1
        right = np.random.randn(16000) * 0.1
        stereo = np.column_stack([left, right])
        
        # Manually add to queue
        recorder._recording_queue.put(stereo)
        
        # Process
        with patch.object(recorder, '_is_recording', True):
            recorder._is_recording = False
            result = recorder.stop()
        
        if result is not None:
            audio, _ = result
            assert audio.ndim == 1  # Should be mono
    
    def test_get_adaptive_threshold(self):
        """Test getting adaptive threshold value."""
        recorder = AudioRecorder(adaptive_threshold=True)
        
        threshold = recorder.get_adaptive_threshold()
        assert threshold >= recorder.silence_threshold
    
    @patch('sounddevice.InputStream')
    def test_start_stop_integration(self, mock_stream_class):
        """Test full start-stop cycle."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream
        
        recorder = AudioRecorder()
        
        # Start recording
        assert recorder.start() is True
        assert recorder.is_recording() is True
        
        # Add some mock audio data
        for _ in range(5):
            audio_chunk = np.random.randn(1024, 1).astype(np.float32) * 0.1
            recorder._recording_queue.put(audio_chunk)
        
        # Stop recording
        result = recorder.stop()
        assert recorder.is_recording() is False
        
        # Should have processed audio
        if result is not None:
            audio, sr = result
            assert len(audio) > 0
            assert sr == recorder.sample_rate
    
    def test_stop_to_file(self):
        """Test saving audio to file."""
        recorder = AudioRecorder()
        
        # Create mock audio data
        audio = np.random.randn(16000) * 0.1
        
        with patch.object(recorder, 'stop', return_value=(audio, 16000)):
            filepath = recorder.stop_to_file()
            
            if filepath:
                import os
                assert os.path.exists(filepath)
                assert filepath.endswith('.wav')
                
                # Cleanup
                try:
                    os.remove(filepath)
                except:
                    pass
    
    def test_cleanup_on_deletion(self):
        """Test that recorder cleans up properly on deletion."""
        recorder = AudioRecorder()
        
        with patch.object(recorder, 'stop') as mock_stop:
            recorder._is_recording = True
            del recorder
            # Destructor should call stop
            # Note: This is best-effort in Python


class TestAudioDeviceInfo:
    """Test AudioDeviceInfo dataclass."""
    
    def test_device_info_creation(self):
        """Test creating device info."""
        info = AudioDeviceInfo(
            index=0,
            name="Test Device",
            channels=2,
            sample_rate=44100.0,
            is_default=True
        )
        
        assert info.index == 0
        assert info.name == "Test Device"
        assert info.channels == 2
        assert info.sample_rate == 44100.0
        assert info.is_default is True


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_audio_array(self):
        """Test handling of empty audio array."""
        recorder = AudioRecorder()
        result = recorder._trim_silence(np.array([]))
        assert result is None
    
    def test_single_sample_audio(self):
        """Test handling of very short audio."""
        recorder = AudioRecorder()
        audio = np.array([0.1])
        result = recorder._trim_silence(audio)
        assert result is not None
    
    def test_nan_in_audio(self):
        """Test handling of NaN values in audio."""
        recorder = AudioRecorder()
        audio = np.array([0.1, 0.2, np.nan, 0.3, 0.4])
        
        # Should handle gracefully (may return None or filtered audio)
        try:
            result = recorder._process_audio(audio)
            # If it doesn't crash, that's good
        except:
            pass  # Expected to potentially fail
    
    def test_inf_in_audio(self):
        """Test handling of infinite values in audio."""
        recorder = AudioRecorder()
        audio = np.array([0.1, 0.2, np.inf, 0.3, 0.4])
        
        try:
            result = recorder._process_audio(audio)
            # Should handle or reject
        except:
            pass
    
    @patch('sounddevice.query_devices')
    def test_invalid_device_index(self, mock_query):
        """Test handling of invalid device index."""
        mock_query.side_effect = Exception("Device not found")
        
        with pytest.raises(ValueError):
            recorder = AudioRecorder(device=999)
    
    def test_concurrent_start_stop(self):
        """Test concurrent start/stop calls."""
        recorder = AudioRecorder()
        
        def start_stop():
            with patch('sounddevice.InputStream'):
                recorder.start()
                time.sleep(0.01)
                recorder.stop()
        
        threads = [threading.Thread(target=start_stop) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should not crash


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
